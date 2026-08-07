from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import artifact_path
from .multiplicity_identity import candidate_multiplicity_label
from .temporal_objects import (
    TemporalComponents,
    TemporalPairEvidence,
    build_object_descriptor,
    component_path_confidence,
    conditional_cell_probability,
    detect_division_edges,
    match_timepoint_objects,
    multiplicity_rank,
)
from .v2_instance_dataset import _crop
from .v2_instance_inference import decode_rle


TIMEPOINTS = ("T0", "T1", "T2")
CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}
LOW_CELL_NONCELL_RESOLUTION_THRESHOLD = 0.30


def _probability_logit(value: float) -> float:
    value = float(np.clip(value, 1e-5, 1.0 - 1e-5))
    return float(np.log(value / (1.0 - value)))


def _probability_sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


def _finite_row_value(row: pd.Series, key: str, default: float) -> float:
    """Read an optional numeric inference feature without letting NaN pass a gate."""

    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _row_flag(row: pd.Series, key: str, default: bool = True) -> bool:
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _component_temporal_outputs(
    frame: pd.DataFrame,
    nodes: list[int],
    edges: list[TemporalPairEvidence],
    settings: dict[str, Any],
    track_id: str,
) -> dict[int, dict[str, Any]]:
    """Fuse object identity, growth and later morphology without forcing a label."""

    local = frame.loc[nodes]
    timepoint_nodes = {
        timepoint: local.index[local["timepoint"].astype(str) == timepoint].tolist()
        for timepoint in TIMEPOINTS
    }
    available_timepoints = [key for key, values in timepoint_nodes.items() if values]
    continuation_edges = [edge for edge in edges if edge.kind == "continuation"]

    def pre_temporal_label(index: int) -> str:
        row = frame.loc[index]
        return str(
            row.get("v2_pre_temporal_integrated_label", row.get("integrated_label", ""))
        )

    def pre_temporal_cell(index: int) -> bool:
        row = frame.loc[index]
        return (
            pre_temporal_label(index) in CELL_LABELS
            and conditional_cell_probability(row) >= 0.50
        )

    # A one-to-many geometric hypothesis is not biological division evidence
    # unless both its parent and children were already cell-like before any
    # temporal correction.  This prevents a debris candidate's multiplicity
    # head from creating the evidence that subsequently promotes it to a cell.
    division_edges = [
        edge
        for edge in edges
        if edge.kind == "division"
        and pre_temporal_cell(edge.left)
        and pre_temporal_cell(edge.right)
    ]
    identity_score = float(np.mean([edge.identity for edge in edges])) if edges else 0.0
    foreground_quality = float(np.min([edge.foreground_quality for edge in edges])) if edges else 0.0
    static_score = float(np.mean([edge.static for edge in continuation_edges])) if continuation_edges else 0.0
    shape_score = float(np.mean([edge.tolerant_shape for edge in continuation_edges])) if continuation_edges else 0.0
    change_score = float(np.mean([1.0 - edge.static for edge in continuation_edges])) if continuation_edges else 0.0
    motion_score = float(
        np.mean([np.clip(edge.distance_px / 72.0, 0.0, 1.0) for edge in continuation_edges])
    ) if continuation_edges else 0.0

    labels = {index: pre_temporal_label(index) for index in nodes}
    growth = bool(division_edges)
    if not growth:
        for edge in continuation_edges:
            if (
                pre_temporal_cell(edge.left)
                and pre_temporal_cell(edge.right)
                and multiplicity_rank(labels[edge.right])
                > multiplicity_rank(labels[edge.left])
            ):
                growth = True
                break
    growth_score = 1.0 if division_edges else (0.72 if growth else 0.0)

    static_threshold = float(settings.get("object_static_similarity_threshold", 0.82))
    identity_threshold = float(settings.get("object_identity_threshold", 0.58))
    simple_three = all(len(timepoint_nodes[timepoint]) == 1 for timepoint in TIMEPOINTS)
    adjacent_pairs = {
        (str(frame.at[edge.left, "timepoint"]), str(frame.at[edge.right, "timepoint"])): edge
        for edge in continuation_edges
    }
    stable_adjacent = [
        adjacent_pairs.get(("T0", "T1")),
        adjacent_pairs.get(("T1", "T2")),
    ]
    areas = pd.to_numeric(local["v2_instance_area_px"], errors="coerce").fillna(0.0).clip(lower=1.0)
    area_ratio = float(areas.max() / max(float(areas.min()), 1.0))
    conditional = {index: conditional_cell_probability(frame.loc[index]) for index in nodes}
    masses = {
        index: float(frame.at[index, "cell_probability"]) + float(frame.at[index, "debris_probability"])
        for index in nodes
    }
    consensus_weights = {
        index: 0.50
        + 0.50
        * max(
            float(frame.at[index, "cell_probability"]),
            float(frame.at[index, "debris_probability"]),
            float(frame.at[index, "invalid_probability"]),
        )
        for index in nodes
    }
    morphology_consensus_cell = float(
        np.average(
            [conditional[index] for index in nodes],
            weights=[consensus_weights[index] for index in nodes],
        )
    )
    conditional_debris = {
        index: float(
            frame.at[index, "debris_probability"]
            / max(
                float(frame.at[index, "cell_probability"])
                + float(frame.at[index, "debris_probability"]),
                1e-6,
            )
        )
        for index in nodes
    }
    cell_anchor_threshold = float(settings.get("temporal_cell_anchor_threshold", 0.80))
    cell_anchor_maximum_invalid = float(
        settings.get("temporal_cell_anchor_maximum_invalid_probability", 0.20)
    )
    cell_anchor_minimum_instance_confidence = float(
        settings.get("temporal_cell_anchor_minimum_instance_confidence", 0.55)
    )
    cell_anchor_minimum_objectness = float(
        settings.get("temporal_cell_anchor_minimum_objectness", 0.70)
    )
    cell_anchor_maximum_wall_overlap = float(
        settings.get("temporal_cell_anchor_maximum_wall_overlap", 0.85)
    )
    high_cell_nodes = []
    for index in nodes:
        row = frame.loc[index]
        if (
            conditional[index] >= cell_anchor_threshold
            and masses[index] >= 0.45
            and _finite_row_value(row, "invalid_probability", 1.0)
            <= cell_anchor_maximum_invalid
            and _finite_row_value(row, "v2_instance_confidence", 1.0)
            >= cell_anchor_minimum_instance_confidence
            and _finite_row_value(row, "v2_objectness", 1.0)
            >= cell_anchor_minimum_objectness
            and _finite_row_value(row, "v2_wall_overlap", 0.0)
            <= cell_anchor_maximum_wall_overlap
            and _row_flag(row, "v2_is_unique_instance")
            and _row_flag(row, "v2_is_counting_instance")
        ):
            high_cell_nodes.append(index)
    high_debris_nodes = [
        index for index in nodes
        if conditional[index] <= 1.0 - float(settings.get("temporal_debris_anchor_threshold", 0.90))
        and masses[index] >= 0.45
        and float(frame.at[index, "invalid_probability"]) < 0.60
    ]
    minimum_pair_static = float(
        settings.get(
            "object_static_minimum_pair_similarity",
            max(0.0, static_threshold - 0.10),
        )
    )
    minimum_static_shape = float(
        settings.get("object_static_minimum_shape_similarity", 0.90)
    )
    three_frame_static = bool(
        simple_three
        and all(edge is not None for edge in stable_adjacent)
        and all(edge.identity >= identity_threshold for edge in stable_adjacent if edge is not None)
        # One frame can have a different focus/halo while the complete object
        # remains recognisably unchanged.  Require a strong triplet mean plus
        # a per-pair floor and stable shape, rather than demanding that both
        # raw appearance pairs independently clear the same hard threshold.
        and static_score >= static_threshold
        and all(
            edge.static >= minimum_pair_static
            for edge in stable_adjacent
            if edge is not None
        )
        and shape_score >= minimum_static_shape
        and all(edge.foreground_quality >= float(settings.get("minimum_foreground_quality", 0.30)) for edge in stable_adjacent if edge is not None)
        and area_ratio <= float(settings.get("static_maximum_area_ratio", 1.65))
        and not growth
    )
    two_frame_static = bool(
        not three_frame_static
        and len(available_timepoints) >= 2
        and any(
            edge.identity >= identity_threshold
            and edge.static >= static_threshold
            and edge.foreground_quality >= float(settings.get("minimum_foreground_quality", 0.30))
            for edge in continuation_edges
        )
        and not growth
    )
    ordered_three_frame_nodes = [
        timepoint_nodes[timepoint][0]
        for timepoint in TIMEPOINTS
        if len(timepoint_nodes[timepoint]) == 1
    ]
    debris_sequence = [
        conditional_debris[index] for index in ordered_three_frame_nodes
    ]
    debris_trend_delta = (
        float(debris_sequence[-1] - debris_sequence[0])
        if len(debris_sequence) == 3
        else 0.0
    )
    debris_probability_trend = bool(
        simple_three
        and len(debris_sequence) == 3
        and all(edge is not None for edge in stable_adjacent)
        and all(
            edge.identity >= identity_threshold
            and edge.foreground_quality
            >= float(settings.get("minimum_foreground_quality", 0.30))
            for edge in stable_adjacent
            if edge is not None
        )
        and static_score
        >= float(settings.get("debris_trend_minimum_static_similarity", 0.58))
        and shape_score
        >= float(settings.get("debris_trend_minimum_shape_similarity", 0.78))
        and debris_sequence[1] >= debris_sequence[0] - 0.05
        and debris_sequence[2] >= debris_sequence[1] - 0.05
        and debris_trend_delta
        >= float(settings.get("debris_trend_minimum_probability_rise", 0.18))
        and debris_sequence[-1]
        >= float(settings.get("debris_trend_minimum_final_probability", 0.48))
        and not growth
    )
    debris_trend_score = float(
        np.clip(
            debris_trend_delta
            / max(
                float(settings.get("debris_trend_full_strength_rise", 0.35)),
                1e-6,
            ),
            0.0,
            1.0,
        )
        * identity_score
        if debris_probability_trend
        else 0.0
    )

    wall_overlap = pd.to_numeric(local.get("v2_wall_overlap", 0.0), errors="coerce").fillna(0.0)
    maximum_motion = max((edge.distance_px for edge in continuation_edges), default=np.inf)
    static_wall = bool(
        three_frame_static
        and float(wall_overlap.min()) >= float(settings.get("static_wall_overlap_threshold", 0.90))
        and maximum_motion <= float(settings.get("static_wall_maximum_motion_px", 5.0))
        and area_ratio <= float(settings.get("static_wall_maximum_area_ratio", 2.20))
    )

    suspected_dead_morphology_threshold = float(
        settings.get("suspected_dead_cell_morphology_threshold", 0.80)
    )
    suspected_dead_cell = bool(
        three_frame_static
        and all(pre_temporal_cell(index) for index in nodes)
        and morphology_consensus_cell >= suspected_dead_morphology_threshold
    )
    suspected_dead_cell_score = float(
        identity_score * static_score * morphology_consensus_cell
        if suspected_dead_cell
        else 0.0
    )
    strong_static_debris = bool(three_frame_static and not suspected_dead_cell)

    outputs: dict[int, dict[str, Any]] = {}
    for index in nodes:
        row = frame.loc[index]
        # Preserve the exact unary label unless temporal evidence actually
        # decides that this object is a cell. candidate_multiplicity_label()
        # always returns a cell subtype and must never be used as a generic
        # no-op fallback for debris/uncertain candidates.
        original_label = str(row["integrated_label"])
        cell = float(row["cell_probability"])
        debris = float(row["debris_probability"])
        invalid = float(row["invalid_probability"])
        unary_debris_dominant = bool(
            original_label in CELL_LABELS
            and debris > cell
            and debris >= invalid
        )
        if unary_debris_dominant:
            original_label = "debris"
        mass = cell + debris
        base_cell = conditional[index]
        base_confidence = max(base_cell, 1.0 - base_cell)
        common = {
            "same": identity_score,
            "static": static_score,
            "shape": shape_score,
            "change": change_score,
            "growth": growth_score,
            "foreground_quality": foreground_quality,
            "cell": cell,
            "debris": debris,
            "boost": 0.0,
            "cell_boost": 0.0,
            "applied": False,
            "label": original_label,
            "static_wall": False,
            "track_id": track_id,
            # This field is shown as matched frames in the review UI.  A
            # component may contain more than one child after division, so the
            # number of graph nodes is not the number of available frames.
            "proposal_count": len(available_timepoints),
            "frame_count": len(available_timepoints),
            "pair_count": len(edges),
            "three_frame_static": three_frame_static,
            "morphology_consensus_cell": morphology_consensus_cell,
            "suspected_dead_cell": suspected_dead_cell,
            "suspected_dead_cell_score": suspected_dead_cell_score,
            "debris_probability_trend": debris_probability_trend,
            "debris_trend_score": debris_trend_score,
        }
        if static_wall:
            outputs[index] = {
                **common,
                "label": "invalid",
                "static_wall": True,
                "reason": "static_wall_artifact",
            }
            continue
        if len(available_timepoints) < 2 or not edges:
            outputs[index] = {
                **common,
                "reason": (
                    "unary_debris_probability_dominant"
                    if unary_debris_dominant
                    else "insufficient_parallel_candidates"
                ),
            }
            continue
        # Temporal evidence resolves cell-vs-debris ambiguity only.  A row
        # already rejected by the instance stage must never be revived merely
        # because the underlying background artifact is static across frames.
        if original_label == "invalid" or invalid >= 0.60:
            outputs[index] = {**common, "reason": "invalid_candidate"}
            continue
        if suspected_dead_cell:
            outputs[index] = {**common, "reason": "suspected_dead_cell"}
            continue
        # Usually a confident unary decision is an anchor and is preserved.
        # A complete three-frame static object with non-cell-like consensus is
        # the deliberate exception: the track consensus must be evaluated
        # before an isolated confident frame can short-circuit it.
        if (
            base_confidence
            >= float(settings.get("base_high_confidence_threshold", 0.90))
            and not strong_static_debris
            and not debris_probability_trend
        ):
            outputs[index] = {**common, "reason": "high_confidence_base"}
            continue
        ambiguity_minimum = float(settings.get("temporal_ambiguity_minimum_cell_probability", 0.25))
        ambiguity_maximum = float(settings.get("temporal_ambiguity_maximum_cell_probability", 0.75))
        if (
            not (ambiguity_minimum <= base_cell <= ambiguity_maximum)
            and not strong_static_debris
            and not debris_probability_trend
        ):
            outputs[index] = {**common, "reason": "outside_temporal_ambiguity_band"}
            continue
        if mass <= 1e-6:
            outputs[index] = {**common, "reason": "no_cell_debris_mass"}
            continue

        minimum_cell_anchor_path_confidence = float(
            settings.get("temporal_cell_anchor_minimum_path_confidence", 0.72)
        )
        cell_anchor_values = []
        for anchor in high_cell_nodes:
            if anchor == index:
                continue
            path_confidence = component_path_confidence(index, anchor, edges)
            if path_confidence < minimum_cell_anchor_path_confidence:
                continue
            cell_anchor_values.append(
                path_confidence
                * float(np.clip((conditional[anchor] - 0.5) * 2.0, 0.0, 1.0))
            )
        cell_anchor = max(cell_anchor_values, default=0.0)
        debris_anchor = max(
            (
                component_path_confidence(index, anchor, edges)
                * float(np.clip((0.5 - conditional[anchor]) * 2.0, 0.0, 1.0))
                for anchor in high_debris_nodes
                if anchor != index
            ),
            default=0.0,
        )
        cell_evidence = 0.0
        debris_evidence = 0.0
        reason = "no_decisive_object_evidence"
        if growth:
            cell_evidence += float(settings.get("growth_cell_logit", 3.20)) * growth_score
            reason = "division_or_growth_cell_evidence"
        if cell_anchor > 0 and not strong_static_debris:
            cell_evidence += float(settings.get("cell_anchor_logit", 3.10)) * cell_anchor
            if not growth:
                reason = "high_confidence_cell_anchor"
        if strong_static_debris:
            debris_evidence += float(settings.get("three_frame_static_debris_logit", 2.60)) * max(static_score, 0.80)
            reason = "three_frame_static_debris_consensus"
        elif two_frame_static:
            debris_evidence += float(settings.get("two_frame_static_debris_logit", 0.35)) * max(static_score, 0.75)
            reason = "two_frame_static_object"
        if debris_probability_trend:
            debris_evidence += float(
                settings.get("debris_probability_trend_logit", 2.15)
            ) * debris_trend_score
            reason = "multi_frame_debris_probability_trend"
        if debris_anchor > 0:
            debris_evidence += float(settings.get("debris_anchor_logit", 2.20)) * debris_anchor
            if not three_frame_static:
                reason = "high_confidence_debris_anchor"
        consensus_low = float(settings.get("temporal_debris_consensus_threshold", 0.35))
        consensus_high = float(settings.get("temporal_cell_consensus_threshold", 0.65))
        if morphology_consensus_cell <= consensus_low:
            consensus_strength = np.clip(
                (consensus_low - morphology_consensus_cell) / max(consensus_low, 1e-6),
                0.0,
                1.0,
            )
            debris_evidence += float(settings.get("debris_consensus_logit", 1.30)) * (0.55 + 0.45 * consensus_strength)
            if reason == "no_decisive_object_evidence":
                reason = "multi_frame_debris_consensus"
        elif (
            morphology_consensus_cell >= consensus_high
            and not strong_static_debris
        ):
            consensus_strength = np.clip(
                (morphology_consensus_cell - consensus_high) / max(1.0 - consensus_high, 1e-6),
                0.0,
                1.0,
            )
            cell_evidence += float(settings.get("cell_consensus_logit", 1.30)) * (0.55 + 0.45 * consensus_strength)
            if reason == "no_decisive_object_evidence":
                reason = "multi_frame_cell_consensus"
        # Motion/change is intentionally weak: debris can drift and masks can
        # fluctuate. It matters only after a reliable object identity exists.
        if not growth and not three_frame_static and identity_score >= identity_threshold:
            weak_change = identity_score * (0.70 * change_score + 0.30 * motion_score)
            if weak_change >= 0.35:
                cell_evidence += float(settings.get("change_cell_logit", 0.55)) * weak_change
                if reason == "no_decisive_object_evidence":
                    reason = "object_change_weak_cell_evidence"

        if (
            not strong_static_debris
            and cell_evidence > 0
            and debris_evidence > 0
            and abs(cell_evidence - debris_evidence) < 0.45
        ):
            outputs[index] = {**common, "reason": "conflicting_temporal_evidence"}
            continue
        net_evidence = cell_evidence - debris_evidence
        if abs(net_evidence) < 0.30:
            outputs[index] = {**common, "reason": "no_decisive_object_evidence"}
            continue
        # Report the evidence that actually determined the direction.  Static
        # evidence and a cell anchor may coexist; two-frame stability is weak
        # and must not hide a stronger, well-matched cell anchor in the UI.
        if net_evidence > 0:
            if growth:
                reason = "division_or_growth_cell_evidence"
            elif cell_anchor > 0:
                reason = "high_confidence_cell_anchor"
            elif morphology_consensus_cell >= consensus_high:
                reason = "multi_frame_cell_consensus"
            else:
                reason = "object_change_weak_cell_evidence"
        else:
            if strong_static_debris:
                reason = "three_frame_static_debris_consensus"
            elif debris_probability_trend:
                reason = "multi_frame_debris_probability_trend"
            elif debris_anchor > 0:
                reason = "high_confidence_debris_anchor"
            elif morphology_consensus_cell <= consensus_low:
                reason = "multi_frame_debris_consensus"
            else:
                reason = "two_frame_static_object"
        target_cell = _probability_sigmoid(_probability_logit(base_cell) + net_evidence)
        if net_evidence > 0:
            maximum_probability_shift = (
                0.45 if growth else 0.34 if cell_anchor > 0 else 0.12
            )
            conditional_shift = min(max(target_cell - base_cell, 0.0), maximum_probability_shift)
            target_cell = base_cell + conditional_shift
        else:
            maximum_probability_shift = (
                0.55
                if strong_static_debris
                else 0.34
                if debris_probability_trend
                else 0.30
                if debris_anchor > 0
                else 0.15
            )
            conditional_shift = min(max(base_cell - target_cell, 0.0), maximum_probability_shift)
            target_cell = base_cell - conditional_shift
        if strong_static_debris:
            target_cell = min(
                target_cell,
                1.0 - float(settings.get("temporal_debris_decision_threshold", 0.62)),
            )
        new_cell = mass * target_cell
        new_debris = mass - new_cell
        if strong_static_debris:
            adjusted_label = "debris"
        elif target_cell >= float(settings.get("temporal_cell_decision_threshold", 0.62)):
            adjusted_label = candidate_multiplicity_label(row)
        elif target_cell <= 1.0 - float(settings.get("temporal_debris_decision_threshold", 0.62)):
            adjusted_label = "debris"
        else:
            adjusted_label = "uncertain"
        outputs[index] = {
            **common,
            "cell": new_cell,
            "debris": new_debris,
            "boost": max(0.0, new_debris - debris),
            "cell_boost": max(0.0, new_cell - cell),
            "applied": True,
            "reason": reason,
            "label": adjusted_label,
        }
    return outputs


def _revoke_suspected_dead_for_well(
    frame: pd.DataFrame,
    group_indices: list[int],
    outputs: dict[int, dict[str, Any]],
) -> bool:
    """Revoke a dead-cell hint when the only T0 source later divides."""

    group_set = set(group_indices)
    t0_cell_nodes = [
        index
        for index in group_indices
        if str(frame.at[index, "timepoint"]) == "T0"
        and str(frame.at[index, "v2_pre_temporal_integrated_label"]) in CELL_LABELS
        and conditional_cell_probability(frame.loc[index]) >= 0.50
    ]
    well_has_later_division = any(
        float(output.get("growth", 0.0)) > 0.0
        for index, output in outputs.items()
        if index in group_set
    )
    if len(t0_cell_nodes) != 1 or not well_has_later_division:
        return False
    output = outputs.get(t0_cell_nodes[0])
    if not output or not output.get("suspected_dead_cell"):
        return False
    output["suspected_dead_cell"] = False
    output["suspected_dead_cell_score"] = 0.0
    output["reason"] = "suspected_dead_revoked_by_later_division"
    return True


def _normalized_crop(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    raw = _crop(image, x, y, size, int(np.median(image))).astype(np.float32)
    lo, hi = np.percentile(raw, [2, 98])
    return np.clip((raw - lo) / max(float(hi - lo), 1.0), 0, 1)


def _local_patch_similarity(first: np.ndarray, second: np.ndarray, maximum_shift: int = 4) -> float:
    """Maximum normalized correlation near the target, robust to small registration error."""

    margin = maximum_shift + 10
    reference = first[margin:-margin, margin:-margin].astype(np.float64)
    reference = reference - reference.mean()
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm <= 1e-8:
        return 0.0
    best = -1.0
    for dy in range(-maximum_shift, maximum_shift + 1):
        for dx in range(-maximum_shift, maximum_shift + 1):
            top, left = margin + dy, margin + dx
            candidate = second[top : top + reference.shape[0], left : left + reference.shape[1]].astype(np.float64)
            candidate = candidate - candidate.mean()
            denominator = reference_norm * float(np.linalg.norm(candidate))
            if denominator > 1e-8:
                best = max(best, float((reference * candidate).sum() / denominator))
    return float(np.clip((best + 1.0) / 2.0, 0.0, 1.0))


def _adjust_cell_debris_probabilities(
    cell_probability: float,
    debris_probability: float,
    invalid_probability: float,
    same_object_score: float,
    static_similarity_score: float,
    frame_count: int,
    *,
    same_threshold: float = 0.80,
    static_threshold: float = 0.75,
    high_confidence_threshold: float = 0.90,
    beta: float = 2.0,
    maximum_shift: float = 0.30,
) -> tuple[float, float, float, bool, str]:
    """Move ambiguous cell/debris mass toward debris using static temporal evidence.

    Invalid probability is intentionally excluded from the redistribution.  Temporal
    evidence can refine an uncertain cell/debris decision, but cannot rescue or
    create an invalid/background decision.
    """

    cell_probability = float(np.clip(cell_probability, 0.0, 1.0))
    debris_probability = float(np.clip(debris_probability, 0.0, 1.0))
    invalid_probability = float(np.clip(invalid_probability, 0.0, 1.0))
    if frame_count < 2:
        return (
            cell_probability,
            debris_probability,
            0.0,
            False,
            "insufficient_parallel_candidates",
        )
    if invalid_probability >= 0.60:
        return cell_probability, debris_probability, 0.0, False, "invalid_candidate"
    if same_object_score < same_threshold:
        return cell_probability, debris_probability, 0.0, False, "low_match"
    if static_similarity_score < static_threshold:
        return cell_probability, debris_probability, 0.0, False, "low_similarity"

    mass = cell_probability + debris_probability
    if mass <= 1e-6:
        return cell_probability, debris_probability, 0.0, False, "no_cell_debris_mass"
    conditional_cell = float(np.clip(cell_probability / mass, 1e-5, 1.0 - 1e-5))
    base_confidence = max(conditional_cell, 1.0 - conditional_cell)
    if base_confidence >= high_confidence_threshold:
        return cell_probability, debris_probability, 0.0, False, "high_confidence_base"

    ambiguity = 1.0 - abs(2.0 * conditional_cell - 1.0)
    static_excess = (static_similarity_score - static_threshold) / max(1.0 - static_threshold, 1e-6)
    frame_factor = 1.0 if frame_count >= 3 else 0.70
    strength = float(np.clip(same_object_score * static_excess * ambiguity * frame_factor, 0.0, 1.0))
    logit = np.log(conditional_cell / (1.0 - conditional_cell))
    target_cell = 1.0 / (1.0 + np.exp(-(logit - beta * strength)))
    requested_shift = mass * max(0.0, conditional_cell - target_cell)
    shift = float(min(requested_shift, maximum_shift, cell_probability))
    if shift <= 1e-6:
        return cell_probability, debris_probability, 0.0, False, "no_effect"
    return cell_probability - shift, debris_probability + shift, shift, True, "applied"


def _is_static_wall_artifact(
    present: list[float],
    coords: list[tuple[float, float]],
    areas: list[float],
    morphology: list[tuple[float, float, float, float]],
    same_object_score: float,
    static_similarity_score: float,
    *,
    same_threshold: float = 0.80,
    static_threshold: float = 0.80,
    wall_overlap_threshold: float = 0.90,
    maximum_motion_px: float = 5.0,
    maximum_area_ratio: float = 2.20,
) -> bool:
    available = [index for index, value in enumerate(present) if value]
    if len(available) != 3 or same_object_score < same_threshold or static_similarity_score < static_threshold:
        return False
    reference = available[0]
    maximum_displacement = max(
        float(np.hypot(coords[index][0] - coords[reference][0], coords[index][1] - coords[reference][1]))
        for index in available[1:]
    )
    available_areas = [max(float(areas[index]), 1.0) for index in available]
    area_ratio = max(available_areas) / min(available_areas)
    minimum_wall_overlap = min(float(morphology[index][3]) for index in available)
    return (
        maximum_displacement <= maximum_motion_px
        and area_ratio <= maximum_area_ratio
        and minimum_wall_overlap >= wall_overlap_threshold
    )


def _adjusted_label(original_label: str, cell_probability: float, debris_probability: float) -> str:
    if debris_probability >= 0.60 and debris_probability - cell_probability >= 0.10:
        return "debris"
    if cell_probability >= 0.60 and cell_probability - debris_probability >= 0.10:
        return original_label if original_label in CELL_LABELS else "single"
    return "uncertain"


def _track_fused_probabilities(
    morphology: list[tuple[float, float, float, float]],
    proposal_present: list[float],
) -> tuple[float, float, float]:
    """Reliability-weighted morphology probabilities for one temporal track."""

    available = [index for index, value in enumerate(proposal_present) if value]
    if not available:
        return 0.0, 0.0, 1.0
    values = np.asarray([morphology[index][:3] for index in available], dtype=np.float64)
    reliability = 0.50 + 0.50 * values.max(axis=1)
    fused = np.average(values, axis=0, weights=reliability)
    total = float(fused.sum())
    if total > 1e-6:
        fused /= total
    return float(fused[0]), float(fused[1]), float(fused[2])


def preserve_temporal_multiplicity_labels(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Restore per-candidate multiplicity after temporal cell decisions."""

    output = frame.copy()
    adjusted = output.get(
        "v2_temporal_adjusted_label", output["integrated_label"]
    ).fillna("").astype(str)
    temporal_cells = adjusted.isin(CELL_LABELS)
    changed = 0
    for index in output.index[temporal_cells]:
        label = candidate_multiplicity_label(output.loc[index])
        if str(output.at[index, "integrated_label"]) != label:
            changed += 1
        output.at[index, "integrated_label"] = label
        if "v2_temporal_adjusted_label" in output:
            output.at[index, "v2_temporal_adjusted_label"] = label
    output["v2_temporal_multiplicity_preserved"] = temporal_cells
    return output, changed


def resolve_low_cell_noncell_labels(
    frame: pd.DataFrame,
    threshold: float = LOW_CELL_NONCELL_RESOLUTION_THRESHOLD,
) -> tuple[pd.DataFrame, int]:
    """Resolve low-cell ambiguity directly as debris or invalid.

    The uncertain state is retained only when the remaining ambiguity includes
    a meaningful possibility that the object is a cell.
    """

    output = frame.copy()
    cell = pd.to_numeric(
        output.get("v2_adjusted_cell_probability", output["cell_probability"]),
        errors="coerce",
    ).fillna(0.0)
    debris = pd.to_numeric(
        output.get("v2_adjusted_debris_probability", output["debris_probability"]),
        errors="coerce",
    ).fillna(0.0)
    invalid = pd.to_numeric(output["invalid_probability"], errors="coerce").fillna(0.0)
    eligible = output["integrated_label"].isin(["uncertain", "unmarked"]) & cell.lt(threshold)
    as_debris = eligible & debris.ge(invalid)
    as_invalid = eligible & ~as_debris
    output.loc[as_debris, "integrated_label"] = "debris"
    output.loc[as_invalid, "integrated_label"] = "invalid"
    if "v2_temporal_adjusted_label" in output:
        output.loc[as_debris, "v2_temporal_adjusted_label"] = "debris"
        output.loc[as_invalid, "v2_temporal_adjusted_label"] = "invalid"
    output["v2_low_cell_noncell_resolved"] = eligible
    output["v2_noncell_resolution_label"] = np.where(
        as_debris, "debris", np.where(as_invalid, "invalid", "")
    )
    if "v2_temporal_reason" in output:
        output.loc[as_debris, "v2_temporal_reason"] = "noncell_debris_resolution"
        output.loc[as_invalid, "v2_temporal_reason"] = "noncell_invalid_resolution"
    return output, int(eligible.sum())


def infer_v2_temporal_evidence(config: dict[str, Any], checkpoint_path: str | Path) -> Path:
    started = time.perf_counter()
    source = artifact_path(config, "predictions", "latest_v2_predictions.csv")
    base_source = source.with_name("latest_v2_pre_temporal_predictions.csv")
    frame = pd.read_csv(base_source if base_source.exists() else source, low_memory=False)
    if "candidate_source" in frame:
        # Synthetic rows are regenerated from the current model on every run;
        # never let a previous temporal recovery become a new proposal seed.
        frame = frame[
            frame["candidate_source"].fillna("").astype(str) != "temporal_recovery"
        ].copy().reset_index(drop=True)
    settings = config.get("v2_temporal_model", {})
    maximum_correspondence_distance = float(settings.get("maximum_correspondence_distance_px", 128.0))
    identity_threshold = float(settings.get("object_identity_threshold", 0.58))
    descriptor_shift = int(settings.get("object_descriptor_maximum_shift_px", 3))
    instance_patch_size = int(config.get("v2_instance_segmentation", {}).get("patch_size_px", 96))
    descriptor_size = int(settings.get("object_descriptor_size_px", 48))

    if "v2_pre_temporal_integrated_label" not in frame:
        frame["v2_pre_temporal_integrated_label"] = frame["integrated_label"].astype(str)
    else:
        frame["integrated_label"] = frame["v2_pre_temporal_integrated_label"].fillna(frame["integrated_label"])
    if not base_source.exists():
        # Preserve an immutable segmentation/morphology stage so temporal-only
        # policy changes never rerun the expensive instance model.
        frame.to_csv(base_source, index=False)

    output_columns: dict[str, Any] = {
        "v2_temporal_same_object_score": 0.0,
        "v2_temporal_static_similarity_score": 0.0,
        "v2_temporal_candidate_count": 0,
        "v2_temporal_debris_boost": 0.0,
        "v2_adjusted_cell_probability": frame["cell_probability"].astype(float),
        "v2_adjusted_debris_probability": frame["debris_probability"].astype(float),
        "v2_temporal_adjustment_applied": False,
        "v2_temporal_reason": "not_evaluated",
        "v2_temporal_adjusted_label": frame["integrated_label"].astype(str),
        "v2_static_wall_artifact": False,
        "v2_temporal_cell_boost": 0.0,
        "v2_temporal_foreground_similarity": 0.0,
        "v2_temporal_shape_similarity": 0.0,
        "v2_temporal_change_score": 0.0,
        "v2_temporal_growth_score": 0.0,
        "v2_temporal_foreground_quality": 0.0,
        "v2_temporal_evidence_frame_count": 0,
        "v2_temporal_pair_count": 0,
        "v2_temporal_three_frame_static": False,
        "v2_temporal_morphology_consensus_cell_probability": 0.0,
        "v2_suspected_dead_cell": False,
        "v2_suspected_dead_cell_score": 0.0,
        "v2_temporal_debris_probability_trend": False,
        "v2_temporal_debris_trend_score": 0.0,
    }
    for name, values in output_columns.items():
        frame[name] = values

    if "v2_is_unique_instance" not in frame:
        frame["v2_is_unique_instance"] = (
            frame["v2_mask_valid"].fillna(False).astype(bool)
            & ~frame["v2_wall_rejected"].fillna(False).astype(bool)
            & ~frame["v2_is_suppressed"].fillna(False).astype(bool)
        )
    if "v2_is_temporal_candidate" not in frame:
        frame["v2_is_temporal_candidate"] = (
            frame["v2_is_unique_instance"].fillna(False).astype(bool)
            & frame["invalid_probability"].fillna(1.0).astype(float).lt(0.60)
        )
    frame["v2_is_reviewable_instance"] = (
        frame["v2_is_unique_instance"].fillna(False).astype(bool)
        & frame["integrated_label"].isin(
            ["single", "touching_doublet", "cluster_3plus", "debris", "uncertain"]
        )
    )
    frame["v2_is_counting_instance"] = (
        frame["v2_is_unique_instance"].fillna(False).astype(bool)
        & frame["integrated_label"].isin(
            ["single", "touching_doublet", "cluster_3plus", "debris"]
        )
    )
    frame["v2_temporal_recovered"] = False
    frame["v2_temporal_track_id"] = ""

    valid = frame[frame["v2_is_temporal_candidate"].fillna(False).astype(bool)].copy()
    outputs: dict[int, dict[str, Any]] = {}

    for well, group in valid.groupby("well", sort=False):
        all_well = frame[frame["well"] == well]
        images: dict[str, np.ndarray] = {}
        image_fill_values: dict[str, float] = {}
        for timepoint, local in all_well.groupby("timepoint"):
            with Image.open(str(local.iloc[0]["raw_image_path"])) as image:
                images[str(timepoint)] = np.asarray(image.convert("L"), dtype=np.uint8)
            image_fill_values[str(timepoint)] = float(np.median(images[str(timepoint)]))

        descriptors = {}
        for index, row in group.iterrows():
            image = images.get(str(row.timepoint))
            if image is None:
                continue
            descriptor = build_object_descriptor(
                image,
                row,
                instance_patch_size=instance_patch_size,
                descriptor_size=descriptor_size,
                image_fill_value=image_fill_values[str(row.timepoint)],
            )
            if descriptor is not None:
                descriptors[index] = descriptor
        if not descriptors:
            continue
        by_timepoint = {
            timepoint: [
                descriptors[index]
                for index in group.index[group["timepoint"].astype(str) == timepoint]
                if index in descriptors
            ]
            for timepoint in TIMEPOINTS
        }
        components = TemporalComponents(list(descriptors))
        edges: list[TemporalPairEvidence] = []
        degree = {index: 0 for index in descriptors}
        for left_timepoint, right_timepoint in (("T0", "T1"), ("T1", "T2")):
            matches, pair_evidence = match_timepoint_objects(
                by_timepoint[left_timepoint],
                by_timepoint[right_timepoint],
                maximum_distance_px=maximum_correspondence_distance,
                minimum_identity=identity_threshold,
                maximum_shift=descriptor_shift,
            )
            division = detect_division_edges(
                frame,
                by_timepoint[left_timepoint],
                by_timepoint[right_timepoint],
                matches,
                pair_evidence,
                division_radius_px=float(settings.get("division_correspondence_radius_px", 88.0)),
            )
            for edge in matches + division:
                edges.append(edge)
                components.union(edge.left, edge.right)
                degree[edge.left] += 1
                degree[edge.right] += 1

        # A direct T0-T2 edge is useful only when T1 did not provide a path.
        skip_matches, _ = match_timepoint_objects(
            by_timepoint["T0"],
            by_timepoint["T2"],
            maximum_distance_px=maximum_correspondence_distance * 1.15,
            minimum_identity=min(identity_threshold + 0.05, 0.95),
            maximum_shift=descriptor_shift,
        )
        for edge in skip_matches:
            if degree[edge.left] or degree[edge.right]:
                continue
            edge = TemporalPairEvidence(**{**edge.__dict__, "kind": "skip_continuation"})
            edges.append(edge)
            components.union(edge.left, edge.right)
            degree[edge.left] += 1
            degree[edge.right] += 1

        for component_number, nodes in enumerate(components.groups(), start=1):
            node_set = set(nodes)
            local_edges = [
                edge for edge in edges if edge.left in node_set and edge.right in node_set
            ]
            ordered = sorted(
                nodes,
                key=lambda index: (TIMEPOINTS.index(str(frame.at[index, "timepoint"])), str(frame.at[index, "candidate_id"])),
            )
            track_id = f"{well}:O{component_number:03d}:" + "|".join(
                str(frame.at[index, "candidate_id"]) for index in ordered
            )
            outputs.update(
                _component_temporal_outputs(frame, nodes, local_edges, settings, track_id)
            )

        # A T0 object cannot remain labelled as a suspected dead cell when it
        # is the well's only credible T0 cell source and the same well later
        # contains pre-temporal biological division evidence.  This is applied
        # after every component has been evaluated because the division may be
        # represented by a neighbouring child component.
        _revoke_suspected_dead_for_well(
            frame,
            list(group.index),
            outputs,
        )

    for index, output in outputs.items():
        frame.at[index, "v2_temporal_same_object_score"] = output["same"]
        frame.at[index, "v2_temporal_static_similarity_score"] = output["static"]
        frame.at[index, "v2_temporal_candidate_count"] = output["proposal_count"]
        frame.at[index, "v2_temporal_debris_boost"] = output["boost"]
        frame.at[index, "v2_adjusted_cell_probability"] = output["cell"]
        frame.at[index, "v2_adjusted_debris_probability"] = output["debris"]
        frame.at[index, "v2_temporal_adjustment_applied"] = output["applied"]
        frame.at[index, "v2_temporal_reason"] = output["reason"]
        frame.at[index, "v2_temporal_adjusted_label"] = output["label"]
        frame.at[index, "v2_static_wall_artifact"] = output["static_wall"]
        frame.at[index, "integrated_label"] = output["label"]
        frame.at[index, "v2_temporal_track_id"] = output["track_id"]
        frame.at[index, "v2_temporal_cell_boost"] = output["cell_boost"]
        frame.at[index, "v2_temporal_foreground_similarity"] = output["static"]
        frame.at[index, "v2_temporal_shape_similarity"] = output["shape"]
        frame.at[index, "v2_temporal_change_score"] = output["change"]
        frame.at[index, "v2_temporal_growth_score"] = output["growth"]
        frame.at[index, "v2_temporal_foreground_quality"] = output["foreground_quality"]
        frame.at[index, "v2_temporal_evidence_frame_count"] = output["frame_count"]
        frame.at[index, "v2_temporal_pair_count"] = output["pair_count"]
        frame.at[index, "v2_temporal_three_frame_static"] = output["three_frame_static"]
        frame.at[index, "v2_temporal_morphology_consensus_cell_probability"] = output["morphology_consensus_cell"]
        frame.at[index, "v2_suspected_dead_cell"] = output["suspected_dead_cell"]
        frame.at[index, "v2_suspected_dead_cell_score"] = output["suspected_dead_cell_score"]
        frame.at[index, "v2_temporal_debris_probability_trend"] = output[
            "debris_probability_trend"
        ]
        frame.at[index, "v2_temporal_debris_trend_score"] = output[
            "debris_trend_score"
        ]

    frame, low_cell_noncell_resolution_count = resolve_low_cell_noncell_labels(frame)
    frame, multiplicity_preserved_count = preserve_temporal_multiplicity_labels(frame)

    frame["v2_temporal_recovered"] = (
        frame["v2_is_temporal_candidate"].fillna(False).astype(bool)
        & frame["v2_pre_temporal_integrated_label"].isin(["unmarked", "uncertain"])
        & frame["integrated_label"].isin(
            ["single", "touching_doublet", "cluster_3plus", "debris"]
        )
        & frame["v2_temporal_reason"].isin(["applied", "high_confidence_base"])
    )
    frame["v2_is_reviewable_instance"] = (
        frame["v2_is_unique_instance"].fillna(False).astype(bool)
        & frame["integrated_label"].isin(
            ["single", "touching_doublet", "cluster_3plus", "debris", "uncertain"]
        )
    )
    frame["v2_is_counting_instance"] = (
        frame["v2_is_unique_instance"].fillna(False).astype(bool)
        & frame["integrated_label"].isin(
            ["single", "touching_doublet", "cluster_3plus", "debris"]
        )
    )

    round_id = datetime.now().strftime("v2-temporal-round-%Y%m%d-%H%M%S")
    frame["integrated_round_id"] = round_id
    frame.to_csv(source, index=False)
    round_directory = source.parent / round_id
    round_directory.mkdir(parents=True, exist_ok=False)
    frame.to_csv(round_directory / "predictions.csv", index=False)
    evaluated = frame.loc[list(outputs)] if outputs else frame.iloc[0:0]
    summary = {
        "algorithm_version": "v2.1-object-graph",
        "round_id": round_id,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "checkpoint_retained_for_retraining": str(Path(checkpoint_path).resolve()),
        "evaluated_instances": len(outputs),
        "adjustment_applied": int(evaluated["v2_temporal_adjustment_applied"].fillna(False).astype(bool).sum()),
        "static_wall_invalid": int(evaluated["v2_static_wall_artifact"].fillna(False).astype(bool).sum()),
        "temporally_recovered_instances": int(frame["v2_temporal_recovered"].fillna(False).astype(bool).sum()),
        "synthetic_temporal_recovery_instances": 0,
        "low_cell_noncell_resolution_count": low_cell_noncell_resolution_count,
        "temporal_multiplicity_labels_corrected": multiplicity_preserved_count,
        "reason_counts": evaluated["v2_temporal_reason"].value_counts().to_dict(),
        "mean_debris_boost_when_applied": float(
            evaluated.loc[evaluated["v2_temporal_adjustment_applied"].fillna(False).astype(bool), "v2_temporal_debris_boost"].mean()
        ) if evaluated["v2_temporal_adjustment_applied"].fillna(False).astype(bool).any() else 0.0,
        "object_components": int(evaluated["v2_temporal_track_id"].replace("", np.nan).nunique()),
        "three_frame_static_instances": int(evaluated["v2_temporal_three_frame_static"].fillna(False).astype(bool).sum()),
        "growth_evidence_instances": int(evaluated["v2_temporal_growth_score"].fillna(0.0).astype(float).gt(0).sum()),
        "suspected_dead_cell_instances": int(
            evaluated["v2_suspected_dead_cell"].fillna(False).astype(bool).sum()
        ),
        "policy": "foreground-only object matching; global one-to-one continuation with conservative one-to-many division; strong three-frame stability supports debris; missing or weak identity preserves the base label",
    }
    source.with_name("latest_v2_temporal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return source


def refinalize_v2_temporal_noncell_labels(config: dict[str, Any]) -> Path:
    """Apply the noncell-resolution policy without rerunning neural inference."""

    source = artifact_path(config, "predictions", "latest_v2_predictions.csv")
    frame = pd.read_csv(source, low_memory=False)
    settings = config.get("v2_temporal_model", {})
    cached_trend_updates = 0
    if "v2_temporal_track_id" in frame:
        track_ids = frame["v2_temporal_track_id"].fillna("").astype(str)
        for _, local in frame[track_ids.ne("")].groupby(track_ids[track_ids.ne("")]):
            by_timepoint = {
                timepoint: local[local["timepoint"].astype(str) == timepoint]
                for timepoint in TIMEPOINTS
            }
            if any(len(rows) != 1 for rows in by_timepoint.values()):
                continue
            ordered = pd.concat([by_timepoint[timepoint] for timepoint in TIMEPOINTS])
            mass = (
                pd.to_numeric(ordered["cell_probability"], errors="coerce").fillna(0.0)
                + pd.to_numeric(ordered["debris_probability"], errors="coerce").fillna(0.0)
            ).clip(lower=1e-6)
            debris_sequence = (
                pd.to_numeric(ordered["debris_probability"], errors="coerce").fillna(0.0)
                / mass
            ).to_numpy(float)
            identity = float(
                pd.to_numeric(
                    ordered["v2_temporal_same_object_score"], errors="coerce"
                ).fillna(0.0).mean()
            )
            static = float(
                pd.to_numeric(
                    ordered["v2_temporal_static_similarity_score"], errors="coerce"
                ).fillna(0.0).mean()
            )
            shape = float(
                pd.to_numeric(
                    ordered.get(
                        "v2_temporal_shape_similarity",
                        pd.Series(0.0, index=ordered.index),
                    ),
                    errors="coerce",
                ).fillna(0.0).mean()
            )
            growth = float(
                pd.to_numeric(
                    ordered.get(
                        "v2_temporal_growth_score",
                        pd.Series(0.0, index=ordered.index),
                    ),
                    errors="coerce",
                ).fillna(0.0).max()
            )
            rise = float(debris_sequence[-1] - debris_sequence[0])
            eligible = bool(
                identity >= float(settings.get("object_identity_threshold", 0.58))
                and static >= float(settings.get("debris_trend_minimum_static_similarity", 0.58))
                and shape >= float(settings.get("debris_trend_minimum_shape_similarity", 0.78))
                and debris_sequence[1] >= debris_sequence[0] - 0.05
                and debris_sequence[2] >= debris_sequence[1] - 0.05
                and rise >= float(settings.get("debris_trend_minimum_probability_rise", 0.18))
                and debris_sequence[-1] >= float(settings.get("debris_trend_minimum_final_probability", 0.48))
                and growth <= 0.0
            )
            if not eligible:
                continue
            score = float(
                np.clip(
                    rise / max(float(settings.get("debris_trend_full_strength_rise", 0.35)), 1e-6),
                    0.0,
                    1.0,
                )
                * identity
            )
            evidence = float(settings.get("debris_probability_trend_logit", 2.15)) * score
            for index in ordered.index:
                cell = float(frame.at[index, "cell_probability"])
                debris = float(frame.at[index, "debris_probability"])
                cell_debris_mass = max(cell + debris, 1e-6)
                base_cell = cell / cell_debris_mass
                target_cell = _probability_sigmoid(_probability_logit(base_cell) - evidence)
                target_cell = base_cell - min(max(base_cell - target_cell, 0.0), 0.34)
                new_cell = cell_debris_mass * target_cell
                new_debris = cell_debris_mass - new_cell
                frame.at[index, "v2_adjusted_cell_probability"] = new_cell
                frame.at[index, "v2_adjusted_debris_probability"] = new_debris
                frame.at[index, "v2_temporal_debris_boost"] = max(0.0, new_debris - debris)
                frame.at[index, "v2_temporal_adjustment_applied"] = True
                frame.at[index, "v2_temporal_reason"] = "multi_frame_debris_probability_trend"
                frame.at[index, "v2_temporal_debris_probability_trend"] = True
                frame.at[index, "v2_temporal_debris_trend_score"] = score
                if target_cell <= 1.0 - float(
                    settings.get("temporal_debris_decision_threshold", 0.62)
                ):
                    frame.at[index, "integrated_label"] = "debris"
                    frame.at[index, "v2_temporal_adjusted_label"] = "debris"
                cached_trend_updates += 1

    # Revoke stale suspected-dead annotations using already cached, trusted
    # division evidence.  This is a metadata-only pass and does not reload any
    # source TIFFs.
    for _, local in frame.groupby("well", sort=False):
        visible = local[
            local.get(
                "v2_is_unique_instance", pd.Series(True, index=local.index)
            ).fillna(False).astype(bool)
        ]
        t0_cells = visible[
            visible["timepoint"].astype(str).eq("T0")
            & visible["v2_pre_temporal_integrated_label"].isin(CELL_LABELS)
        ]
        later_division = pd.to_numeric(
            visible.get(
                "v2_temporal_growth_score", pd.Series(0.0, index=visible.index)
            ),
            errors="coerce",
        ).fillna(0.0).gt(0.0).any()
        if len(t0_cells) == 1 and later_division:
            index = t0_cells.index[0]
            if bool(frame.at[index, "v2_suspected_dead_cell"]):
                frame.at[index, "v2_suspected_dead_cell"] = False
                frame.at[index, "v2_suspected_dead_cell_score"] = 0.0
                frame.at[index, "v2_temporal_reason"] = (
                    "suspected_dead_revoked_by_later_division"
                )
    frame, resolved_count = resolve_low_cell_noncell_labels(frame)
    frame, multiplicity_preserved_count = preserve_temporal_multiplicity_labels(frame)
    frame["v2_is_reviewable_instance"] = (
        frame["v2_is_unique_instance"].fillna(False).astype(bool)
        & frame["integrated_label"].isin(
            ["single", "touching_doublet", "cluster_3plus", "debris", "uncertain"]
        )
    )
    frame["v2_is_counting_instance"] = (
        frame["v2_is_unique_instance"].fillna(False).astype(bool)
        & frame["integrated_label"].isin(
            ["single", "touching_doublet", "cluster_3plus", "debris"]
        )
    )
    round_id = datetime.now().strftime("v2-noncell-round-%Y%m%d-%H%M%S")
    frame["integrated_round_id"] = round_id
    frame.to_csv(source, index=False)
    round_directory = source.parent / round_id
    round_directory.mkdir(parents=True, exist_ok=False)
    frame.to_csv(round_directory / "predictions.csv", index=False)
    summary_path = source.with_name("latest_v2_temporal_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update({
        "round_id": round_id,
        "low_cell_noncell_resolution_count": resolved_count,
        "temporal_multiplicity_labels_corrected": multiplicity_preserved_count,
        "uncertain_policy": "uncertain is reserved for cell-vs-noncell ambiguity",
        "cached_debris_trend_updates": cached_trend_updates,
    })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return source

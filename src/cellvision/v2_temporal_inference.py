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
    compare_object_descriptors,
    conditional_cell_probability,
    detect_division_edges,
    match_timepoint_objects,
    multiplicity_rank,
)
from .temporal_behavior import evaluate_temporal_behavior
from .temporal_pairwise import load_temporal_pairwise_scorer
from .v2_instance_dataset import _crop
from .v2_instance_inference import decode_rle


TIMEPOINTS = ("T0", "T1", "T2")
CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}
LOW_CELL_NONCELL_RESOLUTION_THRESHOLD = 0.30
V3_NON_FUSING_BEHAVIORS = frozenset(
    {
        "no_decisive_temporal_evidence",
        "wall_uncertain",
        "decline_without_morphology_evidence",
    }
)


def _v3_proposal_is_decisive(output: dict[str, Any]) -> bool:
    """Return whether a V3 proposal may overwrite the V2 final label."""

    behavior = str(output.get("v3_track_behavior", ""))
    return bool(behavior and behavior not in V3_NON_FUSING_BEHAVIORS)


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


def _strong_wall_cell_observation(
    row: pd.Series,
    settings: dict[str, Any],
) -> bool:
    """Return whether unary evidence is strong enough to protect a wall cell.

    Wall overlap and ``v2_is_counting_instance`` are deliberately absent from
    this gate: both can be consequences of the wall policy that this evidence
    is meant to veto.  The stricter absolute semantic and instance-quality
    requirements keep a weak wall response from receiving the same rescue.
    """

    if _finite_row_value(row, "cell_probability", 0.0) < float(
        settings.get("wall_cell_protection_minimum_cell_probability", 0.80)
    ):
        return False
    if _finite_row_value(row, "debris_probability", 1.0) > float(
        settings.get("wall_cell_protection_maximum_debris_probability", 0.20)
    ):
        return False
    if _finite_row_value(row, "invalid_probability", 1.0) > float(
        settings.get("wall_cell_protection_maximum_invalid_probability", 0.20)
    ):
        return False
    if _finite_row_value(row, "v2_instance_confidence", 1.0) < float(
        settings.get("wall_cell_protection_minimum_instance_confidence", 0.55)
    ):
        return False
    if _finite_row_value(row, "v2_objectness", 1.0) < float(
        settings.get("wall_cell_protection_minimum_objectness", 0.70)
    ):
        return False
    if not _row_flag(row, "v2_mask_valid", True):
        return False
    if not _row_flag(row, "v2_is_unique_instance", True):
        return False
    if _row_flag(row, "v2_is_suppressed", False):
        return False
    if _row_flag(row, "v2_wall_rejected", False):
        return False
    return True


def _cross_track_division_cell_candidate(
    row: pd.Series,
    minimum_probability: float,
    settings: dict[str, Any],
    *,
    allow_wall_attached: bool = False,
) -> bool:
    """Return whether a unique V2 instance is safe to use as a split child.

    Cross-track rescue deliberately uses the pre-temporal semantic label and
    the existing instance-quality gates.  It must not revive duplicates,
    rejected wall residues, invalid rows, or a debris candidate merely because
    another nearby object looks cell-like.
    """

    label = str(
        row.get("v2_pre_temporal_integrated_label", row.get("integrated_label", ""))
    )
    if label not in CELL_LABELS:
        return False
    if conditional_cell_probability(row) < float(minimum_probability):
        return False
    if not _row_flag(row, "v2_is_unique_instance", True):
        return False
    strong_wall_cell = _strong_wall_cell_observation(row, settings)
    if (
        not _row_flag(row, "v2_is_counting_instance", True)
        and not (allow_wall_attached and strong_wall_cell)
    ):
        return False
    if _row_flag(row, "v2_wall_rejected", False):
        return False
    if _finite_row_value(row, "invalid_probability", 1.0) > float(
        settings.get("cross_track_division_rescue_maximum_invalid_probability", 0.60)
    ):
        return False
    if _finite_row_value(row, "v2_instance_confidence", 1.0) < float(
        settings.get("cross_track_division_rescue_minimum_instance_confidence", 0.55)
    ):
        return False
    if _finite_row_value(row, "v2_objectness", 1.0) < float(
        settings.get("cross_track_division_rescue_minimum_objectness", 0.70)
    ):
        return False
    if (
        _finite_row_value(row, "v2_wall_overlap", 0.0)
        > float(
            settings.get("cross_track_division_rescue_maximum_wall_overlap", 0.85)
        )
        and not (allow_wall_attached and strong_wall_cell)
    ):
        return False
    return True


def _pair_evidence_for_cross_track_rescue(
    parent: Any,
    child: Any,
    pair_evidence: dict[tuple[int, int], TemporalPairEvidence],
    settings: dict[str, Any],
    pairwise_scorer: Any = None,
    *,
    correspondence_distance_px: float | None = None,
) -> TemporalPairEvidence:
    """Reuse cached evidence and score only the extra cross-track pair."""

    pair = pair_evidence.get((int(parent.index), int(child.index)))
    if pair is None:
        pair = compare_object_descriptors(
            parent,
            child,
            maximum_shift=int(settings.get("object_descriptor_maximum_shift_px", 2)),
            correspondence_distance_px=float(
                correspondence_distance_px
                if correspondence_distance_px is not None
                else settings.get("maximum_correspondence_distance_px", 128.0)
            ),
        )
        if pairwise_scorer is not None:
            pair = pairwise_scorer(parent, child, pair)
        pair_evidence[(int(parent.index), int(child.index))] = pair
    return pair


def _detect_cross_track_division_rescues(
    frame: pd.DataFrame,
    group_indices: list[int],
    by_timepoint: dict[str, list[Any]],
    descriptors: dict[int, Any],
    edges: list[TemporalPairEvidence],
    pair_evidence: dict[tuple[int, int], TemporalPairEvidence],
    node_to_component: dict[int, int],
    component_nodes: dict[int, list[int]],
    component_track_ids: dict[int, str],
    v3_outputs: dict[int, dict[str, Any]],
    settings: dict[str, Any],
    pairwise_scorer: Any = None,
) -> list[dict[str, Any]]:
    """Find a one-to-many split that one-to-one matching assigned elsewhere.

    The normal graph is intentionally conservative and globally one-to-one.
    That is useful for stable debris, but it can consume a genuine child with
    a neighbouring track before division detection sees it.  This pass only
    runs in wells with exactly one credible T0 cell and adds a veto relation;
    it does not merge unrelated components or count suppressed duplicates.
    """

    if not bool(settings.get("cross_track_division_rescue_enabled", True)):
        return []

    t0_cells = [
        index
        for index in group_indices
        if str(frame.at[index, "timepoint"]) == "T0"
        and _cross_track_division_cell_candidate(
            frame.loc[index],
            0.50,
            settings,
            allow_wall_attached=True,
        )
    ]
    if len(t0_cells) != 1:
        return []
    anchor_component = node_to_component.get(t0_cells[0])
    if anchor_component is None:
        return []

    division_radius = float(
        settings.get("cross_track_division_rescue_radius_px", 88.0)
    )
    wall_division_radius = float(
        settings.get("cross_track_division_rescue_wall_radius_px", 224.0)
    )
    minimum_identity = float(
        settings.get("cross_track_division_rescue_minimum_identity", 0.62)
    )
    wall_minimum_identity = float(
        settings.get("cross_track_division_rescue_wall_minimum_identity", 0.25)
    )
    wall_minimum_shape = float(
        settings.get(
            "cross_track_division_rescue_wall_minimum_shape_similarity", 0.70
        )
    )
    minimum_foreground_quality = float(
        settings.get("cross_track_division_rescue_minimum_foreground_quality", 0.30)
    )
    minimum_child_separation = float(
        settings.get("cross_track_division_rescue_minimum_child_separation_px", 5.0)
    )
    minimum_area_ratio = float(
        settings.get("cross_track_division_rescue_minimum_area_ratio", 0.65)
    )
    maximum_area_ratio = float(
        settings.get("cross_track_division_rescue_maximum_area_ratio", 5.0)
    )
    wall_maximum_area_ratio = float(
        settings.get("cross_track_division_rescue_wall_maximum_area_ratio", 10.0)
    )
    wall_minimum_secondary_strong_frames = int(
        settings.get(
            "cross_track_division_rescue_wall_minimum_secondary_strong_frames", 2
        )
    )
    maximum_component_nodes = int(
        settings.get("cross_track_division_rescue_maximum_secondary_component_nodes", 3)
    )
    descriptor_lookup = {int(index): descriptor for index, descriptor in descriptors.items()}
    continuation_by_parent: dict[tuple[int, str, str], list[TemporalPairEvidence]] = {}
    division_parents: set[tuple[int, str, str]] = set()
    for edge in edges:
        left_timepoint = str(frame.at[edge.left, "timepoint"])
        right_timepoint = str(frame.at[edge.right, "timepoint"])
        interval = (int(edge.left), left_timepoint, right_timepoint)
        if (left_timepoint, right_timepoint) not in {("T0", "T1"), ("T1", "T2")}:
            continue
        if edge.kind == "division":
            division_parents.add(interval)
        elif edge.kind == "continuation":
            continuation_by_parent.setdefault(interval, []).append(edge)

    best_by_parent: dict[tuple[int, str, str], dict[str, Any]] = {}
    for left_timepoint, right_timepoint in (("T0", "T1"), ("T1", "T2")):
        for parent in by_timepoint[left_timepoint]:
            parent_index = int(parent.index)
            if node_to_component.get(parent_index) != anchor_component:
                continue
            if not _cross_track_division_cell_candidate(
                frame.loc[parent_index],
                0.50,
                settings,
                allow_wall_attached=True,
            ):
                continue
            parent_row = frame.loc[parent_index]
            wall_overlap_limit = float(
                settings.get(
                    "cross_track_division_rescue_maximum_wall_overlap", 0.85
                )
            )
            strong_wall_parent = bool(
                _strong_wall_cell_observation(parent_row, settings)
                and (
                    _finite_row_value(parent_row, "v2_wall_overlap", 0.0)
                    > wall_overlap_limit
                    or _finite_row_value(parent_row, "radial_fraction", 0.0)
                    >= float(
                        settings.get(
                            "wall_cell_protection_band_start_fraction", 0.42
                        )
                    )
                    or not _row_flag(parent_row, "v2_is_counting_instance", True)
                )
            )
            local_division_radius = (
                wall_division_radius if strong_wall_parent else division_radius
            )
            local_minimum_identity = (
                wall_minimum_identity if strong_wall_parent else minimum_identity
            )
            local_maximum_area_ratio = (
                wall_maximum_area_ratio
                if strong_wall_parent
                else maximum_area_ratio
            )
            parent_key = (parent_index, left_timepoint, right_timepoint)
            if parent_key in division_parents:
                continue
            primary_edges = continuation_by_parent.get(parent_key, [])
            if not primary_edges:
                continue

            primary_edge = max(primary_edges, key=lambda edge: float(edge.identity))
            primary_child = descriptor_lookup.get(int(primary_edge.right))
            if primary_child is None or not _cross_track_division_cell_candidate(
                frame.loc[primary_child.index],
                0.45,
                settings,
                allow_wall_attached=True,
            ):
                continue
            if float(primary_edge.identity) < minimum_identity:
                continue
            if float(primary_edge.foreground_quality) < minimum_foreground_quality:
                continue

            for alternative_child in by_timepoint[right_timepoint]:
                alternative_index = int(alternative_child.index)
                if alternative_index == int(primary_child.index):
                    continue
                alternative_component = node_to_component.get(alternative_index)
                if alternative_component is None or alternative_component == anchor_component:
                    continue
                if len(component_nodes.get(alternative_component, [])) > maximum_component_nodes:
                    continue
                alternative_row = frame.loc[alternative_index]
                alternative_strong_wall_cell = _strong_wall_cell_observation(
                    alternative_row, settings
                )
                if strong_wall_parent:
                    secondary_strong_timepoints = {
                        str(frame.at[index, "timepoint"])
                        for index in component_nodes.get(alternative_component, [])
                        if _strong_wall_cell_observation(frame.loc[index], settings)
                    }
                    if (
                        len(secondary_strong_timepoints)
                        < wall_minimum_secondary_strong_frames
                    ):
                        continue
                if any(
                    str(v3_outputs.get(index, {}).get("v3_track_behavior", ""))
                    == "wall_structure_invalid"
                    for index in component_nodes.get(alternative_component, [])
                ) and not alternative_strong_wall_cell:
                    continue
                if not _cross_track_division_cell_candidate(
                    alternative_row,
                    0.45,
                    settings,
                    allow_wall_attached=True,
                ):
                    continue
                distance = float(
                    np.hypot(
                        alternative_child.aligned_x - parent.aligned_x,
                        alternative_child.aligned_y - parent.aligned_y,
                    )
                )
                if distance > local_division_radius:
                    continue
                child_separation = float(
                    np.hypot(
                        alternative_child.aligned_x - primary_child.aligned_x,
                        alternative_child.aligned_y - primary_child.aligned_y,
                    )
                )
                if child_separation < minimum_child_separation:
                    continue
                alternative_pair = _pair_evidence_for_cross_track_rescue(
                    parent,
                    alternative_child,
                    pair_evidence,
                    settings,
                    pairwise_scorer,
                    correspondence_distance_px=local_division_radius,
                )
                if float(alternative_pair.identity) < local_minimum_identity:
                    continue
                if float(alternative_pair.foreground_quality) < minimum_foreground_quality:
                    continue
                if (
                    strong_wall_parent
                    and float(alternative_pair.tolerant_shape) < wall_minimum_shape
                ):
                    continue
                children = [primary_child, alternative_child]
                combined_area_ratio = sum(float(child.area) for child in children) / max(
                    float(parent.area), 1.0
                )
                if not (
                    minimum_area_ratio
                    <= combined_area_ratio
                    <= local_maximum_area_ratio
                ):
                    continue
                centroid_x = float(
                    np.average(
                        [child.aligned_x for child in children],
                        weights=[child.area for child in children],
                    )
                )
                centroid_y = float(
                    np.average(
                        [child.aligned_y for child in children],
                        weights=[child.area for child in children],
                    )
                )
                centroid_distance = float(
                    np.hypot(centroid_x - parent.aligned_x, centroid_y - parent.aligned_y)
                )
                if centroid_distance > local_division_radius:
                    continue
                alternative_support = max(
                    float(alternative_pair.identity),
                    0.45 * float(alternative_pair.tolerant_shape)
                    + 0.30 * float(alternative_pair.foreground_quality)
                    + 0.25 * _finite_row_value(
                        alternative_row, "cell_probability", 0.0
                    ),
                )
                score = float(
                    min(float(primary_edge.identity), alternative_support)
                    * min(
                        1.0,
                        float(primary_edge.foreground_quality),
                        float(alternative_pair.foreground_quality),
                    )
                )
                rescue = {
                    "parent_index": parent_index,
                    "primary_child_index": int(primary_child.index),
                    "secondary_child_index": alternative_index,
                    "parent_component": anchor_component,
                    "secondary_component": alternative_component,
                    "left_timepoint": left_timepoint,
                    "right_timepoint": right_timepoint,
                    "interval": f"{left_timepoint}->{right_timepoint}",
                    "score": score,
                    "parent_track_id": component_track_ids.get(parent_index, ""),
                    "child_candidate_ids": "|".join(
                        [
                            str(frame.at[primary_child.index, "candidate_id"]),
                            str(frame.at[alternative_index, "candidate_id"]),
                        ]
                    ),
                }
                current = best_by_parent.get(parent_key)
                if current is None or float(rescue["score"]) > float(current["score"]):
                    best_by_parent[parent_key] = rescue

    return list(best_by_parent.values())


def _apply_cross_track_division_rescues(
    frame: pd.DataFrame,
    rescues: list[dict[str, Any]],
    outputs: dict[int, dict[str, Any]],
    v3_outputs: dict[int, dict[str, Any]],
    component_nodes: dict[int, list[int]],
    v3_settings: dict[str, Any],
) -> None:
    """Apply the rescue as a division veto without merging graph components."""

    if not rescues:
        return
    cell_threshold = float(v3_settings.get("division_cell_decision_threshold", 0.62))
    debris_threshold = float(v3_settings.get("division_debris_decision_threshold", 0.38))
    for rescue in rescues:
        related_nodes = set(component_nodes.get(int(rescue["parent_component"]), []))
        related_nodes.update(component_nodes.get(int(rescue["secondary_component"]), []))
        rescue_reason = "cross_track_division_rescue_vetoes_static_debris_and_dead_cell"
        for index in related_nodes:
            output = outputs.get(index)
            if output is not None:
                output["growth"] = max(float(output.get("growth", 0.0)), 1.0)
                output["suspected_dead_cell"] = False
                output["suspected_dead_cell_score"] = 0.0
                output["reason"] = "cross_track_division_rescue"

            v3_output = v3_outputs.get(index)
            if v3_output is not None:
                base_label = str(
                    v3_output.get("v3_proposed_label", frame.at[index, "integrated_label"])
                )
                conditional = conditional_cell_probability(frame.loc[index])
                if conditional >= cell_threshold:
                    proposed_label = candidate_multiplicity_label(frame.loc[index])
                    frame_state = "cell"
                elif base_label == "debris" or conditional <= debris_threshold:
                    proposed_label = "debris"
                    frame_state = "debris"
                else:
                    proposed_label = base_label
                    frame_state = "uncertain"
                intervals = [
                    value
                    for value in str(v3_output.get("v3_division_interval", "")).split("|")
                    if value
                ]
                if rescue["interval"] not in intervals:
                    intervals.append(rescue["interval"])
                v3_output.update(
                    {
                        "v3_track_behavior": "division_or_growth",
                        "v3_behavior_score": max(
                            float(v3_output.get("v3_behavior_score", 0.0)),
                            float(np.clip(0.65 + 0.35 * rescue["score"], 0.0, 1.0)),
                        ),
                        "v3_reason": rescue_reason,
                        "v3_division_interval": "|".join(intervals),
                        "v3_division_veto": True,
                        "v3_frame_state": frame_state,
                        "v3_proposed_label": proposed_label,
                        "v3_would_change": proposed_label != base_label,
                        "v3_division_rescue": True,
                        "v3_division_rescue_parent_candidate_id": str(
                            frame.at[rescue["parent_index"], "candidate_id"]
                        ),
                        "v3_division_rescue_child_candidate_ids": str(
                            rescue["child_candidate_ids"]
                        ),
                        "v3_division_rescue_score": float(rescue["score"]),
                    }
                )

            if "v3_division_rescue" in frame.columns:
                frame.at[index, "v3_division_rescue"] = True
                frame.at[index, "v3_division_rescue_parent_candidate_id"] = str(
                    frame.at[rescue["parent_index"], "candidate_id"]
                )
                frame.at[index, "v3_division_rescue_child_candidate_ids"] = str(
                    rescue["child_candidate_ids"]
                )
                frame.at[index, "v3_division_rescue_score"] = float(rescue["score"])


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
                + float(frame.at[index, "debris_probability"])
                + float(frame.at[index, "invalid_probability"]),
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
    # A stable object may change brightness, halo or local foreground
    # response between days.  Keep the strict pixel-static flag above for
    # auditability, but also recognize a morphology-stable triplet when
    # identity, shape, foreground quality and instance area agree.
    morphology_stable_three_frame = bool(
        simple_three
        and all(edge is not None for edge in stable_adjacent)
        and all(
            edge.identity
            >= float(settings.get("morphology_stable_identity_threshold", 0.70))
            for edge in stable_adjacent
            if edge is not None
        )
        and static_score
        >= float(settings.get("morphology_stable_static_similarity_threshold", 0.68))
        and shape_score
        >= float(settings.get("morphology_stable_shape_similarity_threshold", 0.86))
        and all(
            edge.foreground_quality
            >= float(settings.get("minimum_foreground_quality", 0.30))
            for edge in stable_adjacent
            if edge is not None
        )
        and area_ratio
        <= float(settings.get("morphology_stable_maximum_area_ratio", 1.80))
        and not growth
    )
    two_frame_static = bool(
        not three_frame_static
        and not morphology_stable_three_frame
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
    radial_fraction = pd.to_numeric(
        local.get("radial_fraction", pd.Series(0.0, index=local.index)),
        errors="coerce",
    ).fillna(0.0)
    wall_cell_context = bool(
        wall_overlap.ge(float(settings.get("static_wall_overlap_threshold", 0.90))).any()
        or radial_fraction.ge(
            float(settings.get("wall_cell_protection_band_start_fraction", 0.42))
        ).any()
    )
    strong_wall_cell_timepoints = {
        str(frame.at[index, "timepoint"])
        for index in nodes
        if _strong_wall_cell_observation(frame.loc[index], settings)
    }
    strong_wall_cell_frame_count = len(strong_wall_cell_timepoints)
    strong_wall_cell_track = bool(
        wall_cell_context
        and strong_wall_cell_frame_count
        >= int(settings.get("wall_cell_protection_minimum_frames", 2))
    )
    maximum_motion = max((edge.distance_px for edge in continuation_edges), default=np.inf)
    static_wall = bool(
        (three_frame_static or morphology_stable_three_frame)
        and float(wall_overlap.min()) >= float(settings.get("static_wall_overlap_threshold", 0.90))
        and maximum_motion <= float(settings.get("static_wall_maximum_motion_px", 5.0))
        and area_ratio <= float(settings.get("static_wall_maximum_area_ratio", 2.20))
        and not strong_wall_cell_track
    )

    # V2 no longer infers a dead-cell state.  Keep the legacy output fields
    # false/zero so older CSV readers remain compatible without allowing a
    # static object to veto a genuine T0 cell.
    suspected_dead_cell = False
    suspected_dead_cell_score = 0.0
    # Static, strongly cell-like tracks remain cell candidates.  This is a
    # label-preservation guard only; it deliberately does not create a
    # dead-cell category or score.
    static_cell_like_track = bool(
        (three_frame_static or morphology_stable_three_frame)
        and all(pre_temporal_cell(index) for index in nodes)
        and morphology_consensus_cell
        >= float(settings.get("static_cell_preservation_minimum_probability", 0.80))
        and not strong_wall_cell_track
    )
    morphology_stable_debris = bool(
        morphology_stable_three_frame
        and not debris_probability_trend
        and morphology_consensus_cell
        <= float(
            settings.get(
                "morphology_stable_debris_maximum_cell_probability", 0.62
            )
        )
    )
    strong_static_debris = bool(
        not strong_wall_cell_track
        and not static_cell_like_track
        and (
            three_frame_static
            or morphology_stable_debris
        )
    )

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
        base_confidence = max(cell, debris, invalid)
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
            "static_wall_cell_veto": strong_wall_cell_track,
            "strong_cell_frame_count": strong_wall_cell_frame_count,
            "track_id": track_id,
            # This field is shown as matched frames in the review UI.  A
            # component may contain more than one child after division, so the
            # number of graph nodes is not the number of available frames.
            "proposal_count": len(available_timepoints),
            "frame_count": len(available_timepoints),
            "pair_count": len(edges),
            "three_frame_static": three_frame_static,
            "morphology_stable_three_frame": morphology_stable_three_frame,
            "morphology_stable_debris": morphology_stable_debris,
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
            reason = (
                "three_frame_morphology_stable_debris_consensus"
                if morphology_stable_debris and not three_frame_static
                else "three_frame_static_debris_consensus"
            )
        elif two_frame_static and cell_anchor <= 0.0:
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
                reason = (
                    "three_frame_morphology_stable_debris_consensus"
                    if morphology_stable_debris and not three_frame_static
                    else "three_frame_static_debris_consensus"
                )
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
                # A high-quality cell anchor is allowed to overcome the
                # lower raw cell mass of its matched partner.  The cap is
                # still on the full three-class cell probability, so invalid
                # mass cannot be converted into cell evidence.
                0.45 if growth else 0.40 if cell_anchor > 0 else 0.12
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
        total_probability_mass = mass + invalid
        new_cell = float(min(total_probability_mass * target_cell, mass))
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

    Invalid probability is never used as cell/debris evidence and is preserved
    during redistribution. Temporal evidence can refine an uncertain
    cell/debris decision, but cannot rescue or create an invalid/background
    decision.
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
    total_probability_mass = mass + invalid_probability
    if mass <= 1e-6 or total_probability_mass <= 1e-6:
        return cell_probability, debris_probability, 0.0, False, "no_cell_debris_mass"
    conditional_cell = float(
        np.clip(
            cell_probability / total_probability_mass,
            1e-5,
            1.0 - 1e-5,
        )
    )
    base_confidence = max(cell_probability, debris_probability, invalid_probability)
    if base_confidence >= high_confidence_threshold:
        return cell_probability, debris_probability, 0.0, False, "high_confidence_base"

    ambiguity = 1.0 - abs(2.0 * conditional_cell - 1.0)
    static_excess = (static_similarity_score - static_threshold) / max(1.0 - static_threshold, 1e-6)
    frame_factor = 1.0 if frame_count >= 3 else 0.70
    strength = float(np.clip(same_object_score * static_excess * ambiguity * frame_factor, 0.0, 1.0))
    logit = np.log(conditional_cell / (1.0 - conditional_cell))
    target_cell = 1.0 / (1.0 + np.exp(-(logit - beta * strength)))
    requested_shift = total_probability_mass * max(0.0, conditional_cell - target_cell)
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
    strong_cell_probability: float = 0.80,
    strong_cell_maximum_debris_probability: float = 0.20,
    strong_cell_maximum_invalid_probability: float = 0.20,
    strong_cell_minimum_frames: int = 2,
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
    strong_cell_frames = sum(
        float(morphology[index][0]) >= strong_cell_probability
        and float(morphology[index][1]) <= strong_cell_maximum_debris_probability
        and float(morphology[index][2]) <= strong_cell_maximum_invalid_probability
        for index in available
    )
    return (
        maximum_displacement <= maximum_motion_px
        and area_ratio <= maximum_area_ratio
        and minimum_wall_overlap >= wall_overlap_threshold
        and strong_cell_frames < strong_cell_minimum_frames
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
    v3_settings = config.get("v3_temporal_behavior", {})
    v3_enabled = bool(v3_settings.get("enabled", False))
    v3_state_fusion = str(v3_settings.get("state_fusion", "off")).lower()
    v3_shadow = v3_enabled and v3_state_fusion in {"shadow", "active"}
    v3_active = v3_shadow and v3_state_fusion == "active"
    pairwise_scorer = None
    pairwise_checkpoint_status = "not_requested"
    v3_backend = str(v3_settings.get("backend", "heuristic_behavior_v1")).lower()
    if v3_shadow and v3_backend in {"pairwise", "temporal_pairwise"}:
        configured_pairwise_path = str(
            v3_settings.get("pairwise_checkpoint_path", "")
        ).strip()
        pairwise_path = (
            Path(configured_pairwise_path)
            if configured_pairwise_path
            else artifact_path(config, "v2", "models", "latest_temporal_pairwise.pt")
        )
        if not pairwise_path.exists():
            raise FileNotFoundError(
                "V3 pairwise backend is enabled but its checkpoint does not exist: "
                f"{pairwise_path}"
            )
        pairwise_scorer = load_temporal_pairwise_scorer(pairwise_path)
        pairwise_checkpoint_status = str(pairwise_path.resolve())
    elif v3_shadow:
        pairwise_checkpoint_status = "heuristic_descriptor_evidence"
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
        "v2_static_wall_cell_veto": False,
        "v2_strong_cell_evidence_frame_count": 0,
        "v2_temporal_cell_boost": 0.0,
        "v2_temporal_foreground_similarity": 0.0,
        "v2_temporal_shape_similarity": 0.0,
        "v2_temporal_change_score": 0.0,
        "v2_temporal_growth_score": 0.0,
        "v2_temporal_foreground_quality": 0.0,
        "v2_temporal_evidence_frame_count": 0,
        "v2_temporal_pair_count": 0,
        "v2_temporal_three_frame_static": False,
        "v2_temporal_morphology_stable_three_frame": False,
        "v2_temporal_morphology_consensus_cell_probability": 0.0,
        "v2_suspected_dead_cell": False,
        "v2_suspected_dead_cell_score": 0.0,
        "v2_temporal_debris_probability_trend": False,
        "v2_temporal_debris_trend_score": 0.0,
        # V3 behavior proposals are initialized to the legacy values.  They
        # become populated only when the explicitly enabled shadow/active
        # backend evaluates an identity component.
        "v3_track_behavior": "disabled",
        "v3_track_conclusion": "",
        "v3_unified_label": "",
        "v3_label_mode": "per_frame_evidence",
        "v3_wall_origin": "none",
        "v3_wall_cell_veto": False,
        "v3_wall_strong_cell_frame_count": 0,
        "v3_behavior_score": 0.0,
        "v3_division_interval": "",
        "v3_division_veto": False,
        "v3_division_rescue": False,
        "v3_division_rescue_parent_candidate_id": "",
        "v3_division_rescue_child_candidate_ids": "",
        "v3_division_rescue_score": 0.0,
        "v3_reason": "disabled",
        "v3_frame_state": "preserved",
        "v3_proposed_label": frame["integrated_label"].astype(str),
        "v3_proposed_cell_probability": frame["cell_probability"].astype(float),
        "v3_proposed_debris_probability": frame["debris_probability"].astype(float),
        "v3_proposed_invalid_probability": frame["invalid_probability"].astype(float),
        "v3_would_change": False,
        "v3_identity_score": 0.0,
        "v3_static_similarity": 0.0,
        "v3_shape_similarity": 0.0,
        "v3_morphology_change_score": 0.0,
        "v3_semantic_degradation": False,
        "v3_degradation_evidence_score": 0.0,
        "v3_foreground_quality": 0.0,
        "v3_track_frame_count": 0,
        "v3_track_pair_count": 0,
        "v3_valid_observations": False,
        "v3_persistent_cell_evidence": False,
        "v3_cell_to_debris_candidate": False,
        "v3_track_id": "",
        "v3_timepoint": frame["timepoint"].astype(str),
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
    v3_outputs: dict[int, dict[str, Any]] = {}
    cross_track_division_rescue_count = 0

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
        all_pair_evidence: dict[tuple[int, int], TemporalPairEvidence] = {}
        degree = {index: 0 for index in descriptors}
        for left_timepoint, right_timepoint in (("T0", "T1"), ("T1", "T2")):
            matches, pair_evidence = match_timepoint_objects(
                by_timepoint[left_timepoint],
                by_timepoint[right_timepoint],
                maximum_distance_px=maximum_correspondence_distance,
                minimum_identity=identity_threshold,
                maximum_shift=descriptor_shift,
                pairwise_scorer=pairwise_scorer,
            )
            all_pair_evidence.update(pair_evidence)
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
        skip_matches, skip_pair_evidence = match_timepoint_objects(
            by_timepoint["T0"],
            by_timepoint["T2"],
            maximum_distance_px=maximum_correspondence_distance * 1.15,
            minimum_identity=min(identity_threshold + 0.05, 0.95),
            maximum_shift=descriptor_shift,
            pairwise_scorer=pairwise_scorer,
        )
        all_pair_evidence.update(skip_pair_evidence)
        for edge in skip_matches:
            if degree[edge.left] or degree[edge.right]:
                continue
            edge = TemporalPairEvidence(**{**edge.__dict__, "kind": "skip_continuation"})
            edges.append(edge)
            components.union(edge.left, edge.right)
            degree[edge.left] += 1
            degree[edge.right] += 1

        component_groups = components.groups()
        node_to_component: dict[int, int] = {}
        component_nodes: dict[int, list[int]] = {}
        component_track_ids: dict[int, str] = {}
        for component_number, nodes in enumerate(component_groups, start=1):
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
            component_nodes[component_number] = list(nodes)
            for index in nodes:
                node_to_component[int(index)] = component_number
                component_track_ids[int(index)] = track_id
            outputs.update(
                _component_temporal_outputs(frame, nodes, local_edges, settings, track_id)
            )
            if v3_shadow:
                v3_outputs.update(
                    evaluate_temporal_behavior(
                        frame,
                        nodes,
                        local_edges,
                        v3_settings,
                        track_id,
                    )
                )

        cross_track_rescues = _detect_cross_track_division_rescues(
            frame,
            list(group.index),
            by_timepoint,
            descriptors,
            edges,
            all_pair_evidence,
            node_to_component,
            component_nodes,
            component_track_ids,
            v3_outputs,
            settings,
            pairwise_scorer,
        )
        _apply_cross_track_division_rescues(
            frame,
            cross_track_rescues,
            outputs,
            v3_outputs,
            component_nodes,
            v3_settings,
        )
        cross_track_division_rescue_count += len(cross_track_rescues)

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
        frame.at[index, "v2_static_wall_cell_veto"] = output[
            "static_wall_cell_veto"
        ]
        frame.at[index, "v2_strong_cell_evidence_frame_count"] = output[
            "strong_cell_frame_count"
        ]
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
        frame.at[index, "v2_temporal_morphology_stable_three_frame"] = output[
            "morphology_stable_three_frame"
        ]
        frame.at[index, "v2_temporal_morphology_consensus_cell_probability"] = output["morphology_consensus_cell"]
        frame.at[index, "v2_suspected_dead_cell"] = output["suspected_dead_cell"]
        frame.at[index, "v2_suspected_dead_cell_score"] = output["suspected_dead_cell_score"]
        frame.at[index, "v2_temporal_debris_probability_trend"] = output[
            "debris_probability_trend"
        ]
        frame.at[index, "v2_temporal_debris_trend_score"] = output[
            "debris_trend_score"
        ]
        v3_output = v3_outputs.get(index)
        if v3_output is not None:
            for key, value in v3_output.items():
                if key in frame.columns:
                    frame.at[index, key] = value
            if v3_active and _v3_proposal_is_decisive(v3_output):
                # Active fusion is deliberately opt-in.  The proposal module
                # still cannot revive a row that the instance stage marked
                # invalid; it can only choose among the existing vocabulary.
                # Review-only states (for example an uncertain wall site)
                # must preserve the already-computed V2 result instead of
                # restoring a pre-temporal label.
                proposed_label = str(v3_output["v3_proposed_label"])
                if proposed_label in {
                    "single",
                    "touching_doublet",
                    "cluster_3plus",
                    "debris",
                    "uncertain",
                    "invalid",
                    "unmarked",
                }:
                    frame.at[index, "integrated_label"] = proposed_label
                    frame.at[index, "v2_adjusted_cell_probability"] = float(
                        v3_output["v3_proposed_cell_probability"]
                    )
                    frame.at[index, "v2_adjusted_debris_probability"] = float(
                        v3_output["v3_proposed_debris_probability"]
                    )

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
        "algorithm_version": (
            "v2.1-object-graph+v3-active"
            if v3_active
            else "v2.1-object-graph+v3-shadow"
            if v3_shadow
            else "v2.1-object-graph"
        ),
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
        "morphology_stable_three_frame_instances": int(
            evaluated["v2_temporal_morphology_stable_three_frame"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "growth_evidence_instances": int(evaluated["v2_temporal_growth_score"].fillna(0.0).astype(float).gt(0).sum()),
        "cross_track_division_rescues": cross_track_division_rescue_count,
        "suspected_dead_cell_instances": int(
            evaluated["v2_suspected_dead_cell"].fillna(False).astype(bool).sum()
        ),
        "v3_enabled": v3_enabled,
        "v3_backend": str(v3_settings.get("backend", "disabled")),
        "v3_state_fusion": v3_state_fusion,
        "v3_pairwise_checkpoint": pairwise_checkpoint_status,
        "v3_evaluated_instances": len(v3_outputs),
        "v3_would_change_instances": int(
            frame.loc[list(v3_outputs), "v3_would_change"].fillna(False).astype(bool).sum()
        ) if v3_outputs else 0,
        "v3_behavior_counts": (
            frame.loc[list(v3_outputs), "v3_track_behavior"].value_counts().to_dict()
            if v3_outputs
            else {}
        ),
        "v3_wall_origin_counts": (
            frame.loc[list(v3_outputs), "v3_wall_origin"].value_counts().to_dict()
            if v3_outputs
            else {}
        ),
        "v3_reason_counts": (
            frame.loc[list(v3_outputs), "v3_reason"].value_counts().to_dict()
            if v3_outputs
            else {}
        ),
        "v3_division_veto_instances": int(
            frame.loc[list(v3_outputs), "v3_division_veto"].fillna(False).astype(bool).sum()
        ) if v3_outputs else 0,
        "v3_policy": "division/growth vetoes static-debris and dead-cell overrides; cell_to_debris requires a strong T0 cell, monotonic probability decline, valid observations, and morphology degradation; wall invalid requires positive stable wall-structure evidence",
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
            total_probability = (
                pd.to_numeric(ordered["cell_probability"], errors="coerce").fillna(0.0)
                + pd.to_numeric(ordered["debris_probability"], errors="coerce").fillna(0.0)
                + pd.to_numeric(ordered["invalid_probability"], errors="coerce").fillna(0.0)
            ).clip(lower=1e-6)
            mass = (
                pd.to_numeric(ordered["cell_probability"], errors="coerce").fillna(0.0)
                + pd.to_numeric(ordered["debris_probability"], errors="coerce").fillna(0.0)
            ).clip(lower=1e-6)
            debris_sequence = (
                pd.to_numeric(ordered["debris_probability"], errors="coerce").fillna(0.0)
                / total_probability
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
                invalid = float(frame.at[index, "invalid_probability"])
                cell_debris_mass = max(cell + debris, 1e-6)
                total_probability_mass = max(cell_debris_mass + invalid, 1e-6)
                base_cell = cell / total_probability_mass
                target_cell = _probability_sigmoid(_probability_logit(base_cell) - evidence)
                target_cell = base_cell - min(max(base_cell - target_cell, 0.0), 0.34)
                new_cell = min(total_probability_mass * target_cell, cell_debris_mass)
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

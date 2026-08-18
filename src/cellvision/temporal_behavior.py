from __future__ import annotations

"""Behavior-level temporal policy for ambiguous cell/debris candidates.

This module intentionally sits above object matching and below final label
fusion.  It consumes the existing foreground-only pair evidence and produces
an auditable proposal.  The default pipeline keeps this proposal in shadow
columns, so the policy can be evaluated before it is allowed to change
production labels.

The policy separates three questions that were previously mixed together:

* identity: are the observations the same foreground object;
* behavior: did the object divide, change/degrade, or remain stable;
* scene origin: is a wall candidate an independent compact object or a
  wall-derived texture/structure.

The cell-to-debris branch also exposes a track-level conclusion.  Its frame
labels remain evidence (cell / degenerating / debris), while the unified
conclusion is ``dead_cell`` so a review or downstream consumer does not turn
one biological event into three contradictory labels.

In particular, three-frame stability is not allowed to override a division
hypothesis, and a falling cell probability is not enough to call a cell dead:
the cell-to-debris transition also requires valid observations and genuine
morphology degradation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


TIMEPOINTS = ("T0", "T1", "T2")
CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}
DEFAULT_CELL_TO_DEBRIS_UNIFIED_LABEL = "dead_cell"


@dataclass(frozen=True)
class _TrackSignals:
    nodes_by_timepoint: dict[str, list[int]]
    pair_by_interval: dict[tuple[str, str], Any]
    continuation_edges: list[Any]
    division_intervals: tuple[str, ...]
    identity: float
    static: float
    shape: float
    foreground_quality: float
    motion: float
    morphology_change: float
    complete_triplet: bool
    valid_observations: bool
    division_veto: bool


def _value(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _flag(row: pd.Series, key: str, default: bool = True) -> bool:
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _conditional(row: pd.Series) -> float:
    """Return full three-class cell evidence, including invalid probability."""

    cell = max(_value(row, "cell_probability"), 0.0)
    debris = max(_value(row, "debris_probability"), 0.0)
    invalid = max(_value(row, "invalid_probability"), 0.0)
    return float(np.clip(cell / max(cell + debris + invalid, 1e-8), 0.0, 1.0))


def _base_label(row: pd.Series) -> str:
    value = row.get(
        "v2_pre_temporal_integrated_label",
        row.get("integrated_label", "uncertain"),
    )
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "uncertain"
    return str(value)


def _is_valid_observation(row: pd.Series, settings: dict[str, Any]) -> bool:
    maximum_invalid = float(settings.get("maximum_invalid_probability", 0.60))
    if _value(row, "invalid_probability", 0.0) >= maximum_invalid:
        return False
    if not _flag(row, "v2_mask_valid"):
        return False
    if _flag(row, "v2_wall_rejected", False):
        return False
    if not _flag(row, "v2_is_suppressed", False):
        return True
    return False


def _strong_wall_cell_observation(
    row: pd.Series,
    settings: dict[str, Any],
) -> bool:
    """Return whether absolute unary evidence can veto a wall-structure label."""

    if _value(row, "cell_probability") < float(
        settings.get("wall_cell_protection_minimum_cell_probability", 0.80)
    ):
        return False
    if _value(row, "debris_probability", 1.0) > float(
        settings.get("wall_cell_protection_maximum_debris_probability", 0.20)
    ):
        return False
    if _value(row, "invalid_probability", 1.0) > float(
        settings.get("wall_cell_protection_maximum_invalid_probability", 0.20)
    ):
        return False
    if _value(row, "v2_instance_confidence", 1.0) < float(
        settings.get("wall_cell_protection_minimum_instance_confidence", 0.55)
    ):
        return False
    if _value(row, "v2_objectness", 1.0) < float(
        settings.get("wall_cell_protection_minimum_objectness", 0.70)
    ):
        return False
    if not _flag(row, "v2_mask_valid", True):
        return False
    if not _flag(row, "v2_is_unique_instance", True):
        return False
    if _flag(row, "v2_is_suppressed", False):
        return False
    if _flag(row, "v2_wall_rejected", False):
        return False
    return True


def _semantic_label(label: str) -> str:
    value = str(label)
    if value in CELL_LABELS:
        return "cell"
    if value in {"debris", "debris_artifact"}:
        return "debris"
    if value == "invalid":
        return "invalid"
    return value


def _cell_label(row: pd.Series) -> str:
    label = _base_label(row)
    return label if label in CELL_LABELS else "single"


def _set_conditional_probability(
    row: pd.Series, target: float
) -> tuple[float, float]:
    """Set full cell probability while preserving invalid probability.

    Temporal evidence may redistribute only the cell/debris mass.  Invalid
    mass is never silently converted into a cell or debris observation.
    ``target`` is a full three-class cell probability.
    """

    cell = max(_value(row, "cell_probability"), 0.0)
    debris = max(_value(row, "debris_probability"), 0.0)
    invalid = max(_value(row, "invalid_probability"), 0.0)
    non_invalid_mass = cell + debris
    total_mass = non_invalid_mass + invalid
    if total_mass <= 1e-8 or non_invalid_mass <= 1e-8:
        return _value(row, "cell_probability"), _value(row, "debris_probability")
    target_cell = float(
        np.clip(total_mass * float(np.clip(target, 0.0, 1.0)), 0.0, non_invalid_mass)
    )
    return target_cell, non_invalid_mass - target_cell


def _nodes_by_timepoint(frame: pd.DataFrame, nodes: list[int]) -> dict[str, list[int]]:
    local = frame.loc[nodes]
    return {
        timepoint: local.index[
            local["timepoint"].astype(str) == timepoint
        ].tolist()
        for timepoint in TIMEPOINTS
    }


def _track_signals(
    frame: pd.DataFrame,
    nodes: list[int],
    edges: list[Any],
    settings: dict[str, Any],
) -> _TrackSignals:
    nodes_by_timepoint = _nodes_by_timepoint(frame, nodes)
    continuation_edges = [
        edge
        for edge in edges
        if str(getattr(edge, "kind", "continuation"))
        in {"continuation", "skip_continuation"}
    ]
    pair_by_interval: dict[tuple[str, str], Any] = {}
    for edge in continuation_edges:
        interval = (
            str(frame.at[edge.left, "timepoint"]),
            str(frame.at[edge.right, "timepoint"]),
        )
        if interval not in {("T0", "T1"), ("T1", "T2")}:
            continue
        previous = pair_by_interval.get(interval)
        if previous is None or float(edge.identity) > float(previous.identity):
            pair_by_interval[interval] = edge

    division_by_parent: dict[tuple[int, str, str], list[Any]] = {}
    division_min_cell = float(settings.get("division_minimum_cell_probability", 0.45))
    for edge in edges:
        if str(getattr(edge, "kind", "")) != "division":
            continue
        left_timepoint = str(frame.at[edge.left, "timepoint"])
        right_timepoint = str(frame.at[edge.right, "timepoint"])
        if (left_timepoint, right_timepoint) not in {
            ("T0", "T1"),
            ("T1", "T2"),
        }:
            continue
        if (
            _conditional(frame.loc[edge.left]) < division_min_cell
            or _conditional(frame.loc[edge.right]) < division_min_cell
        ):
            continue
        key = (int(edge.left), left_timepoint, right_timepoint)
        division_by_parent.setdefault(key, []).append(edge)
    division_intervals = tuple(
        sorted(
            {
                f"{left_timepoint}->{right_timepoint}"
                for (_parent, left_timepoint, right_timepoint), children in division_by_parent.items()
                if len({int(edge.right) for edge in children}) >= 2
            }
        )
    )
    # Multiplicity growth remains a useful fallback when the detector did not
    # emit a complete two-child division group.
    growth_intervals: set[str] = set()
    rank = {"single": 1, "touching_doublet": 2, "cluster_3plus": 3}
    for edge in continuation_edges:
        left_label = _base_label(frame.loc[edge.left])
        right_label = _base_label(frame.loc[edge.right])
        if (
            left_label in CELL_LABELS
            and right_label in CELL_LABELS
            and rank.get(right_label, 1) > rank.get(left_label, 1)
        ):
            growth_intervals.add(
                f"{frame.at[edge.left, 'timepoint']}->{frame.at[edge.right, 'timepoint']}"
            )
    all_division_intervals = tuple(sorted(set(division_intervals) | growth_intervals))
    division_veto = bool(all_division_intervals)

    identity = float(np.mean([float(edge.identity) for edge in continuation_edges])) if continuation_edges else 0.0
    static = float(np.mean([float(edge.static) for edge in continuation_edges])) if continuation_edges else 0.0
    shape = float(np.mean([float(edge.tolerant_shape) for edge in continuation_edges])) if continuation_edges else 0.0
    foreground_quality = (
        float(np.min([float(edge.foreground_quality) for edge in continuation_edges]))
        if continuation_edges
        else 0.0
    )
    motion = (
        float(np.mean([np.clip(float(edge.distance_px) / 72.0, 0.0, 1.0) for edge in continuation_edges]))
        if continuation_edges
        else 0.0
    )
    area_change = (
        float(np.mean([1.0 - float(edge.area_similarity) for edge in continuation_edges]))
        if continuation_edges
        else 0.0
    )
    # Intensity/brightness variation is deliberately excluded from the main
    # degradation score.  Brightness changes are a known nuisance for debris;
    # shape and area are the stronger death/degradation signals.
    morphology_change = float(
        np.clip(0.75 * (1.0 - shape) + 0.25 * area_change, 0.0, 1.0)
    )
    explicit_change = pd.to_numeric(
        frame.loc[nodes].get(
            "v3_morphology_degradation_score",
            pd.Series(np.nan, index=nodes),
        ),
        errors="coerce",
    ).dropna()
    if not explicit_change.empty:
        morphology_change = max(morphology_change, float(explicit_change.mean()))

    valid_observations = all(
        _is_valid_observation(frame.loc[index], settings) for index in nodes
    )
    complete_triplet = all(len(nodes_by_timepoint[timepoint]) == 1 for timepoint in TIMEPOINTS)
    return _TrackSignals(
        nodes_by_timepoint=nodes_by_timepoint,
        pair_by_interval=pair_by_interval,
        continuation_edges=continuation_edges,
        division_intervals=all_division_intervals,
        identity=identity,
        static=static,
        shape=shape,
        foreground_quality=foreground_quality,
        motion=motion,
        morphology_change=morphology_change,
        complete_triplet=complete_triplet,
        valid_observations=valid_observations,
        division_veto=division_veto,
    )


def _wall_signal(
    frame: pd.DataFrame,
    nodes: list[int],
    signals: _TrackSignals,
    settings: dict[str, Any],
) -> tuple[str, float, str, int]:
    """Return (origin, score, reason, strong-cell frames) for a wall site.

    Being near the wall or having a large mask overlap is only context.  The
    invalid decision requires a stable same-site object plus evidence that the
    foreground is part of a directional/connected wall structure.  A compact
    rescue object, division, or genuine morphology change wins over the
    texture-invalid hypothesis.
    """

    local = frame.loc[nodes]
    wall_start = float(settings.get("wall_band_start_fraction", 0.42))
    overlap_threshold = float(settings.get("wall_overlap_threshold", 0.90))
    minimum_neighbors = int(settings.get("wall_minimum_neighbor_count", 3))
    anisotropy_threshold = float(settings.get("wall_anisotropy_threshold", 0.55))
    chain_anisotropy_threshold = float(
        settings.get("wall_chain_anisotropy_threshold", 0.40)
    )
    maximum_motion = float(settings.get("wall_maximum_motion_px", 5.0))
    compact_blobness = float(settings.get("wall_compact_blobness_threshold", 0.34))

    radial = pd.to_numeric(
        local.get("radial_fraction", pd.Series(0.0, index=local.index)),
        errors="coerce",
    ).fillna(0.0)
    wall_overlap = pd.to_numeric(
        local.get("v2_wall_overlap", pd.Series(0.0, index=local.index)),
        errors="coerce",
    ).fillna(0.0)
    neighbour_count = pd.to_numeric(
        local.get("wall_neighbor_count", pd.Series(0.0, index=local.index)),
        errors="coerce",
    ).fillna(0.0)
    anisotropy_columns = []
    for column in ("review_anisotropy", "background_anisotropy"):
        if column in local:
            anisotropy_columns.append(
                pd.to_numeric(local[column], errors="coerce").fillna(0.0)
            )
    anisotropy = (
        pd.concat(anisotropy_columns, axis=1).max(axis=1)
        if anisotropy_columns
        else pd.Series(0.0, index=local.index)
    )
    blobness = pd.to_numeric(
        local.get("wall_rescue_blobness", pd.Series(0.0, index=local.index)),
        errors="coerce",
    ).fillna(0.0)
    sources = local.get(
        "candidate_source", pd.Series("", index=local.index)
    ).fillna("").astype(str)
    strong_cell_timepoints = {
        str(frame.at[index, "timepoint"])
        for index in nodes
        if _strong_wall_cell_observation(frame.loc[index], settings)
    }
    strong_cell_frame_count = len(strong_cell_timepoints)
    strong_cell_track = bool(
        strong_cell_frame_count
        >= int(settings.get("wall_cell_protection_minimum_frames", 2))
    )

    wall_context = bool(
        radial.ge(wall_start).any() or wall_overlap.ge(overlap_threshold).any()
    )
    if not wall_context:
        return "none", 0.0, "not_in_wall_context", 0

    # A wall-attached cell can overlap the wall mask completely.  Repeated
    # high-quality absolute cell predictions therefore have explicit veto
    # priority over geometric wall context.  One isolated high frame remains
    # insufficient, which preserves the stable-wall rejection path for B8-
    # like texture candidates.
    if strong_cell_track:
        strong_cell_score = float(
            np.mean(
                [
                    _value(frame.loc[index], "cell_probability")
                    for index in nodes
                    if _strong_wall_cell_observation(frame.loc[index], settings)
                ]
            )
        )
        score = float(
            np.clip(0.70 * strong_cell_score + 0.30 * signals.identity, 0.0, 1.0)
        )
        return (
            "wall_independent_object",
            score,
            "strong_multiframe_cell_evidence_overrides_wall_structure",
            strong_cell_frame_count,
        )

    adjacent_motion = max(
        (float(edge.distance_px) for edge in signals.continuation_edges),
        default=np.inf,
    )
    same_site = bool(
        signals.complete_triplet
        and signals.identity >= float(settings.get("wall_minimum_identity", 0.70))
        and signals.static >= float(settings.get("wall_minimum_static_similarity", 0.78))
        and signals.shape >= float(settings.get("wall_minimum_shape_similarity", 0.82))
        and adjacent_motion <= maximum_motion
    )
    high_overlap_site = bool(
        radial.ge(wall_start).all()
        and wall_overlap.ge(overlap_threshold).all()
    )
    wall_chain = bool(neighbour_count.ge(minimum_neighbors).all())
    directional = bool(anisotropy.ge(anisotropy_threshold).all())
    chain_directional = bool(
        wall_chain and anisotropy.mean() >= chain_anisotropy_threshold
    )
    structure_evidence = bool(
        same_site
        and high_overlap_site
        and (directional or chain_directional)
    )

    # ``wall_residual_peak`` and its dense response are candidate-generation
    # signals, not proof that the candidate is an object independent of the
    # wall.  Requiring compact morphology in repeated frames prevents a
    # stable high-contrast wall texture from overriding positive wall-
    # structure evidence.  A dedicated cell-rescue proposal or biological
    # change can still protect a genuine wall-attached cell.
    compact_frame_count = int(blobness.ge(compact_blobness).sum())
    minimum_compact_frames = min(
        len(local),
        max(1, int(settings.get("wall_compact_minimum_frames", 2))),
    )
    compact_object = bool(
        compact_frame_count >= minimum_compact_frames
        or sources.eq("wall_cell_rescue_peak").any()
        or (
            _conditional(local.loc[local.index[0]])
            >= float(settings.get("wall_compact_cell_probability", 0.80))
            and signals.morphology_change
            >= float(settings.get("wall_compact_change_threshold", 0.35))
        )
    )
    biological_change = bool(
        signals.division_veto
        or signals.morphology_change
        >= float(settings.get("wall_biological_change_threshold", 0.50))
    )
    if structure_evidence and not compact_object and not biological_change:
        score = float(
            np.clip(
                0.35 * signals.identity
                + 0.25 * signals.static
                + 0.20 * wall_overlap.mean()
                + 0.20 * anisotropy.mean(),
                0.0,
                1.0,
            )
        )
        return (
            "wall_structure",
            score,
            "stable_wall_site_structure",
            strong_cell_frame_count,
        )
    if compact_object or biological_change:
        score = float(
            np.clip(
                0.45 * max(_conditional(local.loc[index]) for index in local.index)
                + 0.30 * signals.morphology_change
                + 0.25 * float(blobness.max()),
                0.0,
                1.0,
            )
        )
        return (
            "wall_independent_object",
            score,
            "wall_object_requires_biological_branch",
            strong_cell_frame_count,
        )
    return (
        "wall_uncertain",
        float(np.clip(0.5 * signals.identity + 0.5 * wall_overlap.mean(), 0.0, 1.0)),
        "wall_site_insufficient_structure_evidence",
        strong_cell_frame_count,
    )


def _preserved_label(row: pd.Series) -> str:
    return _base_label(row)


def evaluate_temporal_behavior(
    frame: pd.DataFrame,
    nodes: list[int],
    edges: list[Any],
    settings: dict[str, Any] | None = None,
    track_id: str = "",
) -> dict[int, dict[str, Any]]:
    """Evaluate one identity component and return per-frame shadow proposals.

    The returned labels are proposals, not an implicit mutation of ``frame``.
    ``state``/``track_behavior`` carry the richer interpretation while the
    label stays within the existing production vocabulary.
    """

    settings = dict(settings or {})
    signals = _track_signals(frame, nodes, edges, settings)
    conditional = {index: _conditional(frame.loc[index]) for index in nodes}
    valid = {index: _is_valid_observation(frame.loc[index], settings) for index in nodes}
    timepoint_for = {
        index: str(frame.at[index, "timepoint"]) for index in nodes
    }
    ordered = {
        timepoint: values[0]
        for timepoint, values in signals.nodes_by_timepoint.items()
        if len(values) == 1
    }
    pair_identity_ok = bool(
        all(
            interval in signals.pair_by_interval
            and float(signals.pair_by_interval[interval].identity)
            >= float(settings.get("minimum_identity", 0.58))
            and float(signals.pair_by_interval[interval].foreground_quality)
            >= float(settings.get("minimum_foreground_quality", 0.30))
            for interval in (("T0", "T1"), ("T1", "T2"))
        )
    )
    stable_pairs = bool(
        pair_identity_ok
        and signals.static >= float(settings.get("stable_static_similarity", 0.82))
        and signals.shape >= float(settings.get("stable_shape_similarity", 0.90))
        and signals.foreground_quality
        >= float(settings.get("minimum_foreground_quality", 0.30))
    )
    # Raw foreground similarity is sensitive to illumination, focus/halo and
    # small registration differences.  Keep the strict static gate above for
    # the strongest evidence, but add a morphology-only gate for objects
    # whose identity, shape and area remain stable across all three frames.
    # This is the path used for ambiguous cell/debris candidates such as a
    # stable dark artifact whose pixel appearance changes between days.
    morphology_stable_pairs = bool(
        pair_identity_ok
        and not signals.division_veto
        and signals.static
        >= float(settings.get("morphology_stable_static_similarity", 0.68))
        and signals.shape
        >= float(settings.get("morphology_stable_shape_similarity", 0.86))
        and signals.foreground_quality
        >= float(settings.get("minimum_foreground_quality", 0.30))
        and signals.morphology_change
        <= float(settings.get("morphology_stable_maximum_change", 0.25))
    )
    t0 = ordered.get("T0")
    t1 = ordered.get("T1")
    t2 = ordered.get("T2")
    cell_strong = float(settings.get("cell_strong_threshold", 0.80))
    final_cell_max = float(settings.get("cell_to_debris_final_cell_max", 0.50))
    probability_drop = float(settings.get("cell_to_debris_minimum_probability_drop", 0.30))
    monotonic_tolerance = float(settings.get("cell_to_debris_monotonic_tolerance", 0.05))
    degradation_threshold = float(
        settings.get("cell_to_debris_minimum_morphology_degradation", 0.50)
    )
    strong_t0_cell = bool(
        t0 is not None
        and valid.get(t0, False)
        and conditional[t0] >= cell_strong
        and _value(frame.loc[t0], "cell_probability")
        + _value(frame.loc[t0], "debris_probability")
        >= float(settings.get("minimum_cell_debris_mass", 0.45))
    )
    declining_cell = bool(
        t0 is not None
        and t1 is not None
        and t2 is not None
        and conditional[t0] - conditional[t2] >= probability_drop
        and conditional[t1] <= conditional[t0] + monotonic_tolerance
        and conditional[t2] <= conditional[t1] + monotonic_tolerance
        and conditional[t2] <= final_cell_max
    )
    later_labels = {
        _semantic_label(_base_label(frame.loc[index]))
        for index in (t1, t2)
        if index is not None
    }
    semantic_degradation = bool(
        signals.complete_triplet
        and signals.valid_observations
        and pair_identity_ok
        and declining_cell
        and conditional[t0] - conditional[t2]
        >= float(settings.get("cell_to_debris_minimum_semantic_drop", 0.30))
        and (
            conditional[t1]
            <= float(settings.get("cell_to_debris_semantic_midpoint_max", 0.70))
            or "debris" in later_labels
            or "uncertain" in later_labels
        )
    )
    semantic_degradation_score = float(
        np.clip(
            (conditional[t0] - conditional[t2])
            if semantic_degradation
            else 0.0,
            0.0,
            1.0,
        )
    )
    degradation_evidence_score = max(
        signals.morphology_change,
        semantic_degradation_score,
    )
    cell_to_debris = bool(
        signals.complete_triplet
        and signals.valid_observations
        # Identity must remain reliable, but morphology is expected to change
        # in this branch, so do not require the high static/shape gates used
        # by the stable-debris rule.
        and pair_identity_ok
        and not signals.division_veto
        and strong_t0_cell
        and declining_cell
        and (
            signals.morphology_change >= degradation_threshold
            or semantic_degradation
        )
    )
    decline_without_change = bool(
        signals.complete_triplet
        and strong_t0_cell
        and declining_cell
        and not cell_to_debris
        and not signals.division_veto
        and signals.morphology_change < degradation_threshold
    )

    mean_cell = float(np.mean(list(conditional.values()))) if conditional else 0.0
    persistent_cell = bool(
        sum(value >= float(settings.get("persistent_cell_threshold", 0.72)) for value in conditional.values())
        >= int(settings.get("persistent_cell_minimum_frames", 2))
    )
    strict_stable_debris = bool(
        signals.complete_triplet
        and signals.valid_observations
        and stable_pairs
        and not signals.division_veto
        and signals.morphology_change
        <= float(settings.get("stable_debris_maximum_morphology_change", 0.25))
        and not persistent_cell
        and mean_cell <= float(settings.get("stable_debris_maximum_cell_probability", 0.55))
    )
    morphology_stable_debris = bool(
        signals.complete_triplet
        and signals.valid_observations
        and morphology_stable_pairs
        and not persistent_cell
        and mean_cell
        <= float(
            settings.get(
                "morphology_stable_debris_maximum_cell_probability", 0.62
            )
        )
    )
    stable_debris = bool(strict_stable_debris or morphology_stable_debris)
    static_cell_conflict = bool(
        signals.complete_triplet
        and (stable_pairs or morphology_stable_pairs)
        and not signals.division_veto
        and not stable_debris
        and not cell_to_debris
        and (
            any(value >= cell_strong for value in conditional.values())
            or persistent_cell
        )
    )

    wall_origin, wall_score, wall_reason, wall_strong_cell_frame_count = _wall_signal(
        frame, nodes, signals, settings
    )
    if wall_origin == "wall_structure" and not (cell_to_debris or signals.division_veto):
        track_behavior = "wall_structure_invalid"
        reason = wall_reason
        behavior_score = wall_score
    elif signals.division_veto:
        track_behavior = "division_or_growth"
        reason = "division_or_growth_vetoes_static_debris_and_dead_cell"
        behavior_score = float(np.clip(0.65 + 0.35 * signals.morphology_change, 0.0, 1.0))
    elif cell_to_debris:
        track_behavior = "cell_to_debris"
        reason = "strong_t0_cell_monotonic_decline_with_morphology_degradation"
        behavior_score = float(
            np.clip(
                0.35 * (conditional[t0] - conditional[t2])
                + 0.35 * signals.morphology_change
                + 0.30 * signals.identity,
                0.0,
                1.0,
            )
        )
    elif stable_debris:
        track_behavior = "stable_debris"
        reason = (
            "three_frame_morphology_stable_noncell_without_persistent_cell_evidence"
            if morphology_stable_debris and not strict_stable_debris
            else "three_frame_stable_noncell_without_persistent_cell_evidence"
        )
        behavior_score = float(
            np.clip(
                0.40 * signals.identity
                + 0.35 * signals.static
                + 0.25 * (1.0 - mean_cell),
                0.0,
                1.0,
            )
        )
    elif decline_without_change:
        track_behavior = "decline_without_morphology_evidence"
        reason = "cell_probability_decline_without_morphology_degradation"
        behavior_score = float(np.clip(conditional[t0] - conditional[t2], 0.0, 1.0))
    elif static_cell_conflict:
        track_behavior = "stable_cell_or_conflict"
        reason = "stable_track_has_strong_cell_evidence_no_debris_override"
        behavior_score = float(np.clip(0.5 * signals.identity + 0.5 * mean_cell, 0.0, 1.0))
    elif wall_origin == "wall_uncertain":
        track_behavior = "wall_uncertain"
        reason = wall_reason
        behavior_score = wall_score
    elif wall_origin == "wall_independent_object":
        track_behavior = "wall_independent_object"
        reason = wall_reason
        behavior_score = wall_score
    else:
        track_behavior = "no_decisive_temporal_evidence"
        reason = "no_decisive_temporal_behavior"
        behavior_score = float(np.clip(0.50 * signals.identity + 0.50 * signals.morphology_change, 0.0, 1.0))

    division_interval = "|".join(signals.division_intervals)
    unified_label = (
        str(
            settings.get(
                "cell_to_debris_unified_label",
                DEFAULT_CELL_TO_DEBRIS_UNIFIED_LABEL,
            )
        ).strip()
        if track_behavior == "cell_to_debris"
        else ""
    )
    label_mode = "unified_track" if unified_label else "per_frame_evidence"
    outputs: dict[int, dict[str, Any]] = {}
    for index in nodes:
        row = frame.loc[index]
        base_label = _preserved_label(row)
        base_cell = _value(row, "cell_probability")
        base_debris = _value(row, "debris_probability")
        proposed_label = base_label
        proposed_cell = base_cell
        proposed_debris = base_debris
        frame_state = "preserved"
        proposed_invalid = _value(row, "invalid_probability")

        if not valid[index] or _semantic_label(base_label) == "invalid":
            proposed_label = "invalid"
            frame_state = "invalid"
        elif track_behavior == "wall_structure_invalid":
            proposed_label = "invalid"
            frame_state = "invalid"
            proposed_invalid = max(proposed_invalid, 1.0)
        elif track_behavior == "division_or_growth":
            # A division veto protects cell-like parents/children from the
            # static-debris and suspected-dead rules.  It does not invent a
            # cell from a clearly invalid or clearly debris-only row.
            if conditional[index] >= float(settings.get("division_cell_decision_threshold", 0.62)):
                proposed_label = _cell_label(row)
                frame_state = "cell"
            elif _semantic_label(base_label) == "debris" or conditional[index] <= float(settings.get("division_debris_decision_threshold", 0.38)):
                proposed_label = "debris"
                frame_state = "debris"
            else:
                proposed_label = base_label
                frame_state = "uncertain"
        elif track_behavior == "cell_to_debris":
            if index == t0:
                proposed_label = _cell_label(row)
                frame_state = "cell"
            elif index == t2:
                proposed_cell, proposed_debris = _set_conditional_probability(
                    row,
                    min(
                        conditional[index],
                        float(settings.get("cell_to_debris_output_cell_max", 0.45)),
                    ),
                )
                proposed_label = "debris"
                frame_state = "debris"
            else:
                # T1 is intentionally an observable transitional state.  It
                # is not labelled as a healthy cell merely because T0 was one,
                # and it is not forced to debris before the final frame.
                proposed_label = (
                    _cell_label(row)
                    if conditional[index] >= float(settings.get("cell_to_debris_t1_cell_threshold", 0.62))
                    else "uncertain"
                )
                frame_state = "degenerating"
        elif track_behavior == "stable_debris":
            proposed_cell, proposed_debris = _set_conditional_probability(
                row,
                min(
                    conditional[index],
                    float(settings.get("stable_debris_output_cell_max", 0.38)),
                ),
            )
            proposed_label = "debris"
            frame_state = "debris"
        elif track_behavior == "stable_cell_or_conflict":
            if conditional[index] >= float(settings.get("stable_cell_decision_threshold", 0.62)):
                proposed_label = _cell_label(row)
                frame_state = "cell"
            else:
                proposed_label = base_label
                frame_state = "uncertain"
        elif track_behavior == "wall_uncertain":
            # Insufficient wall evidence is a review state, not permission to
            # overwrite a pre-existing cell/debris decision.  This keeps the
            # active switch from turning unresolved wall cells into negatives.
            proposed_label = base_label
            frame_state = "uncertain"
        elif track_behavior == "wall_independent_object":
            if conditional[index] >= float(settings.get("stable_cell_decision_threshold", 0.62)):
                proposed_label = _cell_label(row)
                frame_state = "cell"
            elif _semantic_label(base_label) == "debris":
                proposed_label = "debris"
                frame_state = "debris"
            else:
                proposed_label = "uncertain"
                frame_state = "uncertain"

        outputs[index] = {
            "v3_track_behavior": track_behavior,
            "v3_track_conclusion": unified_label,
            "v3_unified_label": unified_label,
            "v3_label_mode": label_mode,
            "v3_wall_origin": wall_origin,
            "v3_wall_cell_veto": bool(
                wall_strong_cell_frame_count
                >= int(settings.get("wall_cell_protection_minimum_frames", 2))
            ),
            "v3_wall_strong_cell_frame_count": wall_strong_cell_frame_count,
            "v3_behavior_score": behavior_score,
            "v3_division_interval": division_interval,
            "v3_division_veto": signals.division_veto,
            "v3_reason": reason,
            "v3_frame_state": frame_state,
            "v3_proposed_label": proposed_label,
            "v3_proposed_cell_probability": proposed_cell,
            "v3_proposed_debris_probability": proposed_debris,
            "v3_proposed_invalid_probability": proposed_invalid,
            "v3_would_change": _semantic_label(proposed_label) != _semantic_label(base_label),
            "v3_identity_score": signals.identity,
            "v3_static_similarity": signals.static,
            "v3_shape_similarity": signals.shape,
            "v3_morphology_change_score": signals.morphology_change,
            "v3_semantic_degradation": semantic_degradation,
            "v3_degradation_evidence_score": degradation_evidence_score,
            "v3_foreground_quality": signals.foreground_quality,
            "v3_track_frame_count": len(signals.nodes_by_timepoint),
            "v3_track_pair_count": len(signals.continuation_edges),
            "v3_valid_observations": signals.valid_observations,
            "v3_persistent_cell_evidence": persistent_cell,
            "v3_cell_to_debris_candidate": cell_to_debris,
            "v3_track_id": track_id,
            "v3_timepoint": timepoint_for[index],
        }
    return outputs

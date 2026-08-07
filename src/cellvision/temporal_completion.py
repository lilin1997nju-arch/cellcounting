from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import artifact_path
from .multiplicity_identity import candidate_multiplicity_label


CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}
TIME_INDEX = {"T0": 0, "T1": 1, "T2": 2}


def _column(
    frame: pd.DataFrame,
    name: str,
    default: float | str,
) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _anchor_mask(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.Series:
    probability = _column(frame, "cell_probability", 0.0).fillna(0).astype(float)
    confidence = _column(
        frame, "integrated_confidence", 0.0
    ).fillna(0).astype(float)
    manual = _column(frame, "candidate_source", "").astype(str).eq(
        "manual_cell_anchor"
    )
    recovered = _column(
        frame, "temporal_completion_status", ""
    ).astype(str).isin(["auto_promoted", "confidence_recovered"])
    return frame["integrated_label"].isin(CELL_LABELS) & (
        manual
        | recovered
        | (
            probability
            >= float(settings.get("anchor_cell_probability", 0.82))
        )
        | (
            confidence
            >= float(settings.get("anchor_integrated_confidence", 0.82))
        )
    )


def _recoverable_mask(
    frame: pd.DataFrame,
    config: dict[str, Any],
    settings: dict[str, Any],
) -> pd.Series:
    filters = config.get("candidate_filter", {})
    hard_wall = float(filters.get("hard_wall_exclusion_fraction", 0.44))
    radial = _column(frame, "radial_fraction", 0.0).fillna(0).astype(float)
    area = _column(frame, "area_px", 0.0).fillna(0).astype(float)
    source = _column(frame, "candidate_source", "").astype(str)
    auto_status = _column(frame, "auto_status", "").astype(str)
    manual = source.eq("manual_cell_anchor")
    cell_probability = _column(
        frame, "cell_probability", 0.0
    ).fillna(0).astype(float)
    return (
        area.between(
            float(settings.get("minimum_area_px", 8)),
            float(settings.get("maximum_area_px", 1200)),
        )
        & (manual | radial.lt(hard_wall))
        & ~auto_status.eq("deterministic_wall_invalid")
        & (
            ~frame["integrated_label"].eq("invalid")
            | source.isin(
                [
                    "manual_cell_anchor",
                    "manual_annotation_anchor",
                    "raw_dense_peak",
                ]
            )
            | cell_probability.ge(
                float(settings.get("invalid_cell_probability_floor", 0.12))
            )
        )
    )


def _shape_score(source: pd.Series, candidate: pd.Series) -> float:
    area_ratio = abs(
        np.log(
            max(float(candidate.get("area_px", 1.0)), 1.0)
            / max(float(source.get("area_px", 1.0)), 1.0)
        )
    )
    shape_delta = (
        abs(
            float(candidate.get("circularity", 0.5))
            - float(source.get("circularity", 0.5))
        )
        + 0.6
        * abs(
            float(candidate.get("eccentricity", 0.5))
            - float(source.get("eccentricity", 0.5))
        )
        + 0.6
        * abs(
            float(candidate.get("solidity", 0.5))
            - float(source.get("solidity", 0.5))
        )
    )
    return float(
        np.exp(-area_ratio / 1.35) * np.exp(-shape_delta / 0.75)
    )


def _aligned(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [float(row["aligned_x_px"]), float(row["aligned_y_px"])],
        dtype=float,
    )


def _matching_third_anchor(
    source: pd.Series,
    anchors: pd.DataFrame,
    source_timepoint: str,
    target_timepoint: str,
    maximum_motion: float,
) -> pd.Series | None:
    remaining = [
        timepoint
        for timepoint in TIME_INDEX
        if timepoint not in {source_timepoint, target_timepoint}
    ]
    if not remaining:
        return None
    local = anchors[anchors["timepoint"] == remaining[0]]
    if local.empty:
        return None
    distances = np.linalg.norm(
        local[["aligned_x_px", "aligned_y_px"]].to_numpy(float)
        - _aligned(source),
        axis=1,
    )
    eligible = np.flatnonzero(distances <= maximum_motion)
    if not len(eligible):
        return None
    best_index = min(
        eligible,
        key=lambda position: (
            0.65 * distances[position]
            + 180.0
            * (1.0 - _shape_score(source, local.iloc[position]))
        ),
    )
    return local.iloc[int(best_index)]


def _predicted_position(
    source: pd.Series,
    third: pd.Series | None,
    source_timepoint: str,
    target_timepoint: str,
) -> tuple[np.ndarray, str]:
    source_xy = _aligned(source)
    if third is None:
        return source_xy, "registered_source_position"
    third_timepoint = str(third["timepoint"])
    denominator = TIME_INDEX[third_timepoint] - TIME_INDEX[source_timepoint]
    if denominator == 0:
        return source_xy, "registered_source_position"
    ratio = (
        TIME_INDEX[target_timepoint] - TIME_INDEX[source_timepoint]
    ) / denominator
    predicted = source_xy + ratio * (_aligned(third) - source_xy)
    mode = "interpolated_motion" if 0 < ratio < 1 else "extrapolated_motion"
    return predicted, mode


def _candidate_score(
    source: pd.Series,
    candidate: pd.Series,
    predicted_xy: np.ndarray,
    third: pd.Series | None,
    settings: dict[str, Any],
) -> dict[str, float]:
    distance = float(np.linalg.norm(_aligned(candidate) - predicted_xy))
    distance_scale = float(settings.get("distance_score_scale_px", 300))
    distance_score = float(np.exp(-distance / max(distance_scale, 1.0)))
    shape_score = _shape_score(source, candidate)
    cell_probability = float(
        np.clip(candidate.get("cell_probability", 0.0), 0.0, 1.0)
    )
    debris_probability = float(
        np.clip(candidate.get("debris_probability", 0.0), 0.0, 1.0)
    )
    source_confidence = float(
        np.clip(source.get("integrated_confidence", 0.0), 0.0, 1.0)
    )
    support_score = 1.0 if third is not None else 0.55
    label = str(candidate.get("integrated_label", "uncertain"))
    label_penalty = {
        "debris": 0.05,
        "invalid": 0.10,
        "uncertain": 0.0,
    }.get(label, 0.0)
    score = (
        0.30 * cell_probability
        + 0.22 * shape_score
        + 0.25 * distance_score
        + 0.13 * support_score
        + 0.10 * source_confidence
        - 0.18 * debris_probability
        - label_penalty
    )
    return {
        "completion_score": float(np.clip(score, 0.0, 1.0)),
        "distance_px": distance,
        "distance_score": distance_score,
        "shape_score": shape_score,
        "cell_probability": cell_probability,
        "debris_probability": debris_probability,
        "source_confidence": source_confidence,
        "support_score": support_score,
    }


def _promoted_label(candidate: pd.Series) -> str:
    return candidate_multiplicity_label(candidate)


def complete_temporal_candidates(
    config: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Recover missing temporal cell candidates before lineage construction.

    The function never revives deterministic wall candidates. It promotes only
    mutually unambiguous candidates; medium-confidence conflicts are exported
    for review without changing their original class.
    """
    result = frame.copy()
    settings = config.get("temporal_completion", {})
    result["original_integrated_label"] = result["integrated_label"].astype(str)
    result["temporal_completion_score"] = 0.0
    result["temporal_completion_status"] = "not_evaluated"
    result["temporal_completion_source_id"] = ""
    result["temporal_completion_direction"] = ""
    recoverable = _recoverable_mask(result, config, settings)
    search_radii = [
        float(value)
        for value in settings.get("search_radii_px", [96, 192, 384, 768])
    ]
    maximum_radius = max(search_radii)
    existing_radius = float(settings.get("existing_match_radius_px", 640))
    maximum_motion = float(settings.get("maximum_anchor_motion_px", 1400))
    auto_threshold = float(settings.get("automatic_promotion_score", 0.78))
    review_threshold = float(settings.get("review_score", 0.58))
    ambiguity_margin = float(settings.get("ambiguity_margin", 0.10))
    strong_debris = float(settings.get("strong_debris_probability", 0.72))
    passes = int(settings.get("passes", 2))
    directions = [
        ("T2", "T0"),
        ("T1", "T0"),
        ("T2", "T1"),
        ("T0", "T1"),
        ("T1", "T2"),
        ("T0", "T2"),
    ]
    suggestions: list[dict[str, Any]] = []
    promoted_ids: set[str] = set()
    evaluated_gaps: set[tuple[str, str, str]] = set()

    for pass_index in range(passes):
        pass_proposals: list[dict[str, Any]] = []
        anchors = result[_anchor_mask(result, settings)].copy()
        for well, well_frame in result.groupby("well", sort=True):
            if str(well).upper() == "A1":
                continue
            well_anchors = anchors[anchors["well"] == well]
            for source_timepoint, target_timepoint in directions:
                sources = well_anchors[
                    well_anchors["timepoint"] == source_timepoint
                ]
                target_anchors = well_anchors[
                    well_anchors["timepoint"] == target_timepoint
                ]
                target_pool = result[
                    (result["well"] == well)
                    & (result["timepoint"] == target_timepoint)
                    & recoverable
                ]
                for source_index, source in sources.iterrows():
                    gap_key = (
                        str(source["candidate_id"]),
                        source_timepoint,
                        target_timepoint,
                    )
                    third = _matching_third_anchor(
                        source,
                        well_anchors,
                        source_timepoint,
                        target_timepoint,
                        maximum_motion,
                    )
                    predicted_xy, prediction_mode = _predicted_position(
                        source,
                        third,
                        source_timepoint,
                        target_timepoint,
                    )
                    if not target_anchors.empty:
                        anchor_distances = np.linalg.norm(
                            target_anchors[
                                ["aligned_x_px", "aligned_y_px"]
                            ].to_numpy(float)
                            - predicted_xy,
                            axis=1,
                        )
                        if float(anchor_distances.min()) <= existing_radius:
                            continue
                    evaluated_gaps.add(gap_key)
                    if target_pool.empty:
                        continue
                    distances = np.linalg.norm(
                        target_pool[
                            ["aligned_x_px", "aligned_y_px"]
                        ].to_numpy(float)
                        - predicted_xy,
                        axis=1,
                    )
                    local_positions = np.flatnonzero(
                        distances <= maximum_radius
                    )
                    if not len(local_positions):
                        continue
                    scored = []
                    for position in local_positions:
                        candidate = target_pool.iloc[int(position)]
                        evidence = _candidate_score(
                            source,
                            candidate,
                            predicted_xy,
                            third,
                            settings,
                        )
                        scored.append((int(position), evidence))
                    scored.sort(
                        key=lambda item: item[1]["completion_score"],
                        reverse=True,
                    )
                    best_position, best_evidence = scored[0]
                    second_score = (
                        scored[1][1]["completion_score"]
                        if len(scored) > 1
                        else 0.0
                    )
                    candidate = target_pool.iloc[best_position]
                    pass_proposals.append(
                        {
                            "pass_index": pass_index,
                            "well": well,
                            "source_index": int(source_index),
                            "source_candidate_id": str(source["candidate_id"]),
                            "source_timepoint": source_timepoint,
                            "target_timepoint": target_timepoint,
                            "candidate_index": int(candidate.name),
                            "candidate_id": str(candidate["candidate_id"]),
                            "predicted_x_px": float(predicted_xy[0]),
                            "predicted_y_px": float(predicted_xy[1]),
                            "prediction_mode": prediction_mode,
                            "original_label": str(
                                candidate["integrated_label"]
                            ),
                            "proposed_label": _promoted_label(candidate),
                            "source_margin": float(
                                best_evidence["completion_score"] - second_score
                            ),
                            **best_evidence,
                        }
                    )

        if not pass_proposals:
            continue
        proposal_frame = pd.DataFrame(pass_proposals)
        proposal_frame["ownership_margin"] = 1.0
        for _, local in proposal_frame.groupby(
            ["target_timepoint", "candidate_id"], sort=False
        ):
            ordered = local.sort_values(
                "completion_score", ascending=False
            )
            best_score = float(ordered.iloc[0]["completion_score"])
            second_score = (
                float(ordered.iloc[1]["completion_score"])
                if len(ordered) > 1
                else 0.0
            )
            proposal_frame.loc[
                local.index, "ownership_margin"
            ] = proposal_frame.loc[local.index, "completion_score"].map(
                lambda score: (
                    best_score - second_score
                    if abs(float(score) - best_score) < 1e-12
                    else float(score) - best_score
                )
            )
        best_for_candidate = proposal_frame.groupby(
            ["target_timepoint", "candidate_id"]
        )["completion_score"].transform("max")
        proposal_frame["candidate_best_score"] = best_for_candidate
        for proposal in proposal_frame.itertuples(index=False):
            candidate = result.loc[int(proposal.candidate_index)]
            score = float(proposal.completion_score)
            source_margin = float(proposal.source_margin)
            owner_margin = float(proposal.ownership_margin)
            is_owner = (
                abs(score - float(proposal.candidate_best_score)) < 1e-12
            )
            threshold = auto_threshold
            if float(candidate.get("debris_probability", 0.0)) >= strong_debris:
                threshold += float(
                    settings.get("strong_debris_extra_threshold", 0.10)
                )
            if str(candidate["integrated_label"]) == "invalid":
                threshold += float(
                    settings.get("invalid_extra_threshold", 0.06)
                )
            unambiguous = (
                source_margin >= ambiguity_margin
                and owner_margin >= ambiguity_margin
                and is_owner
            )
            decision = "below_review_threshold"
            if score >= threshold and unambiguous:
                candidate_index = int(proposal.candidate_index)
                original_label = str(result.at[candidate_index, "integrated_label"])
                if original_label not in CELL_LABELS:
                    result.at[candidate_index, "integrated_label"] = (
                        proposal.proposed_label
                    )
                result.at[
                    candidate_index, "integrated_confidence"
                ] = max(
                    float(result.at[candidate_index, "integrated_confidence"]),
                    score,
                )
                result.at[
                    candidate_index, "temporal_completion_score"
                ] = score
                result.at[
                    candidate_index, "temporal_completion_status"
                ] = "auto_promoted" if original_label not in CELL_LABELS else "confidence_recovered"
                result.at[
                    candidate_index, "temporal_completion_source_id"
                ] = proposal.source_candidate_id
                result.at[
                    candidate_index, "temporal_completion_direction"
                ] = f"{proposal.source_timepoint}->{proposal.target_timepoint}"
                promoted_ids.add(str(proposal.candidate_id))
                decision = str(
                    result.at[
                        candidate_index, "temporal_completion_status"
                    ]
                )
            elif score >= review_threshold:
                decision = (
                    "ambiguous_temporal_candidate"
                    if not unambiguous
                    else "temporal_reclassification_review"
                )
                candidate_index = int(proposal.candidate_index)
                if score > float(
                    result.at[
                        candidate_index, "temporal_completion_score"
                    ]
                ):
                    result.at[
                        candidate_index, "temporal_completion_score"
                    ] = score
                    result.at[
                        candidate_index, "temporal_completion_status"
                    ] = decision
                    result.at[
                        candidate_index, "temporal_completion_source_id"
                    ] = proposal.source_candidate_id
                    result.at[
                        candidate_index, "temporal_completion_direction"
                    ] = (
                        f"{proposal.source_timepoint}->"
                        f"{proposal.target_timepoint}"
                    )
                    if "integrated_review_priority" in result:
                        current_priority = float(
                            np.nan_to_num(
                                result.at[
                                    candidate_index,
                                    "integrated_review_priority",
                                ],
                                nan=0.0,
                            )
                        )
                        result.at[
                            candidate_index,
                            "integrated_review_priority",
                        ] = max(current_priority, 1.5 - score)
            if score >= review_threshold or decision.startswith("auto") or decision == "confidence_recovered":
                evidence = {
                    key: float(getattr(proposal, key))
                    for key in [
                        "distance_px",
                        "distance_score",
                        "shape_score",
                        "cell_probability",
                        "debris_probability",
                        "source_confidence",
                        "support_score",
                    ]
                }
                suggestions.append(
                    {
                        "well": proposal.well,
                        "source_candidate_id": proposal.source_candidate_id,
                        "source_timepoint": proposal.source_timepoint,
                        "target_timepoint": proposal.target_timepoint,
                        "candidate_id": proposal.candidate_id,
                        "predicted_x_px": proposal.predicted_x_px,
                        "predicted_y_px": proposal.predicted_y_px,
                        "prediction_mode": proposal.prediction_mode,
                        "original_label": proposal.original_label,
                        "proposed_label": proposal.proposed_label,
                        "completion_score": score,
                        "source_margin": source_margin,
                        "ownership_margin": owner_margin,
                        "decision": decision,
                        "evidence_json": json.dumps(
                            evidence, ensure_ascii=False
                        ),
                    }
                )

    suggestion_frame = pd.DataFrame(suggestions)
    if not suggestion_frame.empty:
        suggestion_frame = (
            suggestion_frame.sort_values(
                ["completion_score", "source_margin"],
                ascending=[False, False],
            )
            .drop_duplicates(
                [
                    "source_candidate_id",
                    "source_timepoint",
                    "target_timepoint",
                    "candidate_id",
                    "decision",
                ],
                keep="first",
            )
            .reset_index(drop=True)
        )
    status_counts = {
        str(key): int(value)
        for key, value in result[
            "temporal_completion_status"
        ].value_counts().items()
    }
    review_counts = (
        {
            str(key): int(value)
            for key, value in suggestion_frame["decision"].value_counts().items()
        }
        if not suggestion_frame.empty
        else {}
    )
    report = {
        "candidate_count": int(len(result)),
        "evaluated_gap_count": int(len(evaluated_gaps)),
        "promoted_candidate_count": int(len(promoted_ids)),
        "status_counts": status_counts,
        "review_decision_counts": review_counts,
        "automatic_promotion_score": auto_threshold,
        "review_score": review_threshold,
        "ambiguity_margin": ambiguity_margin,
        "search_radii_px": search_radii,
        "method": (
            "bidirectional registered search with interpolation/extrapolation, "
            "joint cell/debris scoring, and mutual ambiguity gating"
        ),
    }
    return result, suggestion_frame, report


def write_temporal_completion_artifacts(
    config: dict[str, Any],
    frame: pd.DataFrame,
    suggestions: pd.DataFrame,
    report: dict[str, Any],
) -> dict[str, Any]:
    predictions_path = artifact_path(
        config,
        "predictions",
        "latest_temporally_completed_predictions.csv",
    )
    suggestions_path = artifact_path(
        config,
        "predictions",
        "latest_temporal_completion_review.csv",
    )
    report_path = artifact_path(
        config,
        "predictions",
        "latest_temporal_completion_summary.json",
    )
    frame.to_csv(predictions_path, index=False, encoding="utf-8")
    suggestions.to_csv(suggestions_path, index=False, encoding="utf-8")
    report = {
        **report,
        "predictions": str(predictions_path),
        "review_queue": str(suggestions_path),
    }
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report

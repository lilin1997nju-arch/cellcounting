from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import artifact_path
from .temporal_completion import (
    complete_temporal_candidates,
    write_temporal_completion_artifacts,
)


CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}
MULTIPLICITY_COUNT = {
    "single": 1,
    "touching_doublet": 2,
    "cluster_3plus": 3,
}


def _reviewed_lineages(database: str | Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    try:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT canonical_target_id, object_type, viability,
                       timepoint_points_json
                FROM lineage_reviews
                """
            ).fetchall()
        for row in rows:
            points = json.loads(row["timepoint_points_json"] or "{}")
            cell_counts = {}
            for timepoint, point in points.items():
                labels = []
                if point.get("present"):
                    labels.append(point.get("object_label"))
                    labels.extend(
                        child.get("object_label")
                        for child in point.get("additional_points", [])
                    )
                cell_counts[timepoint] = sum(label == "cell" for label in labels)
            result[str(row["canonical_target_id"])] = {
                "object_type": str(row["object_type"]),
                "viability": str(row["viability"]),
                "cell_counts": cell_counts,
            }
    except (sqlite3.OperationalError, json.JSONDecodeError):
        pass
    return result


def _candidate_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "x_px": float(row["x_px"]),
        "y_px": float(row["y_px"]),
        "aligned_x_px": float(row["aligned_x_px"]),
        "aligned_y_px": float(row["aligned_y_px"]),
        "area_px": float(row["area_px"]),
        "marker_diameter_px": float(row["diameter_px"]),
        "multiplicity": str(row["integrated_label"]),
        "cell_count": int(
            MULTIPLICITY_COUNT.get(str(row["integrated_label"]), 1)
        ),
        "confidence": float(row["integrated_confidence"]),
    }


def _predicted_aligned(track: dict[str, Any]) -> np.ndarray:
    current = np.asarray(track["current_aligned"], dtype=float)
    velocity = np.asarray(
        track.get("last_motion_vector", [0.0, 0.0]), dtype=float
    )
    return current + velocity


def _assign_debris_candidates(
    tracks: list[dict[str, Any]],
    candidates: pd.DataFrame,
    maximum_distance: float | None,
) -> dict[int, int]:
    """One-to-one debris linking that permits drift but preserves morphology."""
    if not tracks or candidates.empty:
        return {}
    costs = np.full((len(tracks), len(candidates)), 1e6, dtype=float)
    candidate_xy = candidates[
        ["aligned_x_px", "aligned_y_px"]
    ].to_numpy(float)
    candidate_area = np.maximum(
        candidates["area_px"].to_numpy(float), 1.0
    )
    for track_index, track in enumerate(tracks):
        distances = np.linalg.norm(
            candidate_xy - _predicted_aligned(track), axis=1
        )
        area_change = np.abs(
            np.log(candidate_area / max(float(track["current_area"]), 1.0))
        )
        shape_change = (
            18.0
            * np.abs(
                candidates["circularity"].to_numpy(float)
                - float(track["circularity"])
            )
            + 12.0
            * np.abs(
                candidates["eccentricity"].to_numpy(float)
                - float(track["eccentricity"])
            )
            + 12.0
            * np.abs(
                candidates["solidity"].to_numpy(float)
                - float(track["solidity"])
            )
        )
        cost = distances + 16.0 * area_change + shape_change
        invalid = area_change > np.log(6.0)
        if maximum_distance is not None:
            invalid |= distances > maximum_distance
        else:
            invalid |= shape_change > 34.0
        cost[invalid] = 1e6
        costs[track_index] = cost
    track_indices, candidate_indices = linear_sum_assignment(costs)
    return {
        int(track_index): int(candidate_index)
        for track_index, candidate_index in zip(
            track_indices, candidate_indices, strict=True
        )
        if costs[track_index, candidate_index] < 1e5
    }


def _track_debris_for_well(
    well: str,
    debris: pd.DataFrame,
    search_radii: list[float],
    minimum_link_confidence: float,
    distance_scale: float,
    ambiguity_scale: float,
    resolution: float,
) -> list[dict[str, Any]]:
    roots = debris[debris["timepoint"] == "T0"].copy()
    tracks: list[dict[str, Any]] = []
    for number, (_, row) in enumerate(roots.iterrows(), start=1):
        tracks.append(
            {
                "well": well,
                "track_id": f"{well}:debris:{number}",
                "candidate_id": str(row["candidate_id"]),
                "current_aligned": [
                    float(row["aligned_x_px"]),
                    float(row["aligned_y_px"]),
                ],
                "current_area": float(row["area_px"]),
                "circularity": float(row["circularity"]),
                "eccentricity": float(row["eccentricity"]),
                "solidity": float(row["solidity"]),
                "path": [
                    {"timepoint": "T0", **_candidate_payload(row)}
                ],
                "missing_timepoints": [],
                "unlinked_reasons": {},
                "last_motion_vector": [0.0, 0.0],
                "maximum_motion_px": 0.0,
                "confidence": float(row["integrated_confidence"]),
            }
        )
    for timepoint in ("T1", "T2"):
        candidates = (
            debris[debris["timepoint"] == timepoint]
            .copy()
            .reset_index(drop=True)
        )
        assignments: dict[int, int] = {}
        assignment_evidence: dict[int, dict[str, float]] = {}
        assignment_scopes: dict[int, float | None] = {}
        remaining_tracks = list(range(len(tracks)))
        remaining_candidates = list(range(len(candidates)))
        for radius in [*search_radii, None]:
            if not remaining_tracks or not remaining_candidates:
                break
            local_tracks = [tracks[index] for index in remaining_tracks]
            local_candidates = candidates.iloc[remaining_candidates]
            partial = _assign_debris_candidates(
                local_tracks, local_candidates, radius
            )
            matched_tracks = []
            matched_candidates = []
            for local_track, local_candidate in partial.items():
                track_index = remaining_tracks[local_track]
                candidate_index = remaining_candidates[local_candidate]
                evidence = _link_confidence(
                    tracks[track_index],
                    candidates.iloc[candidate_index],
                    tracks,
                    distance_scale=distance_scale,
                    ambiguity_scale=ambiguity_scale,
                )
                if (
                    evidence["link_confidence"]
                    < minimum_link_confidence
                ):
                    continue
                assignments[track_index] = candidate_index
                assignment_evidence[track_index] = evidence
                assignment_scopes[track_index] = radius
                matched_tracks.append(track_index)
                matched_candidates.append(candidate_index)
            remaining_tracks = [
                index
                for index in remaining_tracks
                if index not in matched_tracks
            ]
            remaining_candidates = [
                index
                for index in remaining_candidates
                if index not in matched_candidates
            ]
        for track_index, track in enumerate(tracks):
            if track_index not in assignments:
                track["missing_timepoints"].append(timepoint)
                options = [
                    _link_confidence(
                        track,
                        candidate,
                        tracks,
                        distance_scale=distance_scale,
                        ambiguity_scale=ambiguity_scale,
                    )
                    for _, candidate in candidates.iterrows()
                ]
                best = (
                    max(
                        options,
                        key=lambda value: value["link_confidence"],
                    )
                    if options
                    else None
                )
                track["unlinked_reasons"][timepoint] = {
                    "reason": "best_link_below_confidence_threshold",
                    "minimum_link_confidence": minimum_link_confidence,
                    "best_link_confidence": (
                        best["link_confidence"] if best else None
                    ),
                    "best_link_distance_px": (
                        best["link_distance_px"] if best else None
                    ),
                    "predicted_x_px": (
                        best["predicted_x_px"]
                        if best
                        else float(_predicted_aligned(track)[0])
                    ),
                    "predicted_y_px": (
                        best["predicted_y_px"]
                        if best
                        else float(_predicted_aligned(track)[1])
                    ),
                }
                continue
            row = candidates.iloc[assignments[track_index]]
            aligned = np.asarray(
                [float(row["aligned_x_px"]), float(row["aligned_y_px"])]
            )
            previous = np.asarray(track["current_aligned"], dtype=float)
            motion = float(
                np.linalg.norm(aligned - previous)
            )
            track["maximum_motion_px"] = max(
                track["maximum_motion_px"], motion
            )
            track["last_motion_vector"] = (aligned - previous).tolist()
            track["current_aligned"] = aligned.tolist()
            track["current_area"] = float(row["area_px"])
            track["circularity"] = float(row["circularity"])
            track["eccentricity"] = float(row["eccentricity"])
            track["solidity"] = float(row["solidity"])
            track["confidence"] = min(
                track["confidence"],
                float(row["integrated_confidence"]),
                float(
                    assignment_evidence[track_index]["link_confidence"]
                ),
            )
            track["path"].append(
                {
                    "timepoint": timepoint,
                    "match_scope": (
                        "global_confidence"
                        if assignment_scopes[track_index] is None
                        else (
                            f"radius_{assignment_scopes[track_index]:g}"
                        )
                    ),
                    **assignment_evidence[track_index],
                    **_candidate_payload(row),
                }
            )
    rows = []
    for track in tracks:
        linked_timepoints = len(track["path"])
        rows.append(
            {
                "well": well,
                "track_id": track["track_id"],
                "candidate_id": track["candidate_id"],
                "status": (
                    "linked_T0_T1_T2"
                    if linked_timepoints == 3
                    else "partially_linked"
                    if linked_timepoints >= 2
                    else "T0_only"
                ),
                "path_json": json.dumps(
                    track["path"], ensure_ascii=False
                ),
                "missing_timepoints_json": json.dumps(
                    track["missing_timepoints"], ensure_ascii=False
                ),
                "unlinked_reasons_json": json.dumps(
                    track["unlinked_reasons"], ensure_ascii=False
                ),
                "maximum_motion_um": (
                    track["maximum_motion_px"] * resolution
                ),
                "confidence": track["confidence"],
            }
        )
    return rows


def _points_payload(
    payloads: list[dict[str, Any]], object_label: str
) -> dict[str, Any]:
    if not payloads:
        return {"present": False, "additional_points": []}

    def point(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "present": True,
            "x_px": payload["x_px"],
            "y_px": payload["y_px"],
            "candidate_id": payload["candidate_id"],
            "area_px": payload["area_px"],
            "object_label": object_label,
            "marker_diameter_px": payload["marker_diameter_px"],
            "multiplicity": payload.get("multiplicity", "single"),
            "cell_count": payload.get("cell_count", 1),
            "additional_points": [],
        }

    primary = point(payloads[0])
    primary["additional_points"] = [
        point(payload) for payload in payloads[1:]
    ]
    return primary


def _build_tracking_proposals(
    config: dict[str, Any],
    frame: pd.DataFrame,
    root_frame: pd.DataFrame,
    debris_frame: pd.DataFrame,
    database: str | Path,
) -> pd.DataFrame:
    round_id = str(frame.iloc[0]["integrated_round_id"])
    minimum_link_confidence = float(
        config.get("multiplicity", {}).get(
            "minimum_link_confidence", 0.58
        )
    )
    debris_minimum_link_confidence = float(
        config.get("debris_tracking", {}).get(
            "minimum_link_confidence", 0.56
        )
    )
    excluded = {
        str(value).upper()
        for value in config.get("review_queue", {}).get(
            "excluded_wells", []
        )
    }
    t0_lookup = frame[frame["timepoint"] == "T0"].set_index(
        "candidate_id", drop=False
    )
    proposals: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(database) as connection:
            manual_cells = pd.read_sql_query(
                """
                SELECT object_id, well, timepoint, x_px, y_px
                FROM annotations
                WHERE object_type = 'cell'
                  AND timepoint IN ('T0', 'T1', 'T2')
                """,
                connection,
            )
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        manual_cells = pd.DataFrame(
            columns=["object_id", "well", "timepoint", "x_px", "y_px"]
        )

    def expand_reviewed_cells(
        well: str,
        timepoint: str,
        payloads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        local_manual = manual_cells[
            (manual_cells["well"] == well)
            & (manual_cells["timepoint"] == timepoint)
        ]
        for payload in payloads:
            cell_count = int(payload.get("cell_count", 1))
            if cell_count <= 1 or local_manual.empty:
                expanded.append(payload)
                continue
            distances = np.hypot(
                local_manual["x_px"].to_numpy(float)
                - float(payload["x_px"]),
                local_manual["y_px"].to_numpy(float)
                - float(payload["y_px"]),
            )
            radius = max(
                24.0, float(payload.get("marker_diameter_px", 8.0)) * 2.5
            )
            positions = np.flatnonzero(distances <= radius)
            if len(positions) < cell_count:
                expanded.append(payload)
                continue
            positions = positions[np.argsort(distances[positions])][
                :cell_count
            ]
            for position in positions:
                reviewed = local_manual.iloc[int(position)]
                expanded.append(
                    {
                        **payload,
                        "candidate_id": str(reviewed["object_id"]),
                        "x_px": float(reviewed["x_px"]),
                        "y_px": float(reviewed["y_px"]),
                        "multiplicity": "single",
                        "cell_count": 1,
                    }
                )
        return expanded

    def base_row(
        candidate_id: str,
        well: str,
        model_label: str,
        confidence: float,
    ) -> dict[str, Any]:
        source = t0_lookup.loc[candidate_id]
        return {
            "round_id": round_id,
            "proposal_round_id": round_id,
            "candidate_id": candidate_id,
            "canonical_target_id": f"{round_id}::{candidate_id}",
            "well": well,
            "timepoint": "T0",
            "x_px": float(source["x_px"]),
            "y_px": float(source["y_px"]),
            "area_px": float(source["area_px"]),
            "diameter_px": float(source["diameter_px"]),
            "auto_label": model_label,
            "integrated_label": str(source["integrated_label"]),
            "auto_status": "tracking_proposal_ready",
            "cell_probability": float(source["cell_probability"]),
            "debris_probability": float(source["debris_probability"]),
            "invalid_probability": float(source["invalid_probability"]),
            "confidence": float(confidence),
            "model_confidence": float(confidence),
            "foreground_fraction": float(source["area_px"]) / 96**2,
            "decision_source": "latest_integrated_tracking",
            "review_priority": float(confidence),
        }

    for row in root_frame.itertuples(index=False):
        if str(row.well).upper() in excluded:
            continue
        path = json.loads(row.path_json)
        division_children = json.loads(row.first_division_children_json)
        confirmations = json.loads(row.confirmation_positions_json)
        unlinked = json.loads(
            getattr(row, "unlinked_reasons_json", "{}") or "{}"
        )
        timepoint_payloads: dict[str, list[dict[str, Any]]] = {
            timepoint: [] for timepoint in ("T0", "T1", "T2")
        }
        for payload in path:
            timepoint_payloads[payload["timepoint"]].append(payload)
        if row.first_division_timepoint:
            timepoint_payloads[str(row.first_division_timepoint)] = (
                division_children
            )
        for timepoint, payloads in confirmations.items():
            timepoint_payloads[timepoint] = payloads
        for timepoint in ("T0", "T1", "T2"):
            timepoint_payloads[timepoint] = expand_reviewed_cells(
                str(row.well), timepoint, timepoint_payloads[timepoint]
            )
        points = {
            timepoint: _points_payload(
                timepoint_payloads[timepoint], "cell"
            )
            for timepoint in ("T0", "T1", "T2")
        }
        link_scores: dict[str, float | None] = {}
        link_reasons: dict[str, str] = {}
        for link_id, target_timepoint in (
            ("T0-T1", "T1"),
            ("T1-T2", "T2"),
        ):
            payloads = timepoint_payloads[target_timepoint]
            scores = [
                float(payload["link_confidence"])
                for payload in payloads
                if payload.get("link_confidence") is not None
            ]
            if target_timepoint in confirmations:
                link_scores[link_id] = None
                link_reasons[link_id] = (
                    "首次分裂后仅展示生长位置，不继续建立谱系边"
                )
            elif scores:
                link_scores[link_id] = min(scores)
                link_reasons[link_id] = (
                    f"链接置信度 {min(scores):.2f}，"
                    f"阈值 {minimum_link_confidence:.2f}"
                )
            elif target_timepoint in unlinked:
                best = unlinked[target_timepoint].get(
                    "best_link_confidence"
                )
                link_scores[link_id] = (
                    float(best) if best is not None else None
                )
                link_reasons[link_id] = (
                    "最佳候选低于链接阈值，已保留为未链接"
                    + (
                        f"（{float(best):.2f} < "
                        f"{minimum_link_confidence:.2f}）"
                        if best is not None
                        else ""
                    )
                )
            else:
                link_scores[link_id] = None
                link_reasons[link_id] = "未发现可接受的后续细胞候选"
        final_label = (
            "live_cell"
            if row.status == "first_division_confirmed"
            else "dead_cell"
            if row.status == "dead"
            else "cell_unknown"
        )
        links = {
            "T0-T1": (
                "correct"
                if (
                    points["T0"]["present"]
                    and points["T1"]["present"]
                    and link_scores["T0-T1"] is not None
                    and link_scores["T0-T1"]
                    >= minimum_link_confidence
                )
                else "uncertain"
            ),
            "T1-T2": (
                "correct"
                if (
                    points["T1"]["present"]
                    and points["T2"]["present"]
                    and link_scores["T1-T2"] is not None
                    and link_scores["T1-T2"]
                    >= minimum_link_confidence
                )
                else "uncertain"
            ),
        }
        proposal = base_row(
            str(row.candidate_id),
            str(row.well),
            "cell",
            float(row.confidence),
        )
        proposal.update(
            {
                "candidate_source": "tracked_cell_root",
                "temporal_label": final_label,
                "temporal_evidence": str(row.status),
                "proposal_points_json": json.dumps(
                    points, ensure_ascii=False
                ),
                "proposed_final_label": final_label,
                "proposed_morphology": "cell_like",
                "proposed_division": (
                    "divided"
                    if row.status == "first_division_confirmed"
                    else "none"
                ),
                "proposed_lineage_status": (
                    "correct_lineage"
                    if row.status
                    in {"first_division_confirmed", "dead", "non_growing"}
                    else "needs_review"
                ),
                "proposed_links_json": json.dumps(links),
                "proposed_link_scores_json": json.dumps(link_scores),
                "proposed_link_reasons_json": json.dumps(
                    link_reasons, ensure_ascii=False
                ),
                "tracking_status": str(row.status),
            }
        )
        proposals.append(proposal)

    for row in debris_frame.itertuples(index=False):
        if str(row.well).upper() in excluded:
            continue
        path = json.loads(row.path_json)
        unlinked = json.loads(
            getattr(row, "unlinked_reasons_json", "{}") or "{}"
        )
        by_timepoint = {
            timepoint: [
                payload
                for payload in path
                if payload["timepoint"] == timepoint
            ]
            for timepoint in ("T0", "T1", "T2")
        }
        points = {
            timepoint: _points_payload(by_timepoint[timepoint], "debris")
            for timepoint in ("T0", "T1", "T2")
        }
        link_scores: dict[str, float | None] = {}
        link_reasons: dict[str, str] = {}
        for link_id, target_timepoint in (
            ("T0-T1", "T1"),
            ("T1-T2", "T2"),
        ):
            scores = [
                float(payload["link_confidence"])
                for payload in by_timepoint[target_timepoint]
                if payload.get("link_confidence") is not None
            ]
            if scores:
                link_scores[link_id] = min(scores)
                link_reasons[link_id] = (
                    f"运动预测杂质链接 {min(scores):.2f}，"
                    f"阈值 {debris_minimum_link_confidence:.2f}"
                )
            elif target_timepoint in unlinked:
                best = unlinked[target_timepoint].get(
                    "best_link_confidence"
                )
                link_scores[link_id] = (
                    float(best) if best is not None else None
                )
                link_reasons[link_id] = (
                    "最佳杂质候选低于链接阈值，已保持未链接"
                    + (
                        f"（{float(best):.2f} < "
                        f"{debris_minimum_link_confidence:.2f}）"
                        if best is not None
                        else ""
                    )
                )
            else:
                link_scores[link_id] = None
                link_reasons[link_id] = "未发现可接受的杂质候选"
        links = {
            "T0-T1": (
                "correct"
                if (
                    points["T0"]["present"]
                    and points["T1"]["present"]
                    and link_scores["T0-T1"] is not None
                    and link_scores["T0-T1"]
                    >= debris_minimum_link_confidence
                )
                else "uncertain"
            ),
            "T1-T2": (
                "correct"
                if (
                    points["T1"]["present"]
                    and points["T2"]["present"]
                    and link_scores["T1-T2"] is not None
                    and link_scores["T1-T2"]
                    >= debris_minimum_link_confidence
                )
                else "uncertain"
            ),
        }
        proposal = base_row(
            str(row.candidate_id),
            str(row.well),
            "debris",
            float(row.confidence),
        )
        proposal.update(
            {
                "candidate_source": "tracked_drifting_debris",
                "temporal_label": "debris",
                "temporal_evidence": (
                    f"{row.status}; 最大漂移 {row.maximum_motion_um:.1f} µm"
                ),
                "proposal_points_json": json.dumps(
                    points, ensure_ascii=False
                ),
                "proposed_final_label": "debris",
                "proposed_morphology": "debris_like",
                "proposed_division": "none",
                "proposed_lineage_status": "debris_lineage",
                "proposed_links_json": json.dumps(links),
                "proposed_link_scores_json": json.dumps(link_scores),
                "proposed_link_reasons_json": json.dumps(
                    link_reasons, ensure_ascii=False
                ),
                "tracking_status": str(row.status),
            }
        )
        proposals.append(proposal)
    return pd.DataFrame(proposals)


def _assign_primary_candidates(
    roots: list[dict[str, Any]],
    candidates: pd.DataFrame,
    maximum_distance: float | None,
) -> dict[int, int]:
    if not roots or candidates.empty:
        return {}
    costs = np.full((len(roots), len(candidates)), 1e6, dtype=float)
    candidate_xy = candidates[["aligned_x_px", "aligned_y_px"]].to_numpy(float)
    for root_index, root in enumerate(roots):
        predicted = _predicted_aligned(root)
        distances = np.linalg.norm(candidate_xy - predicted, axis=1)
        area_penalty = np.abs(
            np.log(
                np.maximum(candidates["area_px"].to_numpy(float), 1.0)
                / max(float(root["current_area"]), 1.0)
            )
        )
        shape_penalty = (
            24.0
            * np.abs(
                candidates["circularity"].to_numpy(float)
                - float(root["circularity"])
            )
            + 16.0
            * np.abs(
                candidates["eccentricity"].to_numpy(float)
                - float(root["eccentricity"])
            )
            + 16.0
            * np.abs(
                candidates["solidity"].to_numpy(float)
                - float(root["solidity"])
            )
        )
        costs[root_index] = (
            0.18 * distances + 18.0 * area_penalty + shape_penalty
        )
        invalid = area_penalty > np.log(6.0)
        if maximum_distance is not None:
            invalid |= distances > maximum_distance
        else:
            invalid |= shape_penalty > 34.0
        costs[root_index, invalid] = 1e6
    root_indices, candidate_indices = linear_sum_assignment(costs)
    return {
        int(root_index): int(candidate_index)
        for root_index, candidate_index in zip(
            root_indices, candidate_indices, strict=True
        )
        if costs[root_index, candidate_index] < 1e5
    }


def _link_confidence(
    root: dict[str, Any],
    candidate: pd.Series,
    all_roots: list[dict[str, Any]],
    *,
    distance_scale: float,
    ambiguity_scale: float,
) -> dict[str, float]:
    root_xy = _predicted_aligned(root)
    candidate_xy = np.asarray(
        [candidate["aligned_x_px"], candidate["aligned_y_px"]],
        dtype=float,
    )
    distance = float(np.linalg.norm(candidate_xy - root_xy))
    distance_score = float(np.exp(-distance / max(distance_scale, 1.0)))
    area_log_ratio = float(
        abs(
            np.log(
                max(float(candidate["area_px"]), 1.0)
                / max(float(root["current_area"]), 1.0)
            )
        )
    )
    area_score = float(np.exp(-area_log_ratio))
    shape_delta = float(
        abs(float(candidate["circularity"]) - float(root["circularity"]))
        + 0.6
        * abs(
            float(candidate["eccentricity"]) - float(root["eccentricity"])
        )
        + 0.6
        * abs(float(candidate["solidity"]) - float(root["solidity"]))
    )
    shape_score = float(np.exp(-shape_delta / 0.55))
    model_score = float(
        np.clip(float(candidate["integrated_confidence"]), 0.0, 1.0)
    )
    other_distances = [
        float(
            np.linalg.norm(
                candidate_xy
                - _predicted_aligned(other)
            )
        )
        for other in all_roots
        if other is not root
    ]
    if other_distances:
        nearest_other = min(other_distances)
        margin = nearest_other - distance
        ambiguity_score = float(
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        margin / max(ambiguity_scale, 1.0), -30.0, 30.0
                    )
                )
            )
        )
    else:
        nearest_other = float("inf")
        margin = float("inf")
        ambiguity_score = 1.0
    base_score = (
        0.35 * distance_score
        + 0.25 * area_score
        + 0.20 * shape_score
        + 0.20 * model_score
    )
    confidence = float(
        np.clip(
            base_score * (0.45 + 0.55 * ambiguity_score),
            0.0,
            1.0,
        )
    )
    return {
        "link_confidence": confidence,
        "link_distance_px": distance,
        "distance_score": distance_score,
        "area_score": area_score,
        "shape_score": shape_score,
        "model_score": model_score,
        "ambiguity_score": ambiguity_score,
        "nearest_other_distance_px": nearest_other,
        "nearest_root_margin_px": margin,
        "predicted_x_px": float(root_xy[0]),
        "predicted_y_px": float(root_xy[1]),
    }


def build_first_division_lineages(
    config: dict[str, Any], database: str | Path
) -> dict[str, Any]:
    source = artifact_path(
        config, "predictions", "latest_integrated_predictions.csv"
    )
    frame = pd.read_csv(source)
    frame, completion_review, completion_report = (
        complete_temporal_candidates(config, frame)
    )
    completion_report = write_temporal_completion_artifacts(
        config,
        frame,
        completion_review,
        completion_report,
    )
    reviewed = _reviewed_lineages(database)
    settings = config.get("multiplicity", {})
    search_radii = [
        float(value) for value in settings.get("search_radii_px", [24, 48, 80])
    ]
    minimum_link_confidence = float(
        settings.get("minimum_link_confidence", 0.58)
    )
    link_distance_scale = float(
        settings.get("link_distance_scale_px", 1000)
    )
    link_ambiguity_scale = float(
        settings.get("link_ambiguity_scale_px", 140)
    )
    division_radius = float(
        settings.get("first_division_radius_px", 48)
    )
    confirmation_radius = float(
        settings.get("growth_confirmation_radius_px", 120)
    )
    debris_settings = config.get("debris_tracking", {})
    debris_search_radii = [
        float(value)
        for value in debris_settings.get(
            "search_radii_px", [180, 360, 720, 1200]
        )
    ]
    debris_minimum_confidence = float(
        debris_settings.get("minimum_link_confidence", 0.56)
    )
    debris_distance_scale = float(
        debris_settings.get("link_distance_scale_px", 1600)
    )
    debris_ambiguity_scale = float(
        debris_settings.get("link_ambiguity_scale_px", 140)
    )
    resolution = float(config["calibration"]["resolution_um_per_pixel"])

    root_rows: list[dict[str, Any]] = []
    debris_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    for well, well_frame in frame.groupby("well", sort=True):
        cells = well_frame[
            well_frame["integrated_label"].isin(CELL_LABELS)
        ].copy()
        debris = well_frame[
            well_frame["integrated_label"] == "debris"
        ].copy()
        roots_frame = cells[cells["timepoint"] == "T0"].copy()
        root_states: list[dict[str, Any]] = []
        for root_number, (_, row) in enumerate(
            roots_frame.iterrows(), start=1
        ):
            root_states.append(
                {
                    "root_id": f"{well}:root:{root_number}",
                    "candidate_id": str(row["candidate_id"]),
                    "well": str(well),
                    "initial_label": str(row["integrated_label"]),
                    "initial_cell_count": int(
                        MULTIPLICITY_COUNT[str(row["integrated_label"])]
                    ),
                    "current_aligned": [
                        float(row["aligned_x_px"]),
                        float(row["aligned_y_px"]),
                    ],
                    "current_area": float(row["area_px"]),
                    "circularity": float(row["circularity"]),
                    "eccentricity": float(row["eccentricity"]),
                    "solidity": float(row["solidity"]),
                    "start_aligned": [
                        float(row["aligned_x_px"]),
                        float(row["aligned_y_px"]),
                    ],
                    "status": "tracking",
                    "first_division_timepoint": None,
                    "first_division_children": [],
                    "confirmation_positions": {},
                    "path": [
                        {
                            "timepoint": "T0",
                            **_candidate_payload(row),
                        }
                    ],
                    "missing_timepoints": [],
                    "unlinked_reasons": {},
                    "last_motion_vector": [0.0, 0.0],
                    "maximum_motion_px": 0.0,
                    "confidence": float(row["integrated_confidence"]),
                    "temporal_label": str(
                        row.get("temporal_label", "uncertain")
                    ),
                }
            )

        for timepoint in ("T1", "T2"):
            active = [
                root
                for root in root_states
                if root["status"] == "tracking"
                and root["initial_cell_count"] == 1
            ]
            if not active:
                continue
            candidates = (
                cells[cells["timepoint"] == timepoint]
                .copy()
                .reset_index(drop=True)
            )
            if candidates.empty:
                for root in active:
                    root["missing_timepoints"].append(timepoint)
                continue

            assignments: dict[int, int] = {}
            assignment_scopes: dict[int, float | None] = {}
            assignment_evidence: dict[int, dict[str, float]] = {}
            remaining_roots = list(range(len(active)))
            remaining_candidates = list(range(len(candidates)))
            for radius in [*search_radii, None]:
                if not remaining_roots or not remaining_candidates:
                    break
                local_roots = [active[index] for index in remaining_roots]
                local_candidates = candidates.iloc[
                    remaining_candidates
                ].copy()
                partial = _assign_primary_candidates(
                    local_roots, local_candidates, radius
                )
                matched_root_indices = []
                matched_candidate_indices = []
                for local_root, local_candidate in partial.items():
                    root_index = remaining_roots[local_root]
                    candidate_index = remaining_candidates[local_candidate]
                    evidence = _link_confidence(
                        active[root_index],
                        candidates.iloc[candidate_index],
                        active,
                        distance_scale=link_distance_scale,
                        ambiguity_scale=link_ambiguity_scale,
                    )
                    if (
                        evidence["link_confidence"]
                        < minimum_link_confidence
                    ):
                        continue
                    assignments[root_index] = candidate_index
                    assignment_scopes[root_index] = radius
                    assignment_evidence[root_index] = evidence
                    matched_root_indices.append(root_index)
                    matched_candidate_indices.append(candidate_index)
                remaining_roots = [
                    index
                    for index in remaining_roots
                    if index not in matched_root_indices
                ]
                remaining_candidates = [
                    index
                    for index in remaining_candidates
                    if index not in matched_candidate_indices
                ]
            candidate_xy = candidates[
                ["aligned_x_px", "aligned_y_px"]
            ].to_numpy(float)
            owner_xy = np.asarray(
                [
                    (
                        candidates.iloc[assignments[index]][
                            ["aligned_x_px", "aligned_y_px"]
                        ].to_numpy(float)
                        if index in assignments
                        else _predicted_aligned(root)
                    )
                    for index, root in enumerate(active)
                ],
                dtype=float,
            )
            all_distances = np.linalg.norm(
                candidate_xy[:, None, :] - owner_xy[None, :, :], axis=2
            )
            nearest_root = all_distances.argmin(axis=1)

            for active_index, root in enumerate(active):
                if active_index not in assignments:
                    root["missing_timepoints"].append(timepoint)
                    evidence_options = [
                        _link_confidence(
                            root,
                            candidate,
                            active,
                            distance_scale=link_distance_scale,
                            ambiguity_scale=link_ambiguity_scale,
                        )
                        for _, candidate in candidates.iterrows()
                    ]
                    best = (
                        max(
                            evidence_options,
                            key=lambda value: value["link_confidence"],
                        )
                        if evidence_options
                        else None
                    )
                    root["unlinked_reasons"][timepoint] = {
                        "reason": (
                            "best_link_below_confidence_threshold"
                            if best
                            else "no_cell_candidate"
                        ),
                        "minimum_link_confidence": minimum_link_confidence,
                        "best_link_confidence": (
                            best["link_confidence"] if best else None
                        ),
                        "best_link_distance_px": (
                            best["link_distance_px"] if best else None
                        ),
                        "best_ambiguity_score": (
                            best["ambiguity_score"] if best else None
                        ),
                        "predicted_x_px": (
                            best["predicted_x_px"]
                            if best
                            else float(_predicted_aligned(root)[0])
                        ),
                        "predicted_y_px": (
                            best["predicted_y_px"]
                            if best
                            else float(_predicted_aligned(root)[1])
                        ),
                    }
                    continue
                primary_position = assignments[active_index]
                primary = candidates.iloc[primary_position]
                local_mask = (
                    (nearest_root == active_index)
                    & (all_distances[:, active_index] <= division_radius)
                )
                child_positions = list(np.flatnonzero(local_mask))
                if primary_position not in child_positions:
                    child_positions.insert(0, primary_position)
                cluster_roots = [
                    {
                        **candidate_root,
                        "current_aligned": owner_xy[index].tolist(),
                        "last_motion_vector": [0.0, 0.0],
                    }
                    for index, candidate_root in enumerate(active)
                ]
                child_evidence = {
                    position: {
                        **(
                            local_evidence := _link_confidence(
                                cluster_roots[active_index],
                                candidates.iloc[position],
                                cluster_roots,
                                distance_scale=max(
                                    division_radius * 3.0, 1.0
                                ),
                                ambiguity_scale=link_ambiguity_scale,
                            )
                        ),
                        "link_confidence": min(
                            assignment_evidence[active_index][
                                "link_confidence"
                            ],
                            local_evidence["link_confidence"],
                        ),
                        "source_link_confidence": (
                            assignment_evidence[active_index][
                                "link_confidence"
                            ]
                        ),
                        "source_displacement_px": (
                            assignment_evidence[active_index][
                                "link_distance_px"
                            ]
                        ),
                        "cluster_distance_px": local_evidence[
                            "link_distance_px"
                        ],
                    }
                    for position in child_positions
                }
                child_positions = [
                    position
                    for position in child_positions
                    if child_evidence[position]["link_confidence"]
                    >= minimum_link_confidence
                    or position == primary_position
                ]
                children = candidates.iloc[child_positions].copy()
                observed_child_count = int(
                    sum(
                        MULTIPLICITY_COUNT.get(str(label), 1)
                        for label in children["integrated_label"]
                    )
                )
                reviewed_root = reviewed.get(root["candidate_id"], {})
                reviewed_count = int(
                    reviewed_root.get("cell_counts", {}).get(timepoint, 0)
                )
                child_area_ratio = float(
                    children["area_px"].astype(float).sum()
                    / max(float(root["current_area"]), 1.0)
                )
                division_confirmed = (
                    (
                        observed_child_count >= 2
                        and 0.45 <= child_area_ratio <= 4.5
                    )
                    or reviewed_count >= 2
                )
                if division_confirmed:
                    root["status"] = "first_division_confirmed"
                    root["first_division_timepoint"] = timepoint
                    root["first_division_children"] = [
                        {
                            **_candidate_payload(child),
                            "match_scope": (
                                "global_confidence"
                                if assignment_scopes[active_index] is None
                                else (
                                    f"radius_"
                                    f"{assignment_scopes[active_index]:g}"
                                )
                            ),
                            **child_evidence[int(position)],
                        }
                        for position, child in children.iterrows()
                    ]
                    root["confidence"] = (
                        1.0
                        if reviewed_count >= 2
                        else min(
                            root["confidence"],
                            float(children["integrated_confidence"].mean()),
                            float(
                                min(
                                    child_evidence[int(position)][
                                        "link_confidence"
                                    ]
                                    for position in children.index
                                )
                            ),
                        )
                    )
                    continue

                new_aligned = np.asarray(
                    [
                        float(primary["aligned_x_px"]),
                        float(primary["aligned_y_px"]),
                    ]
                )
                motion = float(
                    np.linalg.norm(
                        new_aligned - np.asarray(root["current_aligned"])
                    )
                )
                root["maximum_motion_px"] = max(
                    root["maximum_motion_px"], motion
                )
                previous_aligned = np.asarray(
                    root["current_aligned"], dtype=float
                )
                root["last_motion_vector"] = (
                    new_aligned - previous_aligned
                ).tolist()
                root["current_aligned"] = new_aligned.tolist()
                root["current_area"] = float(primary["area_px"])
                root["circularity"] = float(primary["circularity"])
                root["eccentricity"] = float(primary["eccentricity"])
                root["solidity"] = float(primary["solidity"])
                root["confidence"] = min(
                    root["confidence"],
                    float(primary["integrated_confidence"]),
                    float(
                        assignment_evidence[active_index][
                            "link_confidence"
                        ]
                    ),
                )
                root["path"].append(
                    {
                        "timepoint": timepoint,
                        "search_radius_px": assignment_scopes[
                            active_index
                        ],
                        "match_scope": (
                            "global_confidence"
                            if assignment_scopes[active_index] is None
                            else (
                                f"radius_"
                                f"{assignment_scopes[active_index]:g}"
                            )
                        ),
                        **assignment_evidence[active_index],
                        **_candidate_payload(primary),
                    }
                )

        # After the first confirmed division, the child centers form a union
        # search region. Later frames are only growth confirmation positions;
        # they do not create deeper parent-child edges.
        timepoint_order = {"T1": 1, "T2": 2}
        for timepoint in ("T1", "T2"):
            divided_roots = [
                root
                for root in root_states
                if root["first_division_timepoint"] is not None
                and timepoint_order[root["first_division_timepoint"]]
                < timepoint_order[timepoint]
            ]
            later_candidates = cells[
                cells["timepoint"] == timepoint
            ].copy()
            if not divided_roots or later_candidates.empty:
                continue
            candidate_xy = later_candidates[
                ["aligned_x_px", "aligned_y_px"]
            ].to_numpy(float)
            distance_columns = []
            for root in divided_roots:
                seed_xy = np.asarray(
                    [
                        [
                            child["aligned_x_px"],
                            child["aligned_y_px"],
                        ]
                        for child in root["first_division_children"]
                    ],
                    dtype=float,
                )
                collective_motion = (
                    seed_xy.mean(axis=0)
                    - np.asarray(root["start_aligned"], dtype=float)
                )
                predicted_seed_xy = seed_xy + collective_motion
                current_distances = np.linalg.norm(
                    candidate_xy[:, None, :] - seed_xy[None, :, :],
                    axis=2,
                ).min(axis=1)
                predicted_distances = np.linalg.norm(
                    candidate_xy[:, None, :]
                    - predicted_seed_xy[None, :, :],
                    axis=2,
                ).min(axis=1)
                distances = np.minimum(
                    current_distances, predicted_distances
                )
                distance_columns.append(distances)
            distance_matrix = np.stack(distance_columns, axis=1)
            nearest_roots = distance_matrix.argmin(axis=1)
            for root_index, root in enumerate(divided_roots):
                assigned = later_candidates[
                    (nearest_roots == root_index)
                    & (
                        distance_matrix[:, root_index]
                        <= confirmation_radius
                    )
                ]
                root["confirmation_positions"][timepoint] = [
                    _candidate_payload(candidate)
                    for _, candidate in assigned.iterrows()
                ]

        for root in root_states:
            human = reviewed.get(root["candidate_id"], {})
            if root["status"] == "tracking":
                if (
                    human.get("object_type") == "cell"
                    and human.get("viability") == "dead"
                ):
                    root["status"] = "dead"
                elif len(root["path"]) >= 3:
                    # Low motion without division is insufficient to call a
                    # cell dead. Keep it as non-growing until a reviewer
                    # confirms death from the whole temporal sequence.
                    root["status"] = "non_growing"
                elif root["missing_timepoints"]:
                    root["status"] = "lost_or_uncertain"
                else:
                    root["status"] = "non_growing"
            if root["initial_cell_count"] > 1:
                root["status"] = "multi_cell_at_T0"
            root_rows.append(
                {
                    "well": well,
                    "root_id": root["root_id"],
                    "candidate_id": root["candidate_id"],
                    "initial_multiplicity": root["initial_label"],
                    "initial_cell_count": root["initial_cell_count"],
                    "status": root["status"],
                    "first_division_timepoint": root[
                        "first_division_timepoint"
                    ],
                    "first_division_children_json": json.dumps(
                        root["first_division_children"], ensure_ascii=False
                    ),
                    "path_json": json.dumps(root["path"], ensure_ascii=False),
                    "confirmation_positions_json": json.dumps(
                        root["confirmation_positions"],
                        ensure_ascii=False,
                    ),
                    "unlinked_reasons_json": json.dumps(
                        root["unlinked_reasons"],
                        ensure_ascii=False,
                    ),
                    "maximum_motion_um": root["maximum_motion_px"] * resolution,
                    "confidence": root["confidence"],
                }
            )

        debris_rows.extend(
            _track_debris_for_well(
                str(well),
                debris,
                debris_search_radii,
                debris_minimum_confidence,
                debris_distance_scale,
                debris_ambiguity_scale,
                resolution,
            )
        )

        proliferating = [
            root
            for root in root_states
            if root["status"] == "first_division_confirmed"
        ]
        dead = [root for root in root_states if root["status"] == "dead"]
        t0_cell_count = int(
            sum(root["initial_cell_count"] for root in root_states)
        )
        if len(proliferating) >= 2:
            origin = "multiple_cell_origin"
        elif len(proliferating) == 1:
            growing_root = proliferating[0]
            others = [
                root
                for root in root_states
                if root["root_id"] != growing_root["root_id"]
            ]
            if t0_cell_count == 1:
                origin = "single_cell_origin"
            elif (
                growing_root["initial_cell_count"] == 1
                and others
                and all(root["status"] == "dead" for root in others)
            ):
                origin = "functional_single_cell_origin_with_dead_T0_cells"
            else:
                origin = "single_growing_lineage_in_multi_cell_T0_uncertain"
        elif t0_cell_count == 0:
            origin = "no_T0_cell"
        else:
            origin = "no_confirmed_growth_or_uncertain"

        later_positions = {}
        for timepoint in ("T1", "T2"):
            later_positions[timepoint] = [
                _candidate_payload(row)
                for _, row in cells[cells["timepoint"] == timepoint].iterrows()
            ]
        debris_counts = {
            timepoint: int(
                (debris["timepoint"] == timepoint).sum()
            )
            for timepoint in ("T0", "T1", "T2")
        }
        well_rows.append(
            {
                "well": well,
                "growth_status": (
                    "confirmed_growth"
                    if proliferating
                    else "no_confirmed_growth"
                ),
                "origin_conclusion": origin,
                "t0_cell_count": t0_cell_count,
                "t0_root_count": int(len(root_states)),
                "proliferating_root_count": int(len(proliferating)),
                "dead_root_count": int(len(dead)),
                "debris_present": bool(len(debris)),
                "debris_counts_json": json.dumps(debris_counts),
                "later_cell_positions_json": json.dumps(
                    later_positions, ensure_ascii=False
                ),
            }
        )

    root_frame = pd.DataFrame(root_rows)
    debris_frame = pd.DataFrame(debris_rows)
    well_frame = pd.DataFrame(well_rows)
    root_path = artifact_path(
        config, "predictions", "latest_first_division_lineages.csv"
    )
    well_path = artifact_path(
        config, "predictions", "latest_well_conclusions.csv"
    )
    debris_path = artifact_path(
        config, "predictions", "latest_debris_tracks.csv"
    )
    proposal_path = artifact_path(
        config, "predictions", "latest_tracking_proposals.csv"
    )
    root_frame.to_csv(root_path, index=False, encoding="utf-8")
    debris_frame.to_csv(debris_path, index=False, encoding="utf-8")
    well_frame.to_csv(well_path, index=False, encoding="utf-8")
    proposal_frame = _build_tracking_proposals(
        config, frame, root_frame, debris_frame, database
    )
    proposal_frame.to_csv(proposal_path, index=False, encoding="utf-8")
    summary = {
        "root_count": int(len(root_frame)),
        "well_count": int(len(well_frame)),
        "root_status_counts": (
            {
                key: int(value)
                for key, value in root_frame["status"].value_counts().items()
            }
            if not root_frame.empty
            else {}
        ),
        "growth_status_counts": {
            key: int(value)
            for key, value in well_frame["growth_status"].value_counts().items()
        },
        "origin_conclusion_counts": {
            key: int(value)
            for key, value in well_frame["origin_conclusion"].value_counts().items()
        },
        "roots": str(root_path),
        "debris_tracks": str(debris_path),
        "debris_track_count": int(len(debris_frame)),
        "debris_track_status_counts": (
            {
                key: int(value)
                for key, value in debris_frame["status"].value_counts().items()
            }
            if not debris_frame.empty
            else {}
        ),
        "review_proposals": str(proposal_path),
        "review_proposal_count": int(len(proposal_frame)),
        "wells": str(well_path),
        "tracking_scope": (
            "T2 uses the previous motion vector; division children are "
            "clustered around the displaced primary cell; global search is "
            "confidence-gated; cell roots stop lineage edges after first "
            "division and later cells are position-only growth confirmation; "
            f"cell links below {minimum_link_confidence:.2f} remain unlinked; "
            "debris uses the same motion-aware confidence logic"
        ),
        "minimum_link_confidence": minimum_link_confidence,
        "temporal_completion": completion_report,
        "unlinked_timepoint_count": int(
            sum(
                len(json.loads(value or "{}"))
                for value in root_frame.get(
                    "unlinked_reasons_json", pd.Series(dtype=str)
                )
            )
        ),
    }
    artifact_path(
        config, "predictions", "latest_lineage_summary.json"
    ).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary

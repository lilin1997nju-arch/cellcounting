from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from .config import artifact_path
from .multiplicity import ensure_integrated_review_table, ensure_multiplicity_table
from .review_helpers import _visible_v2_review_instances, _with_final_decisions
from .review_storage import _summary_json_safe
from .review_summary import (
    SUMMARY_VERSION,
    latest_prediction_path,
    read_summary,
    summary_path,
    summary_signature,
    write_summary,
)
from .well_screening import build_well_screening


def build_quick_review_service(
    config: dict[str, Any],
    database: str | Path,
    images_manifest: pd.DataFrame,
    prediction_cache: dict[str, Any],
    proposal_cache: dict[str, Any],
    completion_review_cache: dict[str, Any],
    quick_frame_cache: dict[str, Any],
    quick_summary_cache: dict[str, Any],
    quick_summary_lock: Any,
    quick_summary_file: str | Path,
    gated_lookup: Any,
    gated_report_path: Any,
    ui_screening_status: Any,
    ui_screening_status_label: Any,
    ui_status_aliases: dict[str, str],
) -> SimpleNamespace:
    """Build the quick-review data-layer closures for one plate."""

    quick_summary_file = Path(quick_summary_file)

    def latest_integrated_predictions() -> pd.DataFrame:
        source = latest_prediction_path(config["paths"]["artifact_root"])
        if source is None:
            return pd.DataFrame()
        modified = source.stat().st_mtime_ns
        if (
            prediction_cache["mtime"] != modified
            or prediction_cache["source"] != str(source)
        ):
            prediction_cache["frame"] = pd.read_csv(source)
            prediction_cache["mtime"] = modified
            prediction_cache["source"] = str(source)
        return prediction_cache["frame"]

    def latest_tracking_proposals() -> pd.DataFrame:
        source = artifact_path(
            config, "predictions", "latest_tracking_proposals.csv"
        )
        if not source.exists():
            return pd.DataFrame()
        modified = source.stat().st_mtime_ns
        if proposal_cache["mtime"] != modified:
            proposal_cache["frame"] = pd.read_csv(source)
            proposal_cache["mtime"] = modified
        return proposal_cache["frame"]

    def latest_completion_review() -> pd.DataFrame:
        source = artifact_path(
            config,
            "predictions",
            "latest_temporal_completion_review.csv",
        )
        if not source.exists():
            return pd.DataFrame()
        modified = source.stat().st_mtime_ns
        if completion_review_cache["mtime"] != modified:
            completion_review_cache["frame"] = pd.read_csv(source)
            completion_review_cache["mtime"] = modified
        return completion_review_cache["frame"]

    def quick_review_frame() -> pd.DataFrame:
        frame = latest_integrated_predictions()
        if frame.empty:
            return frame.copy()
        cache_key = (
            prediction_cache["source"],
            prediction_cache["mtime"],
            database.stat().st_mtime_ns if database.exists() else None,
        )
        if quick_frame_cache["key"] == cache_key:
            return quick_frame_cache["frame"]
        frame = frame.copy()
        temporal_defaults = {
            "temporal_completion_score": 0.0,
            "temporal_completion_status": "not_evaluated",
            "temporal_completion_source_id": "",
            "temporal_completion_direction": "",
            "v3_track_behavior": "disabled",
            "v3_track_conclusion": "",
            "v3_unified_label": "",
            "v3_label_mode": "per_frame_evidence",
            "v3_reason": "",
            "v3_track_id": "",
            "v3_cell_to_debris_candidate": False,
        }
        for column, default in temporal_defaults.items():
            if column not in frame:
                frame[column] = default
        ordinary_review_target = frame["integrated_label"].isin(
            [
                "single",
                "touching_doublet",
                "cluster_3plus",
                "debris",
                "uncertain",
            ]
        )
        # V3 wall-structure invalidations are deliberately kept visible.  They
        # are high-value temporal corrections that reviewers must be able to
        # confirm instead of silently disappearing from the audit interface.
        v3_wall_structure_target = frame["v3_track_behavior"].astype(str).eq(
            "wall_structure_invalid"
        )
        reviewable = frame[
            (ordinary_review_target | v3_wall_structure_target)
            & (frame["well"].astype(str).str.upper() != "A1")
        ].copy()
        round_id = str(reviewable.iloc[0]["integrated_round_id"])
        ensure_integrated_review_table(database)
        with sqlite3.connect(database) as connection:
            reviews = pd.read_sql_query(
                """
                SELECT candidate_id, reviewed_label, decision
                FROM integrated_training_reviews
                WHERE round_id = ?
                """,
                connection,
                params=(round_id,),
            )
        reviewable = reviewable.merge(
            reviews, on="candidate_id", how="left"
        )
        reviewable["current_label"] = reviewable[
            "reviewed_label"
        ].fillna(reviewable["integrated_label"])
        track_reviews = pd.DataFrame()
        if "v3_track_id" in reviewable.columns:
            try:
                with sqlite3.connect(database) as connection:
                    track_reviews = pd.read_sql_query(
                        """
                        SELECT track_id, label AS v3_reviewed_label
                        FROM temporal_track_reviews
                        WHERE round_id = ?
                        """,
                        connection,
                        params=(round_id,),
                    )
            except (sqlite3.OperationalError, pd.errors.DatabaseError):
                track_reviews = pd.DataFrame()
            if not track_reviews.empty:
                track_reviews["track_id"] = track_reviews["track_id"].astype(str)
                reviewable["v3_track_id"] = reviewable["v3_track_id"].fillna("").astype(str)
                reviewable = reviewable.merge(
                    track_reviews,
                    left_on="v3_track_id",
                    right_on="track_id",
                    how="left",
                ).drop(columns=["track_id"])
            else:
                reviewable["v3_reviewed_label"] = None
        else:
            reviewable["v3_reviewed_label"] = None
        with sqlite3.connect(database) as connection:
            manual = pd.read_sql_query(
                """
                SELECT candidate_id, well, timepoint, x_px, y_px,
                       diameter_px, reviewed_label, reviewer, updated_at
                FROM quick_missed_objects
                WHERE round_id = ?
                """,
                connection,
                params=(round_id,),
            )
        if not manual.empty:
            manual["integrated_round_id"] = round_id
            manual["integrated_label"] = manual["reviewed_label"]
            manual["current_label"] = manual["reviewed_label"]
            manual["decision"] = "approved"
            manual["integrated_confidence"] = 1.0
            manual["integrated_review_priority"] = 0.0
            manual["area_px"] = np.pi * (
                manual["diameter_px"].astype(float) / 2
            ) ** 2
            cell_like = manual["reviewed_label"].isin(
                ["single", "touching_doublet", "cluster_3plus"]
            )
            manual["cell_probability"] = cell_like.astype(float)
            manual["debris_probability"] = (
                manual["reviewed_label"] == "debris"
            ).astype(float)
            manual["invalid_probability"] = 0.0
            manual["single_probability"] = (
                manual["reviewed_label"] == "single"
            ).astype(float)
            manual["touching_doublet_probability"] = (
                manual["reviewed_label"] == "touching_doublet"
            ).astype(float)
            manual["cluster_3plus_probability"] = (
                manual["reviewed_label"] == "cluster_3plus"
            ).astype(float)
            manual["is_manual_missed"] = True
            reviewable["is_manual_missed"] = False
            manual = manual[
                ~manual["candidate_id"].astype(str).isin(
                    reviewable["candidate_id"].astype(str)
                )
            ]
            reviewable = pd.concat(
                [reviewable, manual], ignore_index=True, sort=False
            )
        elif "is_manual_missed" not in reviewable:
            reviewable["is_manual_missed"] = False
        if "v2_instance_id" in reviewable:
            # V2 masks are authoritative for instance ownership. Re-applying
            # V1 circle-distance suppression would split/erase valid contours.
            visible = _visible_v2_review_instances(reviewable)
        else:
            reviewable = suppress_nested_single_candidates(
                reviewable,
                config,
                label_column="current_label",
                confidence_column="integrated_confidence",
            )
            visible = reviewable[
                ~reviewable["is_hierarchy_suppressed"]
                & ~reviewable["is_duplicate_suppressed"]
            ].copy()
        visible = _with_final_decisions(visible)
        quick_frame_cache["key"] = cache_key
        quick_frame_cache["frame"] = visible
        return visible

    def quick_review_well_rows(
        mode: str = "pending",
        search: str | None = None,
        frame: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        if frame is None:
            frame = quick_review_frame()
        if frame.empty:
            return []
        screening_path = artifact_path(
            config, "predictions", "latest_well_screening.csv"
        )
        screening_lookup: dict[str, dict[str, Any]] = {}
        if screening_path.exists():
            screening_frame = pd.read_csv(screening_path)
            screening_lookup = {
                str(row.well).upper(): row._asdict()
                for row in screening_frame.itertuples(index=False)
            }
        report_lookup = gated_lookup()
        rows: list[dict[str, Any]] = []
        for well, local in frame.groupby("well", sort=False):
            reviewed = int(local["reviewed_label"].notna().sum())
            total = int(len(local))
            completed = reviewed == total
            review_labels = local["final_review_label"].fillna(
                local["current_label"]
            )
            track_ids = local["v3_track_id"].fillna("").astype(str)
            v3_tracks = local.loc[track_ids.ne(""), ["v3_track_id"]].drop_duplicates()
            v3_cell_to_debris = local[
                local["v3_track_behavior"].astype(str).eq("cell_to_debris")
            ]
            screening = screening_lookup.get(str(well).upper(), {})
            report = report_lookup.get(str(well).upper(), {})
            status = ui_screening_status(
                report,
                str(screening.get("screening_status", "ambiguous")),
            )
            rows.append(
                {
                    "well": str(well),
                    "object_count": total,
                    "reviewed_count": reviewed,
                    "completed": completed,
                    "corrected_count": int(
                        (local["decision"] == "corrected").sum()
                    ),
                    "uncertain_count": int(
                        (review_labels == "uncertain").sum()
                    ),
                    "cell_count": int(
                        review_labels.isin(
                            [
                                "single",
                                "touching_doublet",
                                "cluster_3plus",
                            ]
                        ).sum()
                    ),
                    "debris_count": int(
                        (review_labels == "debris").sum()
                    ),
                    "temporal_review_count": int(
                        local["temporal_completion_status"].isin(
                            [
                                "ambiguous_temporal_candidate",
                                "temporal_reclassification_review",
                            ]
                        ).sum()
                    ),
                    "v3_track_count": int(len(v3_tracks)),
                    "v3_cell_to_debris_count": int(
                        v3_cell_to_debris["v3_track_id"].fillna("").astype(str).replace("", np.nan).nunique()
                    ),
                    "priority": float(
                        local["integrated_review_priority"].max()
                    ),
                    "screening_status": status,
                    "base_screening_status": str(
                        ui_status_aliases.get(
                            str(screening.get("base_screening_status", "ambiguous")),
                            str(screening.get("base_screening_status", "ambiguous")),
                        )
                    ),
                    "late_growth_status": str(
                        screening.get("late_growth_status", "unavailable")
                    ),
                    "high_confidence_single_active": bool(
                        screening.get("high_confidence_single_active", False)
                    ),
                    "report_category": report.get("final_category"),
                    "report_category_label": ui_screening_status_label(status),
                    "report_reason": report.get("undetermined_reason"),
                    "report_reason_label": report.get("undetermined_reason_label"),
                }
            )
        if mode == "pending":
            rows = [row for row in rows if not row["completed"]]
        elif mode == "reviewed":
            rows = [row for row in rows if row["completed"]]
        if search:
            needle = search.strip().upper()
            rows = [
                row for row in rows if needle in row["well"].upper()
            ]

        def well_key(row: dict[str, Any]) -> tuple[int, int]:
            value = str(row["well"])
            try:
                return ord(value[0].upper()) - ord("A"), int(value[1:])
            except (ValueError, IndexError):
                return 99, 999

        return sorted(rows, key=well_key)

    def quick_review_summary_signature() -> dict[str, Any]:
        return summary_signature(
            config["paths"]["artifact_root"],
            database_path=database,
            screening_path=artifact_path(
                config, "predictions", "latest_well_screening.csv"
            ),
            report_path=gated_report_path(),
        )

    def quick_review_summary(*, force: bool = False) -> dict[str, Any]:
        """Load or rebuild the persistent list summary for this plate."""

        with quick_summary_lock:
            signature = quick_review_summary_signature()
            if (
                not force
                and quick_summary_cache["signature"] == signature
                and quick_summary_cache["payload"] is not None
            ):
                return quick_summary_cache["payload"]

            if not force:
                persisted = read_summary(quick_summary_file, signature)
                if persisted is not None:
                    quick_summary_cache["signature"] = signature
                    quick_summary_cache["payload"] = persisted
                    return persisted

            frame = quick_review_frame()
            if frame.empty:
                payload: dict[str, Any] = {
                    "version": SUMMARY_VERSION,
                    "signature": quick_review_summary_signature(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "status": "not_generated",
                    "wells": [],
                }
            else:
                wells = quick_review_well_rows(mode="all", frame=frame)
                label_counts = {
                    str(key): int(value)
                    for key, value in frame["current_label"].value_counts().items()
                }
                payload = {
                    "version": SUMMARY_VERSION,
                    "signature": quick_review_summary_signature(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "status": "ready",
                    "round_id": str(frame.iloc[0]["integrated_round_id"]),
                    "well_count": int(len(wells)),
                    "completed_well_count": int(
                        sum(bool(row["completed"]) for row in wells)
                    ),
                    "pending_well_count": int(
                        sum(not bool(row["completed"]) for row in wells)
                    ),
                    "object_count": int(len(frame)),
                    "reviewed_object_count": int(
                        frame["reviewed_label"].notna().sum()
                    ),
                    "corrected_object_count": int(
                        (frame["decision"] == "corrected").sum()
                    ),
                    "temporal_review_object_count": int(
                        frame["temporal_completion_status"].isin(
                            [
                                "ambiguous_temporal_candidate",
                                "temporal_reclassification_review",
                            ]
                        ).sum()
                    ),
                    "label_counts": label_counts,
                    "wells": wells,
                }

            payload = _summary_json_safe(payload)
            write_summary(quick_summary_file, payload)
            quick_summary_cache["signature"] = payload["signature"]
            quick_summary_cache["payload"] = payload
            return payload

    def lineage_representative_view(
        well: str,
        timepoint: str,
        size: int,
        target_id: str,
        registration: dict[str, dict[str, float]],
    ) -> dict[str, Any] | None:
        proposals = latest_tracking_proposals()
        if proposals.empty:
            return None
        matched = proposals[
            (proposals["well"] == well.upper())
            & (
                (proposals["candidate_id"].astype(str) == target_id)
                | (
                    proposals["canonical_target_id"].astype(str)
                    == target_id
                )
            )
        ]
        if matched.empty:
            return None
        proposal = matched.iloc[0]
        payload = json.loads(proposal["proposal_points_json"] or "{}")
        primary = payload.get(timepoint, {})
        points = []
        if primary.get("present"):
            points = [primary, *(primary.get("additional_points") or [])]
        image = images_manifest[
            (images_manifest["well"] == well.upper())
            & (images_manifest["timepoint"] == timepoint.upper())
        ]
        if image.empty:
            return None
        if not points:
            lineage_sources = [
                artifact_path(
                    config,
                    "predictions",
                    "latest_first_division_lineages.csv",
                ),
                artifact_path(
                    config,
                    "predictions",
                    "latest_debris_tracks.csv",
                ),
            ]
            predicted = None
            for source in lineage_sources:
                if not source.exists():
                    continue
                lineage = pd.read_csv(source)
                local = lineage[
                    lineage["candidate_id"].astype(str) == target_id
                ]
                if local.empty or "unlinked_reasons_json" not in local:
                    continue
                reasons = json.loads(
                    local.iloc[0]["unlinked_reasons_json"] or "{}"
                )
                predicted = reasons.get(timepoint)
                if predicted:
                    break
            if predicted:
                shift = registration.get(
                    timepoint,
                    {
                        "align_shift_x_px": 0.0,
                        "align_shift_y_px": 0.0,
                    },
                )
                return {
                    "center_x_px": float(
                        predicted["predicted_x_px"]
                        - shift["align_shift_x_px"]
                    ),
                    "center_y_px": float(
                        predicted["predicted_y_px"]
                        - shift["align_shift_y_px"]
                    ),
                    "cell_count": 0,
                    "whole_lineage_cell_count": 0,
                    "whole_lineage_object_count": 0,
                    "window_size_px": int(size),
                    "scope": "lineage_motion_prediction",
                }
            return None
        half = size / 2
        coordinates = np.asarray(
            [[float(point["x_px"]), float(point["y_px"])] for point in points],
            dtype=float,
        )
        cell_counts = np.asarray(
            [
                int(point.get("cell_count", 1))
                if point.get("object_label") == "cell"
                else 1
                for point in points
            ],
            dtype=int,
        )
        confidences = np.ones(len(points), dtype=float)
        width = float(image.iloc[0]["width_px"])
        height = float(image.iloc[0]["height_px"])
        center_x_options = np.unique(
            np.clip(
                np.concatenate(
                    [
                        coordinates[:, 0] - half,
                        coordinates[:, 0] + half,
                        coordinates[:, 0],
                    ]
                ),
                half,
                width - half,
            )
        )
        center_y_options = np.unique(
            np.clip(
                np.concatenate(
                    [
                        coordinates[:, 1] - half,
                        coordinates[:, 1] + half,
                        coordinates[:, 1],
                    ]
                ),
                half,
                height - half,
            )
        )
        best_key = (-1, -1.0, float("-inf"))
        center_x, center_y = width / 2, height / 2
        for candidate_x in center_x_options:
            inside_x = np.abs(coordinates[:, 0] - candidate_x) <= half
            for candidate_y in center_y_options:
                inside = inside_x & (
                    np.abs(coordinates[:, 1] - candidate_y) <= half
                )
                center_cost = float(
                    (
                        (
                            (coordinates[inside, 0] - candidate_x) ** 2
                            + (coordinates[inside, 1] - candidate_y) ** 2
                        )
                        * cell_counts[inside]
                    ).sum()
                )
                key = (
                    int(cell_counts[inside].sum()),
                    float(confidences[inside].sum()),
                    -center_cost,
                )
                if key > best_key:
                    best_key = key
                    center_x = float(candidate_x)
                    center_y = float(candidate_y)
        return {
            "center_x_px": center_x,
            "center_y_px": center_y,
            "cell_count": int(best_key[0]),
            "whole_lineage_cell_count": int(
                sum(
                    int(point.get("cell_count", 1))
                    for point in points
                    if point.get("object_label") == "cell"
                )
            ),
            "whole_lineage_object_count": int(len(points)),
            "window_size_px": int(size),
            "scope": "current_lineage",
        }

    def add_model_candidates(
        context: dict[str, Any],
        well: str,
    ) -> dict[str, Any]:
        frame = latest_integrated_predictions()
        for timepoint, info in context["timepoints"].items():
            if not info.get("available"):
                continue
            local = frame[
                (frame["well"] == well.upper())
                & (frame["timepoint"] == timepoint)
            ].copy()
            cell_mask = local["integrated_label"].isin(
                ["single", "touching_doublet", "cluster_3plus"]
            )
            all_cells = local[cell_mask]
            info["all_model_cells"] = all_cells[
                [
                    "candidate_id",
                    "x_px",
                    "y_px",
                    "diameter_px",
                    "integrated_label",
                    "integrated_confidence",
                ]
            ].replace({np.nan: None}).to_dict(orient="records")
            within = local[
                local["integrated_label"].isin(
                    [
                        "single",
                        "touching_doublet",
                        "cluster_3plus",
                        "debris",
                    ]
                )
                & local["x_px"].between(
                    info["origin_x_px"],
                    info["origin_x_px"] + context["search_size_px"],
                )
                & local["y_px"].between(
                    info["origin_y_px"],
                    info["origin_y_px"] + context["search_size_px"],
                )
            ].copy()
            info["model_candidates"] = (
                within[
                    [
                        "candidate_id",
                        "x_px",
                        "y_px",
                        "diameter_px",
                        "integrated_label",
                        "integrated_confidence",
                    ]
                ]
                .rename(columns={"diameter_px": "equivalent_diameter_px"})
                .replace({np.nan: None})
                .to_dict(orient="records")
            )
        return context
    return SimpleNamespace(
        latest_integrated_predictions=latest_integrated_predictions,
        latest_tracking_proposals=latest_tracking_proposals,
        latest_completion_review=latest_completion_review,
        quick_review_frame=quick_review_frame,
        quick_review_well_rows=quick_review_well_rows,
        quick_review_summary_signature=quick_review_summary_signature,
        quick_review_summary=quick_review_summary,
        lineage_representative_view=lineage_representative_view,
        add_model_candidates=add_model_candidates,
    )

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
from .multiplicity import (
    ensure_integrated_review_table,
    ensure_multiplicity_table,
    save_integrated_reviews,
    save_multiplicity_labels,
)
from .hierarchy import suppress_nested_single_candidates
from .review_helpers import _boolean_series, _visible_v2_review_instances, _with_final_decisions
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
from fastapi import FastAPI, HTTPException
from .review_context import build_review_context
from .review_payloads import QuickReviewObjectItem, QuickReviewUndoPayload, QuickReviewWellPayload
from .review_storage import (
    _capture_quick_review_undo_snapshot,
    _restore_quick_review_undo_snapshot,
    _summary_json_safe,
    save_annotation,
)
from .teaching import save_teaching_labels


V3_TRACK_REVIEW_LABELS = {
    "cell",
    "dead_cell",
    # Retain legacy exact cell-subtype values for existing saved reviews.
    "single",
    "touching_doublet",
    "cluster_3plus",
    "debris",
    "invalid",
    "uncertain",
    "unmarked",
}

CELL_UNIT_WEIGHTS = {
    "single": 1,
    "touching_doublet": 2,
    "cluster_3plus": 3,
}


def _quick_review_filter_metrics(
    local: pd.DataFrame,
    screening: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Build the well-level values used by the pre-review filter UI."""

    labels = local["final_review_label"].fillna(local["current_label"])
    timepoints = local["timepoint"].astype(str).str.upper()

    def cell_count(timepoint: str) -> int:
        selected_labels = labels[timepoints.eq(timepoint)]
        calculated = int(
            sum(CELL_UNIT_WEIGHTS.get(str(label), 0) for label in selected_labels)
        )
        try:
            screening_count = float(screening.get(f"{timepoint.lower()}_cell_units"))
        except (TypeError, ValueError):
            screening_count = float("nan")
        return calculated if not np.isfinite(screening_count) else int(round(screening_count))

    t2_labels = labels[timepoints.eq("T2")]
    raw_coverage = report.get(
        "day14_sheet_coverage_pct",
        report.get("endpoint_sheet_coverage_pct"),
    )
    try:
        endpoint_coverage = float(raw_coverage)
    except (TypeError, ValueError):
        endpoint_coverage = None
    if endpoint_coverage is not None and not np.isfinite(endpoint_coverage):
        endpoint_coverage = None
    manual_decision = str(screening.get("review_decision", "unclassified")).lower()
    if manual_decision not in {"approved", "pending", "rejected", "unclassified"}:
        manual_decision = "unclassified"
    return {
        "endpoint_coverage_pct": endpoint_coverage,
        "day0_cell_count": cell_count("T0"),
        "day1_cell_count": cell_count("T1"),
        "day2_cell_count": cell_count("T2"),
        "day2_debris_count": int(t2_labels.eq("debris").sum()),
        "manual_review_decision": manual_decision,
    }


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
                SELECT candidate_id, reviewed_label, decision, updated_at, integrated_review_id
                FROM integrated_training_reviews
                ORDER BY updated_at, integrated_review_id
                """,
                connection,
            )
        # A later inference round changes integrated_round_id without migrating
        # human decisions.  Match by candidate_id and keep the newest decision
        # so reviewed progress survives a round-id change.
        if not reviews.empty:
            reviews = reviews.drop_duplicates("candidate_id", keep="last")
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
                        SELECT track_id, label AS v3_reviewed_label, updated_at, temporal_track_review_id
                        FROM temporal_track_reviews
                        ORDER BY updated_at, temporal_track_review_id
                        """,
                        connection,
                    )
                if not track_reviews.empty:
                    track_reviews = track_reviews.drop_duplicates("track_id", keep="last")
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
        # candidate_id is UNIQUE in quick_missed_objects, so manual additions
        # survive a round-id change without filtering by round_id.
        with sqlite3.connect(database) as connection:
            manual = pd.read_sql_query(
                """
                SELECT candidate_id, well, timepoint, x_px, y_px,
                       diameter_px, reviewed_label, reviewer, updated_at
                FROM quick_missed_objects
                """,
                connection,
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
        try:
            with sqlite3.connect(database) as connection:
                manual_decision_lookup = {
                    str(well).upper(): str(decision).lower()
                    for well, decision in connection.execute(
                        "SELECT well, decision FROM well_screening_reviews"
                    ).fetchall()
                }
        except sqlite3.Error:
            manual_decision_lookup = {}
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
            screening = dict(screening_lookup.get(str(well).upper(), {}))
            screening["review_decision"] = manual_decision_lookup.get(
                str(well).upper(), "unclassified"
            )
            report = report_lookup.get(str(well).upper(), {})
            status = ui_screening_status(
                report,
                str(screening.get("screening_status", "ambiguous")),
            )
            filter_metrics = _quick_review_filter_metrics(local, screening, report)
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
                    **filter_metrics,
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


def register_quick_review_routes(
    app: FastAPI,
    config: dict[str, Any],
    database: str | Path,
    images_manifest: pd.DataFrame,
    quick_review_service: SimpleNamespace,
    gated_lookup: Any,
    sync_catalog_after_review: Any,
    refresh_gated_report: Any,
) -> None:
    """Register the quick-review routes for one plate."""

    @app.get("/api/quick-review-stats")
    def quick_review_stats() -> dict[str, Any]:
        summary = quick_review_service.quick_review_summary()
        if summary.get("status") != "ready":
            return {"status": "not_generated"}
        return {
            "status": "ready",
            **{
                key: summary[key]
                for key in (
                    "round_id",
                    "well_count",
                    "completed_well_count",
                    "pending_well_count",
                    "object_count",
                    "reviewed_object_count",
                    "corrected_object_count",
                    "temporal_review_object_count",
                    "label_counts",
                )
                if key in summary
            },
        }

    @app.get("/api/quick-review-wells")
    def quick_review_wells(
        mode: str = "pending",
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        if mode not in {"pending", "reviewed", "all"}:
            raise HTTPException(
                status_code=422, detail="Invalid quick review mode"
            )
        summary = quick_review_service.quick_review_summary()
        rows = list(summary.get("wells") or [])
        if mode == "pending":
            rows = [row for row in rows if not row.get("completed")]
        elif mode == "reviewed":
            rows = [row for row in rows if row.get("completed")]
        if search:
            needle = search.strip().upper()
            rows = [
                row for row in rows if needle in str(row.get("well", "")).upper()
            ]
        return rows

    @app.get("/api/quick-review-well/{well}")
    def quick_review_well(well: str) -> dict[str, Any]:
        normalized_well = well.upper()
        frame = quick_review_service.quick_review_frame()
        local = frame[frame["well"] == normalized_well].copy()
        if local.empty:
            raise HTTPException(
                status_code=404, detail="Reviewable well unavailable"
            )
        screening_path = artifact_path(
            config, "predictions", "latest_well_screening.csv"
        )
        screening: dict[str, Any] = {}
        roi: dict[str, Any] = {}
        if screening_path.exists():
            screening_frame = pd.read_csv(screening_path)
            selected_screening = screening_frame[
                screening_frame["well"].astype(str).str.upper()
                == normalized_well
            ]
            if not selected_screening.empty:
                screening = selected_screening.iloc[0].to_dict()
                try:
                    roi = json.loads(str(screening.get("roi_json", "{}")))
                except json.JSONDecodeError:
                    roi = {}
        try:
            with sqlite3.connect(database) as connection:
                saved_decision = connection.execute(
                    "SELECT decision FROM well_screening_reviews WHERE well = ?",
                    (normalized_well,),
                ).fetchone()
        except sqlite3.Error:
            saved_decision = None
        screening["review_decision"] = (
            str(saved_decision[0]).lower() if saved_decision else "unclassified"
        )
        report = gated_lookup().get(normalized_well, {})
        display_names = config.get("review", {}).get(
            "timepoint_display_names",
            {"T0": "T0", "T1": "T1", "T2": "T2", "T3": "Day7", "T4": "Day14"},
        )
        available_images = images_manifest[
            (images_manifest["well"] == normalized_well)
            & (images_manifest["timepoint"].isin(["T0", "T1", "T2", "T3", "T4"]))
            & (images_manifest["decode_status"] == "ok")
        ]
        images: dict[str, Any] = {}
        for timepoint in ["T0", "T1", "T2", "T3", "T4"]:
            selected = available_images[
                available_images["timepoint"] == timepoint
            ]
            if selected.empty:
                images[timepoint] = {"available": False}
                continue
            row = selected.iloc[0]
            late_decision = (
                str(screening.get(f"{timepoint.lower()}_growth_decision", "pending"))
                if timepoint in {"T3", "T4"}
                else ""
            )
            # Late images are review evidence only.  The former Day7 density
            # locator was not accurate enough to justify automatic panning or
            # zooming, so T3/T4 always open as a complete well field.
            representative = (
                None if timepoint in {"T3", "T4"} else roi.get(timepoint) or None
            )
            if timepoint == "T4" and report:
                day14_positive = str(
                    report.get("day14_obvious_growth", "")
                ).strip().lower() in {"1", "true", "yes"}
                late_decision = (
                    "obvious_growth"
                    if day14_positive
                    else "no_growth"
                )
            # The endpoint CF mask is not sufficiently reliable for a visual
            # overlay.  Keep the raw image and human growth decision, but do
            # not calculate or return expensive/ambiguous shadow contours.
            growth_regions: list[dict[str, Any]] = []
            images[timepoint] = {
                "available": True,
                "display_label": str(display_names.get(timepoint, timepoint)),
                "width_px": int(row["width_px"]),
                "height_px": int(row["height_px"]),
                "url": (
                    f"/api/well-image?well={normalized_well}"
                    f"&timepoint={timepoint}&max_size=1400"
                ),
                "hires_url": (
                    f"/api/well-image?well={normalized_well}"
                    f"&timepoint={timepoint}&max_size=4096"
                ),
                "annotatable": timepoint in {"T0", "T1", "T2"},
                "late_growth_decision": late_decision,
                "late_growth_source": (
                    "pending"
                    if pd.isna(
                        screening.get(
                            f"{timepoint.lower()}_growth_source", "pending"
                        )
                    )
                    else str(
                        screening.get(
                            f"{timepoint.lower()}_growth_source", "pending"
                        )
                    )
                ) if timepoint in {"T3", "T4"} else "",
                "late_growth_search_stage": (
                    ""
                    if pd.isna(
                        screening.get(
                            f"{timepoint.lower()}_growth_search_stage", ""
                        )
                    )
                    else str(
                        screening.get(
                            f"{timepoint.lower()}_growth_search_stage", ""
                        )
                    )
                ) if timepoint in {"T3", "T4"} else "",
                "representative_view": representative,
                "growth_regions": growth_regions,
                "growth_overlay_style": "none",
                "default_zoom": 1.0,
            }
        columns = [
            "candidate_id",
            "well",
            "timepoint",
            "x_px",
            "y_px",
            "area_px",
            "diameter_px",
            "integrated_label",
            "integrated_confidence",
            "cell_probability",
            "debris_probability",
            "invalid_probability",
            "single_probability",
            "touching_doublet_probability",
            "cluster_3plus_probability",
            "reviewed_label",
            "decision",
            "current_label",
            "final_label",
            "final_review_label",
            "final_source",
            "final_reason_code",
            "final_reason_text",
            "final_confidence",
            "final_status",
            "is_manual_missed",
            "temporal_completion_score",
            "temporal_completion_status",
            "temporal_completion_source_id",
            "temporal_completion_direction",
            "temporal_appearance_status",
            "temporal_appearance_confidence",
            "temporal_appearance_match_count",
            "temporal_mean_patch_similarity",
            "temporal_mean_mask_iou",
            "temporal_max_area_ratio",
            "temporal_appearance_matches",
            "instance_component_id",
            "instance_component_distance_px",
            "instance_footprint_diameter_px",
            "v2_instance_id",
            "v2_contour_json",
            "v2_instance_area_px",
            "v2_instance_diameter_px",
            "v2_instance_confidence",
            "v2_objectness",
            "v2_wall_overlap",
            "v2_is_unique_instance",
            "v2_is_temporal_candidate",
            "v2_is_reviewable_instance",
            "v2_is_counting_instance",
            "v2_auto_invalid_probability_rule",
            "v2_temporal_same_object_score",
            "v2_temporal_static_similarity_score",
            "v2_temporal_candidate_count",
            "v2_temporal_foreground_similarity",
            "v2_temporal_shape_similarity",
            "v2_temporal_change_score",
            "v2_temporal_growth_score",
            "v2_temporal_foreground_quality",
            "v2_temporal_evidence_frame_count",
            "v2_temporal_pair_count",
            "v2_temporal_three_frame_static",
            "v2_temporal_morphology_stable_three_frame",
            "v2_temporal_morphology_consensus_cell_probability",
            "v2_temporal_debris_boost",
            "v2_temporal_cell_boost",
            "v2_adjusted_cell_probability",
            "v2_adjusted_debris_probability",
            "v2_temporal_adjustment_applied",
            "v2_temporal_reason",
            "v2_temporal_adjusted_label",
            "v2_static_wall_artifact",
            "v2_static_wall_cell_veto",
            "v2_strong_cell_evidence_frame_count",
            "v2_temporal_recovered",
            "v2_temporal_track_id",
            "v2_low_cell_noncell_resolved",
            "v2_noncell_resolution_label",
            "v3_track_behavior",
            "v3_track_conclusion",
            "v3_unified_label",
            "v3_label_mode",
            "v3_wall_origin",
            "v3_wall_cell_veto",
            "v3_wall_strong_cell_frame_count",
            "v3_behavior_score",
            "v3_division_interval",
            "v3_division_veto",
            "v3_division_rescue",
            "v3_division_rescue_parent_candidate_id",
            "v3_division_rescue_child_candidate_ids",
            "v3_division_rescue_score",
            "v3_reason",
            "v3_frame_state",
            "v3_proposed_label",
            "v3_proposed_cell_probability",
            "v3_proposed_debris_probability",
            "v3_proposed_invalid_probability",
            "v3_would_change",
            "v3_identity_score",
            "v3_static_similarity",
            "v3_shape_similarity",
            "v3_morphology_change_score",
            "v3_semantic_degradation",
            "v3_degradation_evidence_score",
            "v3_foreground_quality",
            "v3_track_frame_count",
            "v3_track_pair_count",
            "v3_valid_observations",
            "v3_persistent_cell_evidence",
            "v3_cell_to_debris_candidate",
            "v3_track_id",
            "v3_timepoint",
            "v3_reviewed_label",
        ]
        columns = [column for column in columns if column in local.columns]
        timepoint_order = pd.Categorical(
            local["timepoint"], categories=["T0", "T1", "T2"], ordered=True
        )
        local = (
            local.assign(_timepoint_order=timepoint_order)
            .sort_values(["_timepoint_order", "y_px", "x_px"])
            .drop(columns="_timepoint_order")
        )
        v3_tracks: list[dict[str, Any]] = []
        if "v3_track_id" in local.columns:
            track_frame = local[local["v3_track_id"].fillna("").astype(str).ne("")]
            for track_id, track in track_frame.groupby("v3_track_id", sort=False):
                def first_text(column: str) -> str:
                    if column not in track:
                        return ""
                    values = track[column].fillna("").astype(str)
                    return next((value for value in values if value), "")

                v3_tracks.append(
                    {
                        "track_id": str(track_id),
                        "well": normalized_well,
                        "behavior": first_text("v3_track_behavior"),
                        "conclusion": first_text("v3_track_conclusion"),
                        "unified_label": first_text("v3_unified_label"),
                        "label_mode": first_text("v3_label_mode"),
                        "reason": first_text("v3_reason"),
                        "division_rescue": bool(
                            _boolean_series(
                                track.get(
                                    "v3_division_rescue",
                                    pd.Series(False, index=track.index),
                                )
                            ).any()
                        )
                        if "v3_division_rescue" in track
                        else False,
                        "division_rescue_parent_candidate_id": first_text(
                            "v3_division_rescue_parent_candidate_id"
                        ),
                        "division_rescue_child_candidate_ids": first_text(
                            "v3_division_rescue_child_candidate_ids"
                        ),
                        "division_rescue_score": float(
                            pd.to_numeric(
                                track.get(
                                    "v3_division_rescue_score",
                                    pd.Series(0.0, index=track.index),
                                ),
                                errors="coerce",
                            ).fillna(0.0).max()
                        ),
                        "reviewed_label": first_text("v3_reviewed_label"),
                        "candidate_ids": track["candidate_id"].astype(str).tolist(),
                        "timepoints": track["timepoint"].astype(str).tolist(),
                        "frame_count": int(len(track)),
                    }
                )
        search_hints: list[dict[str, Any]] = []
        try:
            with sqlite3.connect(database) as connection:
                count_override_rows = connection.execute(
                    """
                    SELECT timepoint, cell_count
                    FROM well_timepoint_cell_count_reviews
                    WHERE well = ?
                    """,
                    (normalized_well,),
                ).fetchall()
        except sqlite3.Error:
            count_override_rows = []
        count_overrides = {
            str(timepoint).upper(): max(0, int(cell_count))
            for timepoint, cell_count in count_override_rows
        }
        cell_count_totals: dict[str, dict[str, Any]] = {}
        final_labels = local["final_review_label"].fillna(local["current_label"])
        for timepoint in ("T0", "T1", "T2"):
            selected_labels = final_labels[
                local["timepoint"].astype(str).str.upper().eq(timepoint)
            ]
            automatic_count = int(
                sum(CELL_UNIT_WEIGHTS.get(str(label), 0) for label in selected_labels)
            )
            override = count_overrides.get(timepoint)
            cell_count_totals[timepoint] = {
                "automatic_count": automatic_count,
                "cell_count": automatic_count if override is None else override,
                "source": "automatic" if override is None else "human",
            }
        return {
            "round_id": str(local.iloc[0]["integrated_round_id"]),
            "well": normalized_well,
            "images": images,
            "objects": (
                local[columns]
                .replace({np.nan: None})
                .to_dict(orient="records")
            ),
            "v3_tracks": v3_tracks,
            "search_hints": search_hints,
            "cell_count_totals": cell_count_totals,
            "screening": {
                key: (None if pd.isna(value) else value)
                for key, value in screening.items()
                if key != "roi_json"
            },
            "report": {
                key: (None if pd.isna(value) else value)
                for key, value in report.items()
                if key != "day7_regions_json"
            },
            "counts_by_timepoint": {
                timepoint: int((local["timepoint"] == timepoint).sum())
                for timepoint in ["T0", "T1", "T2"]
            },
        }


    @app.post("/api/quick-review-well-labels")
    def quick_review_well_labels(
        payload: QuickReviewWellPayload,
    ) -> dict[str, Any]:
        affected_wells = {item.well.upper() for item in payload.items}
        if payload.well:
            affected_wells.add(payload.well.upper())
        allowed = {
            "single",
            "touching_doublet",
            "cluster_3plus",
            "debris",
            "invalid",
            "uncertain",
        }
        for item in payload.items:
            if (
                item.predicted_label not in allowed
                or item.reviewed_label not in allowed
            ):
                raise HTTPException(
                    status_code=422,
                    detail="Invalid quick review label",
                )
        for track_review in payload.v3_track_reviews:
            if track_review.label not in V3_TRACK_REVIEW_LABELS:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid V3 track review label",
                )
        primary_well = (
            payload.well.upper()
            if payload.well
            else next(iter(affected_wells), "")
        )
        undo_snapshot = _capture_quick_review_undo_snapshot(
            database,
            payload.round_id,
            [item.candidate_id for item in payload.items],
            [item.track_id for item in payload.v3_track_reviews],
        )
        undo_snapshot["well"] = primary_well
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            existing_manual = {
                str(row["candidate_id"]): dict(row)
                for row in connection.execute(
                    """
                    SELECT quick_missed_id, annotation_id, candidate_id
                    FROM quick_missed_objects
                    WHERE round_id = ?
                    """,
                    (payload.round_id,),
                ).fetchall()
            }
        standard_items: list[dict[str, Any]] = []
        mappings: list[dict[str, Any]] = []
        created_session_ids: list[int] = []
        updated = datetime.now(timezone.utc).isoformat()
        cell_labels = {
            "single",
            "touching_doublet",
            "cluster_3plus",
        }

        def object_type_for(label: str) -> str:
            if label in cell_labels:
                return "cell"
            if label == "debris":
                return "debris"
            if label == "invalid":
                return "irrelevant"
            return "uncertain"

        def sync_training_labels(
            candidate_id: str,
            item: QuickReviewObjectItem,
        ) -> None:
            if item.reviewed_label in cell_labels:
                save_teaching_labels(
                    database,
                    [
                        {
                            "candidate_id": candidate_id,
                            "well": item.well.upper(),
                            "timepoint": item.timepoint.upper(),
                            "x_px": item.x_px,
                            "y_px": item.y_px,
                            "label": "cell",
                            "source": "quick_missed_annotation",
                        }
                    ],
                    payload.reviewer,
                )
                save_multiplicity_labels(
                    database,
                    [
                        {
                            "candidate_id": candidate_id,
                            "well": item.well.upper(),
                            "timepoint": item.timepoint.upper(),
                            "x_px": item.x_px,
                            "y_px": item.y_px,
                            "label": item.reviewed_label,
                            "source": "quick_missed_annotation",
                        }
                    ],
                    payload.reviewer,
                )
            elif item.reviewed_label == "debris":
                save_teaching_labels(
                    database,
                    [
                        {
                            "candidate_id": candidate_id,
                            "well": item.well.upper(),
                            "timepoint": item.timepoint.upper(),
                            "x_px": item.x_px,
                            "y_px": item.y_px,
                            "label": "debris",
                            "source": "quick_missed_annotation",
                        }
                    ],
                    payload.reviewer,
                )
                with sqlite3.connect(database) as connection:
                    connection.execute(
                        """
                        DELETE FROM multiplicity_labels
                        WHERE candidate_id = ?
                        """,
                        (candidate_id,),
                    )
            else:
                with sqlite3.connect(database) as connection:
                    connection.execute(
                        "DELETE FROM teaching_labels WHERE candidate_id = ?",
                        (candidate_id,),
                    )
                    connection.execute(
                        """
                        DELETE FROM multiplicity_labels
                        WHERE candidate_id = ?
                        """,
                        (candidate_id,),
                    )

        try:
            for item in payload.items:
                manual_row = existing_manual.get(item.candidate_id)
                if item.is_new:
                    if item.reviewed_label == "invalid":
                        continue
                    image = images_manifest[
                        (images_manifest["well"] == item.well.upper())
                        & (
                            images_manifest["timepoint"]
                            == item.timepoint.upper()
                        )
                    ]
                    if image.empty:
                        raise ValueError("Image unavailable for missed target")
                    image_row = image.iloc[0]
                    temporary_id = item.candidate_id
                    annotation_id = save_annotation(
                        database,
                        {
                            "sequence_id": str(
                                image_row["experiment_id"]
                            ),
                            "plate_id": str(image_row["plate_id"]),
                            "well": item.well.upper(),
                            "timepoint": item.timepoint.upper(),
                            "object_id": temporary_id,
                            "canonical_target_id": temporary_id,
                            "track_id": temporary_id,
                            "parent_track_id": None,
                            "x_px": item.x_px,
                            "y_px": item.y_px,
                            "object_type": object_type_for(
                                item.reviewed_label
                            ),
                            "viability": (
                                "unknown"
                                if item.reviewed_label in cell_labels
                                else "not_applicable"
                            ),
                            "division_state": "unknown",
                            "duplicate_of": None,
                            "reviewer": payload.reviewer,
                            "confidence": 1.0,
                            "notes": "quick_review_missed",
                        },
                    )
                    candidate_id = (
                        f"{item.well.upper()}:{item.timepoint.upper()}:"
                        f"manual:{annotation_id}"
                    )
                    with sqlite3.connect(database) as connection:
                        connection.execute(
                            """
                            INSERT INTO quick_missed_objects (
                              round_id, annotation_id, candidate_id, well,
                              timepoint, x_px, y_px, diameter_px,
                              reviewed_label, reviewer, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                payload.round_id,
                                annotation_id,
                                candidate_id,
                                item.well.upper(),
                                item.timepoint.upper(),
                                item.x_px,
                                item.y_px,
                                max(4.0, min(item.diameter_px, 96.0)),
                                item.reviewed_label,
                                payload.reviewer,
                                updated,
                            ),
                        )
                    sync_training_labels(candidate_id, item)
                    mappings.append(
                        {
                            "temporary_id": temporary_id,
                            "candidate_id": candidate_id,
                            "annotation_id": annotation_id,
                        }
                    )
                elif manual_row:
                    candidate_id = item.candidate_id
                    annotation_id = int(manual_row["annotation_id"])
                    if item.reviewed_label == "invalid":
                        with sqlite3.connect(database) as connection:
                            connection.execute(
                                """
                                DELETE FROM quick_missed_objects
                                WHERE candidate_id = ?
                                """,
                                (candidate_id,),
                            )
                            connection.execute(
                                """
                                DELETE FROM annotations
                                WHERE annotation_id = ?
                                """,
                                (annotation_id,),
                            )
                            connection.execute(
                                """
                                DELETE FROM teaching_labels
                                WHERE candidate_id = ?
                                """,
                                (candidate_id,),
                            )
                            connection.execute(
                                """
                                DELETE FROM multiplicity_labels
                                WHERE candidate_id = ?
                                """,
                                (candidate_id,),
                            )
                        continue
                    with sqlite3.connect(database) as connection:
                        connection.execute(
                            """
                            UPDATE quick_missed_objects
                            SET x_px=?, y_px=?, diameter_px=?,
                                reviewed_label=?, reviewer=?, updated_at=?
                            WHERE candidate_id=?
                            """,
                            (
                                item.x_px,
                                item.y_px,
                                max(4.0, min(item.diameter_px, 96.0)),
                                item.reviewed_label,
                                payload.reviewer,
                                updated,
                                candidate_id,
                            ),
                        )
                        connection.execute(
                            """
                            UPDATE annotations
                            SET x_px=?, y_px=?, object_type=?, reviewer=?,
                                notes='quick_review_missed',
                                updated_at=?
                            WHERE annotation_id=?
                            """,
                            (
                                item.x_px,
                                item.y_px,
                                object_type_for(item.reviewed_label),
                                payload.reviewer,
                                updated,
                                annotation_id,
                            ),
                        )
                    sync_training_labels(candidate_id, item)
                else:
                    standard_items.append(
                        {
                            "candidate_id": item.candidate_id,
                            "predicted_label": item.predicted_label,
                            "reviewed_label": item.reviewed_label,
                        }
                    )
            saved_standard = (
                save_integrated_reviews(
                    database,
                    payload.round_id,
                    standard_items,
                    payload.reviewer,
                )
                if standard_items
                else 0
            )
            saved_v3_tracks = 0
            if payload.v3_track_reviews:
                with sqlite3.connect(database) as connection:
                    connection.executemany(
                        """
                        INSERT INTO temporal_track_reviews (
                          round_id, track_id, well, label, behavior,
                          reviewer, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(round_id, track_id) DO UPDATE SET
                          well=excluded.well,
                          label=excluded.label,
                          behavior=excluded.behavior,
                          reviewer=excluded.reviewer,
                          updated_at=excluded.updated_at
                        """,
                        [
                            (
                                payload.round_id,
                                item.track_id,
                                item.well.upper(),
                                item.label,
                                item.behavior,
                                payload.reviewer,
                                updated,
                            )
                            for item in payload.v3_track_reviews
                        ],
                    )
                    saved_v3_tracks = len(payload.v3_track_reviews)
            if payload.duration_ms is not None:
                corrected_count = sum(
                    item.reviewed_label != item.predicted_label for item in payload.items
                )
                with sqlite3.connect(database) as connection:
                    session_cursor = connection.execute(
                        "INSERT INTO quick_review_sessions (round_id, well, reviewer, duration_ms, object_count, corrected_count, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            payload.round_id,
                            (
                                payload.well.upper()
                                if payload.well
                                else payload.items[0].well.upper()
                                if payload.items
                                else ""
                            ),
                            payload.reviewer,
                            max(0, int(payload.duration_ms)),
                            len(payload.items),
                            corrected_count,
                            updated,
                        ),
                    )
                    if session_cursor.lastrowid is not None:
                        created_session_ids.append(int(session_cursor.lastrowid))
            undo_snapshot["created_manual"] = mappings
            undo_snapshot["created_session_ids"] = created_session_ids
            undo_snapshot_json = json.dumps(
                _summary_json_safe(undo_snapshot),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            with sqlite3.connect(database) as connection:
                undo_cursor = connection.execute(
                    """
                    INSERT INTO quick_review_undo_actions (
                      round_id, well, reviewer, snapshot_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        payload.round_id,
                        primary_well,
                        payload.reviewer,
                        undo_snapshot_json,
                        updated,
                    ),
                )
                undo_action_id = int(undo_cursor.lastrowid)
            screening = build_well_screening(
                config,
                database,
                selected_wells=affected_wells or None,
            )
            gated_summary, updated_report = refresh_gated_report()
            screening_source = artifact_path(
                config, "predictions", "latest_well_screening.csv"
            )
            screening_row = None
            selected_screening = pd.DataFrame()
            if screening_source.exists() and primary_well:
                screening_frame = pd.read_csv(screening_source, low_memory=False)
                selected_screening = screening_frame[
                    screening_frame["well"].astype(str).str.upper() == primary_well
                ]
            if not selected_screening.empty:
                screening_row = selected_screening.replace({np.nan: None}).iloc[0].to_dict()
            try:
                # Refresh the persistent list index once while the save
                # request already owns the authoritative updated state.  The
                # next stats/list requests can then reuse it without parsing
                # the full prediction table again.
                quick_review_service.quick_review_summary(force=True)
            except Exception:  # pragma: no cover - cache failure must not undo a save
                quick_summary_cache["signature"] = None
                quick_summary_cache["payload"] = None
            sync_catalog_after_review(
                "quick_review",
                wells=affected_wells,
                reviewer=payload.reviewer,
                action_id=str(undo_action_id),
                operation="quick_review_save",
            )
            return {
                "status": "saved",
                "saved_standard": saved_standard,
                "saved_missed": len(mappings),
                "saved_v3_tracks": saved_v3_tracks,
                "mappings": mappings,
                "undo_action": {
                    "action_id": undo_action_id,
                    "well": primary_well,
                    "round_id": payload.round_id,
                    "created_at": updated,
                },
                "well_screening": screening,
                "screening": screening_row,
                "gated_report_summary": (
                    None
                    if gated_summary is None
                    else {key: value for key, value in gated_summary.items() if key != "wells"}
                ),
                "report": updated_report.get(primary_well),
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


    @app.get("/api/quick-review-undo")
    def quick_review_undo_status(reviewer: str = "local_user") -> dict[str, Any]:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT undo_action_id, round_id, well, reviewer, created_at
                FROM quick_review_undo_actions
                WHERE reviewer = ? AND undone_at IS NULL
                ORDER BY undo_action_id DESC
                LIMIT 1
                """,
                (reviewer,),
            ).fetchone()
        if row is None:
            return {"available": False}
        return {
            "available": True,
            "action": {
                "action_id": int(row["undo_action_id"]),
                "round_id": str(row["round_id"]),
                "well": str(row["well"]),
                "reviewer": str(row["reviewer"]),
                "created_at": str(row["created_at"]),
            },
        }

    @app.post("/api/quick-review-undo")
    def quick_review_undo(payload: QuickReviewUndoPayload) -> dict[str, Any]:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            if payload.action_id is None:
                row = connection.execute(
                    """
                    SELECT *
                    FROM quick_review_undo_actions
                    WHERE reviewer = ? AND undone_at IS NULL
                    ORDER BY undo_action_id DESC
                    LIMIT 1
                    """,
                    (payload.reviewer,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT *
                    FROM quick_review_undo_actions
                    WHERE undo_action_id = ?
                      AND reviewer = ?
                      AND undone_at IS NULL
                    """,
                    (int(payload.action_id), payload.reviewer),
                ).fetchone()
        if row is None:
            return {"status": "empty", "available": False}
        try:
            snapshot = json.loads(str(row["snapshot_json"]))
        except (TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="Undo snapshot is invalid") from exc
        _restore_quick_review_undo_snapshot(database, snapshot)
        undone_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE quick_review_undo_actions SET undone_at = ? WHERE undo_action_id = ?",
                (undone_at, int(row["undo_action_id"])),
            )
        affected_wells = {
            str(snapshot.get("well", row["well"])).upper()
        }
        screening = build_well_screening(
            config,
            database,
            selected_wells=affected_wells,
        )
        gated_summary, updated_report = refresh_gated_report()
        try:
            quick_review_service.quick_review_summary(force=True)
        except Exception:  # pragma: no cover - cache failure must not undo a restore
            quick_summary_cache["signature"] = None
            quick_summary_cache["payload"] = None
        sync_catalog_after_review(
            "quick_review",
            wells=affected_wells,
            reviewer=payload.reviewer,
            action_id=str(row["undo_action_id"]),
            operation="quick_review_undo",
        )
        undone_well = str(row["well"]).upper()
        screening_row = None
        screening_source = artifact_path(
            config, "predictions", "latest_well_screening.csv"
        )
        if screening_source.exists():
            screening_frame = pd.read_csv(screening_source, low_memory=False)
            selected_screening = screening_frame[
                screening_frame["well"].astype(str).str.upper() == undone_well
            ]
            if not selected_screening.empty:
                screening_row = selected_screening.replace({np.nan: None}).iloc[0].to_dict()
        return {
            "status": "undone",
            "available": True,
            "undone_action": {
                "action_id": int(row["undo_action_id"]),
                "round_id": str(row["round_id"]),
                "well": undone_well,
                "created_at": str(row["created_at"]),
                "undone_at": undone_at,
            },
            "well_screening": screening,
            "screening": screening_row,
            "gated_report_summary": (
                None
                if gated_summary is None
                else {key: value for key, value in gated_summary.items() if key != "wells"}
            ),
            "report": updated_report.get(undone_well),
        }


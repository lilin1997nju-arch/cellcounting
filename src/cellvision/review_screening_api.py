from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from .config import artifact_path
from .review_payloads import (
    LateGrowthReviewPayload,
    TimepointCellCountReviewPayload,
    WellScreeningReviewPayload,
)
from .well_screening import (
    build_well_screening,
    ensure_well_timepoint_cell_count_review_table,
    save_late_growth_review,
)


def register_screening_routes(
    app: FastAPI,
    config: dict[str, Any],
    database: str | Path,
    images_manifest: pd.DataFrame,
    gated_lookup: Any,
    ui_screening_status: Any,
    ui_screening_status_label: Any,
    ui_status_aliases: dict[str, str],
    refresh_gated_report: Any,
    sync_catalog_after_review: Any,
) -> None:
    """Register well-conclusions, screening and late-growth routes."""

    @app.get("/api/well-conclusions")
    def well_conclusions(limit: int = 200) -> list[dict[str, Any]]:
        source = artifact_path(
            config, "predictions", "latest_well_conclusions.csv"
        )
        if not source.exists():
            return []
        frame = pd.read_csv(source).head(max(1, min(limit, 500)))
        return frame.replace({pd.NA: None}).where(
            pd.notna(frame), None
        ).to_dict(orient="records")

    @app.get("/api/screening-wells")
    def screening_wells() -> list[dict[str, Any]]:
        source = artifact_path(
            config, "predictions", "latest_well_screening.csv"
        )
        if not source.exists():
            build_well_screening(config, database)
        frame = pd.read_csv(source)
        base = {
            str(row.well).upper(): row._asdict()
            for row in frame.itertuples(index=False)
        }
        report = gated_lookup()
        if report:
            for well, report_row in report.items():
                row = base.setdefault(well, {"well": well})
                status = ui_screening_status(
                    report_row,
                    str(row.get("screening_status", "ambiguous")),
                )
                row.update({
                    "screening_status": status,
                    "report_category": report_row.get("final_category"),
                    "report_category_label": ui_screening_status_label(status),
                    "report_reason": report_row.get("undetermined_reason"),
                    "report_reason_label": report_row.get("undetermined_reason_label"),
                    "day14_obvious_growth": report_row.get("day14_obvious_growth"),
                })
        def sort_key(item: dict[str, Any]) -> tuple[int, int]:
            well = str(item.get("well", ""))
            try:
                return ord(well[0]) - ord("A"), int(well[1:])
            except (IndexError, ValueError):
                return 99, 99
        for row in base.values():
            row["screening_status"] = ui_status_aliases.get(
                str(row.get("screening_status", "ambiguous")),
                str(row.get("screening_status", "t0_missing_late_cells")),
            )
        return [
            {key: (None if pd.isna(value) else value) for key, value in row.items()}
            for row in sorted(base.values(), key=sort_key)
        ]

    @app.post("/api/screening-review")
    def screening_review(
        payload: WellScreeningReviewPayload,
    ) -> dict[str, Any]:
        if payload.decision not in {"approved", "rejected", "pending", "unclassified"}:
            raise HTTPException(status_code=422, detail="Invalid screening decision")
        well = payload.well.upper()
        if well not in set(images_manifest["well"].astype(str).str.upper()):
            raise HTTPException(status_code=404, detail="well unavailable")
        updated = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(database) as connection:
            connection.execute(
                """
                INSERT INTO well_screening_reviews (
                  well, decision, reviewer, notes, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(well) DO UPDATE SET
                  decision=excluded.decision,
                  reviewer=excluded.reviewer,
                  notes=excluded.notes,
                  updated_at=excluded.updated_at
                """,
                (well, payload.decision, payload.reviewer, payload.notes, updated),
            )
        summary = build_well_screening(config, database, selected_wells={well})
        sync_catalog_after_review(
            "screening_review",
            wells={well},
            reviewer=payload.reviewer,
            operation="screening_review_save",
        )
        return {"status": "saved", "well": well, "summary": summary}

    @app.post("/api/late-growth-review")
    def late_growth_review(payload: LateGrowthReviewPayload) -> dict[str, Any]:
        well = payload.well.upper()
        timepoint = payload.timepoint.upper()
        available = images_manifest[
            (images_manifest["well"].astype(str).str.upper() == well)
            & (images_manifest["timepoint"].astype(str).str.upper() == timepoint)
            & (images_manifest["decode_status"] == "ok")
        ]
        if available.empty:
            raise HTTPException(
                status_code=404,
                detail="Late growth image unavailable for this well and timepoint",
            )
        try:
            save_late_growth_review(
                database,
                well,
                timepoint,
                payload.decision,
                payload.reviewer,
                payload.notes,
                datetime.now(timezone.utc).isoformat(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        summary = build_well_screening(config, database, selected_wells={well})
        gated_summary, updated_report = refresh_gated_report()
        sync_catalog_after_review(
            "late_growth_review",
            wells={well},
            reviewer=payload.reviewer,
            operation="late_growth_review_save",
        )
        source = artifact_path(config, "predictions", "latest_well_screening.csv")
        frame = pd.read_csv(source)
        selected = frame[frame["well"].astype(str).str.upper() == well]
        row = (
            selected.replace({np.nan: None}).iloc[0].to_dict()
            if not selected.empty
            else None
        )
        return {
            "status": "saved",
            "well": well,
            "timepoint": timepoint,
            "summary": summary,
            "screening": row,
            "gated_report_summary": (
                None
                if gated_summary is None
                else {key: value for key, value in gated_summary.items() if key != "wells"}
            ),
            "report": updated_report.get(well),
        }

    @app.post("/api/timepoint-cell-count-review")
    def timepoint_cell_count_review(
        payload: TimepointCellCountReviewPayload,
    ) -> dict[str, Any]:
        well = payload.well.upper()
        timepoint = payload.timepoint.upper()
        if timepoint not in {"T0", "T1", "T2"}:
            raise HTTPException(status_code=422, detail="Invalid early timepoint")
        if well not in set(images_manifest["well"].astype(str).str.upper()):
            raise HTTPException(status_code=404, detail="well unavailable")
        ensure_well_timepoint_cell_count_review_table(database)
        updated = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(database) as connection:
            if payload.cell_count is None:
                connection.execute(
                    "DELETE FROM well_timepoint_cell_count_reviews WHERE well = ? AND timepoint = ?",
                    (well, timepoint),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO well_timepoint_cell_count_reviews (
                      well, timepoint, cell_count, reviewer, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(well, timepoint) DO UPDATE SET
                      cell_count=excluded.cell_count,
                      reviewer=excluded.reviewer,
                      updated_at=excluded.updated_at
                    """,
                    (well, timepoint, int(payload.cell_count), payload.reviewer, updated),
                )
        build_well_screening(config, database, selected_wells={well})
        _, updated_report = refresh_gated_report()
        source = artifact_path(config, "predictions", "latest_well_screening.csv")
        frame = pd.read_csv(source)
        selected = frame[frame["well"].astype(str).str.upper() == well]
        screening_row = (
            selected.replace({np.nan: None}).iloc[0].to_dict()
            if not selected.empty
            else None
        )
        sync_catalog_after_review(
            "timepoint_cell_count_review",
            wells={well},
            reviewer=payload.reviewer,
            operation="timepoint_cell_count_review_save",
        )
        return {
            "status": "saved",
            "well": well,
            "timepoint": timepoint,
            "cell_count": payload.cell_count,
            "source": "automatic" if payload.cell_count is None else "human",
            "screening": screening_row,
            "report": updated_report.get(well),
        }


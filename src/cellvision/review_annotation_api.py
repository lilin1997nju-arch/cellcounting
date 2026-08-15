from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from .active_learning import build_review_queue
from .review_context import build_review_context, compute_well_registration
from .review_payloads import AnnotationPayload, LineageReviewPayload
from .review_storage import save_annotation, save_lineage_review


def register_annotation_routes(
    app: FastAPI,
    config: dict[str, Any],
    database: str | Path,
    images_manifest: pd.DataFrame,
    quick_review_service: SimpleNamespace,
    sync_catalog_after_review: Any,
) -> None:
    """Register annotation, review-context and lineage routes."""

    @app.get("/api/annotations")
    def annotations(limit: int = 100) -> list[dict[str, Any]]:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM annotations ORDER BY updated_at DESC LIMIT ?", (min(limit, 1000),)
            ).fetchall()
            return [dict(row) for row in rows]

    @app.get("/api/review-candidates")
    def review_candidates(
        limit: int = 500, include_completed: bool = False
    ) -> list[dict[str, Any]]:
        candidates = build_review_queue(
            config, include_completed=include_completed
        ).head(min(limit, 5000))
        return candidates.to_dict(orient="records")

    @app.get("/api/review-context")
    def review_context(
        well: str,
        x: float,
        y: float,
        search_size: int = 1024,
        view_mode: str = "densest",
        center_tp: str | None = None,
        center_x: float | None = None,
        center_y: float | None = None,
        target_id: str | None = None,
    ) -> dict[str, Any]:
        if view_mode not in {"densest", "lineage"}:
            raise HTTPException(status_code=422, detail="Invalid view mode")
        try:
            centers: dict[str, tuple[float, float]] = {}
            representative: dict[str, Any] = {}
            registration = compute_well_registration(
                config, images_manifest, well
            )
            if view_mode == "densest" and target_id:
                for timepoint in ("T1", "T2"):
                    dense = quick_review_service.lineage_representative_view(
                        well,
                        timepoint,
                        search_size,
                        target_id,
                        registration,
                    )
                    if dense:
                        centers[timepoint] = (
                            dense["center_x_px"],
                            dense["center_y_px"],
                        )
                        representative[timepoint] = dense
            if (
                center_tp in {"T0", "T1", "T2"}
                and center_x is not None
                and center_y is not None
            ):
                centers[center_tp] = (center_x, center_y)
            context = build_review_context(
                config,
                images_manifest,
                well,
                x,
                y,
                search_size=search_size,
                timepoint_centers=centers,
            )
            context["view_mode"] = view_mode
            context["representative_views"] = representative
            return quick_review_service.add_model_candidates(context, well)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


    @app.post("/api/annotations")
    def annotations_create(payload: AnnotationPayload) -> dict[str, Any]:
        try:
            annotation_id = save_annotation(database, payload.model_dump())
            sync_catalog_after_review(
                "annotation_review",
                wells={payload.well.upper()},
                reviewer=payload.reviewer,
            )
            return {"status": "saved", "annotation_id": annotation_id}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/lineage-reviews")
    def lineage_reviews(limit: int = 100) -> list[dict[str, Any]]:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM lineage_reviews ORDER BY updated_at DESC LIMIT ?",
                (min(limit, 1000),),
            ).fetchall()
            return [dict(row) for row in rows]

    @app.get("/api/link-reviews")
    def link_reviews(limit: int = 100) -> list[dict[str, Any]]:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM link_reviews ORDER BY updated_at DESC LIMIT ?",
                (min(limit, 1000),),
            ).fetchall()
            return [dict(row) for row in rows]

    @app.post("/api/lineage-reviews")
    def lineage_reviews_create(payload: LineageReviewPayload) -> dict[str, Any]:
        try:
            review_id = save_lineage_review(database, payload.model_dump())
            sync_catalog_after_review(
                "lineage_review",
                wells={payload.well.upper()},
                reviewer=payload.reviewer,
            )
            return {"status": "saved", "review_id": review_id}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

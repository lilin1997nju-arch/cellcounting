from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException

from .config import artifact_path
from .multiplicity import (
    ensure_multiplicity_table,
    multiplicity_queue,
    multiplicity_stats,
    save_categorized_review_labels,
)
from .review_payloads import MultiplicityLabelsPayload, TeachingLabelsPayload
from .runtime import production_mode_enabled
from .teaching import (
    save_teaching_labels,
    teaching_queue,
    teaching_stats,
    train_teaching_classifier,
)


def register_training_routes(
    app: FastAPI,
    config: dict[str, Any],
    database: str | Path,
    sync_catalog_after_review: Callable[..., None],
) -> None:
    """Register teaching and multiplicity training/review routes."""

    @app.get("/api/teach-candidates")
    def teach_candidates(
        mode: str = "seed",
        limit: int = 30,
        anchor_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if mode not in {"seed", "uncertain", "hard_negative", "similar"}:
            raise HTTPException(status_code=422, detail="Invalid teaching mode")
        try:
            return teaching_queue(
                config, database, mode, limit, anchor_id=anchor_id
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/teach-stats")
    def teach_stats() -> dict[str, Any]:
        stats = teaching_stats(database)
        model_report = artifact_path(
            config, "models", "teaching_classifier.json"
        )
        stats["model"] = (
            json.loads(model_report.read_text(encoding="utf-8"))
            if model_report.exists()
            else {"status": "not_trained"}
        )
        return stats

    @app.post("/api/teach-labels")
    def teach_labels_create(
        payload: TeachingLabelsPayload,
    ) -> dict[str, Any]:
        try:
            saved = save_teaching_labels(
                database,
                [item.model_dump() for item in payload.items],
                payload.reviewer,
            )
            sync_catalog_after_review("teaching_review", reviewer=payload.reviewer)
            return {"status": "saved", "saved": saved}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/teach-train")
    def teach_train() -> dict[str, Any]:
        if production_mode_enabled():
            raise HTTPException(
                status_code=403,
                detail="Model training is disabled in production compute-only mode",
            )
        try:
            return train_teaching_classifier(config, database)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/multiplicity-candidates")
    def multiplicity_candidates(
        mode: str = "likely_doublet",
        limit: int = 30,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        if mode not in {"likely_doublet", "uncertain", "diverse"}:
            raise HTTPException(status_code=422, detail="Invalid queue mode")
        try:
            return multiplicity_queue(
                config, database, mode, limit, category=category
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/multiplicity-stats")
    def multiplicity_label_stats() -> dict[str, Any]:
        return multiplicity_stats(database)

    @app.post("/api/multiplicity-labels")
    def multiplicity_labels_create(
        payload: MultiplicityLabelsPayload,
    ) -> dict[str, Any]:
        try:
            saved = save_categorized_review_labels(
                database,
                [item.model_dump() for item in payload.items],
                payload.reviewer,
            )
            sync_catalog_after_review("multiplicity_review", reviewer=payload.reviewer)
            return {"status": "saved", "saved": saved}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/multiplicity-labels/{candidate_id}")
    def multiplicity_label_delete(candidate_id: str) -> dict[str, Any]:
        """Remove the latest quick-training label so the reviewer can undo."""

        ensure_multiplicity_table(database)
        with sqlite3.connect(database) as connection:
            cursor = connection.execute(
                "DELETE FROM multiplicity_labels WHERE candidate_id = ?",
                (candidate_id,),
            )
            deleted = int(cursor.rowcount or 0)
        return {"status": "deleted", "deleted": deleted}


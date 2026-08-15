from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException

from .review_payloads import MaskComparisonPayload, MaskReviewSavePayload
from .v2_mask_review import (
    create_model_comparison_round,
    list_mask_review_rounds,
    mask_comparison_options,
    mask_review_candidate,
    mask_review_candidates,
    mask_review_summary,
    save_mask_review,
)


def register_mask_routes(
    app: FastAPI,
    config: dict[str, Any],
    database: str | Path,
    sync_catalog_after_review: Callable[..., None],
) -> None:
    """Register the V2 mask-review routes."""

    @app.get("/api/mask-review-rounds")
    def mask_review_rounds() -> list[dict[str, Any]]:
        return list_mask_review_rounds(config, database)

    @app.get("/api/mask-comparison-options")
    def mask_comparison_options_api() -> dict[str, Any]:
        try:
            return mask_comparison_options(config, database)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/mask-comparison-round")
    def mask_comparison_round_api(payload: MaskComparisonPayload) -> dict[str, Any]:
        try:
            return create_model_comparison_round(
                config,
                old_checkpoint=payload.old_checkpoint,
                new_checkpoint=payload.new_checkpoint,
                source_configs=payload.source_configs or None,
                round_id=payload.round_id,
            )
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/mask-review-summary")
    def mask_review_summary_api(round_id: str) -> dict[str, Any]:
        try:
            return mask_review_summary(config, database, round_id)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/mask-review-candidates")
    def mask_review_candidates_api(
        round_id: str, status: str = "pending", limit: int = 500
    ) -> list[dict[str, Any]]:
        try:
            return mask_review_candidates(
                config, database, round_id, status=status, limit=min(int(limit), 5000)
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/mask-review-candidate")
    def mask_review_candidate_api(
        round_id: str, candidate_id: str
    ) -> dict[str, Any]:
        try:
            item = mask_review_candidate(config, database, round_id, candidate_id)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="mask review candidate not found")
        return item

    @app.post("/api/mask-review-save")
    def mask_review_save(payload: MaskReviewSavePayload) -> dict[str, Any]:
        try:
            result = save_mask_review(
                config,
                database,
                round_id=payload.round_id,
                candidate_id=payload.candidate_id,
                decision=payload.decision,
                reviewed_mask_rle=payload.reviewed_mask_rle,
                reviewer=payload.reviewer,
                notes=payload.notes,
            )
            sync_catalog_after_review("mask_review", reviewer=payload.reviewer)
            return result
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


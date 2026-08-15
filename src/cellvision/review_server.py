from __future__ import annotations

import json
import html
import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
from pydantic import BaseModel, Field
from scipy import ndimage, signal
from skimage import measure

from .active_learning import build_review_queue
from .config import artifact_path
from .project_catalog import ProjectCatalog
from .dense_candidates import augment_candidates_with_dense_raw_proposals
from .hierarchy import suppress_nested_single_candidates
from .review_context import build_review_context, compute_well_registration
from .review_summary import (
    SUMMARY_VERSION,
    latest_prediction_path,
    read_summary,
    summary_path,
    summary_signature,
    write_summary,
)
from .multiplicity import (
    ensure_integrated_review_table,
    ensure_multiplicity_table,
    generate_integrated_training_round,
    integrated_review_queue,
    integrated_review_stats,
    multiplicity_queue,
    multiplicity_stats,
    save_integrated_reviews,
    save_categorized_review_labels,
    train_multiplicity_classifier,
)
from .teaching import (
    auto_annotation_queue,
    auto_annotation_stats,
    ensure_teaching_features,
    generate_auto_annotation_round,
    save_teaching_labels,
    save_auto_annotation_reviews,
    teaching_queue,
    teaching_stats,
    train_teaching_classifier,
)
from .well_screening import build_well_screening, save_late_growth_review
from .review_annotation_api import register_annotation_routes
from .review_image_api import register_image_routes
from .review_mask_api import register_mask_routes
from .review_screening_api import register_screening_routes
from .review_training_api import register_round_routes, register_training_routes
from .review_image_cache import patch_cache_path, render_patch, render_review_image, report_image_cache_path, review_image_cache_path
from .gated_screening import build_gated_plate_report
from .v2_mask_review import (
    create_model_comparison_round,
    create_mask_review_round,
    list_mask_review_rounds,
    mask_comparison_options,
    mask_review_candidate,
    mask_review_candidates,
    mask_review_summary,
    save_mask_review,
)
from .runtime import production_mode_enabled


from .review_quick_review import build_quick_review_service, register_quick_review_routes
from .review_storage import (
    SCHEMA,
    _capture_quick_review_undo_snapshot,
    _delete_rows_by_values,
    _ensure_schema_columns,
    _fetch_rows_by_values,
    _restore_quick_review_undo_snapshot,
    _restore_rows,
    _summary_json_safe,
    initialize_database,
    save_annotation,
    save_lineage_review,
)

from .review_helpers import (
    FINAL_REASON_TEXTS,
    _boolean_series,
    _decision_bool,
    _decision_number,
    _decision_text,
    _final_review_label,
    _growth_region_contours,
    _review_images_manifest,
    _visible_v2_review_instances,
    _with_final_decisions,
)

from .review_payloads import (
    AnnotationPayload,
    AutoReviewItem,
    AutoReviewsPayload,
    IntegratedReviewItem,
    IntegratedReviewsPayload,
    LateGrowthReviewPayload,
    LineageReviewPayload,
    LinkReviewPayload,
    MaskComparisonPayload,
    MaskReviewSavePayload,
    MultiplicityLabelItem,
    MultiplicityLabelsPayload,
    PointSelection,
    QuickReviewObjectItem,
    QuickReviewUndoPayload,
    QuickReviewWellPayload,
    TeachingLabelItem,
    TeachingLabelsPayload,
    TimepointSelection,
    V3TrackReviewItem,
    WellScreeningReviewPayload,
)


def create_app(
    config: dict[str, Any],
    *,
    project_back_url: str | None = None,
    review_base_url: str | None = None,
) -> FastAPI:
    catalog_context = config.get("_catalog_context")
    catalog: ProjectCatalog | None = None
    catalog_manifest_path = ""
    catalog_plate_id = ""
    if isinstance(catalog_context, dict):
        catalog_manifest_path = str(catalog_context.get("manifest_path") or "")
        project_id = str(catalog_context.get("project_id") or "")
        plate_slug = str(catalog_context.get("plate_slug") or "")
        catalog_plate_id = f"{project_id}:{plate_slug}" if project_id and plate_slug else ""
        catalog_path = catalog_context.get("catalog_path")
        if catalog_path and catalog_manifest_path and catalog_plate_id:
            try:
                catalog = ProjectCatalog(str(catalog_path))
            except (OSError, sqlite3.Error):  # catalog sync must not block review startup
                catalog = None
    database_path = artifact_path(config, "annotations", "annotations.db")
    images_manifest_path = artifact_path(config, "manifests", "images.csv")
    database = initialize_database(database_path)
    images_manifest = _review_images_manifest(
        config,
        pd.read_csv(images_manifest_path),
    )
    review_ui_dir = Path(__file__).resolve().parents[2] / "review-ui"
    teaching_html_path = (
        Path(__file__).resolve().parents[2] / "review-ui" / "teach.html"
    )
    auto_review_html_path = (
        Path(__file__).resolve().parents[2] / "review-ui" / "auto-review.html"
    )
    multiplicity_html_path = (
        Path(__file__).resolve().parents[2]
        / "review-ui"
        / "doublet-teach.html"
    )
    single_doublet_review_html_path = (
        Path(__file__).resolve().parents[2]
        / "review-ui"
        / "single-doublet-review.html"
    )
    integrated_review_html_path = (
        Path(__file__).resolve().parents[2]
        / "review-ui"
        / "integrated-review.html"
    )
    mask_review_html_path = (
        Path(__file__).resolve().parents[2] / "review-ui" / "mask-review.html"
    )
    app = FastAPI(title="Cell Vision Local Review")
    escaped_project_back_url = html.escape(project_back_url or "", quote=True)
    escaped_review_base_url = html.escape((review_base_url or "").rstrip("/"), quote=True)

    def review_page(path: Path) -> str:
        """Rewrite a plate page so relative assets work on a nested route."""

        page = path.read_text(encoding="utf-8")
        if escaped_review_base_url:
            page = page.replace(
                "</head>",
                f'<meta name="review-base-url" content="{escaped_review_base_url}">\n'
                f'<base href="{escaped_review_base_url}/">\n</head>',
                1,
            )
            page = page.replace(
                'meta name="review-base-url" content=""',
                f'meta name="review-base-url" content="{escaped_review_base_url}"',
                1,
            )
            page = page.replace('href="/assets/', 'href="assets/').replace(
                'src="/assets/', 'src="assets/'
            )
            for route in (
                "auto-review",
                "teach",
                "doublet-teach",
                "single-doublet-review",
                "integrated-review",
                "mask-review",
            ):
                page = page.replace(f'href="/{route}', f'href="{route}')
            if path == mask_review_html_path:
                page = page.replace(
                    'meta name="mask-review-base" content="./"',
                    f'meta name="mask-review-base" content="{escaped_review_base_url}"',
                    1,
                )
            page = page.replace('href="/"', f'href="{escaped_project_back_url}"')
        return page

    def auto_review_page() -> str:
        page = review_page(auto_review_html_path)
        if escaped_project_back_url:
            page = page.replace(
                '<meta name="project-back-url" content="">',
                f'<meta name="project-back-url" content="{escaped_project_back_url}">',
                1,
            )
        return page

    def scoped_page(path: Path) -> str:
        page = review_page(path)
        if escaped_project_back_url:
            page = page.replace(
                '<meta name="project-back-url" content="">',
                f'<meta name="project-back-url" content="{escaped_project_back_url}">',
                1,
            )
        return page
    prediction_cache: dict[str, Any] = {
        "mtime": None,
        "source": None,
        "frame": pd.DataFrame(),
    }
    proposal_cache: dict[str, Any] = {
        "mtime": None,
        "frame": pd.DataFrame(),
    }
    completion_review_cache: dict[str, Any] = {
        "mtime": None,
        "frame": pd.DataFrame(),
    }
    quick_frame_cache: dict[str, Any] = {
        "key": None,
        "frame": pd.DataFrame(),
    }
    quick_summary_file = summary_path(config["paths"]["artifact_root"])
    quick_summary_cache: dict[str, Any] = {
        "signature": None,
        "payload": None,
    }
    quick_summary_lock = RLock()

    def sync_catalog_after_review(
        source: str,
        *,
        wells: set[str] | list[str] | tuple[str, ...] | None = None,
        reviewer: str = "",
        action_id: str = "",
        operation: str = "review_save",
    ) -> None:
        """Best-effort projection update after an authoritative review save."""

        if catalog is None or not catalog_manifest_path or not catalog_plate_id:
            return
        try:
            try:
                quick_review_service.quick_review_summary(force=True)
            except Exception:
                # Some specialized review routes do not have a quick-review
                # candidate table; their primary DB/report save still syncs.
                pass
            catalog.sync_review_update(
                catalog_manifest_path,
                catalog_plate_id,
                wells=wells,
                source=source,
                reviewer=reviewer,
                action_id=action_id,
                operation=operation,
            )
        except (OSError, sqlite3.Error, ValueError, TypeError):
            # The per-plate annotations DB remains the source of truth.  A
            # transient catalog lock or malformed optional report must never
            # turn a successful review save into a failed user action.
            return

    def gated_settings() -> dict[str, Any]:
        return config.get("gated_report", {})

    def gated_report_path() -> Path | None:
        settings = gated_settings()
        output = settings.get("output_dir")
        return Path(str(output)) / "plate_overview.csv" if output else None

    def gated_lookup() -> dict[str, dict[str, Any]]:
        source = gated_report_path()
        if source is None or not source.exists():
            return {}
        frame = pd.read_csv(source, low_memory=False)
        return {
            str(row.well).upper(): row._asdict()
            for row in frame.itertuples(index=False)
        }

    ui_status_aliases = {
        "single_growth_unconfirmed": "single_not_divided",
        "single_not_divided": "single_not_divided",
        "missing_t0_or_late_object": "t0_missing_late_cells",
        "t0_missing_late_cells": "t0_missing_late_cells",
        "ambiguous": "t0_missing_late_cells",
        "no_cell": "no_cell_growth",
    }
    ui_status_labels = {
        "single_active": "单细胞有活性",
        "single_not_divided": "T0-T2未分裂",
        "multi_origin": "多细胞来源",
        "no_cell_growth": "无明显生长",
        "t0_missing_late_cells": "T0缺失但后期出现细胞",
        "positive_control": "阳性对照",
    }

    def ui_screening_status(gated: dict[str, Any], fallback: str) -> str:
        category = str(gated.get("final_category", ""))
        reason = str(gated.get("undetermined_reason", ""))
        mapped = {
            "single_cell_origin": "single_active",
            "multi_cell_origin": "multi_origin",
            "no_obvious_growth": "no_cell_growth",
            "undetermined": (
                "t0_missing_late_cells"
                if reason == "t0_missing_later_detected"
                else "single_not_divided"
                if reason == "t0_t2_no_division"
                else "t0_missing_late_cells"
            ),
            "positive_control": "positive_control",
        }.get(category)
        return mapped or ui_status_aliases.get(str(fallback), str(fallback))

    def ui_screening_status_label(status: str) -> str:
        normalized = ui_status_aliases.get(str(status), str(status))
        return ui_status_labels.get(normalized, "T0缺失但后期出现细胞")

    def refresh_gated_report() -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
        settings = gated_settings()
        endpoint_csv = settings.get("endpoint_csv") or settings.get("day14_csv")
        if not endpoint_csv or not settings.get("group_id") or not settings.get("output_dir"):
            return None, gated_lookup()
        endpoint_timepoint = str(settings.get("endpoint_timepoint", "T4")).upper()
        endpoint_day_label = str(settings.get("endpoint_day_label", "Day14"))
        with sqlite3.connect(database) as connection:
            try:
                late_rows = connection.execute(
                    """
                    SELECT well, decision
                    FROM late_growth_reviews
                    WHERE timepoint = ?
                    """,
                    (endpoint_timepoint,),
                ).fetchall()
            except sqlite3.OperationalError:
                late_rows = []
        day14_overrides = {
            str(well).upper(): str(decision)
            for well, decision in late_rows
        }
        payload = build_gated_plate_report(
            endpoint_csv,
            settings["group_id"],
            settings["output_dir"],
            early_screening_csv=artifact_path(
                config, "predictions", "latest_well_screening.csv"
            ),
            sessions_csv=settings.get("sessions_csv"),
            locate_day7=False,
            day14_growth_overrides=day14_overrides,
            endpoint_day_label=endpoint_day_label,
        )
        return payload, gated_lookup()

    quick_review_service = build_quick_review_service(
        config=config,
        database=database,
        images_manifest=images_manifest,
        prediction_cache=prediction_cache,
        proposal_cache=proposal_cache,
        completion_review_cache=completion_review_cache,
        quick_frame_cache=quick_frame_cache,
        quick_summary_cache=quick_summary_cache,
        quick_summary_lock=quick_summary_lock,
        quick_summary_file=quick_summary_file,
        gated_lookup=gated_lookup,
        gated_report_path=gated_report_path,
        ui_screening_status=ui_screening_status,
        ui_screening_status_label=ui_screening_status_label,
        ui_status_aliases=ui_status_aliases,
    )


    app.mount(
        "/assets",
        StaticFiles(directory=review_ui_dir),
        name="review-assets",
    )

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        return auto_review_page()

    @app.get("/teach", response_class=HTMLResponse)
    def teach() -> str:
        return scoped_page(teaching_html_path)

    @app.get("/auto-review", response_class=HTMLResponse)
    def auto_review() -> str:
        return auto_review_page()

    @app.get("/doublet-teach", response_class=HTMLResponse)
    def doublet_teach() -> str:
        return scoped_page(multiplicity_html_path)

    @app.get("/single-doublet-review", response_class=HTMLResponse)
    def single_doublet_review() -> str:
        return scoped_page(single_doublet_review_html_path)

    @app.get("/integrated-review", response_class=HTMLResponse)
    def integrated_review() -> str:
        return scoped_page(integrated_review_html_path)

    @app.get("/mask-review", response_class=HTMLResponse)
    def mask_review() -> str:
        return scoped_page(mask_review_html_path)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "scope": "configured host"}

    @app.get("/api/ready")
    def ready() -> dict[str, Any]:
        """Readiness probe that fails before a plate manifest is available."""

        if not images_manifest_path.exists():
            raise HTTPException(status_code=503, detail="images manifest is missing")
        return {
            "status": "ready",
            "database": str(database_path),
            "image_count": int(len(images_manifest)),
        }

    register_mask_routes(app, config, database, sync_catalog_after_review)

    register_annotation_routes(app, config, database, images_manifest, quick_review_service, sync_catalog_after_review)

    register_training_routes(app, config, database, sync_catalog_after_review)

    register_round_routes(app, config, database, sync_catalog_after_review)

    register_quick_review_routes(
        app, config, database, images_manifest, quick_review_service,
        gated_lookup, sync_catalog_after_review, refresh_gated_report,
    )

    register_screening_routes(app, config, database, images_manifest, gated_lookup, ui_screening_status, ui_screening_status_label, ui_status_aliases, refresh_gated_report, sync_catalog_after_review)

    register_image_routes(app, config, images_manifest, database)

    return app


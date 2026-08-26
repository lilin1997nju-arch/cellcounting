"""Project-level review hub and local import/task queue.

The existing per-plate review service remains the source of truth for object
labels.  This parent application aggregates gated plate reports, provides a
folder/session parser for new jobs, and lazily mounts each completed plate
under ``/plates/<slug>/`` so a large project does not load every board at
startup.
"""

from __future__ import annotations

import json
import asyncio
import html
import io
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from tempfile import NamedTemporaryFile
from threading import RLock, Thread
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from starlette.background import BackgroundTask
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance
from pydantic import BaseModel, Field, field_validator

from .config import PROJECT_ROOT, artifact_path, load_config
from .project_catalog import ProjectCatalog, catalog_path_for_manifest
from .multiplicity import (
    ensure_multiplicity_table,
    multiplicity_queue,
    multiplicity_stats,
    save_categorized_review_labels,
)
from .offline_review import (
    export_offline_review_results,
    import_offline_review_results,
)
from .review_data_package import (
    DATA_PACKAGE_FORMAT,
    prepare_review_data_package,
    project_export_signature,
)
from .review_server import _visible_v2_review_instances, create_app, initialize_database
from .review_summary import (
    latest_prediction_path,
    read_cached_summary,
    read_summary,
    summary_path,
    summary_signature,
)
from .session_index import parse_sessions_index, summarize_session_groups
from .task_queue import TaskQueueStore, task_id
from .runtime import production_mode_enabled, remove_development_routes

if not production_mode_enabled():
    from .v2_mask_review import (
        create_model_comparison_round,
        list_mask_review_rounds,
        mask_comparison_options,
        mask_review_candidate,
        mask_review_candidates,
        mask_review_summary,
        save_mask_review,
    )


def _production_instance_id() -> str:
    """Return the stable ID of the installed Workspace, when configured."""

    configured = os.environ.get("CELLVISION_INSTANCE_ID", "").strip()
    if configured:
        return configured
    artifact_root = os.environ.get("CELLVISION_ARTIFACT_ROOT", "").strip()
    if not artifact_root:
        return ""
    try:
        return (Path(artifact_root) / ".cellvision-instance-id").read_text(
            encoding="utf-8-sig"
        ).strip()
    except OSError:
        return ""


def _safe(value: Any) -> Any:
    """Convert pandas/numpy values to JSON-safe primitives."""

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except (TypeError, ValueError):
            pass
    if pd.isna(value):
        return None
    return value


def _resolve(path: str | Path) -> Path:
    value = Path(os.path.expandvars(str(path))).expanduser()
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in str(value)).strip("-")


def _report_for_plate(plate: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    candidates = [
        plate.get("report_json"),
        Path(plate.get("artifact_root", "")) / "plate_overview.json"
        if plate.get("artifact_root")
        else None,
        Path(plate.get("gated_output_dir", "")) / "plate_overview.json"
        if plate.get("gated_output_dir")
        else None,
    ]
    for item in candidates:
        if not item:
            continue
        path = _resolve(item)
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8")), path
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


_REVIEWABLE_LABELS = {
    "single",
    "touching_doublet",
    "cluster_3plus",
    "debris",
    "uncertain",
}
_REVIEW_PROGRESS_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}


def _empty_review_progress(*, available: bool) -> dict[str, Any]:
    return {
        "review_data_available": available,
        "reviewable_well_count": 0,
        "reviewed_well_count": 0,
        "reviewable_object_count": 0,
        "reviewed_object_count": 0,
        "review_complete": bool(available),
    }


def _latest_prediction_path(config: dict[str, Any]) -> Path:
    prediction_root = Path(config["paths"]["artifact_root"]) / "predictions"
    return latest_prediction_path(config["paths"]["artifact_root"]) or (
        prediction_root / "latest_integrated_predictions.csv"
    )


def _plate_artifact_root(plate: dict[str, Any]) -> Path | None:
    value = plate.get("artifact_root")
    return _resolve(value) if value else None


def _review_summary_for_plate(plate: dict[str, Any]) -> dict[str, Any] | None:
    """Read a current persisted summary without loading prediction rows."""

    root = _plate_artifact_root(plate)
    if root is None:
        return None
    report_root = plate.get("gated_output_dir")
    report_path = (
        _resolve(report_root) / "plate_overview.csv"
        if report_root
        else root / "gated" / "plate_overview.csv"
    )
    signature = summary_signature(root, report_path=report_path)
    persisted_path = summary_path(root)
    fresh = read_summary(persisted_path, signature)
    if fresh is not None:
        return fresh
    stale = read_cached_summary(persisted_path)
    if stale is None:
        return None
    return {**stale, "stale": True}


def _review_progress_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "ready":
        return _empty_review_progress(available=False)
    rows = summary.get("wells") if isinstance(summary.get("wells"), list) else []
    well_count = int(summary.get("well_count", len(rows)) or 0)
    completed = int(
        summary.get(
            "completed_well_count",
            sum(bool(row.get("completed")) for row in rows if isinstance(row, dict)),
        )
        or 0
    )
    return {
        "review_data_available": True,
        "reviewable_well_count": well_count,
        "reviewed_well_count": completed,
        "reviewable_object_count": int(summary.get("object_count", 0) or 0),
        "reviewed_object_count": int(summary.get("reviewed_object_count", 0) or 0),
        "review_complete": not rows or completed >= well_count,
    }


def _review_progress_for_plate(plate: dict[str, Any]) -> dict[str, Any]:
    """Read lightweight well-review progress without starting a plate app.

    The project hub must not construct every review application just to render
    a table.  Prefer the persisted per-plate summary and keep the old direct
    prediction/database scan as a compatibility fallback for legacy plates.
    """

    persisted = _review_summary_for_plate(plate)
    if persisted is not None:
        return _review_progress_from_summary(persisted)

    config_value = plate.get("config")
    if not config_value:
        return _empty_review_progress(available=False)
    config_path = _resolve(config_value)
    if not config_path.exists():
        return _empty_review_progress(available=False)
    try:
        config = load_config(config_path)
    except (OSError, ValueError, TypeError):
        return _empty_review_progress(available=False)

    prediction_path = _latest_prediction_path(config)
    database_path = Path(config["paths"]["artifact_root"]) / "annotations" / "annotations.db"
    signature = (
        str(config_path),
        prediction_path.stat().st_mtime_ns if prediction_path.exists() else None,
        database_path.stat().st_mtime_ns if database_path.exists() else None,
    )
    cache_key = str(config_path)
    cached = _REVIEW_PROGRESS_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return dict(cached[1])
    if not prediction_path.exists() or not database_path.exists():
        result = _empty_review_progress(available=False)
        _REVIEW_PROGRESS_CACHE[cache_key] = (signature, result)
        return dict(result)

    try:
        predictions = pd.read_csv(prediction_path, low_memory=False)
        if predictions.empty or "integrated_label" not in predictions.columns:
            result = _empty_review_progress(available=True)
            _REVIEW_PROGRESS_CACHE[cache_key] = (signature, result)
            return dict(result)

        reviewable = predictions[
            predictions["integrated_label"].isin(_REVIEWABLE_LABELS)
            & (predictions["well"].astype(str).str.upper() != "A1")
        ].copy()
        if "v2_instance_id" in reviewable.columns:
            try:
                reviewable = _visible_v2_review_instances(reviewable)
            except (KeyError, TypeError, ValueError):
                # Keep the hub usable with older/incomplete prediction tables.
                pass

        round_id = (
            str(reviewable.iloc[0]["integrated_round_id"])
            if not reviewable.empty and "integrated_round_id" in reviewable.columns
            else ""
        )
        with sqlite3.connect(database_path) as connection:
            try:
                # A later inference round changes integrated_round_id without
                # migrating human decisions; keep the newest decision per
                # candidate_id so reviewed progress survives round-id changes.
                reviews = pd.read_sql_query(
                    """
                    SELECT candidate_id, reviewed_label, updated_at
                    FROM integrated_training_reviews
                    ORDER BY updated_at, integrated_review_id
                    """,
                    connection,
                )
                if not reviews.empty:
                    reviews = reviews.drop_duplicates("candidate_id", keep="last")
            except (sqlite3.OperationalError, pd.errors.DatabaseError):
                reviews = pd.DataFrame()
            try:
                manual = pd.read_sql_query(
                    """
                    SELECT candidate_id, well, reviewed_label
                    FROM quick_missed_objects
                    WHERE round_id = ?
                    """,
                    connection,
                    params=(round_id,),
                )
            except (sqlite3.OperationalError, pd.errors.DatabaseError):
                manual = pd.DataFrame()

        reviewed_ids: set[str] = set()
        if not reviews.empty:
            latest = reviews.drop_duplicates("candidate_id", keep="last")
            reviewed_ids = set(
                latest.loc[latest["reviewed_label"].notna(), "candidate_id"]
                .astype(str)
            )
        reviewable["is_reviewed"] = reviewable["candidate_id"].astype(str).isin(reviewed_ids)

        if not manual.empty:
            existing_ids = set(reviewable["candidate_id"].astype(str))
            manual = manual[~manual["candidate_id"].astype(str).isin(existing_ids)].copy()
            if not manual.empty:
                manual["is_reviewed"] = manual["reviewed_label"].notna()
                manual = manual.rename(columns={"screen_well": "well"})
                reviewable = pd.concat(
                    [reviewable[["well", "is_reviewed"]], manual[["well", "is_reviewed"]]],
                    ignore_index=True,
                )
        else:
            reviewable = reviewable[["well", "is_reviewed"]]

        if reviewable.empty:
            result = _empty_review_progress(available=True)
        else:
            grouped = reviewable.groupby(reviewable["well"].astype(str).str.upper())["is_reviewed"]
            well_progress = grouped.agg(["size", "sum"])
            result = {
                "review_data_available": True,
                "reviewable_well_count": int(len(well_progress)),
                "reviewed_well_count": int((well_progress["size"] == well_progress["sum"]).sum()),
                "reviewable_object_count": int(len(reviewable)),
                "reviewed_object_count": int(reviewable["is_reviewed"].sum()),
                "review_complete": bool((well_progress["size"] == well_progress["sum"]).all()),
            }
    except (OSError, KeyError, ValueError, TypeError, pd.errors.ParserError, sqlite3.Error):
        result = _empty_review_progress(available=False)

    _REVIEW_PROGRESS_CACHE[cache_key] = (signature, result)
    return dict(result)


def _manual_verdict_counts_for_plate(plate: dict[str, Any]) -> dict[str, int]:
    """Count persisted well-level qualified/pending/excluded decisions."""

    counts = {"approved": 0, "pending": 0, "rejected": 0, "unclassified": 0}
    root = _plate_artifact_root(plate)
    if root is None:
        return counts
    screening_source = root / "predictions" / "latest_well_screening.csv"
    total_wells = 0
    try:
        screening = pd.read_csv(screening_source, usecols=lambda column: column == "well")
        total_wells = int(screening["well"].astype(str).str.upper().nunique())
    except (OSError, ValueError, pd.errors.ParserError):
        total_wells = 0
    database = root / "annotations" / "annotations.db"
    explicit: list[str] = []
    if database.exists():
        try:
            with sqlite3.connect(database) as connection:
                explicit = [
                    str(row[0]).lower()
                    for row in connection.execute(
                        "SELECT decision FROM well_screening_reviews"
                    ).fetchall()
                ]
        except sqlite3.Error:
            explicit = []
    for decision in explicit:
        if decision in {"approved", "pending", "rejected"}:
            counts[decision] += 1
    counts["unclassified"] = max(0, total_wells - sum(counts.values()))
    return counts


def _project_detection_dates(value: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return the first/last acquisition dates at day precision."""

    explicit_start = str(value.get("detection_start_date") or "").strip() or None
    explicit_end = str(value.get("detection_end_date") or "").strip() or None
    if explicit_start or explicit_end:
        return explicit_start or explicit_end, explicit_end or explicit_start

    candidates: list[Path] = []
    source = value.get("source_sessions_csv")
    if source:
        candidates.append(_resolve(source))
    for plate in value.get("plates", []):
        if not isinstance(plate, dict) or not plate.get("images_manifest"):
            continue
        candidates.append(_resolve(plate["images_manifest"]))
    for path in candidates:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in {
                "acquisition_date", "acquisition_datetime", "excluded"
            })
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if frame.empty:
            continue
        if "excluded" in frame.columns:
            excluded = frame["excluded"].astype(str).str.strip().str.lower().isin(
                {"1", "true", "yes", "y"}
            )
            frame = frame.loc[~excluded]
        column = "acquisition_date" if "acquisition_date" in frame.columns else "acquisition_datetime"
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
        if not parsed.empty:
            return parsed.min().date().isoformat(), parsed.max().date().isoformat()
    return None, None


def _plate_summary(plate: dict[str, Any]) -> dict[str, Any]:
    report, report_path = _report_for_plate(plate)
    counts: dict[str, int] = {}
    elapsed = None
    pipeline_path = _resolve(plate.get("pipeline_summary", "")) if plate.get("pipeline_summary") else None
    if report:
        counts = {
            str(key): int(value or 0)
            for key, value in (report.get("category_counts") or {}).items()
        }
    if pipeline_path and pipeline_path.exists():
        try:
            pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
            elapsed = pipeline.get("total_elapsed_seconds")
        except (OSError, json.JSONDecodeError):
            pass
    status = str(plate.get("status", "queued"))
    if report:
        status = "completed"
    review_progress = _review_progress_for_plate(plate)
    return {
        "slug": str(plate.get("slug", "")),
        "group_id": str(plate.get("group_id", plate.get("board_id", ""))),
        "board_id": str(plate.get("board_id", "")),
        "status": status,
        "category_counts": counts,
        "well_count": int(report.get("well_count", 0)) if report else 0,
        "positive_well_count": int(report.get("day14_positive_sample_wells", 0)) if report else 0,
        "skipped_well_count": int(report.get("day14_skipped_sample_wells", 0)) if report else 0,
        "endpoint_day_label": report.get("endpoint_day_label", plate.get("endpoint_day_label", "Day14")) if report else plate.get("endpoint_day_label", "Day14"),
        "elapsed_seconds": elapsed,
        "report_path": str(report_path) if report_path else None,
        "config_path": str(_resolve(plate.get("config", ""))) if plate.get("config") else None,
        "manual_verdict_counts": _manual_verdict_counts_for_plate(plate),
        **review_progress,
    }


_EXPORT_TIMEPOINTS = ("T0", "T1", "T2")
_EXPORT_COUNT_LABELS = (
    ("single", "单细胞个数"),
    ("touching_doublet", "双细胞个数"),
    ("cluster_3plus", "多细胞个数"),
)
_EXPORT_CATEGORY_LABELS = {
    "positive_control": "阳性对照",
    "no_obvious_growth": "无明显生长",
    "single_cell_origin": "单细胞来源",
    "multi_cell_origin": "多细胞来源",
    "undetermined": "待确定",
}


def _export_number(value: Any, default: int = 0) -> int:
    """Return a finite, non-negative integer for a cell-count field."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not pd.notna(number):
        return default
    return max(0, int(round(number)))


def _export_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none"} else text


def _export_well_sort_key(well: Any) -> tuple[int, int, str]:
    value = _export_text(well).upper()
    match = re.match(r"^([A-Z]+)(\d+)$", value)
    if match:
        return (ord(match.group(1)[0]) - ord("A"), int(match.group(2)), value)
    return (99, 99, value)


def _export_read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _export_reference_path(value: Any, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _export_plate_csv_paths(plate: dict[str, Any], manifest_path: Path) -> tuple[Path | None, Path | None]:
    base = manifest_path.parent
    artifact_root = _export_reference_path(plate.get("artifact_root"), base)
    gated_root = _export_reference_path(plate.get("gated_output_dir"), base)
    report = _export_reference_path(plate.get("report_json"), base)
    if report is None and gated_root is not None:
        report = gated_root / "plate_overview.json"
    if report is None and artifact_root is not None:
        report = artifact_root / "gated" / "plate_overview.json"
    report_csv = report.with_suffix(".csv") if report is not None else None
    if report_csv is None or not report_csv.exists():
        report_csv = gated_root / "plate_overview.csv" if gated_root is not None else None
    screening_csv = (
        artifact_root / "predictions" / "latest_well_screening.csv"
        if artifact_root is not None
        else None
    )
    return report_csv, screening_csv


def _export_count_columns(objects: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Count final per-object multiplicity labels by timepoint.

    The screening table stores only weighted cell units.  The object table has
    the final ``screen_label``/``predicted_multiplicity`` values needed to
    distinguish one single, one doublet and one 3+ cluster.
    """

    counts: dict[tuple[str, str], int] = {}
    if objects.empty or "well" not in objects.columns or "timepoint" not in objects.columns:
        return counts
    frame = objects.copy()
    label_column = "screen_label" if "screen_label" in frame.columns else "integrated_label"
    if label_column not in frame.columns:
        return counts
    frame["_export_label"] = frame[label_column].map(_export_text).str.lower()
    frame["_export_timepoint"] = frame["timepoint"].map(_export_text).str.upper()
    frame["_export_well"] = frame["well"].map(_export_text).str.upper()
    frame = frame[frame["_export_timepoint"].isin(_EXPORT_TIMEPOINTS)]
    frame = frame[frame["_export_label"].isin({key for key, _ in _EXPORT_COUNT_LABELS})]
    if frame.empty:
        return counts
    grouped = frame.groupby(["_export_well", "_export_timepoint", "_export_label"], sort=False).size()
    for (well, timepoint, label), value in grouped.items():
        counts[(str(well), f"{timepoint}:{label}")] = int(value)
    return counts


def _export_debris_count_columns(objects: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Count final debris labels by well and timepoint for the Excel export."""

    counts: dict[tuple[str, str], int] = {}
    if objects.empty or not {"well", "timepoint"}.issubset(objects.columns):
        return counts
    label_column = "screen_label" if "screen_label" in objects.columns else "integrated_label"
    if label_column not in objects.columns:
        return counts
    frame = objects.copy()
    frame["_export_label"] = frame[label_column].map(_export_text).str.lower()
    frame["_export_timepoint"] = frame["timepoint"].map(_export_text).str.upper()
    frame["_export_well"] = frame["well"].map(_export_text).str.upper()
    frame = frame[
        frame["_export_timepoint"].eq("T2")
        & frame["_export_label"].eq("debris")
    ]
    if not frame.empty:
        for well, value in frame.groupby("_export_well", sort=False).size().items():
            counts[(str(well), "T2:debris")] = int(value)
    return counts


def _export_percentage(value: Any) -> float | None:
    """Return a finite percentage value without rounding it to an integer."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(number):
        return None
    return number


def _export_manual_verdicts(
    plate: dict[str, Any], manifest_path: Path
) -> dict[str, str]:
    artifact_root = _export_reference_path(plate.get("artifact_root"), manifest_path.parent)
    if artifact_root is None:
        return {}
    database = artifact_root / "annotations" / "annotations.db"
    if not database.exists():
        return {}
    labels = {"approved": "合格", "pending": "待定", "rejected": "排除"}
    try:
        with sqlite3.connect(database) as connection:
            return {
                str(well).upper(): labels.get(str(decision).lower(), "")
                for well, decision in connection.execute(
                    "SELECT well, decision FROM well_screening_reviews"
                ).fetchall()
            }
    except sqlite3.Error:
        return {}


def _project_review_filter_counts(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    coverage_min: float | None = None,
    debris_max: int | None = None,
    day0_cells_min: int | None = None,
    day0_cells_max: int | None = None,
    day1_cells_min: int | None = None,
    day1_cells_max: int | None = None,
    day2_cells_min: int | None = None,
    day2_cells_max: int | None = None,
) -> dict[str, Any]:
    """Count reviewable wells matching the project-level pre-review filters."""

    plate_counts: list[dict[str, Any]] = []
    total_matching = 0
    total_reviewable = 0
    for plate in manifest.get("plates", []):
        if not isinstance(plate, dict):
            continue
        slug = _export_text(plate.get("slug") or plate.get("board_id"))
        report_path, screening_path = _export_plate_csv_paths(plate, manifest_path)
        report = _export_read_csv(report_path)
        screening = _export_read_csv(screening_path)
        object_path = screening_path.parent / "latest_screening_objects.csv" if screening_path else None
        objects = _export_read_csv(object_path)
        reviewable_wells: set[str] = set()
        if not objects.empty and {"well", "timepoint"}.issubset(objects.columns):
            label_column = "screen_label" if "screen_label" in objects.columns else "integrated_label"
            if label_column in objects.columns:
                frame = objects.copy()
                frame["_filter_well"] = frame["well"].map(_export_text).str.upper()
                frame["_filter_timepoint"] = frame["timepoint"].map(_export_text).str.upper()
                frame["_filter_label"] = frame[label_column].map(_export_text).str.lower()
                frame = frame[
                    frame["_filter_timepoint"].isin(_EXPORT_TIMEPOINTS)
                    & frame["_filter_label"].isin(_REVIEWABLE_LABELS)
                ]
                reviewable_wells = set(frame["_filter_well"])
        report_rows = {
            _export_text(row.get("well")).upper(): row
            for row in report.to_dict(orient="records")
            if _export_text(row.get("well"))
        }
        screening_rows = {
            _export_text(row.get("well")).upper(): row
            for row in screening.to_dict(orient="records")
            if _export_text(row.get("well"))
        }
        if not reviewable_wells:
            reviewable_wells = set(screening_rows)
        debris_counts = _export_debris_count_columns(objects)
        matching = 0
        for well in reviewable_wells:
            report_row = report_rows.get(well, {})
            screening_row = screening_rows.get(well, {})
            coverage = _export_percentage(
                report_row.get("day14_sheet_coverage_pct", report_row.get("sheet_coverage_pct"))
            )
            cells = {
                day: _export_number(screening_row.get(f"t{day}_cell_units"))
                for day in (0, 1, 2)
            }
            debris = debris_counts.get((well, "T2:debris"), 0)
            if coverage_min is not None and (coverage is None or coverage <= coverage_min):
                continue
            if debris_max is not None and debris >= debris_max:
                continue
            if day0_cells_min is not None and cells[0] < day0_cells_min:
                continue
            if day0_cells_max is not None and cells[0] > day0_cells_max:
                continue
            if day1_cells_min is not None and cells[1] < day1_cells_min:
                continue
            if day1_cells_max is not None and cells[1] > day1_cells_max:
                continue
            if day2_cells_min is not None and cells[2] < day2_cells_min:
                continue
            if day2_cells_max is not None and cells[2] > day2_cells_max:
                continue
            matching += 1
        reviewable = len(reviewable_wells)
        total_matching += matching
        total_reviewable += reviewable
        plate_counts.append({
            "slug": slug,
            "board_id": _export_text(plate.get("board_id") or plate.get("group_id") or slug),
            "matching_well_count": matching,
            "reviewable_well_count": reviewable,
        })
    return {
        "matching_well_count": total_matching,
        "reviewable_well_count": total_reviewable,
        "plates": plate_counts,
    }


def _export_project_rows(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    task_name: str | None = None,
    category_overrides: dict[tuple[str, str], str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build one flat row per well across every board in a project."""

    project_name = (
        _export_text(task_name)
        or _export_text(manifest.get("project_name"))
        or _export_text(manifest.get("project_id"))
        or manifest_path.parent.name
    )
    category_overrides = category_overrides or {}
    rows: list[dict[str, Any]] = []
    warnings: dict[str, int] = {}
    for plate in manifest.get("plates", []):
        if not isinstance(plate, dict):
            continue
        board_name = _export_text(plate.get("board_id")) or _export_text(
            plate.get("group_id")
        ) or _export_text(plate.get("slug"))
        report_path, screening_path = _export_plate_csv_paths(plate, manifest_path)
        report = _export_read_csv(report_path)
        screening = _export_read_csv(screening_path)
        if report.empty and screening.empty:
            warnings["missing_result_files"] = warnings.get("missing_result_files", 0) + 1
            continue

        object_path = (
            screening_path.parent / "latest_screening_objects.csv"
            if screening_path is not None
            else None
        )
        objects = _export_read_csv(object_path)
        object_counts = _export_count_columns(objects)
        debris_counts = _export_debris_count_columns(objects)
        manual_verdicts = _export_manual_verdicts(plate, manifest_path)
        report_rows = {
            _export_text(row.get("well")).upper(): row
            for row in report.to_dict(orient="records")
            if _export_text(row.get("well"))
        }
        screening_rows = {
            _export_text(row.get("well")).upper(): row
            for row in screening.to_dict(orient="records")
            if _export_text(row.get("well"))
        }
        wells = sorted(set(report_rows) | set(screening_rows), key=_export_well_sort_key)
        if not wells:
            warnings["empty_result_files"] = warnings.get("empty_result_files", 0) + 1
            continue
        for well in wells:
            report_row = report_rows.get(well, {})
            screening_row = screening_rows.get(well, {})
            override_category = _export_text(
                category_overrides.get(
                    (str(plate.get("slug") or plate.get("board_id") or ""), well)
                )
            )
            category = (
                override_category
                or _export_text(report_row.get("final_category"))
                or _export_text(report_row.get("category_code"))
            )
            conclusion = (
                _EXPORT_CATEGORY_LABELS.get(category, "")
                if override_category
                else _export_text(report_row.get("final_category_label"))
            ) or _EXPORT_CATEGORY_LABELS.get(category, "") or _export_text(
                screening_row.get("screening_status")
            )
            output: dict[str, Any] = {
                "任务名称": project_name,
                "板子名称": board_name,
                "孔号": well,
                "孔结论": conclusion,
                "人工判定": manual_verdicts.get(well, ""),
            }
            for timepoint in _EXPORT_TIMEPOINTS:
                weighted = _export_number(
                    screening_row.get(f"{timepoint.lower()}_cell_units")
                )
                # If an older result set has no object table, retain the
                # persisted weighted units as a useful fallback total.
                count_total = (
                    object_counts.get((well, f"{timepoint}:single"), 0)
                    + object_counts.get((well, f"{timepoint}:touching_doublet"), 0) * 2
                    + object_counts.get((well, f"{timepoint}:cluster_3plus"), 0) * 3
                )
                has_object_counts = any(
                    key[0] == well and key[1].startswith(f"{timepoint}:")
                    for key in object_counts
                )
                count_source = _export_text(
                    screening_row.get(f"{timepoint.lower()}_cell_units_source")
                ).lower()
                output[f"{timepoint}细胞总数"] = (
                    weighted if count_source == "human" or not has_object_counts else count_total
                )
            output["末点细胞覆盖率"] = _export_percentage(
                report_row.get("day14_sheet_coverage_pct", report_row.get("sheet_coverage_pct"))
            )
            output["Day2杂质数"] = debris_counts.get((well, "T2:debris"), 0)
            rows.append(output)
    return rows, warnings


def _write_project_result_excel(
    manifest: dict[str, Any],
    manifest_path: Path,
    destination: Path,
    *,
    task_name: str | None = None,
    category_overrides: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    import openpyxl

    rows, warnings = _export_project_rows(
        manifest,
        manifest_path,
        task_name=task_name,
        category_overrides=category_overrides,
    )
    columns = ["任务名称", "板子名称", "孔号", "孔结论", "人工判定"]
    columns += [f"{timepoint}细胞总数" for timepoint in _EXPORT_TIMEPOINTS]
    columns += ["末点细胞覆盖率", "Day2杂质数"]
    frame = pd.DataFrame(rows, columns=columns)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="检测结果")
        worksheet = writer.sheets["检测结果"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.sheet_view.showGridLines = False
        header_fill = "0E3439"
        header_font = "FFFFFF"
        for cell in worksheet[1]:
            cell.fill = openpyxl.styles.PatternFill("solid", fgColor=header_fill)
            cell.font = openpyxl.styles.Font(color=header_font, bold=True)
            cell.alignment = openpyxl.styles.Alignment(horizontal="center", vertical="center")
        for column_cells in worksheet.columns:
            column_letter = column_cells[0].column_letter
            max_length = max(len(str(cell.value or "")) for cell in column_cells)
            worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 24)
        worksheet.row_dimensions[1].height = 28
        for row in worksheet.iter_rows(min_row=2, min_col=6):
            for cell in row:
                cell.number_format = "0"
        coverage_column = columns.index("末点细胞覆盖率") + 1
        for cell in next(worksheet.iter_cols(
            min_col=coverage_column, max_col=coverage_column, min_row=2
        )):
            cell.number_format = '0.00"%"'
    return {"row_count": len(frame), "board_count": len(manifest.get("plates", [])), "warnings": warnings}


def _export_filename(value: Any) -> str:
    name = _export_text(value) or "项目任务"
    name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip(" .")
    return f"{name or '项目任务'}_检测结果.xlsx"


def _remove_export_file(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


class FolderPayload(BaseModel):
    path: str = Field(min_length=1)

    @field_validator("path", mode="before")
    @classmethod
    def validate_required_path(cls, value: Any) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("数据文件夹不能为空")
        return normalized


class TaskPayload(FolderPayload):
    name: str = Field(min_length=1, max_length=120)
    created_by: str = Field(min_length=1, max_length=80)
    selected_timepoint_labels: list[str] = Field(default_factory=list)

    @field_validator("name", "created_by", mode="before")
    @classmethod
    def validate_required_text(cls, value: Any, info: Any) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            label = "任务名称" if info.field_name == "name" else "创建人"
            raise ValueError(f"{label}不能为空")
        return normalized


class ProjectRenamePayload(BaseModel):
    project_name: str = Field(min_length=1, max_length=120)


class ProjectMultiplicityLabelItem(BaseModel):
    plate_slug: str
    candidate_id: str
    well: str
    timepoint: str
    x_px: float
    y_px: float
    label: str
    source: str = "quick_single_doublet"


class ProjectMultiplicityLabelsPayload(BaseModel):
    items: list[ProjectMultiplicityLabelItem]
    reviewer: str = "local_user"


class ProjectMaskReviewSavePayload(BaseModel):
    round_id: str
    candidate_id: str
    decision: str
    reviewed_mask_rle: str | None = None
    reviewer: str = "local_user"
    notes: str = ""


class ProjectMaskComparisonPayload(BaseModel):
    old_checkpoint: str | None = None
    new_checkpoint: str | None = None
    source_configs: list[str] = Field(default_factory=list)
    round_id: str | None = None


_DAY_RE = re.compile(r"^day\s*(?P<day>-?\d+)$", re.IGNORECASE)


def _day_number(value: Any) -> int | None:
    """Return an actual culture-day number from a parsed Day label."""

    match = _DAY_RE.match(str(value or "").strip())
    if match:
        try:
            return int(match.group("day"))
        except ValueError:
            return None
    return None


def _timepoint_options(sessions: pd.DataFrame) -> list[dict[str, Any]]:
    """Collapse session rows into actual calendar-day choices.

    ``timepoint_label`` is an acquisition-order label (T0, T1, ...), while
    ``day_label`` is the culture age.  Task selection must use the latter so a
    task remains correct when a plate has a missing or extra acquisition.
    """

    if sessions.empty:
        return []
    frame = sessions.copy()
    frame["day_number"] = frame["culture_day"].map(_day_number)
    # ``culture_day`` is an Int64 column in pandas; use it when available.
    numeric = pd.to_numeric(frame["culture_day"], errors="coerce")
    frame["day_number"] = numeric.where(numeric.notna(), frame["day_number"])
    rows: list[dict[str, Any]] = []
    for day_number, group in frame.dropna(subset=["day_number"]).groupby("day_number", sort=True):
        day = int(day_number)
        labels = sorted({str(value) for value in group["timepoint_label"].dropna()})
        day_label = f"Day{day}"
        rows.append({
            "day_label": day_label,
            "day_number": day,
            "timepoint_labels": labels,
            "session_count": int(len(group)),
            "group_count": int(group["group_id"].nunique()),
            "complete_session_count": int(group["group_complete"].fillna(False).sum()),
            "required_early": day in {0, 1, 2},
            "eligible_endpoint": day >= 7,
        })
    return rows


def _default_timepoint_selection(options: list[dict[str, Any]]) -> list[str]:
    selected = [str(item["day_label"]) for item in options if int(item["day_number"]) in {0, 1, 2}]
    later = [
        item
        for item in options
        if bool(item.get("eligible_endpoint"))
        or int(item.get("day_number", -1)) >= 7
    ]
    if later:
        selected.append(str(max(later, key=lambda item: int(item["day_number"]))["day_label"]))
    return list(dict.fromkeys(selected))


def _validate_timepoint_selection(
    options: list[dict[str, Any]], selected: list[str] | None,
) -> tuple[list[str], str, int]:
    by_label = {str(item["day_label"]).casefold(): item for item in options}
    default = _default_timepoint_selection(options)
    requested = [str(value).strip() for value in (selected or []) if str(value).strip()]
    chosen: list[dict[str, Any]] = []
    for value in requested or default:
        item = by_label.get(value.casefold())
        if item is None:
            # Accept T0/T1/T2 aliases returned by older clients.
            alias = next((candidate for candidate in options if value.casefold() in {label.casefold() for label in candidate["timepoint_labels"]}), None)
            item = alias
        if item is not None and item not in chosen:
            chosen.append(item)
    chosen.sort(key=lambda item: int(item["day_number"]))
    chosen_labels = [str(item["day_label"]) for item in chosen]
    missing = [f"Day{day}" for day in (0, 1, 2) if f"Day{day}".casefold() not in {label.casefold() for label in chosen_labels}]
    later = [item for item in chosen if int(item["day_number"]) >= 7]
    if missing:
        raise HTTPException(status_code=422, detail=f"必须连续选择 Day0、Day1、Day2；缺少：{', '.join(missing)}")
    if not later:
        raise HTTPException(status_code=422, detail="至少选择一个 Day7 或更晚的时间点作为末点")
    endpoint = max(later, key=lambda item: int(item["day_number"]))
    return chosen_labels, str(endpoint["day_label"]), int(endpoint["day_number"])


def _session_is_complete(row: dict[str, Any]) -> bool:
    value = row.get("group_complete")
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _complete_group_timepoint_selection(
    session_records: list[dict[str, Any]], selected: list[str]
) -> dict[str, Any]:
    """Return only boards that have complete sessions at every selected day."""

    selected_labels = list(dict.fromkeys(str(label) for label in selected))
    selected_lookup = {label.casefold(): label for label in selected_labels}
    by_group: dict[str, dict[str, list[dict[str, Any]]]] = {}
    board_ids: dict[str, str] = {}
    for row in session_records:
        group = str(row.get("group_id") or "").strip()
        label = str(row.get("day_label") or "").casefold()
        if not group:
            continue
        by_group.setdefault(group, {})
        board_id = str(row.get("board_id") or "").strip()
        if board_id:
            board_ids.setdefault(group, board_id)
        if label in selected_lookup:
            by_group[group].setdefault(label, []).append(row)

    included_group_ids: list[str] = []
    excluded_groups: list[dict[str, Any]] = []
    for group, available in sorted(by_group.items()):
        missing = [
            label for key, label in selected_lookup.items() if key not in available
        ]
        incomplete = [
            label
            for key, label in selected_lookup.items()
            if key in available and not any(_session_is_complete(row) for row in available[key])
        ]
        if missing or incomplete:
            reasons = []
            if missing:
                reasons.append(f"缺少 {', '.join(missing)}")
            if incomplete:
                reasons.append(f"{', '.join(incomplete)} 不完整")
            excluded_groups.append({
                "group_id": group,
                "board_id": board_ids.get(group, ""),
                "missing_timepoints": missing,
                "incomplete_timepoints": incomplete,
                "reason": "；".join(reasons),
            })
        else:
            included_group_ids.append(group)

    included = set(included_group_ids)
    selected_sessions = [
        row
        for row in session_records
        if str(row.get("group_id") or "").strip() in included
        and str(row.get("day_label") or "").casefold() in selected_lookup
        and _session_is_complete(row)
    ]
    return {
        "included_group_ids": included_group_ids,
        "included_group_count": len(included_group_ids),
        "excluded_group_count": len(excluded_groups),
        "excluded_groups": excluded_groups,
        "sessions": selected_sessions,
    }


def _parse_folder(path_value: str) -> dict[str, Any]:
    root = _resolve(path_value)
    index = root if root.is_file() and root.name.casefold() == "sessions.idx" else root / "sessions.idx"
    if not index.exists():
        raise HTTPException(status_code=400, detail=f"未找到 sessions.idx: {index}")
    try:
        sessions = parse_sessions_index(index, root if root.is_dir() else index.parent, timepoint_origin=0)
        groups = summarize_session_groups(sessions)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    options = _timepoint_options(sessions)
    default_selected = _default_timepoint_selection(options)
    try:
        _, default_endpoint, default_endpoint_day = _validate_timepoint_selection(options, default_selected)
        selection_valid = True
        selection_error = ""
    except HTTPException as exc:
        default_endpoint = ""
        default_endpoint_day = None
        selection_valid = False
        selection_error = str(exc.detail)
    default_coverage = (
        _complete_group_timepoint_selection(
            [
                _safe(row)
                for row in sessions[
                    ["group_id", "board_id", "timepoint_label", "day_label", "culture_day", "session_path", "group_complete"]
                ].to_dict(orient="records")
            ],
            default_selected,
        )
        if selection_valid
        else {
            "included_group_ids": [],
            "included_group_count": 0,
            "excluded_group_count": 0,
            "excluded_groups": [],
            "sessions": [],
        }
    )
    day_values = sorted(
        {str(value) for value in sessions["day_label"].dropna()},
        key=lambda label: (
            _day_number(label) is None,
            _day_number(label) if _day_number(label) is not None else 10**9,
            label.casefold(),
        ),
    )
    timepoint_values = sorted(
        {str(value) for value in sessions["timepoint_label"].dropna()},
        key=lambda label: (
            not label.casefold().startswith("t"),
            int(label[1:])
            if label.casefold().startswith("t") and label[1:].isdigit()
            else 10**9,
            label.casefold(),
        ),
    )
    return {
        "root": str(root),
        "folder_name": _folder_name(root),
        "index": str(index),
        "session_count": int(len(sessions)),
        "group_count": int(len(groups)),
        "groups": [_safe(row) for row in groups.to_dict(orient="records")],
        "timepoint_labels": timepoint_values,
        "day_labels": day_values,
        "timepoint_options": options,
        "default_selected_timepoint_labels": default_selected,
        "default_endpoint_day_label": default_endpoint,
        "default_endpoint_day_number": default_endpoint_day,
        "selection_valid": selection_valid,
        "selection_error": selection_error,
        "compatible_group_count": default_coverage["included_group_count"],
        "excluded_group_count": default_coverage["excluded_group_count"],
        "excluded_groups": default_coverage["excluded_groups"],
        "session_records": [
            _safe(row)
            for row in sessions[
                ["group_id", "board_id", "timepoint_label", "day_label", "culture_day", "session_path", "group_complete"]
            ].to_dict(orient="records")
        ],
        "complete_groups": int(groups["complete_96_well_sessions"].eq(groups["session_count"]).sum()) if not groups.empty else 0,
    }


def _queue_path(manifest_path: Path) -> Path:
    configured = manifest_path.parent / "task_queue.json"
    configured.parent.mkdir(parents=True, exist_ok=True)
    return configured


def _project_manifest_paths(current: Path) -> list[Path]:
    """Find sibling project manifests for the hub homepage."""

    try:
        current_value = json.loads(current.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current_value = {}
    current_dict = current_value if isinstance(current_value, dict) else {}
    configured = current_dict.get("project_manifest_paths")
    if isinstance(configured, list):
        paths: list[Path] = []
        for value in configured:
            candidate = Path(os.path.expandvars(str(value))).expanduser()
            if not candidate.is_absolute():
                candidate = current.parent / candidate
            candidate = candidate.resolve()
            if candidate.is_file():
                paths.append(candidate)
        if bool(current_dict.get("review_hub")):
            return sorted(set(paths), key=lambda path: path.as_posix().casefold())

    root = current.parent.parent
    paths = [path for path in root.glob("*/project.json") if path.is_file()]
    if current.is_file() and current not in paths:
        paths.append(current)
    return sorted(set(paths), key=lambda path: path.as_posix().casefold())


def _find_project_manifest(current: Path, project_id: str) -> Path | None:
    wanted = _slug(project_id)
    for path in _project_manifest_paths(current):
        value = _read_manifest(path) or {}
        candidate = _slug(str(value.get("project_id") or path.parent.name))
        if candidate == wanted:
            return path
    return None


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    base = path.parent
    for key in (
        "root",
        "source_sessions_csv",
        "source_day14_csv",
        "source_endpoint_csv",
    ):
        item = value.get(key)
        if item:
            candidate = Path(os.path.expandvars(str(item))).expanduser()
            value[key] = str(candidate if candidate.is_absolute() else (base / candidate).resolve())
    for plate in value.get("plates", []):
        if not isinstance(plate, dict):
            continue
        for key in (
            "config",
            "artifact_root",
            "gated_output_dir",
            "images_manifest",
            "pipeline_summary",
            "report_json",
        ):
            item = plate.get(key)
            if not item:
                continue
            candidate = Path(os.path.expandvars(str(item))).expanduser()
            plate[key] = str(candidate if candidate.is_absolute() else (base / candidate).resolve())
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _folder_name(path_value: str | Path) -> str:
    path = Path(os.path.expandvars(str(path_value))).expanduser()
    if path.name.casefold() == "sessions.idx":
        path = path.parent
    return path.name or "新建项目"


def _project_card(path: Path) -> dict[str, Any]:
    """Build a lightweight project card without opening board artifacts."""

    value = _read_manifest(path) or {}
    plates = value.get("plates") if isinstance(value.get("plates"), list) else []
    aggregate: dict[str, int] = {}
    completed = 0
    reviewed = 0
    for plate in plates:
        if not isinstance(plate, dict):
            continue
        stored_counts = plate.get("category_counts")
        if str(plate.get("status", "")).lower() == "completed":
            completed += 1
            if isinstance(stored_counts, dict):
                for key, count in stored_counts.items():
                    aggregate[str(key)] = aggregate.get(str(key), 0) + int(count or 0)
        if bool(plate.get("review_complete")):
            reviewed += 1
    project_id = str(value.get("project_id") or path.parent.name)
    detection_start = value.get("detection_start_date")
    detection_end = value.get("detection_end_date")
    return {
        "project_id": project_id,
        "project_name": str(value.get("project_name") or project_id),
        "root": value.get("root"),
        "manifest_path": str(path),
        "plate_count": len(plates),
        "completed_plate_count": completed,
        "recognized_plate_count": completed,
        "reviewed_plate_count": reviewed,
        "category_counts": aggregate,
        "single_cell_origin_well_count": int(aggregate.get("single_cell_origin", 0)),
        "detection_start_date": detection_start,
        "detection_end_date": detection_end,
        "created_by": str(
            value.get("created_by")
            or value.get("creator")
            or value.get("owner")
            or ""
        ),
        "generated_at": value.get("generated_at"),
        "detail_url": f"/projects/{_slug(project_id)}/",
    }


class _LazyPlateApp:
    """Create a plate review app only when that plate is first opened.

    A project may contain many boards.  Eagerly constructing one full review
    application per board duplicates image/data caches and makes startup scale
    with the whole project.  This ASGI wrapper keeps the same mounted URL while
    deferring the expensive initialization to the first request.
    """

    def __init__(
        self,
        config_path: Path,
        project_back_url: str = "",
        review_base_url: str = "",
        catalog_context: dict[str, Any] | None = None,
    ):
        self.config_path = config_path
        self.project_back_url = project_back_url
        self.review_base_url = review_base_url.rstrip("/")
        self.catalog_context = dict(catalog_context or {})
        self._app = None
        self._error: str | None = None
        self._lock = asyncio.Lock()
        self._active_requests = 0
        self._last_used = time.monotonic()

    @property
    def loaded(self) -> bool:
        return self._app is not None

    @property
    def active_requests(self) -> int:
        return self._active_requests

    @property
    def last_used(self) -> float:
        return self._last_used

    def touch(self) -> None:
        self._last_used = time.monotonic()

    async def unload(self) -> None:
        """Release the in-memory review application after it becomes idle."""

        async with self._lock:
            if self._active_requests:
                return
            self._app = None
            self._error = None
            self.touch()

    async def _ensure_app(self):
        if self._app is not None:
            return self._app
        async with self._lock:
            if self._app is None and self._error is None:
                try:
                    # Review-app creation performs synchronous image/database
                    # discovery.  Keep that work off the event loop.
                    config = load_config(self.config_path)
                    if self.catalog_context:
                        config["_catalog_context"] = dict(self.catalog_context)
                    self._app = await asyncio.to_thread(
                        create_app,
                        config,
                        project_back_url=self.project_back_url,
                        review_base_url=self.review_base_url,
                    )
                except Exception as exc:  # pragma: no cover - startup-only path
                    self._error = f"{type(exc).__name__}: {exc}"
        return self._app

    async def __call__(self, scope, receive, send):
        self._active_requests += 1
        self.touch()
        try:
            app = await self._ensure_app()
            if app is None:
                if scope.get("type") != "http":
                    raise RuntimeError(self._error or "plate review app unavailable")
                body = json.dumps(
                    {"detail": "plate review app unavailable", "error": self._error},
                    ensure_ascii=False,
                ).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [(b"content-type", b"application/json; charset=utf-8")],
                })
                await send({"type": "http.response.body", "body": body})
                return
            await app(scope, receive, send)
        finally:
            self._active_requests = max(0, self._active_requests - 1)
            self.touch()


class _PlateReviewManager:
    """Resolve and lazily serve any project/plate pair from one hub.

    Project manifests are intentionally read only when a board is requested.
    The manager keeps a small LRU of initialized review applications, so the
    hub can expose many projects without loading every board's CSV/database at
    startup.
    """

    def __init__(
        self,
        root_manifest: Path,
        *,
        fixed_project_id: str | None = None,
        legacy_project_id: str | None = None,
        max_loaded: int = 3,
        idle_seconds: float = 20 * 60,
    ) -> None:
        self.root_manifest = root_manifest
        self.fixed_project_id = fixed_project_id
        self.legacy_project_id = legacy_project_id
        self.max_loaded = max(1, int(max_loaded))
        self.idle_seconds = max(30.0, float(idle_seconds))
        self._apps: OrderedDict[tuple[str, str], _LazyPlateApp] = OrderedDict()
        self._lock = asyncio.Lock()
        self._manifest_cache: dict[Path, tuple[int, dict[str, Any]]] = {}
        self._manifest_lock = RLock()

    @property
    def loaded_keys(self) -> list[tuple[str, str]]:
        return [key for key, value in self._apps.items() if value.loaded]

    async def start(self) -> None:
        if getattr(self, "_janitor", None) is None or self._janitor.done():
            self._janitor = asyncio.create_task(self._janitor_loop())

    async def stop(self) -> None:
        janitor = getattr(self, "_janitor", None)
        if janitor is None:
            return
        janitor.cancel()
        try:
            await janitor
        except asyncio.CancelledError:
            pass
        self._janitor = None

    async def _janitor_loop(self) -> None:
        interval = min(60.0, max(5.0, self.idle_seconds / 4))
        try:
            while True:
                await asyncio.sleep(interval)
                await self._evict()
        except asyncio.CancelledError:
            raise

    def resolve_plate(
        self,
        project_id: str,
        plate_slug: str,
    ) -> tuple[str, Path, dict[str, Any]] | None:
        """Return the canonical project id, config path and plate record."""

        requested_project = self.fixed_project_id or project_id
        wanted_project = _slug(requested_project)
        manifest_path = None
        value = None
        with self._manifest_lock:
            for candidate in _project_manifest_paths(self.root_manifest):
                try:
                    modified = candidate.stat().st_mtime_ns
                except OSError:
                    continue
                cached = self._manifest_cache.get(candidate)
                if cached is None or cached[0] != modified:
                    current = _read_manifest(candidate)
                    if current is None:
                        self._manifest_cache.pop(candidate, None)
                        continue
                    self._manifest_cache[candidate] = (modified, current)
                    cached = (modified, current)
                candidate_value = cached[1]
                candidate_id = _slug(str(candidate_value.get("project_id") or candidate.parent.name))
                if candidate_id == wanted_project:
                    manifest_path = candidate
                    value = candidate_value
                    break
            live_paths = set(_project_manifest_paths(self.root_manifest))
            for cached_path in list(self._manifest_cache):
                if cached_path not in live_paths:
                    self._manifest_cache.pop(cached_path, None)
        if value is None:
            return None
        canonical_project = str(value.get("project_id") or manifest_path.parent.name)
        if self.fixed_project_id and _slug(canonical_project) != _slug(self.fixed_project_id):
            return None
        wanted_plate = _slug(plate_slug)
        for plate in value.get("plates", []):
            if not isinstance(plate, dict):
                continue
            candidate = str(plate.get("slug") or _slug(plate.get("board_id", "")))
            if not candidate or _slug(candidate) != wanted_plate:
                continue
            config_value = plate.get("config")
            images_value = plate.get("images_manifest")
            if not config_value or not images_value:
                return None
            config_path = _resolve(config_value)
            images_path = _resolve(images_value)
            if not config_path.exists() or not images_path.exists():
                return None
            return canonical_project, config_path, plate
        return None

    async def _get_app(
        self,
        project_id: str,
        plate_slug: str,
    ) -> tuple[_LazyPlateApp | None, str | None]:
        resolved = self.resolve_plate(project_id, plate_slug)
        if resolved is None:
            return None, "project or plate not found"
        canonical_project, config_path, _ = resolved
        canonical_plate = _slug(plate_slug)
        catalog_manifest = _find_project_manifest(self.root_manifest, canonical_project)
        catalog_context = {
            "catalog_path": str(catalog_path_for_manifest(self.root_manifest)),
            "manifest_path": str(catalog_manifest) if catalog_manifest else "",
            "project_id": canonical_project,
            "plate_slug": canonical_plate,
        }
        key = (canonical_project, canonical_plate)
        async with self._lock:
            app = self._apps.get(key)
            if app is None:
                app = _LazyPlateApp(
                    config_path,
                    project_back_url=f"/projects/{_slug(canonical_project)}/",
                    review_base_url=f"/projects/{_slug(canonical_project)}/plates/{canonical_plate}",
                    catalog_context=catalog_context,
                )
                self._apps[key] = app
            else:
                # A worker may replace a generated config while the server is
                # running.  Re-resolve the path on every request and refresh
                # the wrapper if it changed.
                if app.config_path != config_path:
                    await app.unload()
                    app = _LazyPlateApp(
                        config_path,
                        project_back_url=f"/projects/{_slug(canonical_project)}/",
                        review_base_url=f"/projects/{_slug(canonical_project)}/plates/{canonical_plate}",
                        catalog_context=catalog_context,
                    )
                    self._apps[key] = app
            self._apps.move_to_end(key)
            app.touch()
        await self._evict(exclude=key)
        return app, None

    async def _evict(self, *, exclude: tuple[str, str] | None = None) -> None:
        now = time.monotonic()
        async with self._lock:
            all_loaded = [
                (key, app)
                for key, app in self._apps.items()
                if app.loaded
            ]
            loaded = [
                item for item in all_loaded
                if item[0] != exclude and item[1].active_requests == 0
            ]
            by_age = sorted(loaded, key=lambda item: item[1].last_used)
            needed = max(0, len(all_loaded) - self.max_loaded)
            candidates = by_age[:needed]
            for item in by_age:
                if now - item[1].last_used >= self.idle_seconds and item not in candidates:
                    candidates.append(item)
            for key, app in candidates:
                if self._apps.get(key) is app:
                    self._apps.pop(key, None)
        for _, app in candidates:
            await app.unload()

    async def __call__(self, scope, receive, send):
        params = scope.get("path_params", {})
        project_id = str(
            params.get("project_id")
            or self.fixed_project_id
            or self.legacy_project_id
            or ""
        )
        plate_slug = str(params.get("plate_slug") or "")
        app, error = await self._get_app(project_id, plate_slug)
        if app is None:
            body = json.dumps({"detail": error or "plate not found"}, ensure_ascii=False).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"application/json; charset=utf-8")],
            })
            await send({"type": "http.response.body", "body": body})
            return
        # ``add_route`` performs parameter matching but does not create the
        # child ASGI scope that ``mount`` normally creates.  Strip the hub
        # prefix before handing the request to the per-board FastAPI app so
        # its existing /auto-review, /api/* and /assets/* routes continue to
        # match.  Keep root_path for URL generation and diagnostics.
        path = str(params.get("path") or "")
        child_scope = dict(scope)
        prefix = str(scope.get("root_path") or "")
        if not prefix:
            raw_path = str(scope.get("path") or "")
            marker = f"/plates/{plate_slug}"
            marker_index = raw_path.find(marker)
            prefix = raw_path[: marker_index + len(marker)] if marker_index >= 0 else raw_path
        child_scope["root_path"] = ""
        child_scope["app_root_path"] = prefix
        child_scope["path"] = f"/{path}" if path else "/"
        child_scope["path_params"] = {}
        await app(child_scope, receive, send)


def create_project_app(manifest_path: str | Path) -> FastAPI:
    manifest_file = _resolve(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(manifest_file)
    manifest = _read_manifest(manifest_file)
    if manifest is None:
        raise ValueError(f"Invalid project manifest: {manifest_file}")
    project_name = str(manifest.get("project_name", manifest.get("project_id", "Cell Vision Project")))
    project_id = str(manifest.get("project_id") or manifest_file.parent.name)
    project_back_url = f"/projects/{_slug(project_id)}/"
    ui_root = PROJECT_ROOT / "review-ui"
    app = FastAPI(title=f"Cell Vision Project · {project_name}")
    app.mount("/project-assets", StaticFiles(directory=ui_root), name="project-assets")
    queue_file = _queue_path(manifest_file)
    queue_store = TaskQueueStore(queue_file)
    catalog = ProjectCatalog(catalog_path_for_manifest(manifest_file))
    catalog_refresh_lock = RLock()
    catalog_refreshed_at = 0.0

    def refresh_catalog(*, force: bool = False) -> None:
        """Refresh changed manifests/reports before serving project metadata."""

        nonlocal catalog_refreshed_at
        now = time.monotonic()
        if not force and now - catalog_refreshed_at < 1.0:
            return
        with catalog_refresh_lock:
            now = time.monotonic()
            if not force and now - catalog_refreshed_at < 1.0:
                return
            try:
                catalog.reconcile_all(manifest_file)
                catalog_refreshed_at = now
            except (OSError, sqlite3.Error, ValueError, TypeError):
                # Keep the legacy manifest response available if a catalog
                # file is temporarily locked or an optional report is bad.
                return

    def sync_task_catalog(task: dict[str, Any] | None) -> None:
        if not task:
            return
        try:
            catalog.sync_task(task)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            return

    def sync_catalog_review(
        selected_manifest: Path,
        plate_slug: str | None = None,
        *,
        wells: set[str] | list[str] | tuple[str, ...] | None = None,
        source: str = "human_review",
        reviewer: str = "",
        operation: str = "review_save",
    ) -> None:
        """Project-level review endpoints share the same catalog contract."""

        selected_value = _read_manifest(selected_manifest) or {}
        selected_project_id = str(selected_value.get("project_id") or selected_manifest.parent.name)
        try:
            if plate_slug:
                catalog.sync_review_update(
                    selected_manifest,
                    f"{selected_project_id}:{_slug(plate_slug)}",
                    wells=wells,
                    source=source,
                    reviewer=reviewer,
                    operation=operation,
                )
            else:
                catalog.sync_manifest(selected_manifest, force=True, source=source)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            return

    def manifest_for_project(project_id: str | None = None) -> Path:
        """Resolve the manifest that owns a project-scoped API request."""

        if not project_id:
            return manifest_file
        selected = _find_project_manifest(manifest_file, project_id)
        if selected is None:
            raise HTTPException(status_code=404, detail="project not found")
        return selected

    def task_store_for_manifest(selected_manifest: Path) -> TaskQueueStore:
        """Keep each project's task lifecycle in its own durable store."""

        return TaskQueueStore(_queue_path(selected_manifest))

    def task_plan_paths() -> list[Path]:
        """Find task plans written before queues became project-scoped.

        The first project-hub implementation kept plans beside the startup
        project's queue.  Later versions moved each queue beside its project
        manifest.  Plans are small metadata files, so scanning only the
        project directories (rather than image/artifact trees) is cheap and
        keeps those older tasks recoverable.
        """

        paths: list[Path] = []
        seen: set[Path] = set()
        for selected_manifest in _project_manifest_paths(manifest_file):
            plan_dir = selected_manifest.parent / "task_plans"
            if not plan_dir.is_dir():
                continue
            for plan_path in plan_dir.glob("*.json"):
                if not plan_path.is_file():
                    continue
                if task_plan_deleted_marker(plan_path).exists():
                    continue
                resolved = plan_path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                paths.append(resolved)
        return sorted(paths, key=lambda path: path.as_posix().casefold())

    def task_plan_deleted_marker(plan_path: Path) -> Path:
        """Return the durable tombstone used when a legacy task is deleted."""

        return plan_path.with_name(f"{plan_path.name}.deleted")

    def mark_task_plan_deleted(task: dict[str, Any] | None) -> None:
        """Prevent a deleted legacy plan from being imported again."""

        if not task:
            return
        plan_value = str(task.get("plan_path") or "")
        if not plan_value:
            return
        plan_path = _resolve(plan_value)
        if plan_path.suffix.casefold() != ".json" or plan_path.parent.name != "task_plans":
            return
        try:
            _write_json_atomic(
                task_plan_deleted_marker(plan_path),
                {
                    "task_id": str(task.get("task_id") or plan_path.stem),
                    "deleted_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except OSError:
            # Removing the queue row is still useful even if a read-only plan
            # directory prevents the compatibility tombstone from being made.
            return

    def task_plan_manifest(plan: dict[str, Any], plan_path: Path) -> Path:
        """Resolve the best project manifest for a legacy task plan."""

        explicit = plan.get("project_manifest")
        if explicit:
            candidate = _resolve(str(explicit))
            if candidate.is_file():
                return candidate
        task_key = str(plan.get("task_id") or "")
        if task_key:
            for candidate in _project_manifest_paths(manifest_file):
                value = _read_manifest(candidate) or {}
                if str(value.get("task_id") or "") == task_key:
                    return candidate
        owner = plan_path.parent.parent / "project.json"
        if owner.is_file():
            return owner.resolve()
        return manifest_file

    def legacy_task_from_plan(plan_path: Path, plan: dict[str, Any]) -> dict[str, Any] | None:
        """Convert an old plan-only record into the current task shape."""

        if task_plan_deleted_marker(plan_path).exists():
            return None
        task_key = str(plan.get("task_id") or "")
        if not task_key:
            return None
        selected_manifest = task_plan_manifest(plan, plan_path)
        selected_value = _read_manifest(selected_manifest) or {}
        manifest_task_matches = str(selected_value.get("task_id") or "") == task_key
        sessions = [item for item in plan.get("sessions", []) if isinstance(item, dict)]
        group_ids = {
            str(item.get("group_id"))
            for item in sessions
            if item.get("group_id")
        }
        timepoint_labels = sorted({
            str(item.get("timepoint_label"))
            for item in sessions
            if item.get("timepoint_label")
        })
        day_labels = sorted({
            str(item.get("day_label"))
            for item in sessions
            if item.get("day_label")
        })
        status = str(plan.get("status") or "")
        if not status and manifest_task_matches:
            status = str(selected_value.get("status") or "")
        status = status or "queued"
        project_id = str(
            plan.get("project_id")
            or (selected_value.get("project_id") if selected_value else "")
            or selected_manifest.parent.name
        )
        project_name = str(
            plan.get("project_name")
            or (selected_value.get("project_name") if selected_value else "")
            or project_id
        )
        group_count = int(plan.get("group_count", 0) or len(group_ids))
        record: dict[str, Any] = {
            "task_id": task_key,
            "name": str(plan.get("name") or project_name or task_key),
            "project_name": project_name,
            "created_by": str(
                plan.get("created_by")
                or selected_value.get("created_by")
                or selected_value.get("creator")
                or selected_value.get("owner")
                or ""
            ),
            "path": str(plan.get("root") or selected_value.get("root") or ""),
            "index": str(plan.get("index") or ""),
            "status": status,
            "created_at": str(
                plan.get("created_at")
                or selected_value.get("created_at")
                or selected_value.get("generated_at")
                or ""
            ),
            "updated_at": str(plan.get("updated_at") or plan.get("created_at") or ""),
            "group_count": group_count,
            "session_count": int(plan.get("session_count", 0) or len(sessions)),
            "timepoint_labels": timepoint_labels,
            "day_labels": day_labels,
            "timepoint_options": plan.get("timepoint_options", []),
            "selected_timepoint_labels": list(plan.get("selected_timepoint_labels", [])),
            "endpoint_day_label": plan.get("endpoint_day_label"),
            "endpoint_day_number": plan.get("endpoint_day_number"),
            "endpoint_timepoint_labels": list(plan.get("endpoint_timepoint_labels", [])),
            "early_timepoint_labels": list(plan.get("early_timepoint_labels", [])),
            "plan_path": str(plan_path),
            "project_id": project_id,
            "project_manifest": str(selected_manifest),
        }
        if status == "completed":
            record.update({
                "progress_current": group_count,
                "progress_total": group_count,
                "progress_percent": 100,
                "progress_stage": "completed",
            })
        return record

    def migrate_legacy_task_plans() -> None:
        """Restore queue entries that only exist as legacy task plans."""

        for plan_path in task_plan_paths():
            plan = _read_manifest(plan_path)
            if plan is None:
                continue
            task = legacy_task_from_plan(plan_path, plan)
            if task is None:
                continue
            # A legacy plan lived beside the old startup queue.  Once its
            # owning manifest is known, place the recovered record beside
            # that manifest so a project-scoped worker uses the same queue as
            # the browser lifecycle endpoints.  This only adds a projection;
            # the original plan and source artifacts are left untouched.
            queue_owner_value = str(task.get("project_manifest") or "")
            queue_owner = _resolve(queue_owner_value) if queue_owner_value else Path()
            if not queue_owner.is_file():
                queue_owner = plan_path.parent.parent / "project.json"
            legacy_queue = TaskQueueStore(queue_owner.parent / "task_queue.json")
            restored = legacy_queue.ensure(task)
            sync_task_catalog(restored)

    def task_store_entries() -> list[tuple[Path, TaskQueueStore]]:
        """Return all project stores, including legacy records in the main store."""

        migrate_legacy_task_plans()
        entries: list[tuple[Path, TaskQueueStore]] = []
        seen: set[Path] = set()
        for selected_manifest in _project_manifest_paths(manifest_file):
            resolved_queue = _queue_path(selected_manifest).resolve()
            if resolved_queue in seen:
                continue
            seen.add(resolved_queue)
            entries.append((selected_manifest, TaskQueueStore(resolved_queue)))
        if queue_file.resolve() not in seen:
            entries.append((manifest_file, queue_store))
        return entries

    def remove_task_from_all_queues(task_id_value: str) -> bool:
        """Remove every physical copy of a task from all known queues."""

        removed = False
        for _, other_store in task_store_entries():
            if other_store.delete(task_id_value) is not None:
                removed = True
        return removed

    def disposable_task_project(task: dict[str, Any]) -> Path | None:
        """Return an isolated generated project directory safe to remove."""

        manifest_value = str(task.get("project_manifest") or "").strip()
        if not manifest_value:
            return None
        task_manifest = _resolve(manifest_value)
        projects_root = manifest_file.parent.parent.resolve()
        project_dir = task_manifest.parent.resolve()
        if (
            task_manifest.name.casefold() != "project.json"
            or project_dir == manifest_file.parent.resolve()
            or project_dir.parent != projects_root
        ):
            return None
        value = _read_manifest(task_manifest) if task_manifest.exists() else None
        if value is not None:
            manifest_task_id = str(value.get("task_id") or "")
            manifest_project_id = str(value.get("project_id") or project_dir.name)
            if manifest_task_id and manifest_task_id != str(task.get("task_id") or ""):
                return None
            if str(task.get("project_id") or manifest_project_id) != manifest_project_id:
                return None
        return project_dir

    def tasks_for_project(project_id: str | None = None) -> list[dict[str, Any]]:
        """Read one project's queue, or a de-duplicated hub-wide view."""

        if project_id:
            selected_manifest = manifest_for_project(project_id).resolve()
            selected_value = _read_manifest(selected_manifest) or {}
            selected_project_id = _slug(
                str(selected_value.get("project_id") or selected_manifest.parent.name)
            )
            values: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for owner_manifest, store in task_store_entries():
                for task in store.list():
                    task_manifest = str(task.get("project_manifest") or "")
                    task_project_id = _slug(str(task.get("project_id") or ""))
                    has_task_identity = bool(task_project_id or task_manifest)
                    belongs = (
                        (
                            task_project_id == selected_project_id
                            or (
                                task_manifest
                                and _resolve(task_manifest).resolve() == selected_manifest
                            )
                        )
                        if has_task_identity
                        else owner_manifest.resolve() == selected_manifest
                    )
                    if not belongs:
                        continue
                    key = str(task.get("task_id") or "")
                    if key and key in seen_ids:
                        continue
                    if key:
                        seen_ids.add(key)
                    values.append(task)
                    sync_task_catalog(task)
            return values
        values: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for _, store in task_store_entries():
            for task in store.list():
                key = str(task.get("task_id") or "")
                if key and key in seen_ids:
                    continue
                if key:
                    seen_ids.add(key)
                values.append(task)
                sync_task_catalog(task)
        return values

    def locate_task(task_id_value: str) -> tuple[TaskQueueStore, dict[str, Any]] | None:
        """Find a task without assuming it belongs to the startup project."""

        for _, store in task_store_entries():
            task = store.get(task_id_value)
            if task is not None:
                return store, task
        return None

    # Register only a dynamic dispatcher.  It resolves the requested project
    # and plate at request time, so sibling projects are immediately usable and
    # no board image manifest/database is read during service startup.
    review_manager = _PlateReviewManager(manifest_file, legacy_project_id=project_id)

    @app.on_event("startup")
    async def start_review_manager() -> None:
        await review_manager.start()

    @app.on_event("shutdown")
    async def stop_review_manager() -> None:
        await review_manager.stop()

    app.add_route(
        "/projects/{project_id}/plates/{plate_slug}/{path:path}",
        review_manager,
        methods=None,
        name="project-plate-review",
    )
    # Keep the old URL working for existing bookmarks.  It is restricted to
    # the startup project because the old URL has no project namespace.
    app.add_route(
        "/plates/{plate_slug}/{path:path}",
        review_manager,
        methods=None,
        name="legacy-plate-review",
    )
    mounted: list[str] = []
    for plate in manifest.get("plates", []):
        slug = str(plate.get("slug") or _slug(plate.get("board_id", "")))
        if review_manager.resolve_plate(project_id, slug) is not None:
            mounted.append(slug)
    manifest["mounted_plates"] = mounted

    # Bind the project-level mask-review page to the first mounted plate.  The
    # project hub may contain many boards, but a review batch must never mix a
    # board's coordinates with another board's source images.
    default_mask_config: dict[str, Any] | None = None
    default_mask_database: Path | None = None
    default_mask_images: pd.DataFrame | None = None
    selected_mask_plate = next(
        (
            plate
            for plate in manifest.get("plates", [])
            if str(plate.get("slug") or _slug(plate.get("board_id", ""))) in mounted
        ),
        None,
    )
    try:
        if selected_mask_plate is not None:
            selected_mask_config_path = _resolve(selected_mask_plate["config"])
            default_mask_config = load_config(selected_mask_config_path)
            default_mask_database = initialize_database(
                artifact_path(default_mask_config, "annotations", "annotations.db")
            )
            selected_images_path = _resolve(selected_mask_plate.get("images_manifest", ""))
            if selected_images_path.exists():
                default_mask_images = pd.read_csv(selected_images_path)
    except (OSError, KeyError, TypeError, ValueError, pd.errors.ParserError):
        default_mask_config = None
        default_mask_database = None
        default_mask_images = None

    mask_context_cache: dict[str, tuple[dict[str, Any] | None, Path | None, pd.DataFrame | None]] = {}

    def mask_context(project_id: str | None = None) -> tuple[dict[str, Any] | None, Path | None, pd.DataFrame | None]:
        """Load mask-review inputs lazily for the selected project."""

        if not project_id:
            return default_mask_config, default_mask_database, default_mask_images
        selected_path = manifest_for_project(project_id)
        selected_value = _read_manifest(selected_path)
        if selected_value is None:
            raise HTTPException(status_code=404, detail="project manifest not found")
        selected_id = _slug(str(selected_value.get("project_id") or selected_path.parent.name))
        if selected_id in mask_context_cache:
            return mask_context_cache[selected_id]
        selected_plate = next(
            (
                plate for plate in selected_value.get("plates", [])
                if isinstance(plate, dict) and plate.get("config") and plate.get("images_manifest")
            ),
            None,
        )
        config_value: dict[str, Any] | None = None
        database_value: Path | None = None
        images_value: pd.DataFrame | None = None
        try:
            if selected_plate is not None:
                config_value = load_config(_resolve(selected_plate["config"]))
                database_value = initialize_database(
                    artifact_path(config_value, "annotations", "annotations.db")
                )
                images_path = _resolve(selected_plate["images_manifest"])
                if images_path.exists():
                    images_value = pd.read_csv(images_path)
        except (OSError, KeyError, TypeError, ValueError, pd.errors.ParserError):
            config_value = None
            database_value = None
            images_value = None
        context = (config_value, database_value, images_value)
        mask_context_cache[selected_id] = context
        return context

    @app.get("/mask-review", response_class=HTMLResponse)
    def project_mask_review(project_id: str | None = None) -> str:
        page = (ui_root / "mask-review.html").read_text(encoding="utf-8")
        page = page.replace(
            '<meta name="mask-review-base" content="./">',
            '<meta name="mask-review-base" content="/">',
            1,
        )
        page = page.replace(
            '<meta name="project-id" content="">',
            f'<meta name="project-id" content="{html.escape(_slug(project_id or ""), quote=True)}">',
            1,
        )
        selected_back_url = f"/projects/{_slug(project_id)}/" if project_id else "/"
        page = page.replace(
            '<meta name="project-back-url" content="/">',
            f'<meta name="project-back-url" content="{html.escape(selected_back_url, quote=True)}">',
            1,
        )
        return page.replace('href="assets/', 'href="/project-assets/').replace(
            'src="assets/', 'src="/project-assets/'
        )

    @app.get("/projects/{project_id}/mask-review", response_class=HTMLResponse)
    def project_scoped_mask_review(project_id: str) -> str:
        return project_mask_review(project_id)

    @app.get("/api/mask-review-rounds")
    def project_mask_review_rounds(project_id: str | None = None) -> list[dict[str, Any]]:
        selected_config, selected_database, _ = mask_context(project_id)
        if selected_config is None or selected_database is None:
            return []
        return list_mask_review_rounds(selected_config, selected_database)

    @app.get("/api/mask-comparison-options")
    def project_mask_comparison_options(project_id: str | None = None) -> dict[str, Any]:
        selected_config, selected_database, _ = mask_context(project_id)
        if selected_config is None or selected_database is None:
            raise HTTPException(status_code=503, detail="mask review is unavailable")
        try:
            return mask_comparison_options(selected_config, selected_database)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/mask-comparison-round")
    def project_mask_comparison_round(
        payload: ProjectMaskComparisonPayload,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        selected_config, selected_database, _ = mask_context(project_id)
        if selected_config is None or selected_database is None:
            raise HTTPException(status_code=503, detail="mask review is unavailable")
        try:
            return create_model_comparison_round(
                selected_config,
                old_checkpoint=payload.old_checkpoint,
                new_checkpoint=payload.new_checkpoint,
                source_configs=payload.source_configs or None,
                round_id=payload.round_id,
            )
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/mask-review-summary")
    def project_mask_review_summary(round_id: str, project_id: str | None = None) -> dict[str, Any]:
        selected_config, selected_database, _ = mask_context(project_id)
        if selected_config is None or selected_database is None:
            raise HTTPException(status_code=503, detail="mask review is unavailable")
        try:
            return mask_review_summary(selected_config, selected_database, round_id)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/mask-review-candidates")
    def project_mask_review_candidates(
        round_id: str, status: str = "pending", limit: int = 500,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        selected_config, selected_database, _ = mask_context(project_id)
        if selected_config is None or selected_database is None:
            raise HTTPException(status_code=503, detail="mask review is unavailable")
        try:
            return mask_review_candidates(
                selected_config,
                selected_database,
                round_id,
                status=status,
                limit=min(int(limit), 5000),
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/mask-review-candidate")
    def project_mask_review_candidate(
        round_id: str, candidate_id: str, project_id: str | None = None
    ) -> dict[str, Any]:
        selected_config, selected_database, _ = mask_context(project_id)
        if selected_config is None or selected_database is None:
            raise HTTPException(status_code=503, detail="mask review is unavailable")
        try:
            item = mask_review_candidate(
                selected_config,
                selected_database,
                round_id,
                candidate_id,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="mask review candidate not found")
        return item

    @app.post("/api/mask-review-save")
    def project_mask_review_save(
        payload: ProjectMaskReviewSavePayload,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        selected_config, selected_database, _ = mask_context(project_id)
        if selected_config is None or selected_database is None:
            raise HTTPException(status_code=503, detail="mask review is unavailable")
        try:
            result = save_mask_review(
                selected_config,
                selected_database,
                round_id=payload.round_id,
                candidate_id=payload.candidate_id,
                decision=payload.decision,
                reviewed_mask_rle=payload.reviewed_mask_rle,
                reviewer=payload.reviewer,
                notes=payload.notes,
            )
            sync_catalog_review(
                manifest_for_project(project_id) if project_id else manifest_file,
                source="mask_review",
                reviewer=payload.reviewer,
                operation="mask_review_save",
            )
            return result
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/patch")
    def project_mask_patch(
        well: str,
        timepoint: str,
        x: float,
        y: float,
        size: int = 256,
        source_config: str | None = None,
        project_id: str | None = None,
    ) -> Response:
        _, _, image_manifest = mask_context(project_id)
        if source_config:
            try:
                comparison_config = load_config(_resolve(source_config))
                comparison_manifest = (
                    Path(comparison_config["paths"]["artifact_root"])
                    / "manifests"
                    / "images.csv"
                )
                if comparison_manifest.exists():
                    image_manifest = pd.read_csv(comparison_manifest)
            except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, pd.errors.ParserError):
                image_manifest = None
        if image_manifest is None:
            raise HTTPException(status_code=503, detail="image manifest unavailable")
        selected = image_manifest[
            (image_manifest["well"].astype(str).str.upper() == well.upper())
            & (image_manifest["timepoint"].astype(str).str.upper() == timepoint.upper())
            & (image_manifest["decode_status"].astype(str) == "ok")
        ]
        if selected.empty:
            raise HTTPException(status_code=404, detail="image unavailable")
        size = max(64, min(int(size), 2048))
        half = size // 2
        with Image.open(selected.iloc[0]["raw_image_path"]) as image:
            gray = ImageEnhance.Contrast(image.convert("L")).enhance(1.8)
            crop = gray.crop((int(x) - half, int(y) - half, int(x) + half, int(y) + half))
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=90)
        return Response(content=buffer.getvalue(), media_type="image/jpeg")

    def multiplicity_plate_contexts(selected_manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return the lightweight per-board context for the project queue."""

        selected_value = selected_manifest if selected_manifest is not None else manifest
        contexts: list[dict[str, Any]] = []
        for plate in selected_value.get("plates", []):
            slug = str(plate.get("slug") or _slug(plate.get("board_id", "")))
            config_value = plate.get("config")
            if not slug or not config_value:
                continue
            config_path = _resolve(config_value)
            if not config_path.exists():
                continue
            try:
                config = load_config(config_path)
                database = artifact_path(config, "annotations", "annotations.db")
            except (OSError, KeyError, TypeError, ValueError):
                continue
            contexts.append({
                "slug": slug,
                "label": str(plate.get("board_id") or plate.get("group_id") or slug),
                "config": config,
                "database": database,
            })
        return contexts

    @app.get("/single-doublet-review", response_class=HTMLResponse)
    def single_doublet_review(project_id: str | None = None) -> str:
        selected_manifest = manifest if not project_id else _read_manifest(manifest_for_project(project_id))
        if selected_manifest is None:
            raise HTTPException(status_code=404, detail="project manifest not found")
        selected_project_id = str(selected_manifest.get("project_id") or project_id or manifest_file.parent.name)
        selected_project_name = str(selected_manifest.get("project_name") or selected_project_id)
        selected_back_url = f"/projects/{_slug(selected_project_id)}/" if project_id else "/"
        page = (ui_root / "single-doublet-review.html").read_text(encoding="utf-8")
        page = page.replace('content="plate"', 'content="project"', 1)
        page = page.replace(
            '<meta name="project-id" content="">',
            f'<meta name="project-id" content="{html.escape(_slug(selected_project_id), quote=True)}">',
            1,
        )
        page = page.replace(
            '<meta name="project-back-url" content="">',
            f'<meta name="project-back-url" content="{html.escape(selected_back_url, quote=True)}">',
            1,
        )
        page = page.replace(
            '<meta name="project-name" content="">',
            f'<meta name="project-name" content="{html.escape(selected_project_name, quote=True)}">',
            1,
        )
        page = page.replace('href="assets/', 'href="/project-assets/', 1)
        page = page.replace('src="assets/', 'src="/project-assets/', 1)
        return page

    @app.get("/projects/{project_id}/single-doublet-review", response_class=HTMLResponse)
    def project_single_doublet_review(project_id: str) -> str:
        return single_doublet_review(project_id)

    @app.get("/api/multiplicity-training-candidates")
    def multiplicity_training_candidates(
        mode: str = "likely_doublet",
        limit: int = 48,
        category: str | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if mode not in {"likely_doublet", "uncertain", "diverse"}:
            raise HTTPException(status_code=422, detail="Invalid queue mode")
        if category is not None and category not in {
            "single",
            "touching_doublet",
            "cluster_3plus",
            "debris",
            "invalid",
        }:
            raise HTTPException(status_code=422, detail="Invalid multiplicity category")
        selected_manifest = manifest if not project_id else _read_manifest(manifest_for_project(project_id))
        if selected_manifest is None:
            raise HTTPException(status_code=404, detail="project manifest not found")
        selected_project_id = str(selected_manifest.get("project_id") or project_id or manifest_file.parent.name)
        requested = max(1, min(int(limit), 240))
        candidates: list[dict[str, Any]] = []
        for context in multiplicity_plate_contexts(selected_manifest):
            try:
                rows = multiplicity_queue(
                    context["config"],
                    context["database"],
                    mode,
                    min(100, requested),
                    category=category,
                )
            except (OSError, KeyError, TypeError, ValueError, pd.errors.ParserError):
                continue
            for row in rows:
                item = dict(row)
                item["plate_slug"] = context["slug"]
                item["plate_label"] = context["label"]
                item["patch_base"] = f"/projects/{_slug(selected_project_id)}/plates/{context['slug']}"
                item["review_id"] = f"{context['slug']}::{item['candidate_id']}"
                candidates.append(item)

        def numeric(item: dict[str, Any], key: str, default: float = 0.0) -> float:
            try:
                value = item.get(key)
                return default if value is None else float(value)
            except (TypeError, ValueError):
                return default

        def binary_priority(item: dict[str, Any]) -> float:
            """Prefer single/doublet evidence and demote clear 3+ groups."""

            single = numeric(item, "single_probability", -1.0)
            doublet = numeric(item, "touching_doublet_probability", -1.0)
            cluster = numeric(item, "cluster_3plus_probability", -1.0)
            if min(single, doublet, cluster) < 0:
                return numeric(item, "doublet_priority")
            return max(single, doublet) - 0.35 * cluster

        for item in candidates:
            item["_binary_priority"] = binary_priority(item)

        if category is not None:
            # ``multiplicity_uncertainty`` is unrelated to the debris/wall
            # buckets.  Keep the category-specific low-confidence ordering
            # produced by ``multiplicity_queue`` for every tab.
            candidates.sort(
                key=lambda item: (
                    -numeric(item, "category_priority"),
                    -numeric(item, "multiplicity_uncertainty", 0.0),
                )
            )
        elif mode == "uncertain":
            candidates.sort(
                key=lambda item: (
                    -numeric(item, "multiplicity_uncertainty", 0.0),
                    -numeric(item, "_binary_priority"),
                )
            )
        elif mode == "diverse":
            candidates_by_plate: dict[str, list[dict[str, Any]]] = {}
            for item in candidates:
                candidates_by_plate.setdefault(str(item["plate_slug"]), []).append(item)
            for bucket in candidates_by_plate.values():
                bucket.sort(key=lambda item: -numeric(item, "_binary_priority"))
            candidates = []
            buckets = list(candidates_by_plate.values())
            while buckets and len(candidates) < requested:
                next_buckets: list[list[dict[str, Any]]] = []
                for bucket in buckets:
                    if bucket:
                        candidates.append(bucket.pop(0))
                    if bucket:
                        next_buckets.append(bucket)
                    if len(candidates) >= requested:
                        break
                buckets = next_buckets
        else:
            candidates.sort(
                key=lambda item: (
                    -numeric(item, "_binary_priority"),
                    -numeric(item, "cell_probability"),
                )
            )
        for item in candidates:
            item.pop("_binary_priority", None)
        return [_safe(item) for item in candidates[:requested]]

    @app.get("/api/multiplicity-training-stats")
    def multiplicity_training_stats(project_id: str | None = None) -> dict[str, Any]:
        selected_manifest = manifest if not project_id else _read_manifest(manifest_for_project(project_id))
        if selected_manifest is None:
            raise HTTPException(status_code=404, detail="project manifest not found")
        totals = {
            "single": 0,
            "touching_doublet": 0,
            "cluster_3plus": 0,
            "debris": 0,
            "invalid": 0,
            "approved": 0,
            "not_cell": 0,
            "skip": 0,
        }
        plates: list[dict[str, Any]] = []
        for context in multiplicity_plate_contexts(selected_manifest):
            stats = multiplicity_stats(context["database"])
            counts = stats.get("counts", {})
            for label in totals:
                totals[label] += int(counts.get(label, 0) or 0)
            plates.append({
                "plate_slug": context["slug"],
                "plate_label": context["label"],
                "total": int(stats.get("total", 0) or 0),
                "counts": counts,
            })
        return {
            "total": int(sum(totals.values())),
            "counts": totals,
            "recommended_minimums": {
                "single": 40,
                "touching_doublet": 20,
                "cluster_3plus": 10,
                "debris": 0,
                "invalid": 0,
                "approved": 0,
                "not_cell": 20,
            },
            "plates": plates,
        }

    @app.post("/api/multiplicity-training-labels")
    def multiplicity_training_labels(
        payload: ProjectMultiplicityLabelsPayload,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        selected_manifest = manifest if not project_id else _read_manifest(manifest_for_project(project_id))
        if selected_manifest is None:
            raise HTTPException(status_code=404, detail="project manifest not found")
        contexts = {item["slug"]: item for item in multiplicity_plate_contexts(selected_manifest)}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in payload.items:
            if item.plate_slug not in contexts:
                raise HTTPException(status_code=422, detail="Unknown plate")
            grouped.setdefault(item.plate_slug, []).append(
                item.model_dump(exclude={"plate_slug"})
            )
        saved = 0
        by_plate: dict[str, int] = {}
        for slug, items in grouped.items():
            count = save_categorized_review_labels(
                contexts[slug]["database"], items, payload.reviewer
            )
            saved += int(count)
            by_plate[slug] = int(count)
            sync_catalog_review(
                manifest_for_project(project_id) if project_id else manifest_file,
                slug,
                wells={str(item.get("well") or "").upper() for item in items if item.get("well")},
                source="multiplicity_review",
                reviewer=payload.reviewer,
                operation="multiplicity_review_save",
            )
        return {"status": "saved", "saved": saved, "by_plate": by_plate}

    @app.delete("/api/multiplicity-training-labels/{plate_slug}/{candidate_id}")
    def multiplicity_training_label_delete(
        plate_slug: str, candidate_id: str, project_id: str | None = None
    ) -> dict[str, Any]:
        selected_manifest = manifest if not project_id else _read_manifest(manifest_for_project(project_id))
        if selected_manifest is None:
            raise HTTPException(status_code=404, detail="project manifest not found")
        contexts = {item["slug"]: item for item in multiplicity_plate_contexts(selected_manifest)}
        context = contexts.get(plate_slug)
        if context is None:
            raise HTTPException(status_code=404, detail="Unknown plate")
        ensure_multiplicity_table(context["database"])
        with sqlite3.connect(context["database"]) as connection:
            cursor = connection.execute(
                "DELETE FROM multiplicity_labels WHERE candidate_id = ?",
                (candidate_id,),
            )
            deleted = int(cursor.rowcount or 0)
        return {"status": "deleted", "deleted": deleted}

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        # The landing page is deliberately project-level.  Opening a board is
        # an explicit second click, so a new task can never unexpectedly
        # replace the currently selected project.
        return (ui_root / "project-list.html").read_text(encoding="utf-8")

    @app.get("/projects/{project_id}/", response_class=HTMLResponse)
    def project_detail_page(project_id: str) -> str:
        target = _slug(project_id)
        # Re-scan here instead of using the startup snapshot: creating a task
        # writes a new project manifest while the hub is already running.
        available = {
            _slug(str((_read_manifest(path) or {}).get("project_id", path.parent.name))): path
            for path in _project_manifest_paths(manifest_file)
        }
        if target not in available:
            raise HTTPException(status_code=404, detail="project not found")
        page = (ui_root / "project-dashboard.html").read_text(encoding="utf-8")
        return page.replace("</head>", f'<meta name="project-id" content="{target}"></head>', 1)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        loaded = [
            {"project_id": project, "plate_slug": plate}
            for project, plate in review_manager.loaded_keys
        ]
        return {
            "status": "ok",
            "project": project_name,
            "mounted_plates": mounted,
            "loaded_plates": [item["plate_slug"] for item in loaded if item["project_id"] == project_id],
            "loaded_review_apps": loaded,
            "project_count": len(_project_manifest_paths(manifest_file)),
        }

    @app.get("/api/review-platform/status")
    def installed_review_platform_status() -> dict[str, Any]:
        if os.environ.get("CELLVISION_REVIEW_PLATFORM") != "1":
            raise HTTPException(status_code=404, detail="review platform is not enabled")
        from .review_platform import review_platform_status

        return review_platform_status()

    @app.post("/api/review-platform/import-package")
    def import_review_platform_package() -> dict[str, Any]:
        if os.environ.get("CELLVISION_REVIEW_PLATFORM") != "1":
            raise HTTPException(status_code=404, detail="review platform is not enabled")
        from .review_platform import (
            choose_review_package,
            register_review_package,
            review_platform_status,
            write_review_hub_manifest,
        )

        selected = choose_review_package()
        if selected is None:
            return {"status": "cancelled", **review_platform_status()}
        try:
            entrypoint, metadata = register_review_package(selected)
            write_review_hub_manifest()
            catalog.sync_manifest(entrypoint, force=True, source="offline_review_import")
            refresh_catalog(force=True)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "status": "imported",
            "project_id": str(metadata.get("project_id") or ""),
            "project_name": str(metadata.get("project_name") or metadata.get("project_id") or ""),
            **review_platform_status(),
        }

    @app.get("/api/catalog/status")
    def catalog_status() -> dict[str, Any]:
        refresh_catalog(force=True)
        return catalog.status()

    @app.get("/api/project/catalog-wells")
    def project_catalog_wells(
        project_id: str,
        plate_slug: str,
        category: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        selected_manifest = manifest_for_project(project_id)
        selected_value = _read_manifest(selected_manifest) or {}
        selected_project_id = str(selected_value.get("project_id") or selected_manifest.parent.name)
        plate_id = f"{selected_project_id}:{_slug(plate_slug)}"
        return catalog.list_wells(plate_id, category=category, limit=limit)

    @app.get("/api/ready")
    def ready() -> dict[str, Any]:
        """Readiness probe for a reverse proxy or service manager."""

        missing = []
        if not manifest_file.exists():
            missing.append(str(manifest_file))
        if not ui_root.exists():
            missing.append(str(ui_root))
        if missing:
            raise HTTPException(status_code=503, detail={"missing": missing})
        return {
            "status": "ready",
            "instance_id": _production_instance_id(),
            "project": project_name,
            "plate_count": len(mounted),
            "project_count": len(_project_manifest_paths(manifest_file)),
        }

    @app.get("/api/projects")
    def projects(
        q: str = "",
        page: int = 1,
        page_size: int = 10,
    ) -> dict[str, Any]:
        refresh_catalog()
        try:
            return catalog.list_projects(query=q, page=page, page_size=page_size)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            values = [_project_card(path) for path in _project_manifest_paths(manifest_file)]
            needle = q.strip().casefold()
            if needle:
                values = [
                    item for item in values
                    if needle in json.dumps(item, ensure_ascii=False).casefold()
                ]
            safe_page_size = max(1, min(100, int(page_size or 10)))
            safe_page = max(1, int(page or 1))
            pages = max(1, (len(values) + safe_page_size - 1) // safe_page_size)
            safe_page = min(safe_page, pages)
            start = (safe_page - 1) * safe_page_size
            return {
                "items": values[start : start + safe_page_size],
                "page": safe_page,
                "page_size": safe_page_size,
                "total": len(values),
                "pages": pages,
                "query": q,
                "aggregate": {"project_count": len(values)},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    @app.get("/api/project")
    def project(project_id: str | None = None) -> dict[str, Any]:
        refresh_catalog()
        selected_manifest = manifest
        selected_path = manifest_file
        if project_id:
            wanted = _slug(project_id)
            for path in _project_manifest_paths(manifest_file):
                candidate = _read_manifest(path) or {}
                candidate_id = _slug(str(candidate.get("project_id") or path.parent.name))
                if candidate_id == wanted:
                    selected_manifest = candidate
                    selected_path = path
                    break
            else:
                raise HTTPException(status_code=404, detail="project not found")
        selected_project_id = str(selected_manifest.get("project_id") or selected_path.parent.name)
        try:
            catalog_detail = catalog.project_detail(selected_project_id)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            catalog_detail = None
        if catalog_detail is not None and catalog_detail.get("plates"):
            catalog_plates = catalog_detail["plates"]
            manifest_plates = {
                str(item.get("slug") or _slug(item.get("board_id", ""))): item
                for item in selected_manifest.get("plates", [])
                if isinstance(item, dict)
            }
            verdict_totals = {"approved": 0, "pending": 0, "rejected": 0, "unclassified": 0}
            for item in catalog_plates:
                source_plate = manifest_plates.get(str(item.get("plate_slug") or item.get("slug") or ""), {})
                verdict_counts = _manual_verdict_counts_for_plate(source_plate)
                item["manual_verdict_counts"] = verdict_counts
                for decision, value in verdict_counts.items():
                    verdict_totals[decision] += int(value)
            mounted = [
                str(item.get("plate_slug"))
                for item in catalog_plates
                if review_manager.resolve_plate(selected_project_id, str(item.get("plate_slug"))) is not None
            ]
            return {
                **catalog_detail,
                "root": selected_manifest.get("root"),
                "portable_review": bool(
                    selected_manifest.get("portable_review")
                    or os.environ.get("CELLVISION_PORTABLE_REVIEW") == "1"
                ),
                "mounted_plates": mounted,
                "manual_verdict_counts": verdict_totals,
                "detection_start_date": _project_detection_dates(selected_manifest)[0],
                "detection_end_date": _project_detection_dates(selected_manifest)[1],
            }
        plates = [_plate_summary(item) for item in selected_manifest.get("plates", [])]
        selected_mounted = [
            str(item.get("slug") or _slug(item.get("board_id", "")))
            for item in selected_manifest.get("plates", [])
            if isinstance(item, dict)
            and review_manager.resolve_plate(
                selected_project_id,
                str(item.get("slug") or _slug(item.get("board_id", ""))),
            ) is not None
        ]
        aggregate: dict[str, int] = {}
        verdict_totals = {"approved": 0, "pending": 0, "rejected": 0, "unclassified": 0}
        for item in plates:
            for key, value in item["category_counts"].items():
                aggregate[key] = aggregate.get(key, 0) + int(value)
            for decision, value in item.get("manual_verdict_counts", {}).items():
                verdict_totals[decision] += int(value)
        detection_start, detection_end = _project_detection_dates(selected_manifest)
        return {
            "project_id": selected_manifest.get("project_id", selected_path.parent.name),
            "project_name": selected_manifest.get("project_name", selected_path.parent.name),
            "root": selected_manifest.get("root"),
            "plate_count": len(plates),
            "mounted_plates": selected_mounted,
            "recognized_plate_count": int(sum(item.get("status") == "completed" for item in plates)),
            "reviewed_plate_count": int(sum(bool(item.get("review_complete")) for item in plates)),
            "category_counts": aggregate,
            "manual_verdict_counts": verdict_totals,
            "plates": plates,
            "single_cell_origin_well_count": int(aggregate.get("single_cell_origin", 0)),
            "detection_start_date": detection_start,
            "detection_end_date": detection_end,
            "created_by": str(
                selected_manifest.get("created_by")
                or selected_manifest.get("creator")
                or selected_manifest.get("owner")
                or ""
            ),
            "generated_at": selected_manifest.get("generated_at"),
            "portable_review": bool(
                selected_manifest.get("portable_review")
                or os.environ.get("CELLVISION_PORTABLE_REVIEW") == "1"
            ),
        }

    @app.get("/api/project/review-filter-counts")
    def project_review_filter_counts(
        project_id: str | None = None,
        coverage_min: float | None = None,
        debris_max: int | None = None,
        day0_cells_min: int | None = None,
        day0_cells_max: int | None = None,
        day1_cells_min: int | None = None,
        day1_cells_max: int | None = None,
        day2_cells_min: int | None = None,
        day2_cells_max: int | None = None,
    ) -> dict[str, Any]:
        for day, minimum, maximum in (
            (0, day0_cells_min, day0_cells_max),
            (1, day1_cells_min, day1_cells_max),
            (2, day2_cells_min, day2_cells_max),
        ):
            if minimum is not None and maximum is not None and minimum > maximum:
                raise HTTPException(
                    status_code=422,
                    detail=f"Day{day} cell minimum exceeds maximum",
                )
        selected_path = manifest_for_project(project_id)
        selected_manifest = _read_manifest(selected_path)
        if selected_manifest is None:
            raise HTTPException(status_code=404, detail="project manifest not found")
        return _project_review_filter_counts(
            selected_manifest,
            selected_path,
            coverage_min=coverage_min,
            debris_max=debris_max,
            day0_cells_min=day0_cells_min,
            day0_cells_max=day0_cells_max,
            day1_cells_min=day1_cells_min,
            day1_cells_max=day1_cells_max,
            day2_cells_min=day2_cells_min,
            day2_cells_max=day2_cells_max,
        )

    @app.get("/api/project/export-results")
    def export_project_results(project_id: str | None = None) -> FileResponse:
        """Download one workbook containing every board's well-level results."""

        selected_path = manifest_for_project(project_id)
        selected_manifest = _read_manifest(selected_path)
        if selected_manifest is None:
            raise HTTPException(status_code=404, detail="project manifest not found")
        selected_project_id = str(
            selected_manifest.get("project_id") or selected_path.parent.name
        )

        # A project can be renamed independently from the task that created it.
        # Prefer the latest task label for the workbook's task-name column, then
        # fall back to the project display name for legacy projects.
        task_name = _export_text(selected_manifest.get("task_name"))
        try:
            task_values = task_store_for_manifest(selected_path).list()
        except (OSError, TypeError, ValueError):
            task_values = []
        if task_values:
            latest_task = max(
                task_values,
                key=lambda item: str(item.get("created_at") or item.get("updated_at") or ""),
            )
            task_name = _export_text(latest_task.get("name")) or task_name

        category_overrides: dict[tuple[str, str], str] = {}
        refresh_catalog(force=True)
        for plate in selected_manifest.get("plates", []):
            if not isinstance(plate, dict):
                continue
            slug = _export_text(plate.get("slug")) or _export_text(plate.get("board_id"))
            if not slug:
                continue
            try:
                catalog_rows = catalog.list_wells(
                    f"{selected_project_id}:{slug}", limit=10000
                )
            except (OSError, sqlite3.Error, ValueError, TypeError):
                catalog_rows = []
            for row in catalog_rows:
                well = _export_text(row.get("well")).upper()
                category = _export_text(row.get("category_code"))
                if well and category:
                    category_overrides[(slug, well)] = category

        temporary = NamedTemporaryFile(
            prefix="cellvision-export-", suffix=".xlsx", delete=False
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        try:
            summary = _write_project_result_excel(
                selected_manifest,
                selected_path,
                temporary_path,
                task_name=task_name,
                category_overrides=category_overrides,
            )
        except (OSError, ValueError, KeyError, ImportError) as exc:
            _remove_export_file(str(temporary_path))
            raise HTTPException(status_code=500, detail=f"导出检测结果失败：{exc}") from exc
        if int(summary.get("row_count", 0)) == 0:
            _remove_export_file(str(temporary_path))
            raise HTTPException(status_code=404, detail="当前项目没有可导出的检测结果")
        download_name = _export_filename(task_name or selected_manifest.get("project_name"))
        return FileResponse(
            temporary_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=download_name,
            background=BackgroundTask(_remove_export_file, str(temporary_path)),
        )

    def completed_task_for_offline_review(
        task_id_value: str,
        project_id_value: str | None,
    ) -> tuple[dict[str, Any], Path]:
        """Resolve a completed task and its owning project manifest."""

        matches = [
            item
            for item in tasks_for_project(project_id_value)
            if str(item.get("task_id") or "") == str(task_id_value)
        ]
        if not matches:
            raise HTTPException(status_code=404, detail="任务不存在")
        task = matches[0]
        if str(task.get("status") or "") != "completed":
            raise HTTPException(status_code=409, detail="只能导出已完成任务的离线审核包")
        task_manifest_value = str(task.get("project_manifest") or "")
        selected_manifest = manifest_for_project(project_id_value)
        if task_manifest_value:
            candidate = Path(task_manifest_value).expanduser()
            task_manifest = (
                candidate.resolve()
                if candidate.is_absolute()
                else (selected_manifest.parent / candidate).resolve()
            )
        else:
            task_manifest = selected_manifest
        if not task_manifest.is_file():
            raise HTTPException(status_code=404, detail="任务所属项目清单不存在")
        return task, task_manifest

    def public_offline_export_state(value: Any) -> dict[str, Any]:
        state = dict(value) if isinstance(value, dict) else {}
        state.pop("bundle_path", None)
        return state

    def update_offline_export_state(
        store: TaskQueueStore,
        task_id_value: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        value = {**state, "updated_at": datetime.now(timezone.utc).isoformat()}
        store.update(task_id_value, offline_export=value)
        return value

    @app.post("/api/project/tasks/{task_id_value}/offline-review-export")
    def start_offline_review_export(
        task_id_value: str,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Start one observable offline-review bundle build."""

        task, task_manifest = completed_task_for_offline_review(
            task_id_value, project_id
        )
        store = task_store_for_manifest(task_manifest)
        previous = task.get("offline_export")
        release_commit_file = PROJECT_ROOT / "RELEASE_GIT_COMMIT.txt"
        current_release_commit = (
            release_commit_file.read_text(encoding="utf-8").strip()
            if release_commit_file.is_file()
            else "development"
        )
        current_export_signature = project_export_signature(
            task_manifest.parent, current_release_commit
        )
        if isinstance(previous, dict):
            previous_status = str(previous.get("status") or "")
            previous_path_value = str(previous.get("package_path") or "")
            previous_path = Path(previous_path_value) if previous_path_value else None
            previous_summary = previous.get("summary")
            previous_commit = (
                str(previous_summary.get("git_commit") or "")
                if isinstance(previous_summary, dict)
                else ""
            )
            previous_format = (
                str(previous_summary.get("format") or "")
                if isinstance(previous_summary, dict)
                else ""
            )
            previous_signature = (
                str(previous_summary.get("export_signature") or "")
                if isinstance(previous_summary, dict)
                else ""
            )
            if previous_status == "running":
                return public_offline_export_state(previous)
            if (
                previous_status == "completed"
                and previous_path is not None
                and previous_path.is_dir()
                and previous_commit == current_release_commit
                and previous_format == DATA_PACKAGE_FORMAT
                and previous_signature == current_export_signature
            ):
                return public_offline_export_state(previous)

        job_id = f"offline-{task_id()}"
        initial = update_offline_export_state(store, task_id_value, {
            "job_id": job_id,
            "status": "running",
            "progress_current": 0,
            "progress_total": 0,
            "progress_percent": 0,
            "progress_message": "正在读取项目数据",
            "started_at": datetime.now(timezone.utc).isoformat(),
        })

        def build_bundle() -> None:
            last_percent = -1

            def report_progress(current: int, total: int, message: str) -> None:
                nonlocal last_percent
                percent = int(round(current / total * 100)) if total else 0
                percent = min(99, max(0, percent))
                if percent == last_percent and current < total:
                    return
                last_percent = percent
                update_offline_export_state(store, task_id_value, {
                    **initial,
                    "status": "running",
                    "progress_current": int(current),
                    "progress_total": int(total),
                    "progress_percent": percent,
                    "progress_message": str(message),
                })

            try:
                package_path, summary = prepare_review_data_package(
                    task_manifest,
                    task,
                    git_commit=current_release_commit,
                    progress_callback=report_progress,
                )
                update_offline_export_state(store, task_id_value, {
                    **initial,
                    "status": "completed",
                    "progress_current": int(summary.get("copied_bytes", 1) or 1),
                    "progress_total": int(summary.get("copied_bytes", 1) or 1),
                    "progress_percent": 100,
                    "progress_message": "无环境依赖的 .cvreview 审核数据包已准备好",
                    "package_path": str(package_path),
                    "summary": summary,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as exc:
                update_offline_export_state(store, task_id_value, {
                    **initial,
                    "status": "error",
                    "progress_message": f"生成失败：{type(exc).__name__}: {exc}",
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                })

        Thread(
            target=build_bundle,
            name=f"cellvision-{job_id}",
            daemon=True,
        ).start()
        return public_offline_export_state(initial)

    @app.get("/api/project/tasks/{task_id_value}/offline-review-export")
    def offline_review_export_progress(
        task_id_value: str,
        project_id: str | None = None,
        job_id: str = "",
    ) -> dict[str, Any]:
        task, _ = completed_task_for_offline_review(task_id_value, project_id)
        state = task.get("offline_export")
        if not isinstance(state, dict):
            return {"status": "idle", "progress_percent": 0}
        if job_id and str(state.get("job_id") or "") != job_id:
            raise HTTPException(status_code=404, detail="离线审核包导出任务不存在")
        return public_offline_export_state(state)

    @app.post("/api/project/tasks/{task_id_value}/import-offline-review")
    def import_offline_review(
        task_id_value: str,
        payload: dict[str, Any],
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply a result JSON exported by a portable review bundle."""

        task, task_manifest = completed_task_for_offline_review(
            task_id_value, project_id
        )
        try:
            result = import_offline_review_results(task_manifest, task, payload)
        except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        sync_catalog_review(
            task_manifest,
            source="offline_review",
            reviewer=str(result.get("reviewer") or "offline_reviewer"),
            operation="offline_review_import",
        )
        return result

    @app.get("/api/project/tasks/{task_id_value}/export-offline-review-results")
    def export_offline_review_result_file(
        task_id_value: str,
        project_id: str | None = None,
    ) -> Response:
        """Export reviews created by the normal UI as an importable JSON file."""

        task, task_manifest = completed_task_for_offline_review(
            task_id_value, project_id
        )
        payload = export_offline_review_results(task_manifest, task)
        filename = (
            f"{_slug(str(task.get('name') or task_id_value)) or 'cellvision'}"
            ".cvreview-result.json"
        )
        return Response(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/project/tasks")
    def tasks(project_id: str | None = None) -> list[dict[str, Any]]:
        return tasks_for_project(project_id)

    @app.get("/api/project/worker-runtime")
    def worker_runtime(project_id: str | None = None) -> dict[str, Any]:
        """Expose the adaptive worker's last hardware/status snapshot."""

        selected_manifest = manifest_for_project(project_id)
        shared_runtime_path = (
            catalog_path_for_manifest(selected_manifest).parent / "worker_runtime.json"
        )
        legacy_runtime_path = _queue_path(selected_manifest).parent / "worker_runtime.json"
        runtime_path = (
            shared_runtime_path
            if shared_runtime_path.exists()
            else legacy_runtime_path
        )
        if not runtime_path.exists():
            return {
                "status": "offline",
                "selected_device": "unknown",
                "message": "项目 worker 尚未启动",
            }
        try:
            value = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "status": "unknown",
                "selected_device": "unknown",
                "message": "worker 状态文件暂不可读",
            }
        return value if isinstance(value, dict) else {
            "status": "unknown",
            "selected_device": "unknown",
        }

    @app.patch("/api/project/{project_id}")
    def rename_project(project_id: str, payload: ProjectRenamePayload) -> dict[str, Any]:
        selected_path = _find_project_manifest(manifest_file, project_id)
        if selected_path is None:
            raise HTTPException(status_code=404, detail="project not found")
        value = _read_manifest(selected_path)
        if value is None:
            raise HTTPException(status_code=404, detail="project manifest not found")
        project_name = payload.project_name.strip()
        if not project_name:
            raise HTTPException(status_code=422, detail="project name cannot be empty")
        value["project_name"] = project_name
        value["renamed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(selected_path, value)
        try:
            catalog.sync_manifest(selected_path, force=True, source="project_renamed")
        except (OSError, sqlite3.Error, ValueError, TypeError):
            pass
        return {
            "status": "renamed",
            "project_id": str(value.get("project_id") or selected_path.parent.name),
            "project_name": project_name,
        }

    @app.delete("/api/project/{project_id}")
    def delete_empty_project(project_id: str) -> dict[str, Any]:
        selected_path = _find_project_manifest(manifest_file, project_id)
        if selected_path is None:
            raise HTTPException(status_code=404, detail="project not found")
        if selected_path.resolve() == manifest_file.resolve():
            raise HTTPException(status_code=409, detail="主项目不能删除")
        value = _read_manifest(selected_path)
        if value is None:
            raise HTTPException(status_code=404, detail="project manifest not found")
        plates = value.get("plates") if isinstance(value.get("plates"), list) else []
        if plates:
            raise HTTPException(status_code=409, detail="只有内容为空、没有板子的项目可以删除")

        related_entries: list[tuple[TaskQueueStore, dict[str, Any]]] = []
        for _, store in task_store_entries():
            related_entries.extend(
                (store, task)
                for task in store.list()
                if Path(str(task.get("project_manifest", ""))).expanduser().resolve()
                == selected_path.resolve()
            )
        related = [task for _, task in related_entries]
        blocked = [
            task for task in related
            if str(task.get("status")) in {"running", "completed"}
        ]
        if blocked:
            raise HTTPException(status_code=409, detail="关联任务正在运行或已完成，不能删除项目")

        selected_path.unlink(missing_ok=False)
        selected_project_id = str(value.get("project_id") or project_id)
        try:
            catalog.mark_project_deleted(selected_project_id)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            pass
        deleted_tasks: list[str] = []
        plan_root = (selected_path.parent / "task_plans").resolve()
        for task_store, task in related_entries:
            task_key = str(task.get("task_id", ""))
            if task_key and task_store.delete(task_key) is not None:
                deleted_tasks.append(task_key)
            plan_value = task.get("plan_path")
            if plan_value:
                plan_path = Path(str(plan_value)).expanduser().resolve()
                if plan_path.parent == plan_root:
                    plan_path.unlink(missing_ok=True)
        try:
            selected_path.parent.rmdir()
        except OSError:
            # Generated artifacts or an empty parent retained by the OS are
            # harmless; the manifest is the project registry entry.
            pass
        return {
            "status": "deleted",
            "project_id": str(value.get("project_id") or project_id),
            "project_name": str(value.get("project_name") or project_id),
            "deleted_tasks": deleted_tasks,
        }

    @app.post("/api/project/browse-folder")
    def browse_folder() -> dict[str, str]:
        """Open a Windows folder picker from an explicit user action."""

        if os.environ.get("CELLVISION_MACHINE_SERVICE") == "1":
            return {
                "path": "",
                "error": "当前账号的桌面桥接未运行；请从共享安装目录双击 Start-CellVision.cmd，或直接输入共享数据路径。",
            }
        if os.name != "nt":
            return {"path": ""}
        focus_helper = r"""
using System;
using System.Text;
using System.Threading;
using System.Runtime.InteropServices;

public static class CellVisionWindowFocus
{
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    [DllImport("user32.dll")]
    private static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetClassName(IntPtr hWnd, StringBuilder className, int maxCount);

    [DllImport("user32.dll")]
    private static extern bool BringWindowToTop(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool SetWindowPos(
        IntPtr hWnd,
        IntPtr insertAfter,
        int x,
        int y,
        int width,
        int height,
        uint flags);

    private static readonly IntPtr HwndTopmost = new IntPtr(-1);
    private const uint SwpNoSize = 0x0001;
    private const uint SwpNoMove = 0x0002;
    private const uint SwpShowWindow = 0x0040;
    private static System.Threading.Timer promotionTimer;

    public static void Start(uint targetProcessId)
    {
        Stop();
        promotionTimer = new System.Threading.Timer(
            unused => Promote(targetProcessId),
            null,
            0,
            50);
    }

    public static void Stop()
    {
        var timer = promotionTimer;
        promotionTimer = null;
        if (timer != null)
        {
            timer.Dispose();
        }
    }

    public static void Promote(uint targetProcessId)
    {
        IntPtr dialogHandle = IntPtr.Zero;
        EnumWindows((hWnd, unused) =>
        {
            uint processId;
            GetWindowThreadProcessId(hWnd, out processId);
            if (processId != targetProcessId || !IsWindowVisible(hWnd))
            {
                return true;
            }

            var className = new StringBuilder(128);
            GetClassName(hWnd, className, className.Capacity);
            if (string.Equals(className.ToString(), "#32770", StringComparison.Ordinal))
            {
                dialogHandle = hWnd;
                return false;
            }

            return true;
        }, IntPtr.Zero);

        if (dialogHandle == IntPtr.Zero)
        {
            return;
        }

        SetWindowPos(
            dialogHandle,
            HwndTopmost,
            0,
            0,
            0,
            0,
            SwpNoMove | SwpNoSize | SwpShowWindow);
        BringWindowToTop(dialogHandle);
        SetForegroundWindow(dialogHandle);
    }
}
"""
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "Add-Type -AssemblyName System.Drawing;"
            "Add-Type -TypeDefinition @'\n"
            + focus_helper
            + "'@;"
            "[System.Windows.Forms.Application]::EnableVisualStyles();"
            "$owner=New-Object System.Windows.Forms.Form;"
            "$owner.FormBorderStyle='None';"
            "$owner.StartPosition='Manual';"
            "$owner.Location=New-Object System.Drawing.Point -ArgumentList -10,-10;"
            "$owner.Size=New-Object System.Drawing.Size -ArgumentList 1,1;"
            "$owner.Opacity=0;"
            "$owner.ShowInTaskbar=$false;"
            "$owner.TopMost=$true;"
            "$owner.Show();"
            "$owner.Activate();"
            "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
            "$d.Description='选择包含 sessions.idx 的数据文件夹';"
            "$d.ShowNewFolderButton=$false;"
            "[CellVisionWindowFocus]::Start([uint32]$PID);"
            "try{if($d.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK){$d.SelectedPath}}"
            "finally{[CellVisionWindowFocus]::Stop();$d.Dispose();$owner.Close();$owner.Dispose()}"
        )
        try:
            # The picker is a GUI dialog; its PowerShell host must not create
            # a console window of its own.  CREATE_NO_WINDOW covers launches
            # from a console-less service, while SW_HIDE/Hidden also covers
            # Windows PowerShell's normal startup-window behavior.
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-STA",
                    "-WindowStyle",
                    "Hidden",
                    "-Command",
                    script,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(
                f"Folder picker failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return {"path": "", "error": "无法启动 Windows 文件夹选择器，请直接输入文件夹路径。"}
        selected_path = completed.stdout.strip()
        if selected_path:
            return {"path": selected_path}
        error = completed.stderr.strip()
        return {
            "path": "",
            "error": error or "未选择文件夹；也可以直接输入文件夹路径。",
        }

    @app.post("/api/project/analyze-folder")
    def analyze_folder(payload: FolderPayload) -> dict[str, Any]:
        return _parse_folder(payload.path)

    @app.post("/api/project/tasks")
    def add_task(payload: TaskPayload) -> dict[str, Any]:
        analysis = _parse_folder(payload.path)
        folder_name = str(analysis.get("folder_name") or _folder_name(analysis["root"]))
        task_name = payload.name.strip() or folder_name
        project_name = folder_name
        created_by = payload.created_by.strip() or "未填写"
        selected, endpoint_label, endpoint_day = _validate_timepoint_selection(
            analysis["timepoint_options"], payload.selected_timepoint_labels
        )
        coverage = _complete_group_timepoint_selection(
            analysis.get("session_records", []), selected
        )
        if not coverage["included_group_ids"]:
            raise HTTPException(
                status_code=422,
                detail="所选时间点没有完整一致的板子，无法创建计算任务。",
            )
        selected_sessions = coverage["sessions"]
        endpoint_timepoint_labels = sorted({
            str(row.get("timepoint_label"))
            for row in selected_sessions
            if str(row.get("day_label", "")).casefold() == endpoint_label.casefold()
            and row.get("timepoint_label")
        })
        selected_dates = sorted({
            str(
                row.get("acquisition_date")
                or row.get("acquisition_datetime")
                or ""
            )[:10]
            for row in selected_sessions
            if len(str(
                row.get("acquisition_date")
                or row.get("acquisition_datetime")
                or ""
            )) >= 10
        })
        detection_start_date = selected_dates[0] if selected_dates else None
        detection_end_date = selected_dates[-1] if selected_dates else None
        now = datetime.now(timezone.utc).isoformat()
        new_task_id = task_id()
        project_id = _slug(project_name) or new_task_id
        project_dir = manifest_file.parent.parent / project_id
        project_manifest_path = project_dir / "project.json"
        if project_manifest_path.exists():
            project_id = f"{project_id}-{new_task_id.rsplit('-', 1)[-1]}"
            project_dir = manifest_file.parent.parent / project_id
            project_manifest_path = project_dir / "project.json"
        project_dir.mkdir(parents=True, exist_ok=True)
        plan_dir = project_dir / "task_plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / f"{new_task_id}.json"
        plan = {
            "task_id": new_task_id,
            "name": task_name,
            "project_name": project_name,
            "created_by": created_by,
            "root": analysis["root"],
            "index": analysis["index"],
            "selected_timepoint_labels": selected,
            "endpoint_day_label": endpoint_label,
            "endpoint_day_number": endpoint_day,
            "endpoint_timepoint_labels": endpoint_timepoint_labels,
            "early_timepoint_labels": ["Day0", "Day1", "Day2"],
            "endpoint_gate": {
                "enabled": True,
                "rule": "endpoint_obvious_sheet_growth=false 的孔跳过 T0-T2 深度计算",
                "endpoint_day_label": endpoint_label,
            },
            "sessions": selected_sessions,
            "included_group_ids": coverage["included_group_ids"],
            "excluded_groups": coverage["excluded_groups"],
            "created_at": now,
        }
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        project_manifest_path.write_text(
            json.dumps({
                "project_id": project_id,
                "project_name": project_name,
                "created_by": created_by,
                "root": str(project_dir),
                "source": {
                    "root": analysis["root"],
                    "index": analysis["index"],
                    "access_policy": "ingest_only",
                },
                "image_storage": {
                    "mode": "project_owned_after_endpoint_gate",
                    "root": str(project_dir / "data" / "images"),
                    "no_growth_images_retained": False,
                },
                "generated_at": now,
                "detection_start_date": detection_start_date,
                "detection_end_date": detection_end_date,
                "task_id": new_task_id,
                "status": "queued",
                "selected_timepoint_labels": selected,
                "endpoint_day_label": endpoint_label,
                "endpoint_day_number": endpoint_day,
                "plates": [],
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        task = {
            "task_id": new_task_id,
            "name": task_name,
            "project_name": project_name,
            "created_by": created_by,
            "path": str(project_dir),
            "source_path": analysis["root"],
            "index": analysis["index"],
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "group_count": coverage["included_group_count"],
            "session_count": len(selected_sessions),
            "source_group_count": analysis["group_count"],
            "source_session_count": analysis["session_count"],
            "excluded_group_count": coverage["excluded_group_count"],
            "excluded_groups": coverage["excluded_groups"],
            "timepoint_labels": analysis["timepoint_labels"],
            "day_labels": analysis["day_labels"],
            "timepoint_options": analysis["timepoint_options"],
            "selected_timepoint_labels": selected,
            "endpoint_day_label": endpoint_label,
            "endpoint_day_number": endpoint_day,
            "endpoint_timepoint_labels": endpoint_timepoint_labels,
            "plan_path": str(plan_path),
            "project_id": project_id,
            "project_manifest": str(project_manifest_path),
        }
        created = task_store_for_manifest(project_manifest_path).add(task)
        try:
            catalog.sync_manifest(project_manifest_path, force=True, source="task_created")
            catalog.sync_task(created)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            pass
        return created

    @app.get("/api/project/task-queue")
    def task_queue(project_id: str | None = None) -> dict[str, Any]:
        selected_manifest = manifest_for_project(project_id)
        selected_queue = _queue_path(selected_manifest)
        return {"queue_path": str(selected_queue), "tasks": tasks_for_project(project_id)}

    @app.get("/api/project/tasks/{task_id}")
    def task_detail(task_id: str) -> dict[str, Any]:
        located = locate_task(task_id)
        if located is None:
            raise HTTPException(status_code=404, detail="task not found")
        _, task = located
        return task

    @app.post("/api/project/tasks/{task_id}/start")
    def start_task(task_id: str) -> dict[str, Any]:
        located = locate_task(task_id)
        if located is None:
            raise HTTPException(status_code=404, detail="task not found")
        store, task = located
        if str(task.get("status")) != "queued":
            raise HTTPException(
                status_code=409,
                detail=f"task cannot start from status {task.get('status')}",
            )
        started = store.start(task_id, worker_id="manual")
        if started is None:
            raise HTTPException(status_code=404, detail="task not found")
        sync_task_catalog(started)
        return started

    @app.post("/api/project/tasks/{task_id}/cancel")
    def cancel_task(task_id: str) -> dict[str, Any]:
        located = locate_task(task_id)
        if located is None:
            raise HTTPException(status_code=404, detail="task not found")
        store, current = located
        status = str(current.get("status"))
        if status not in {"queued", "running", "cancelled"}:
            raise HTTPException(
                status_code=409,
                detail=f"task cannot cancel from status {status}",
            )
        task = store.cancel(task_id)
        sync_task_catalog(task or current)
        return task or current

    @app.delete("/api/project/tasks/{task_id}")
    def delete_task(task_id: str) -> dict[str, Any]:
        located = locate_task(task_id)
        if located is None:
            raise HTTPException(status_code=404, detail="task not found")
        store, current = located
        status = str(current.get("status"))
        if status == "completed":
            raise HTTPException(status_code=409, detail="completed tasks cannot be deleted")
        if status == "running":
            raise HTTPException(status_code=409, detail="cancel the running task before deleting it")
        cleanup_dir = disposable_task_project(current)
        if cleanup_dir is None:
            mark_task_plan_deleted(current)
        deleted = store.get(task_id)
        if deleted is None:
            raise HTTPException(status_code=404, detail="task not found")
        # A legacy task may have been projected into more than one queue while
        # the service was upgraded.  Remove every copy so the next refresh
        # cannot expose it again from a sibling queue.
        remove_task_from_all_queues(task_id)
        try:
            catalog.delete_task(task_id)
            if cleanup_dir is not None:
                catalog.mark_project_deleted(
                    str(current.get("project_id") or cleanup_dir.name)
                )
        except (OSError, sqlite3.Error, ValueError, TypeError):
            pass
        cleaned_project = False
        if cleanup_dir is not None and cleanup_dir.is_dir():
            try:
                shutil.rmtree(cleanup_dir)
                cleaned_project = True
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"任务记录已移除，但项目目录清理失败：{exc}",
                ) from exc
        return {
            "status": "deleted",
            "task": deleted,
            "cleaned_project": cleaned_project,
            "cleaned_project_path": str(cleanup_dir) if cleaned_project else "",
        }

    remove_development_routes(app)
    return app

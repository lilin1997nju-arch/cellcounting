from __future__ import annotations

import json
import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
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
from .dense_candidates import augment_candidates_with_dense_raw_proposals
from .hierarchy import suppress_nested_single_candidates
from .review_context import build_review_context, compute_well_registration
from .multiplicity import (
    ensure_integrated_review_table,
    generate_integrated_training_round,
    integrated_review_queue,
    integrated_review_stats,
    multiplicity_queue,
    multiplicity_stats,
    save_integrated_reviews,
    save_multiplicity_labels,
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
from .review_image_cache import render_review_image, review_image_cache_path
from .decode import inspect_tiff
from .gated_screening import build_gated_plate_report


SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id TEXT NOT NULL,
    plate_id TEXT NOT NULL,
    well TEXT NOT NULL,
    timepoint TEXT NOT NULL,
    object_id TEXT NOT NULL,
    canonical_target_id TEXT NOT NULL,
    track_id TEXT,
    parent_track_id TEXT,
    x_px REAL NOT NULL,
    y_px REAL NOT NULL,
    bbox_x REAL,
    bbox_y REAL,
    bbox_width REAL,
    bbox_height REAL,
    mask_path TEXT,
    object_type TEXT NOT NULL,
    viability TEXT NOT NULL,
    division_state TEXT NOT NULL,
    duplicate_of TEXT,
    reviewer TEXT,
    confidence REAL,
    notes TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(canonical_target_id, timepoint)
);
CREATE TABLE IF NOT EXISTS lineage_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id TEXT NOT NULL,
    plate_id TEXT NOT NULL,
    well TEXT NOT NULL,
    canonical_target_id TEXT NOT NULL UNIQUE,
    object_type TEXT NOT NULL,
    viability TEXT NOT NULL,
    division_state TEXT NOT NULL,
    morphology TEXT NOT NULL,
    timepoint_points_json TEXT NOT NULL,
    lineage_status TEXT NOT NULL DEFAULT 'needs_review',
    review_confidence TEXT NOT NULL DEFAULT 'medium',
    issue_tags TEXT NOT NULL DEFAULT '[]',
    reviewer TEXT,
    notes TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS link_reviews (
    link_review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_target_id TEXT NOT NULL,
    link_id TEXT NOT NULL,
    parent_timepoint TEXT NOT NULL,
    child_timepoint TEXT NOT NULL,
    link_label TEXT NOT NULL,
    reviewer TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(canonical_target_id, link_id)
);
CREATE TABLE IF NOT EXISTS teaching_labels (
    teaching_label_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL UNIQUE,
    well TEXT NOT NULL,
    timepoint TEXT NOT NULL,
    x_px REAL NOT NULL,
    y_px REAL NOT NULL,
    label TEXT NOT NULL,
    source TEXT NOT NULL,
    reviewer TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS auto_annotation_reviews (
    auto_review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    reviewed_label TEXT NOT NULL,
    decision TEXT NOT NULL,
    reviewer TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(round_id, candidate_id)
);
CREATE TABLE IF NOT EXISTS quick_missed_objects (
    quick_missed_id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id TEXT NOT NULL,
    annotation_id INTEGER NOT NULL,
    candidate_id TEXT NOT NULL UNIQUE,
    well TEXT NOT NULL,
    timepoint TEXT NOT NULL,
    x_px REAL NOT NULL,
    y_px REAL NOT NULL,
    diameter_px REAL NOT NULL,
    reviewed_label TEXT NOT NULL,
    reviewer TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quick_review_sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id TEXT NOT NULL,
    well TEXT NOT NULL,
    reviewer TEXT,
    duration_ms INTEGER NOT NULL,
    object_count INTEGER NOT NULL,
    corrected_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _review_images_manifest(
    config: dict[str, Any], base: pd.DataFrame
) -> pd.DataFrame:
    """Append Day7/Day14 images for review without expanding inference scope."""

    directories = config.get("review", {}).get("late_timepoint_directories", {})
    if not directories:
        return base
    exp = config["experiment"]
    extra: list[dict[str, Any]] = []
    for timepoint, directory in directories.items():
        if str(timepoint).upper() not in {"T3", "T4"}:
            continue
        folder = Path(str(directory))
        for row_name in "ABCDEFGH":
            for column in range(1, 13):
                well = f"{row_name}{column}"
                raw = folder / f"{well}.tif"
                cf = folder / f"{well}-cf.tif"
                metadata = inspect_tiff(raw) if raw.exists() else {
                    "width_px": 0,
                    "height_px": 0,
                    "bit_depth": 0,
                    "channels": 0,
                    "pyramid_levels": 0,
                    "decode_status": "missing",
                    "decode_error": "raw image missing",
                }
                extra.append({
                    "experiment_id": exp["experiment_id"],
                    "plate_id": exp["plate_id"],
                    "well": well,
                    "timepoint": str(timepoint).upper(),
                    "raw_image_path": str(raw.resolve()) if raw.exists() else "",
                    "cf_image_path": str(cf.resolve()) if cf.exists() else "",
                    "metrics_csv_path": str((folder / "metricsummary.csv").resolve()),
                    "cf_decode_status": "ok" if cf.exists() else "missing",
                    **metadata,
                })
    if not extra:
        return base
    combined = pd.concat([base, pd.DataFrame(extra)], ignore_index=True, sort=False)
    combined["well"] = combined["well"].astype(str).str.upper()
    combined["timepoint"] = combined["timepoint"].astype(str).str.upper()
    return combined.drop_duplicates(["well", "timepoint"], keep="last")


def _growth_region_contours(
    mask_path: Path,
    metrics_path: Path,
    well: str,
    settings: dict[str, Any],
    *,
    raw_image_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Extract thick Day14 sheet contours while rejecting thin wall bands."""

    if not mask_path.exists():
        return []
    downsample = max(1, int(settings.get("downsample", 4)))
    minimum_coverage = float(settings.get("minimum_component_coverage_pct", 1.0))
    minimum_radius = float(settings.get("minimum_radius_px", 40.0))
    minimum_mean_distance = float(settings.get("minimum_mean_distance_px", 10.0))
    with Image.open(mask_path) as opened:
        width = max(1, opened.width // downsample)
        height = max(1, opened.height // downsample)
        mask = np.asarray(
            opened.resize((width, height), Image.Resampling.NEAREST).convert("L")
        ) > 0
    # The CF image often contains the complete circular well rim.  Its very
    # long connected arc can merge with a real colony and make the displayed
    # growth region wrap around the wall.  Estimate the innermost strong
    # circular edge from the raw overview and clip the CF mask to the well
    # interior before connected-component analysis.  This affects only the
    # explanatory overlay, not the trained cell/debris detector.
    if raw_image_path is not None and raw_image_path.exists() and mask.any():
        try:
            with Image.open(raw_image_path) as opened:
                raw = np.asarray(
                    opened.resize((width, height), Image.Resampling.BILINEAR).convert("L"),
                    dtype=np.float32,
                )
            gradient = np.hypot(ndimage.sobel(raw, axis=0), ndimage.sobel(raw, axis=1))
            yy, xx = np.indices(mask.shape)
            center_y = (height - 1) / 2.0
            center_x = (width - 1) / 2.0
            radius_map = np.hypot(xx - center_x, yy - center_y)
            radial_index = np.floor(radius_map).astype(np.int32)
            radial_sum = np.bincount(radial_index.ravel(), weights=gradient.ravel())
            radial_count = np.bincount(radial_index.ravel())
            radial_mean = radial_sum / np.maximum(radial_count, 1)
            radial_mean = ndimage.gaussian_filter1d(radial_mean, sigma=2.0)
            minimum_radius = int(min(width, height) * 0.30)
            maximum_radius = int(min(width, height) * 0.49)
            search = radial_mean[minimum_radius : maximum_radius + 1]
            if search.size:
                peak_floor = float(np.percentile(search, 88))
                peaks, properties = signal.find_peaks(
                    search,
                    height=peak_floor,
                    distance=max(2, int(min(width, height) * 0.008)),
                )
                if len(peaks):
                    # Prefer the inner edge among similarly strong rim peaks.
                    peak_heights = properties["peak_heights"]
                    strong = peaks[peak_heights >= 0.72 * float(peak_heights.max())]
                    rim_radius = float(minimum_radius + int(strong.min()))
                    inner_margin = max(2.0, min(width, height) * 0.006)
                    mask &= radius_map <= rim_radius - inner_margin
        except (OSError, ValueError):
            pass
    foreground = int(mask.sum())
    if foreground == 0:
        return []
    confluence = 0.0
    if metrics_path.exists():
        try:
            metrics = pd.read_csv(metrics_path)
            selected = metrics[metrics["Well"].astype(str).str.upper() == well.upper()]
            if not selected.empty:
                confluence = float(selected.iloc[0]["Cell Confluence"])
        except (KeyError, OSError, ValueError):
            confluence = 0.0
    effective_area = (
        foreground / max(confluence / 100.0, 1e-9)
        if confluence > 0
        else np.pi * (min(width, height) * 0.39) ** 2
    )
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return []
    areas = np.bincount(labels.ravel())[1:]
    objects = ndimage.find_objects(labels)
    contours: list[dict[str, Any]] = []
    for index in np.argsort(areas)[-min(len(areas), 48):][::-1]:
        coverage = float(areas[index] / effective_area * 100.0)
        if coverage < minimum_coverage:
            continue
        slices = objects[index]
        if slices is None:
            continue
        component = labels[slices] == index + 1
        padded_component = np.pad(component, 1, mode="constant")
        distance = ndimage.distance_transform_edt(padded_component)[1:-1, 1:-1]
        radius = float(distance.max() * downsample)
        mean_distance = float(distance[component].mean() * downsample)
        if radius < minimum_radius or mean_distance < minimum_mean_distance:
            continue
        padded = padded_component.astype(np.uint8)
        candidates = measure.find_contours(padded, 0.5)
        if not candidates:
            continue
        contour = max(candidates, key=len)
        stride = max(1, int(np.ceil(len(contour) / 260)))
        sampled = contour[::stride]
        points = [
            [
                float((point[1] - 1 + slices[1].start) * downsample),
                float((point[0] - 1 + slices[0].start) * downsample),
            ]
            for point in sampled
        ]
        if len(points) >= 3:
            contours.append({
                "points": points,
                "coverage_pct": coverage,
                "maximum_radius_px": radius,
            })
    return contours


def _boolean_series(values: pd.Series) -> pd.Series:
    """Return a real boolean mask even after heterogeneous row concatenation.

    Appending manual-review rows changes pandas boolean columns to ``object``.
    Applying ``~`` to Python bools stored in an object series produces -1/-2,
    which are both truthy and therefore accidentally exposes suppressed V2
    proposals.  Normalize explicitly before any visibility filtering.
    """

    def convert(value: Any) -> bool:
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    return values.map(convert).astype(bool)


def _visible_v2_review_instances(reviewable: pd.DataFrame) -> pd.DataFrame:
    """Keep one authoritative review row for each V2 instance."""

    manual = _boolean_series(
        reviewable.get("is_manual_missed", pd.Series(False, index=reviewable.index))
    )
    mask_valid = _boolean_series(reviewable["v2_mask_valid"])
    wall_rejected = _boolean_series(reviewable["v2_wall_rejected"])
    suppressed = _boolean_series(reviewable["v2_is_suppressed"])
    visible = reviewable[manual | (mask_valid & ~wall_rejected & ~suppressed)].copy()

    # Suppression should already guarantee uniqueness.  This final guard keeps
    # the API invariant stable if an older result file contains two owner rows
    # with the same instance id.
    instance_id = visible["v2_instance_id"].fillna("").astype(str)
    has_instance = ~instance_id.isin(["", "nan", "None"])
    with_instance = visible[has_instance].copy()
    without_instance = visible[~has_instance].copy()
    if not with_instance.empty:
        with_instance["_reviewed_priority"] = with_instance.get(
            "reviewed_label", pd.Series(None, index=with_instance.index)
        ).notna().astype(int)
        with_instance["_instance_confidence_priority"] = pd.to_numeric(
            with_instance.get(
                "v2_instance_confidence", pd.Series(0.0, index=with_instance.index)
            ),
            errors="coerce",
        ).fillna(0.0)
        with_instance = (
            with_instance.sort_values(
                ["_reviewed_priority", "_instance_confidence_priority"],
                ascending=False,
                kind="stable",
            )
            .drop_duplicates(["well", "timepoint", "v2_instance_id"], keep="first")
            .drop(columns=["_reviewed_priority", "_instance_confidence_priority"])
        )
    return pd.concat([with_instance, without_instance], axis=0).sort_index()


class AnnotationPayload(BaseModel):
    sequence_id: str
    plate_id: str
    well: str
    timepoint: str
    object_id: str
    canonical_target_id: str
    x_px: float
    y_px: float
    object_type: str
    viability: str = "unknown"
    division_state: str = "unknown"
    duplicate_of: str | None = None
    reviewer: str = "local_user"
    confidence: float | None = None
    notes: str = ""


class PointSelection(BaseModel):
    x_px: float | None = None
    y_px: float | None = None
    candidate_id: str | None = None
    area_px: float | None = None
    object_label: str = "uncertain"
    marker_diameter_px: float | None = None
    orientation_rad: float | None = None


class TimepointSelection(PointSelection):
    present: bool = True
    additional_points: list[PointSelection] = Field(default_factory=list)


class LinkReviewPayload(BaseModel):
    link_id: str
    parent_timepoint: str
    child_timepoint: str
    link_label: str = "uncertain"


class LineageReviewPayload(BaseModel):
    sequence_id: str
    plate_id: str
    well: str
    canonical_target_id: str
    object_type: str
    viability: str
    division_state: str = "unknown"
    morphology: str = "uncertain"
    points: dict[str, TimepointSelection]
    lineage_status: str = "needs_review"
    review_confidence: str = "medium"
    issue_tags: list[str] = Field(default_factory=list)
    links: list[LinkReviewPayload] = Field(default_factory=list)
    reviewer: str = "local_user"
    notes: str = ""


class TeachingLabelItem(BaseModel):
    candidate_id: str
    well: str
    timepoint: str = "T0"
    x_px: float
    y_px: float
    label: str
    source: str = "quick_teaching"


class TeachingLabelsPayload(BaseModel):
    items: list[TeachingLabelItem]
    reviewer: str = "local_user"


class MultiplicityLabelItem(BaseModel):
    candidate_id: str
    well: str
    timepoint: str
    x_px: float
    y_px: float
    label: str
    source: str = "quick_multiplicity"


class MultiplicityLabelsPayload(BaseModel):
    items: list[MultiplicityLabelItem]
    reviewer: str = "local_user"


class AutoReviewItem(BaseModel):
    candidate_id: str
    predicted_label: str
    reviewed_label: str


class AutoReviewsPayload(BaseModel):
    round_id: str
    items: list[AutoReviewItem]
    reviewer: str = "local_user"


class IntegratedReviewItem(BaseModel):
    candidate_id: str
    predicted_label: str
    reviewed_label: str


class IntegratedReviewsPayload(BaseModel):
    round_id: str
    items: list[IntegratedReviewItem]
    reviewer: str = "local_user"


class QuickReviewObjectItem(BaseModel):
    candidate_id: str
    predicted_label: str
    reviewed_label: str
    well: str
    timepoint: str
    x_px: float
    y_px: float
    diameter_px: float = 12.0
    is_new: bool = False


class QuickReviewWellPayload(BaseModel):
    round_id: str
    items: list[QuickReviewObjectItem]
    well: str | None = None
    reviewer: str = "local_user"
    duration_ms: int | None = None


class WellScreeningReviewPayload(BaseModel):
    well: str
    decision: str
    reviewer: str = "local_user"
    notes: str = ""


class LateGrowthReviewPayload(BaseModel):
    well: str
    timepoint: str
    decision: str
    reviewer: str = "local_user"
    notes: str = ""


def _ensure_schema_columns(connection: sqlite3.Connection) -> None:
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(lineage_reviews)").fetchall()
    }
    additions = {
        "lineage_status": "TEXT NOT NULL DEFAULT 'needs_review'",
        "review_confidence": "TEXT NOT NULL DEFAULT 'medium'",
        "issue_tags": "TEXT NOT NULL DEFAULT '[]'",
    }
    for name, declaration in additions.items():
        if name not in columns:
            connection.execute(
                f"ALTER TABLE lineage_reviews ADD COLUMN {name} {declaration}"
            )


def initialize_database(path: str | Path) -> Path:
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.executescript(SCHEMA)
        _ensure_schema_columns(connection)
    return database


def save_annotation(path: str | Path, payload: dict[str, Any]) -> int:
    allowed_object_types = {"cell", "debris", "irrelevant", "uncertain"}
    allowed_viability = {"live", "dead", "unknown", "not_applicable"}
    allowed_division = {"none", "dividing", "divided", "unknown"}
    if payload["object_type"] not in allowed_object_types:
        raise ValueError("Invalid object_type")
    if payload["viability"] not in allowed_viability:
        raise ValueError("Invalid viability")
    if payload["division_state"] not in allowed_division:
        raise ValueError("Invalid division_state")
    updated = datetime.now(timezone.utc).isoformat()
    fields = [
        "sequence_id", "plate_id", "well", "timepoint", "object_id", "canonical_target_id",
        "track_id", "parent_track_id",
        "x_px", "y_px", "object_type", "viability", "division_state", "duplicate_of",
        "reviewer", "confidence", "notes",
    ]
    values = [payload.get(field) for field in fields]
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            f"""
            INSERT INTO annotations ({','.join(fields)}, updated_at)
            VALUES ({','.join('?' for _ in fields)}, ?)
            ON CONFLICT(canonical_target_id, timepoint) DO UPDATE SET
              object_type=excluded.object_type,
              viability=excluded.viability,
              division_state=excluded.division_state,
              duplicate_of=excluded.duplicate_of,
              reviewer=excluded.reviewer,
              confidence=excluded.confidence,
              notes=excluded.notes,
              updated_at=excluded.updated_at
            """,
            values + [updated],
        )
        return int(cursor.lastrowid)


def save_lineage_review(path: str | Path, payload: dict[str, Any]) -> int:
    allowed_morphology = {"cell_like", "debris_like", "uncertain"}
    allowed_object_labels = {"cell", "debris", "irrelevant", "uncertain"}
    allowed_link_labels = {"correct", "wrong", "uncertain"}
    allowed_lineage_status = {"correct_lineage", "debris_lineage", "wrong_link", "needs_review"}
    allowed_confidence = {"high", "medium", "low"}
    allowed_issue_tags = {
        "split_duplicate",
        "edge_false_positive",
        "missed_target",
        "large_motion_mismatch",
        "shape_mismatch",
    }
    if payload["morphology"] not in allowed_morphology:
        raise ValueError("Invalid morphology")
    if payload.get("lineage_status", "needs_review") not in allowed_lineage_status:
        raise ValueError("Invalid lineage_status")
    if payload.get("review_confidence", "medium") not in allowed_confidence:
        raise ValueError("Invalid review_confidence")
    if not set(payload.get("issue_tags", [])).issubset(allowed_issue_tags):
        raise ValueError("Invalid issue_tags")
    for link in payload.get("links", []):
        if link.get("link_label", "uncertain") not in allowed_link_labels:
            raise ValueError("Invalid link_label")
    points = payload["points"]
    if "T0" not in points or not points["T0"].get("present", False):
        raise ValueError("T0 selection is required")
    for point in points.values():
        labels = [point.get("object_label", payload["object_type"])]
        labels.extend(
            child.get("object_label", "uncertain")
            for child in point.get("additional_points", [])
        )
        if not set(labels).issubset(allowed_object_labels):
            raise ValueError("Invalid point object_label")
    updated = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO lineage_reviews (
              sequence_id, plate_id, well, canonical_target_id, object_type,
              viability, division_state, morphology, timepoint_points_json,
              lineage_status, review_confidence, issue_tags, reviewer, notes, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_target_id) DO UPDATE SET
              object_type=excluded.object_type,
              viability=excluded.viability,
              division_state=excluded.division_state,
              morphology=excluded.morphology,
              timepoint_points_json=excluded.timepoint_points_json,
              lineage_status=excluded.lineage_status,
              review_confidence=excluded.review_confidence,
              issue_tags=excluded.issue_tags,
              reviewer=excluded.reviewer,
              notes=excluded.notes,
              updated_at=excluded.updated_at
            """,
            (
                payload["sequence_id"],
                payload["plate_id"],
                payload["well"],
                payload["canonical_target_id"],
                payload["object_type"],
                payload["viability"],
                payload["division_state"],
                payload["morphology"],
                json.dumps(points, ensure_ascii=False),
                payload.get("lineage_status", "needs_review"),
                payload.get("review_confidence", "medium"),
                json.dumps(payload.get("issue_tags", []), ensure_ascii=False),
                payload.get("reviewer"),
                payload.get("notes", ""),
                updated,
            ),
        )
        review_id = int(cursor.lastrowid)
        connection.execute(
            "DELETE FROM link_reviews WHERE canonical_target_id = ?",
            (payload["canonical_target_id"],),
        )
        for link in payload.get("links", []):
            connection.execute(
                """
                INSERT INTO link_reviews (
                  canonical_target_id, link_id, parent_timepoint, child_timepoint,
                  link_label, reviewer, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["canonical_target_id"],
                    link["link_id"],
                    link["parent_timepoint"],
                    link["child_timepoint"],
                    link.get("link_label", "uncertain"),
                    payload.get("reviewer"),
                    updated,
                ),
            )
    for timepoint, point in points.items():
        if not point.get("present") or point.get("x_px") is None or point.get("y_px") is None:
            continue
        point_object_type = point.get("object_label", payload["object_type"])
        point_viability = (
            payload["viability"]
            if point_object_type == "cell"
            else "unknown" if point_object_type == "uncertain" else "not_applicable"
        )
        save_annotation(
            path,
            {
                "sequence_id": payload["sequence_id"],
                "plate_id": payload["plate_id"],
                "well": payload["well"],
                "timepoint": timepoint,
                "object_id": point.get("candidate_id")
                or f"{payload['canonical_target_id']}:{timepoint}",
                "canonical_target_id": payload["canonical_target_id"],
                "track_id": payload["canonical_target_id"],
                "parent_track_id": None if timepoint == "T0" else payload["canonical_target_id"],
                "x_px": point["x_px"],
                "y_px": point["y_px"],
                "object_type": point_object_type,
                "viability": point_viability,
                "division_state": payload["division_state"],
                "duplicate_of": None,
                "reviewer": payload.get("reviewer"),
                "confidence": 1.0,
                "notes": payload.get("notes", ""),
            },
        )
        for child_index, child in enumerate(point.get("additional_points", []), start=1):
            if child.get("x_px") is None or child.get("y_px") is None:
                continue
            child_target_id = (
                f"{payload['canonical_target_id']}:child:{timepoint}:{child_index}"
            )
            child_object_type = child.get("object_label", "uncertain")
            child_viability = (
                payload["viability"]
                if child_object_type == "cell"
                else "unknown" if child_object_type == "uncertain" else "not_applicable"
            )
            save_annotation(
                path,
                {
                    "sequence_id": payload["sequence_id"],
                    "plate_id": payload["plate_id"],
                    "well": payload["well"],
                    "timepoint": timepoint,
                    "object_id": child.get("candidate_id") or child_target_id,
                    "canonical_target_id": child_target_id,
                    "track_id": child_target_id,
                    "parent_track_id": payload["canonical_target_id"],
                    "x_px": child["x_px"],
                    "y_px": child["y_px"],
                    "object_type": child_object_type,
                    "viability": child_viability,
                    "division_state": "divided",
                    "duplicate_of": None,
                    "reviewer": payload.get("reviewer"),
                    "confidence": 1.0,
                    "notes": payload.get("notes", ""),
                },
            )
    return review_id


def create_app(config: dict[str, Any]) -> FastAPI:
    database_path = artifact_path(config, "annotations", "annotations.db")
    images_manifest_path = artifact_path(config, "manifests", "images.csv")
    database = initialize_database(database_path)
    images_manifest = _review_images_manifest(
        config,
        pd.read_csv(images_manifest_path),
    )
    review_html_path = Path(__file__).resolve().parents[2] / "review-ui" / "index.html"
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
    integrated_review_html_path = (
        Path(__file__).resolve().parents[2]
        / "review-ui"
        / "integrated-review.html"
    )
    screening_html_path = (
        Path(__file__).resolve().parents[2] / "review-ui" / "screening.html"
    )
    app = FastAPI(title="Cell Vision Local Review")
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
    growth_contour_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}

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

    def ui_screening_status(gated: dict[str, Any], fallback: str) -> str:
        category = str(gated.get("final_category", ""))
        return {
            "single_cell_origin": "single_active",
            "multi_cell_origin": "multi_origin",
            "no_obvious_growth": "no_cell_growth",
            "undetermined": (
                "missing_t0_or_late_object"
                if gated.get("undetermined_reason") == "t0_missing_later_detected"
                else "ambiguous"
            ),
            "positive_control": "positive_control",
        }.get(category, fallback)

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

    def latest_integrated_predictions() -> pd.DataFrame:
        base_source = artifact_path(
            config, "predictions", "latest_integrated_predictions.csv"
        )
        completed_source = artifact_path(
            config,
            "predictions",
            "latest_temporally_completed_predictions.csv",
        )
        v2_source = artifact_path(config, "predictions", "latest_v2_predictions.csv")
        if not base_source.exists():
            return pd.DataFrame()
        source = (
            completed_source
            if completed_source.exists()
            and completed_source.stat().st_mtime_ns
            >= base_source.stat().st_mtime_ns
            else base_source
        )
        if v2_source.exists() and v2_source.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            source = v2_source
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
        }
        for column, default in temporal_defaults.items():
            if column not in frame:
                frame[column] = default
        reviewable = frame[
            frame["integrated_label"].isin(
                [
                    "single",
                    "touching_doublet",
                    "cluster_3plus",
                    "debris",
                    "uncertain",
                ]
            )
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
                        (local["current_label"] == "uncertain").sum()
                    ),
                    "cell_count": int(
                        local["current_label"].isin(
                            [
                                "single",
                                "touching_doublet",
                                "cluster_3plus",
                            ]
                        ).sum()
                    ),
                    "debris_count": int(
                        (local["current_label"] == "debris").sum()
                    ),
                    "temporal_review_count": int(
                        local["temporal_completion_status"].isin(
                            [
                                "ambiguous_temporal_candidate",
                                "temporal_reclassification_review",
                            ]
                        ).sum()
                    ),
                    "priority": float(
                        local["integrated_review_priority"].max()
                    ),
                    "screening_status": status,
                    "base_screening_status": str(
                        screening.get("base_screening_status", "ambiguous")
                    ),
                    "late_growth_status": str(
                        screening.get("late_growth_status", "unavailable")
                    ),
                    "high_confidence_single_active": bool(
                        screening.get("high_confidence_single_active", False)
                    ),
                    "report_category": report.get("final_category"),
                    "report_category_label": report.get("final_category_label"),
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
    app.mount(
        "/assets",
        StaticFiles(directory=review_html_path.parent),
        name="review-assets",
    )

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        return auto_review_html_path.read_text(encoding="utf-8")

    @app.get("/teach", response_class=HTMLResponse)
    def teach() -> str:
        return teaching_html_path.read_text(encoding="utf-8")

    @app.get("/auto-review", response_class=HTMLResponse)
    def auto_review() -> str:
        return auto_review_html_path.read_text(encoding="utf-8")

    @app.get("/doublet-teach", response_class=HTMLResponse)
    def doublet_teach() -> str:
        return multiplicity_html_path.read_text(encoding="utf-8")

    @app.get("/integrated-review", response_class=HTMLResponse)
    def integrated_review() -> str:
        return integrated_review_html_path.read_text(encoding="utf-8")

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
                    dense = lineage_representative_view(
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
            return add_model_candidates(context, well)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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
            return {"status": "saved", "saved": saved}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/teach-train")
    def teach_train() -> dict[str, Any]:
        try:
            return train_teaching_classifier(config, database)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/multiplicity-candidates")
    def multiplicity_candidates(
        mode: str = "likely_doublet", limit: int = 30
    ) -> list[dict[str, Any]]:
        if mode not in {"likely_doublet", "uncertain", "diverse"}:
            raise HTTPException(status_code=422, detail="Invalid queue mode")
        return multiplicity_queue(config, database, mode, limit)

    @app.get("/api/multiplicity-stats")
    def multiplicity_label_stats() -> dict[str, Any]:
        return multiplicity_stats(database)

    @app.post("/api/multiplicity-labels")
    def multiplicity_labels_create(
        payload: MultiplicityLabelsPayload,
    ) -> dict[str, Any]:
        try:
            saved = save_multiplicity_labels(
                database,
                [item.model_dump() for item in payload.items],
                payload.reviewer,
            )
            return {"status": "saved", "saved": saved}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/integrated-review-stats")
    def integrated_stats() -> dict[str, Any]:
        return integrated_review_stats(config, database)

    @app.get("/api/quick-review-stats")
    def quick_review_stats() -> dict[str, Any]:
        frame = quick_review_frame()
        if frame.empty:
            return {"status": "not_generated"}
        wells = quick_review_well_rows(mode="all", frame=frame)
        reviewed_count = int(frame["reviewed_label"].notna().sum())
        return {
            "status": "ready",
            "round_id": str(frame.iloc[0]["integrated_round_id"]),
            "well_count": int(len(wells)),
            "completed_well_count": int(
                sum(row["completed"] for row in wells)
            ),
            "pending_well_count": int(
                sum(not row["completed"] for row in wells)
            ),
            "object_count": int(len(frame)),
            "reviewed_object_count": reviewed_count,
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
            "label_counts": {
                str(key): int(value)
                for key, value in frame[
                    "current_label"
                ].value_counts().items()
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
        return quick_review_well_rows(mode=mode, search=search)

    @app.get("/api/quick-review-well/{well}")
    def quick_review_well(well: str) -> dict[str, Any]:
        normalized_well = well.upper()
        frame = quick_review_frame()
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
            representative = roi.get(timepoint) or None
            if timepoint == "T3" and report.get("day7_regions_json"):
                try:
                    day7_regions = json.loads(str(report["day7_regions_json"]))
                except (TypeError, json.JSONDecodeError):
                    day7_regions = []
                if day7_regions:
                    representative = {
                        "x": float(day7_regions[0]["center_x"]),
                        "y": float(day7_regions[0]["center_y"]),
                        "size": float(day7_regions[0]["x1"] - day7_regions[0]["x0"]),
                    }
            if timepoint == "T4" and report:
                day14_positive = str(
                    report.get("day14_obvious_growth", "")
                ).strip().lower() in {"1", "true", "yes"}
                late_decision = (
                    "obvious_growth"
                    if day14_positive
                    else "no_growth"
                )
            growth_regions: list[dict[str, Any]] = []
            if timepoint == "T4":
                mask_path = Path(str(row.get("cf_image_path", "")))
                metrics_path = Path(str(row.get("metrics_csv_path", "")))
                if mask_path.exists():
                    cache_key = (str(mask_path), mask_path.stat().st_mtime_ns)
                    if cache_key not in growth_contour_cache:
                        growth_contour_cache[cache_key] = _growth_region_contours(
                            mask_path,
                            metrics_path,
                            normalized_well,
                            config.get("review", {}).get("day14_overlay", {}),
                            raw_image_path=Path(str(row.get("raw_image_path", ""))),
                        )
                    growth_regions = growth_contour_cache[cache_key]
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
                "growth_overlay_style": "translucent_fill_with_contour" if growth_regions else "none",
                "default_zoom": (
                    3.0
                    if timepoint == "T3"
                    and representative
                    else 1.0
                ),
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
            "v2_temporal_morphology_consensus_cell_probability",
            "v2_suspected_dead_cell",
            "v2_suspected_dead_cell_score",
            "v2_temporal_debris_boost",
            "v2_temporal_cell_boost",
            "v2_adjusted_cell_probability",
            "v2_adjusted_debris_probability",
            "v2_temporal_adjustment_applied",
            "v2_temporal_reason",
            "v2_temporal_adjusted_label",
            "v2_static_wall_artifact",
            "v2_temporal_recovered",
            "v2_temporal_track_id",
            "v2_low_cell_noncell_resolved",
            "v2_noncell_resolution_label",
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
        search_hints: list[dict[str, Any]] = []
        return {
            "round_id": str(local.iloc[0]["integrated_round_id"]),
            "well": normalized_well,
            "images": images,
            "objects": (
                local[columns]
                .replace({np.nan: None})
                .to_dict(orient="records")
            ),
            "search_hints": search_hints,
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

    @app.get("/api/integrated-review-candidates")
    def integrated_candidates(
        mode: str = "all", limit: int = 30
    ) -> list[dict[str, Any]]:
        if mode not in {
            "all",
            "cell",
            "doublet",
            "debris",
            "uncertain",
            "reviewed",
        }:
            raise HTTPException(status_code=422, detail="Invalid review mode")
        return integrated_review_queue(
            config, database, mode, min(limit, 100)
        )

    @app.post("/api/integrated-review-labels")
    def integrated_labels_create(
        payload: IntegratedReviewsPayload,
    ) -> dict[str, Any]:
        try:
            saved = save_integrated_reviews(
                database,
                payload.round_id,
                [item.model_dump() for item in payload.items],
                payload.reviewer,
            )
            return {"status": "saved", "saved": saved}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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
            if payload.duration_ms is not None:
                corrected_count = sum(
                    item.reviewed_label != item.predicted_label for item in payload.items
                )
                with sqlite3.connect(database) as connection:
                    connection.execute(
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
            primary_well = (
                payload.well.upper()
                if payload.well
                else next(iter(affected_wells), "")
            )
            if screening_source.exists() and primary_well:
                screening_frame = pd.read_csv(screening_source, low_memory=False)
                selected_screening = screening_frame[
                    screening_frame["well"].astype(str).str.upper() == primary_well
                ]
                if not selected_screening.empty:
                    screening_row = selected_screening.replace({np.nan: None}).iloc[0].to_dict()
            return {
                "status": "saved",
                "saved_standard": saved_standard,
                "saved_missed": len(mappings),
                "mappings": mappings,
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

    @app.post("/api/integrated-review-new-round")
    def integrated_new_round() -> dict[str, Any]:
        try:
            dense_candidates = (
                augment_candidates_with_dense_raw_proposals(
                    config, database
                )
            )
            ensure_teaching_features(config, force=True)
            morphology_training = train_teaching_classifier(
                config, database
            )
            morphology_round = generate_auto_annotation_round(
                config, database
            )
            multiplicity_training = train_multiplicity_classifier(
                config, database
            )
            integrated_round = generate_integrated_training_round(
                config, database
            )
            well_screening = build_well_screening(config, database)
            return {
                "dense_candidates": dense_candidates,
                "morphology_training": morphology_training,
                "morphology_round": morphology_round,
                "multiplicity_training": multiplicity_training,
                "integrated_round": integrated_round,
                "well_screening": well_screening,
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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
                row.update({
                    "screening_status": ui_screening_status(
                        report_row,
                        str(row.get("screening_status", "ambiguous")),
                    ),
                    "report_category": report_row.get("final_category"),
                    "report_category_label": report_row.get("final_category_label"),
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
        return [
            {key: (None if pd.isna(value) else value) for key, value in row.items()}
            for row in sorted(base.values(), key=sort_key)
        ]

    @app.post("/api/screening-review")
    def screening_review(
        payload: WellScreeningReviewPayload,
    ) -> dict[str, Any]:
        if payload.decision not in {"approved", "rejected", "pending"}:
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
        summary = build_well_screening(config, database)
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
        summary = build_well_screening(config, database)
        gated_summary, updated_report = refresh_gated_report()
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

    @app.get("/api/auto-review-stats")
    def auto_review_stats() -> dict[str, Any]:
        return auto_annotation_stats(config, database)

    @app.get("/api/auto-review-candidates")
    def auto_review_candidates(
        mode: str = "needs_review", limit: int = 30
    ) -> list[dict[str, Any]]:
        if mode not in {"needs_review", "cell", "audit", "reviewed"}:
            raise HTTPException(status_code=422, detail="Invalid review mode")
        return auto_annotation_queue(config, database, mode, limit)

    @app.post("/api/auto-review-labels")
    def auto_review_labels_create(
        payload: AutoReviewsPayload,
    ) -> dict[str, Any]:
        try:
            saved = save_auto_annotation_reviews(
                database,
                payload.round_id,
                [item.model_dump() for item in payload.items],
                payload.reviewer,
            )
            return {"status": "saved", "saved": saved}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/auto-review-new-round")
    def auto_review_new_round() -> dict[str, Any]:
        try:
            training = train_teaching_classifier(config, database)
            generated = generate_auto_annotation_round(config, database)
            return {"training": training, "round": generated}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/patch")
    def patch(well: str, timepoint: str, x: float, y: float, size: int = 256) -> Response:
        selected = images_manifest[
            (images_manifest["well"] == well.upper())
            & (images_manifest["timepoint"] == timepoint.upper())
            & (images_manifest["decode_status"] == "ok")
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

    @app.get("/api/well-image")
    def well_image(
        request: Request, well: str, timepoint: str, max_size: int = 1200
    ) -> Response:
        selected = images_manifest[
            (images_manifest["well"] == well.upper())
            & (images_manifest["timepoint"] == timepoint.upper())
            & (images_manifest["decode_status"] == "ok")
        ]
        if selected.empty:
            raise HTTPException(status_code=404, detail="image unavailable")
        max_size = max(400, min(int(max_size), 4096))
        raw_path = Path(str(selected.iloc[0]["raw_image_path"]))
        cache_path, etag = review_image_cache_path(config, raw_path, max_size)
        headers = {"ETag": f'"{etag}"', "Cache-Control": "public, max-age=31536000, immutable"}
        if request.headers.get("if-none-match") == headers["ETag"]:
            return Response(status_code=304, headers=headers)
        if not cache_path.exists():
            render_review_image(config, raw_path, max_size)
        return Response(content=cache_path.read_bytes(), media_type="image/jpeg", headers=headers)

    @app.get("/api/report-image")
    def report_image(
        well: str,
        timepoint: str,
        view: str = "whole",
        max_size: int = 1200,
    ) -> Response:
        if view not in {"whole", "local"}:
            raise HTTPException(status_code=422, detail="invalid view")
        normalized_well = well.upper()
        normalized_timepoint = timepoint.upper()
        selected = images_manifest[
            (images_manifest["well"] == normalized_well)
            & (images_manifest["timepoint"] == normalized_timepoint)
            & (images_manifest["decode_status"] == "ok")
        ]
        if selected.empty:
            raise HTTPException(status_code=404, detail="image unavailable")
        screening_path = artifact_path(
            config, "predictions", "latest_well_screening.csv"
        )
        if not screening_path.exists():
            build_well_screening(config, database)
        screening = pd.read_csv(screening_path)
        row = screening[screening["well"].astype(str).str.upper() == normalized_well]
        if row.empty:
            raise HTTPException(status_code=404, detail="screening ROI unavailable")
        roi_map = json.loads(str(row.iloc[0]["roi_json"]))
        roi = roi_map.get(normalized_timepoint)
        with Image.open(selected.iloc[0]["raw_image_path"]) as opened:
            rendered = ImageEnhance.Contrast(opened.convert("L")).enhance(1.7)
            width, height = rendered.size
            if roi is None:
                roi = {"x": width / 2, "y": height / 2, "size": min(width, height) * 0.45}
            size = max(240, min(int(float(roi["size"])), min(width, height)))
            half = size // 2
            x = float(roi["x"])
            y = float(roi["y"])
            left = max(0, min(width - size, int(round(x - half))))
            top = max(0, min(height - size, int(round(y - half))))
            right, bottom = left + size, top + size
            if view == "local":
                # The report crop is intentionally clean: no object circles.
                rendered = rendered.crop((left, top, right, bottom))
            else:
                draw = ImageDraw.Draw(rendered)
                line_width = max(8, min(width, height) // 350)
                draw.rectangle(
                    (left, top, right, bottom),
                    outline=255,
                    width=line_width,
                )
            max_size = max(360, min(int(max_size), 2200))
            rendered.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        rendered.save(buffer, format="JPEG", quality=94, subsampling=0)
        return Response(content=buffer.getvalue(), media_type="image/jpeg")

    @app.post("/api/annotations")
    def annotations_create(payload: AnnotationPayload) -> dict[str, Any]:
        try:
            annotation_id = save_annotation(database, payload.model_dump())
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
            return {"status": "saved", "review_id": review_id}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app

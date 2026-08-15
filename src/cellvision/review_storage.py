from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .multiplicity import ensure_integrated_review_table, ensure_multiplicity_table


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
CREATE TABLE IF NOT EXISTS temporal_track_reviews (
    temporal_track_review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id TEXT NOT NULL,
    track_id TEXT NOT NULL,
    well TEXT NOT NULL,
    label TEXT NOT NULL,
    behavior TEXT,
    reviewer TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(round_id, track_id)
);
CREATE TABLE IF NOT EXISTS quick_review_undo_actions (
    undo_action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id TEXT NOT NULL,
    well TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    undone_at TEXT
);
CREATE TABLE IF NOT EXISTS v2_mask_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    well TEXT NOT NULL,
    timepoint TEXT NOT NULL,
    model_mask_rle TEXT NOT NULL,
    reviewed_mask_rle TEXT NOT NULL,
    decision TEXT NOT NULL,
    model_area_px INTEGER NOT NULL,
    reviewed_area_px INTEGER NOT NULL,
    model_diameter_px REAL NOT NULL,
    reviewed_diameter_px REAL NOT NULL,
    contour_json TEXT NOT NULL,
    reviewer TEXT,
    notes TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(round_id, candidate_id)
);
"""



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


def _fetch_rows_by_values(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    values: list[str],
    *,
    prefix_sql: str = "",
    prefix_params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    """Read rows for an undo snapshot using only internal SQL identifiers."""

    normalized = list(dict.fromkeys(str(value) for value in values if value is not None))
    if not normalized:
        return []
    placeholders = ", ".join("?" for _ in normalized)
    query = (
        f"SELECT * FROM {table} {prefix_sql}"
        f"{' AND ' if prefix_sql else 'WHERE '}"
        f"{column} IN ({placeholders})"
    )
    return [
        dict(row)
        for row in connection.execute(
            query,
            (*prefix_params, *normalized),
        ).fetchall()
    ]


def _capture_quick_review_undo_snapshot(
    database: str | Path,
    round_id: str,
    candidate_ids: list[str],
    track_ids: list[str],
) -> dict[str, Any]:
    """Capture the database state changed by one quick-review save.

    The quick-review endpoint writes to several small training/review tables.
    Keeping the previous rows together makes Ctrl/Cmd+Z restore a real saved
    decision, including missed targets and unified V3 track labels.
    """

    ensure_integrated_review_table(database)
    ensure_multiplicity_table(database)
    normalized_candidates = list(
        dict.fromkeys(str(value) for value in candidate_ids if value)
    )
    normalized_tracks = list(dict.fromkeys(str(value) for value in track_ids if value))
    snapshot: dict[str, Any] = {
        "round_id": str(round_id),
        "candidate_ids": normalized_candidates,
        "track_ids": normalized_tracks,
        "integrated_training_reviews": [],
        "quick_missed_objects": [],
        "annotations": [],
        "teaching_labels": [],
        "multiplicity_labels": [],
        "temporal_track_reviews": [],
        "created_manual": [],
        "created_session_ids": [],
    }
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        snapshot["integrated_training_reviews"] = _fetch_rows_by_values(
            connection,
            "integrated_training_reviews",
            "candidate_id",
            normalized_candidates,
            prefix_sql="WHERE round_id = ?",
            prefix_params=(str(round_id),),
        )
        snapshot["quick_missed_objects"] = _fetch_rows_by_values(
            connection,
            "quick_missed_objects",
            "candidate_id",
            normalized_candidates,
            prefix_sql="WHERE round_id = ?",
            prefix_params=(str(round_id),),
        )
        manual_ids = [
            str(row["candidate_id"])
            for row in snapshot["quick_missed_objects"]
        ]
        annotation_ids = [
            str(row["annotation_id"])
            for row in snapshot["quick_missed_objects"]
            if row.get("annotation_id") is not None
        ]
        snapshot["manual_candidate_ids"] = manual_ids
        snapshot["annotation_ids"] = annotation_ids
        snapshot["annotations"] = _fetch_rows_by_values(
            connection,
            "annotations",
            "annotation_id",
            annotation_ids,
        )
        snapshot["teaching_labels"] = _fetch_rows_by_values(
            connection,
            "teaching_labels",
            "candidate_id",
            normalized_candidates,
        )
        snapshot["multiplicity_labels"] = _fetch_rows_by_values(
            connection,
            "multiplicity_labels",
            "candidate_id",
            normalized_candidates,
        )
        snapshot["temporal_track_reviews"] = _fetch_rows_by_values(
            connection,
            "temporal_track_reviews",
            "track_id",
            normalized_tracks,
            prefix_sql="WHERE round_id = ?",
            prefix_params=(str(round_id),),
        )
    return snapshot


def _delete_rows_by_values(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    values: list[Any],
    *,
    prefix_sql: str = "",
    prefix_params: tuple[Any, ...] = (),
) -> None:
    normalized = list(dict.fromkeys(value for value in values if value is not None))
    if not normalized:
        return
    placeholders = ", ".join("?" for _ in normalized)
    query = (
        f"DELETE FROM {table} {prefix_sql}"
        f"{' AND ' if prefix_sql else 'WHERE '}"
        f"{column} IN ({placeholders})"
    )
    connection.execute(query, (*prefix_params, *normalized))


def _restore_rows(
    connection: sqlite3.Connection,
    table: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    columns = list(rows[0])
    quoted_columns = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(
        f"INSERT OR REPLACE INTO {table} ({quoted_columns}) VALUES ({placeholders})",
        [tuple(row.get(column) for column in columns) for row in rows],
    )


def _restore_quick_review_undo_snapshot(
    database: str | Path,
    snapshot: dict[str, Any],
) -> None:
    """Restore one quick-review action and remove any newly created targets."""

    ensure_multiplicity_table(database)
    round_id = str(snapshot.get("round_id", ""))
    candidate_ids = [str(value) for value in snapshot.get("candidate_ids", [])]
    manual_ids = [
        str(value) for value in snapshot.get("manual_candidate_ids", [])
    ]
    created_manual = snapshot.get("created_manual", []) or []
    created_candidate_ids = [
        str(row.get("candidate_id"))
        for row in created_manual
        if row.get("candidate_id")
    ]
    all_manual_ids = list(dict.fromkeys([*manual_ids, *created_candidate_ids]))
    annotation_ids = [
        str(value) for value in snapshot.get("annotation_ids", []) if value is not None
    ]
    created_annotation_ids = [
        str(row.get("annotation_id"))
        for row in created_manual
        if row.get("annotation_id") is not None
    ]
    all_annotation_ids = list(
        dict.fromkeys([*annotation_ids, *created_annotation_ids])
    )
    track_ids = [str(value) for value in snapshot.get("track_ids", [])]
    with sqlite3.connect(database) as connection:
        _delete_rows_by_values(
            connection,
            "integrated_training_reviews",
            "candidate_id",
            candidate_ids,
            prefix_sql="WHERE round_id = ?",
            prefix_params=(round_id,),
        )
        _delete_rows_by_values(connection, "quick_missed_objects", "candidate_id", all_manual_ids)
        _delete_rows_by_values(connection, "annotations", "annotation_id", all_annotation_ids)
        _delete_rows_by_values(connection, "teaching_labels", "candidate_id", candidate_ids)
        _delete_rows_by_values(
            connection,
            "multiplicity_labels",
            "candidate_id",
            candidate_ids,
        )
        _delete_rows_by_values(
            connection,
            "temporal_track_reviews",
            "track_id",
            track_ids,
            prefix_sql="WHERE round_id = ?",
            prefix_params=(round_id,),
        )
        _delete_rows_by_values(
            connection,
            "quick_review_sessions",
            "session_id",
            snapshot.get("created_session_ids", []),
        )
        _restore_rows(connection, "annotations", snapshot.get("annotations", []))
        _restore_rows(
            connection,
            "quick_missed_objects",
            snapshot.get("quick_missed_objects", []),
        )
        _restore_rows(
            connection,
            "teaching_labels",
            snapshot.get("teaching_labels", []),
        )
        _restore_rows(
            connection,
            "multiplicity_labels",
            snapshot.get("multiplicity_labels", []),
        )
        _restore_rows(
            connection,
            "integrated_training_reviews",
            snapshot.get("integrated_training_reviews", []),
        )
        _restore_rows(
            connection,
            "temporal_track_reviews",
            snapshot.get("temporal_track_reviews", []),
        )


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


def _summary_json_safe(value: Any) -> Any:
    """Convert pandas/numpy values in a list summary to strict JSON."""

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _summary_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summary_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _summary_json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value



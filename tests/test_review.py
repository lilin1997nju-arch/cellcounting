import sqlite3

import numpy as np
import pandas as pd

from cellvision.review_server import (
    _growth_region_contours,
    _visible_v2_review_instances,
    PointSelection,
    initialize_database,
    save_annotation,
    save_lineage_review,
)
from PIL import Image


def test_v2_review_visibility_removes_suppressed_rows_after_manual_concat():
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "owner",
                "well": "A12",
                "timepoint": "T1",
                "v2_instance_id": "A12-T1-I001",
                "v2_mask_valid": True,
                "v2_wall_rejected": False,
                "v2_is_suppressed": False,
                "v2_instance_confidence": 0.8,
                "reviewed_label": None,
                "is_manual_missed": False,
            },
            {
                "candidate_id": "duplicate",
                "well": "A12",
                "timepoint": "T1",
                "v2_instance_id": "A12-T1-I001",
                "v2_mask_valid": True,
                "v2_wall_rejected": False,
                "v2_is_suppressed": True,
                "v2_instance_confidence": 0.9,
                "reviewed_label": None,
                "is_manual_missed": False,
            },
            {
                "candidate_id": "manual",
                "well": "A12",
                "timepoint": "T1",
                "v2_instance_id": np.nan,
                "v2_mask_valid": np.nan,
                "v2_wall_rejected": np.nan,
                "v2_is_suppressed": np.nan,
                "v2_instance_confidence": np.nan,
                "reviewed_label": "single",
                "is_manual_missed": True,
            },
        ]
    ).astype({"v2_mask_valid": object, "v2_wall_rejected": object, "v2_is_suppressed": object})

    visible = _visible_v2_review_instances(frame)

    assert set(visible["candidate_id"]) == {"owner", "manual"}


def test_point_selection_preserves_display_diameter_and_orientation():
    point = PointSelection(
        x_px=10,
        y_px=20,
        marker_diameter_px=7.5,
        orientation_rad=0.4,
    )
    assert point.marker_diameter_px == 7.5
    assert point.orientation_rad == 0.4


def test_day14_growth_overlay_keeps_thick_sheet_and_rejects_thin_wall(tmp_path):
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[20:492, 35:39] = 255
    mask[260:390, 290:430] = 255
    mask_path = tmp_path / "A2-cf.tif"
    metrics_path = tmp_path / "metricsummary.csv"
    Image.fromarray(mask).save(mask_path)
    pd.DataFrame([
        {"Well": "A2", "Cell Confluence": 18.0, "Cell Count": 1000}
    ]).to_csv(metrics_path, index=False)
    contours = _growth_region_contours(
        mask_path,
        metrics_path,
        "A2",
        {
            "downsample": 2,
            "minimum_component_coverage_pct": 1.0,
            "minimum_radius_px": 30.0,
            "minimum_mean_distance_px": 8.0,
        },
    )
    assert len(contours) == 1
    points = np.asarray(contours[0]["points"])
    assert points[:, 0].mean() > 280
    assert points[:, 1].mean() > 250


def test_annotation_save_and_deduplicate(tmp_path):
    database = initialize_database(tmp_path / "annotations.db")
    payload = {
        "sequence_id": "s1",
        "plate_id": "p1",
        "well": "H6",
        "timepoint": "T0",
        "object_id": "o1",
        "canonical_target_id": "canonical-1",
        "x_px": 10.0,
        "y_px": 20.0,
        "object_type": "cell",
        "viability": "live",
        "division_state": "none",
        "duplicate_of": None,
        "reviewer": "test",
        "confidence": 1.0,
        "notes": "",
    }
    save_annotation(database, payload)
    payload["notes"] = "updated"
    save_annotation(database, payload)
    with sqlite3.connect(database) as connection:
        count = connection.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
        notes = connection.execute("SELECT notes FROM annotations").fetchone()[0]
    assert count == 1
    assert notes == "updated"


def test_lineage_review_saves_clicked_timepoint_positions(tmp_path):
    database = initialize_database(tmp_path / "annotations.db")
    payload = {
        "sequence_id": "s1",
        "plate_id": "p1",
        "well": "H6",
        "canonical_target_id": "track-1",
        "object_type": "cell",
        "viability": "dead",
        "division_state": "none",
        "morphology": "cell_like",
        "points": {
            "T0": {
                "present": True,
                "x_px": 10.0,
                "y_px": 20.0,
                "candidate_id": "t0",
                "area_px": 8,
                "object_label": "cell",
            },
            "T1": {
                "present": True,
                "x_px": 12.0,
                "y_px": 20.0,
                "candidate_id": "t1",
                "area_px": 9,
                "object_label": "debris",
                "additional_points": [
                    {
                        "x_px": 14.0,
                        "y_px": 21.0,
                        "candidate_id": "t1-child",
                        "area_px": 8,
                        "object_label": "cell",
                    }
                ],
            },
            "T2": {"present": False, "x_px": None, "y_px": None, "candidate_id": None, "area_px": None},
        },
        "lineage_status": "wrong_link",
        "review_confidence": "low",
        "issue_tags": ["shape_mismatch"],
        "links": [
            {
                "link_id": "T0-T1",
                "parent_timepoint": "T0",
                "child_timepoint": "T1",
                "link_label": "wrong",
            }
        ],
        "reviewer": "test",
        "notes": "cell-like and static",
    }
    save_lineage_review(database, payload)
    with sqlite3.connect(database) as connection:
        review_count = connection.execute("SELECT COUNT(*) FROM lineage_reviews").fetchone()[0]
        review = connection.execute(
            "SELECT lineage_status,review_confidence,issue_tags FROM lineage_reviews"
        ).fetchone()
        annotations = connection.execute(
            "SELECT timepoint,x_px,y_px,parent_track_id,object_type,viability "
            "FROM annotations ORDER BY timepoint,x_px"
        ).fetchall()
        links = connection.execute(
            "SELECT link_id,link_label FROM link_reviews"
        ).fetchall()
    assert review_count == 1
    assert review == ("wrong_link", "low", '["shape_mismatch"]')
    assert annotations == [
        ("T0", 10.0, 20.0, None, "cell", "dead"),
        ("T1", 12.0, 20.0, "track-1", "debris", "not_applicable"),
        ("T1", 14.0, 21.0, "track-1", "cell", "dead"),
    ]
    assert links == [("T0-T1", "wrong")]


def test_initialize_database_migrates_existing_lineage_review_table(tmp_path):
    database = tmp_path / "annotations.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE lineage_reviews (
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
                reviewer TEXT,
                notes TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
    initialize_database(database)
    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(lineage_reviews)")
        }
    assert {"lineage_status", "review_confidence", "issue_tags"}.issubset(columns)

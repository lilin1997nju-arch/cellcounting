import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from cellvision.review_server import (
    _growth_region_contours,
    _visible_v2_review_instances,
    _with_final_decisions,
    PointSelection,
    initialize_database,
    save_annotation,
    save_lineage_review,
)
from cellvision.review_quick_review import V3_TRACK_REVIEW_LABELS, _quick_review_filter_metrics
from PIL import Image


def test_final_decision_contract_uses_v3_wall_structure_result():
    frame = pd.DataFrame(
        [
            {
                "integrated_label": "invalid",
                "current_label": "invalid",
                "integrated_confidence": 0.67,
                "reviewed_label": None,
                "v3_reviewed_label": None,
                "v3_track_behavior": "wall_structure_invalid",
                "v3_proposed_label": "invalid",
                "v3_reason": "stable_wall_site_structure",
                "v3_behavior_score": 0.89,
                "v3_label_mode": "per_frame_evidence",
            }
        ]
    )

    result = _with_final_decisions(frame).iloc[0]

    assert result["final_label"] == "invalid"
    assert result["final_source"] == "v3_temporal"
    assert result["final_reason_code"] == "stable_wall_site_structure"
    assert "孔壁结构伪目标" in result["final_reason_text"]
    assert result["final_confidence"] == 0.89
    assert result["final_status"] == "determined"


def test_final_decision_contract_prefers_unified_human_track_review():
    frame = pd.DataFrame(
        [
            {
                "integrated_label": "debris",
                "current_label": "debris",
                "integrated_confidence": 0.70,
                "reviewed_label": "debris",
                "v3_reviewed_label": "dead_cell",
                "v3_track_behavior": "cell_to_debris",
                "v3_track_conclusion": "dead_cell",
                "v3_proposed_label": "debris",
                "v3_reason": "strong_t0_cell_monotonic_decline_with_morphology_degradation",
                "v3_behavior_score": 0.81,
                "v3_label_mode": "unified_track",
            }
        ]
    )

    result = _with_final_decisions(frame).iloc[0]

    assert result["final_label"] == "dead_cell"
    assert result["final_review_label"] == "debris"
    assert result["final_source"] == "human_track_review"
    assert result["final_confidence"] == 1.0


def test_cell_family_track_review_preserves_frame_multiplicity():
    frame = pd.DataFrame(
        [
            {
                "integrated_label": "single",
                "current_label": "single",
                "reviewed_label": "single",
                "v3_reviewed_label": "cell",
                "v3_proposed_label": "single",
                "integrated_confidence": 0.9,
            },
            {
                "integrated_label": "touching_doublet",
                "current_label": "touching_doublet",
                "reviewed_label": "touching_doublet",
                "v3_reviewed_label": "cell",
                "v3_proposed_label": "touching_doublet",
                "integrated_confidence": 0.9,
            },
            {
                "integrated_label": "cluster_3plus",
                "current_label": "cluster_3plus",
                "reviewed_label": "cluster_3plus",
                "v3_reviewed_label": "cell",
                "v3_proposed_label": "cluster_3plus",
                "integrated_confidence": 0.9,
            },
        ]
    )

    result = _with_final_decisions(frame)

    assert result["final_label"].tolist() == [
        "single",
        "touching_doublet",
        "cluster_3plus",
    ]
    assert set(result["final_source"]) == {"human_track_review"}
    assert set(result["final_reason_code"]) == {"human_track_cell_review"}


def test_legacy_exact_cell_track_review_no_longer_flattens_multiplicity():
    frame = pd.DataFrame(
        [
            {
                "integrated_label": "single",
                "current_label": "single",
                "reviewed_label": "single",
                "v3_reviewed_label": "single",
                "v3_proposed_label": "single",
                "integrated_confidence": 0.9,
            },
            {
                "integrated_label": "touching_doublet",
                "current_label": "single",
                "reviewed_label": "single",
                "v3_reviewed_label": "single",
                "v3_proposed_label": "touching_doublet",
                "integrated_confidence": 0.9,
            },
        ]
    )

    result = _with_final_decisions(frame)

    assert result["final_label"].tolist() == ["single", "touching_doublet"]


def test_track_review_accepts_cell_family_without_cell_subtype_options():
    html = (
        Path(__file__).parents[1] / "review-ui" / "auto-review.html"
    ).read_text(encoding="utf-8")
    select = html.split('id="v3TrackLabel"', 1)[1].split("</select>", 1)[0]

    assert "cell" in V3_TRACK_REVIEW_LABELS
    assert 'value="cell"' in select
    assert 'value="single"' not in select
    assert 'value="touching_doublet"' not in select
    assert 'value="cluster_3plus"' not in select


def test_quick_review_filter_metrics_use_endpoint_day2_and_manual_verdict():
    local = pd.DataFrame([
        {"timepoint": "T0", "final_review_label": "single", "current_label": "single"},
        {"timepoint": "T1", "final_review_label": "touching_doublet", "current_label": "touching_doublet"},
        {"timepoint": "T2", "final_review_label": "single", "current_label": "single"},
        {"timepoint": "T2", "final_review_label": "debris", "current_label": "debris"},
        {"timepoint": "T1", "final_review_label": "debris", "current_label": "debris"},
    ])

    metrics = _quick_review_filter_metrics(
        local,
        {"t0_cell_units": 5, "t2_cell_units": 4, "review_decision": "approved"},
        {"day14_sheet_coverage_pct": 12.75},
    )

    assert metrics == {
        "endpoint_coverage_pct": 12.75,
        "day0_cell_count": 5,
        "day1_cell_count": 2,
        "day2_cell_count": 4,
        "day2_debris_count": 1,
        "manual_review_decision": "approved",
    }


def test_auto_review_exposes_pre_filter_and_well_verdict_shortcuts():
    root = Path(__file__).parents[1] / "review-ui"
    html = (root / "auto-review.html").read_text(encoding="utf-8")
    script = (root / "auto-review.js").read_text(encoding="utf-8")

    assert "本板筛选" in html
    assert 'id="plateDay0CellsMin"' in html
    assert 'id="plateDay1CellsMin"' in html
    assert 'id="plateManualVerdictFilter"' in html
    assert '<option value="unclassified">未判定</option>' in html
    assert '<option value="approved">合格</option>' in html
    assert '<option value="pending">待定</option>' in html
    assert '<option value="rejected">排除</option>' in html
    assert 'data-well-verdict="approved"' in html
    assert 'data-well-verdict="pending"' in html
    assert 'data-well-verdict="rejected"' in html
    assert 'data-well-verdict="unclassified"' in html
    assert 'q: "approved"' in script
    assert 'w: "pending"' in script
    assert 'e: "rejected"' in script
    assert "verdicts[event.key.toLowerCase()]" in script
    assert '<kbd>Q</kbd> 合格' in html
    assert '<kbd>W</kbd> 待定' in html
    assert '<kbd>E</kbd> 排除' in html
    assert '<kbd>7</kbd> 合格' not in html
    assert 'api("/api/screening-review"' in script
    assert 'unclassified: "未判定"' in script
    assert 'finiteFilter("day0_cells_min")' in script
    assert 'finiteFilter("day1_cells_min")' in script
    assert 'manualVerdict: "manual_verdict"' in script
    assert 'entryFilters.manualVerdict !== "all"' in script


def test_auto_review_supports_project_pending_well_queue():
    root = Path(__file__).parents[1] / "review-ui"
    html = (root / "auto-review.html").read_text(encoding="utf-8")
    script = (root / "auto-review.js").read_text(encoding="utf-8")

    assert 'id="pendingQueueNotice"' in html
    assert 'entryFilterParameters.get("pending_queue")' in script
    assert 'entryFilters.manualVerdict = "pending"' in script
    assert 'target.searchParams.set("manual_verdict", "pending")' in script
    assert 'state.pendingQueueVisited.add(well)' in script
    assert "advancePendingPlateQueue()" in script


def test_auto_review_cell_total_tracks_labels_until_human_override():
    root = Path(__file__).parents[1] / "review-ui"
    html = (root / "auto-review.html").read_text(encoding="utf-8")
    stylesheet = (root / "auto-review.css").read_text(encoding="utf-8")
    script = (root / "auto-review.js").read_text(encoding="utf-8")

    assert 'class="cell-total-control"' in html
    assert 'class="cell-total-down"' in html
    assert 'class="cell-total-up"' in html
    assert 'class="timepoint-control-row"' in html
    assert html.index('class="cell-total-control"') > html.index('class="timepoint-control-row"')
    assert "恢复自动" in html
    assert "function automaticTimepointCellTotal(timepoint)" in script
    assert 'saved?.source === "human"' in script
    assert 'renderCellTotalControl(timepoint, card)' in script
    assert 'api("/api/timepoint-cell-count-review"' in script
    assert "scheduleTimepointCellTotalSave(well, timepoint, cellCount)" in script
    assert "setTimeout(() => flushTimepointCellTotalSave(key), 350)" in script
    assert "await flushPendingCellCountSaves(well);" in script
    assert "state.busy = true;\n  for (const item of reviewTimepoints)" not in script
    assert ".cell-total-control[hidden]{display:none}" in stylesheet
    assert ".timepoint-control-row{grid-column:1 / -1;grid-row:2" in stylesheet
    assert "grid-template-columns: repeat(3, minmax(380px, 1fr))" in stylesheet
    assert "renderWellVerdict();\n    for (const timepoint" in script


def test_late_review_copy_requires_full_well_view_without_default_zoom():
    root = Path(__file__).parents[1] / "review-ui"
    html = (root / "auto-review.html").read_text(encoding="utf-8")
    script = (root / "auto-review.js").read_text(encoding="utf-8")

    assert "晚期图像只展示完整孔视野" in html
    assert "不自动定位或放大" in html
    assert "applyRepresentativeLateView" not in script


def test_final_decision_contract_exposes_v3_override_as_review_label():
    frame = pd.DataFrame(
        [
            {
                "integrated_label": "single",
                "current_label": "single",
                "integrated_confidence": 0.81,
                "reviewed_label": None,
                "v3_reviewed_label": None,
                "v3_track_behavior": "stable_debris",
                "v3_proposed_label": "debris",
                "v3_reason": "stable_three_frame_debris",
                "v3_behavior_score": 0.72,
                "v3_label_mode": "per_frame_evidence",
            }
        ]
    )

    result = _with_final_decisions(frame).iloc[0]

    assert result["current_label"] == "single"
    assert result["final_label"] == "debris"
    assert result["final_review_label"] == "debris"


def test_auto_review_uses_authoritative_label_for_controls_and_save():
    source = (
        Path(__file__).parents[1] / "review-ui" / "auto-review.js"
    ).read_text(encoding="utf-8")

    assert "button.dataset.label === editableDecisionLabel(object)" in source
    assert "const label = editableDecisionLabel(object);" in source
    assert "reviewed_label: editableDecisionLabel(object)" in source
    assert 'v3TrackLabelFor(object) === "cell"' in source
    assert "cellSubtypeLabels.includes(label)" in source
    assert 'state.v3TrackLabels.set(track.track_id, "cell")' in source


def test_final_decision_contract_keeps_v2_label_for_review_only_v3_state():
    frame = pd.DataFrame(
        [
            {
                "integrated_label": "single",
                "current_label": "single",
                "integrated_confidence": 0.74,
                "reviewed_label": None,
                "v3_reviewed_label": None,
                "v3_track_behavior": "wall_uncertain",
                "v3_proposed_label": "uncertain",
                "v3_reason": "wall_site_insufficient_structure_evidence",
                "v3_behavior_score": 0.68,
                "v3_label_mode": "per_frame_evidence",
            }
        ]
    )

    result = _with_final_decisions(frame).iloc[0]

    assert result["final_label"] == "single"
    assert result["final_source"] == "integrated_model"
    assert result["final_status"] == "needs_review"


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

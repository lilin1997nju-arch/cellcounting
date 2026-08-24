import os
import sqlite3

import pandas as pd
import pytest

from cellvision.well_screening import (
    _classify_well_status,
    _latest_prediction_source,
    _manual_missed_objects,
    _merge_selected_well_rows,
    build_well_screening,
    ensure_well_timepoint_cell_count_review_table,
    late_growth_gate,
    save_late_growth_review,
)
from cellvision.multiplicity import ensure_integrated_review_table, save_integrated_reviews
from cellvision.late_growth_inference import (
    DenseGrowthMetrics,
    _representative_instance_center,
    dense_growth_decision,
    late_growth_decision,
)


def test_late_growth_count_gate_expands_before_final_negative():
    assert late_growth_decision(2, 2, final_stage=False) == "continue_search"
    assert late_growth_decision(2, 4, final_stage=False) == "obvious_growth"
    assert late_growth_decision(2, 3, final_stage=True) == "no_growth"
    assert late_growth_decision(0, 3, final_stage=False) == "obvious_growth"


def test_dense_growth_channel_does_not_require_separable_instances():
    baseline = DenseGrowthMetrics(1000, 0.0001, 0.004, 30, 100, 100)
    colony = DenseGrowthMetrics(5000, 0.052, 0.31, 400, 500, 600)
    assert dense_growth_decision(baseline, colony)
    focus_change_only = DenseGrowthMetrics(1100, 0.012, 0.04, 35, 500, 600)
    assert not dense_growth_decision(baseline, focus_change_only)


def test_representative_instance_center_prefers_densest_cell_region():
    frame = pd.DataFrame(
        {
            "x_px": [100.0, 120.0, 140.0, 1800.0],
            "y_px": [100.0, 115.0, 125.0, 1800.0],
            "integrated_label": [
                "single",
                "touching_doublet",
                "cluster_3plus",
                "single",
            ],
        }
    )
    center = _representative_instance_center(frame, radius=100.0)
    assert center is not None
    assert center[0] < 200
    assert center[1] < 200
    assert center[2:] == (3, 6)


def test_late_growth_gate_skips_only_when_all_available_images_are_no_growth():
    assert late_growth_gate(
        {"T3", "T4"}, {"T3": "no_growth", "T4": "no_growth"}
    ) == ("no_growth", True)
    assert late_growth_gate(
        {"T3", "T4"}, {"T3": "no_growth", "T4": "pending"}
    ) == ("pending", False)
    assert late_growth_gate(
        {"T3", "T4"}, {"T3": "no_growth", "T4": "obvious_growth"}
    ) == ("obvious_growth", False)
    assert late_growth_gate(
        {"T3", "T4"}, {"T3": "no_growth", "T4": "uncertain"}
    ) == ("uncertain", False)


def test_late_growth_gate_handles_one_or_no_available_late_image():
    assert late_growth_gate({"T3"}, {"T3": "no_growth"}) == (
        "no_growth",
        True,
    )
    assert late_growth_gate(set(), {}) == ("unavailable", False)


def test_late_growth_confirms_single_origin_activity():
    status = _classify_well_status(
        {"T0": 1, "T1": 1, "T2": 1},
        t0_cell_instances=1,
        t0_has_uncertain=False,
        confidence=0.60,
        late_growth_status="obvious_growth",
    )
    assert status == ("single_active", True, False, True, True)


def test_corrected_single_is_not_multi_origin():
    status = _classify_well_status(
        {"T0": 1, "T1": 2, "T2": 3},
        t0_cell_instances=1,
        t0_has_uncertain=False,
        confidence=1.0,
        late_growth_status="obvious_growth",
    )
    assert status[0] == "single_active"
    assert status[1] is True
    assert status[2] is False


def test_single_origin_followed_by_touching_or_cluster_is_active_even_if_confidence_is_low():
    status = _classify_well_status(
        {"T0": 1, "T1": 2, "T2": 3},
        t0_cell_instances=1,
        t0_has_uncertain=False,
        confidence=0.45,
        late_growth_status="unavailable",
    )
    assert status[0] == "single_active"
    assert status[1] is True
    assert status[3] is True
    assert status[4] is False


def test_incremental_screening_replaces_only_selected_well():
    previous = pd.DataFrame(
        {
            "well": ["A1", "A2", "B1"],
            "screening_status": ["no_cell", "multi_origin", "no_cell"],
        }
    )
    updated = pd.DataFrame(
        {
            "well": ["A2"],
            "screening_status": ["single_active"],
        }
    )
    merged = _merge_selected_well_rows(previous, updated, {"a2"})
    assert merged.to_dict(orient="records") == [
        {"well": "A1", "screening_status": "no_cell"},
        {"well": "A2", "screening_status": "single_active"},
        {"well": "B1", "screening_status": "no_cell"},
    ]


def test_well_screening_uses_latest_published_v3_predictions(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    v2 = predictions / "latest_v2_predictions.csv"
    v3 = predictions / "latest_v3_predictions.csv"
    v2.write_text("candidate_id\nold\n", encoding="utf-8")
    v3.write_text("candidate_id\nnew\n", encoding="utf-8")
    os.utime(v2, ns=(1, 1))

    assert _latest_prediction_source(
        {"paths": {"artifact_root": str(tmp_path)}}
    ) == v3


def test_save_late_growth_review_upserts_by_well_and_timepoint(tmp_path):
    database = tmp_path / "annotations.db"
    save_late_growth_review(
        database, "a7", "t3", "no_growth", "tester", "first", "2026-08-02T00:00:00Z"
    )
    save_late_growth_review(
        database, "A7", "T3", "obvious_growth", "tester", "updated", "2026-08-02T00:01:00Z"
    )
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT well, timepoint, decision, notes FROM late_growth_reviews"
        ).fetchall()
    assert rows == [("A7", "T3", "obvious_growth", "updated")]


def test_save_late_growth_review_rejects_unsupported_values(tmp_path):
    database = tmp_path / "annotations.db"
    with pytest.raises(ValueError):
        save_late_growth_review(
            database, "A7", "T2", "no_growth", "tester", "", "now"
        )
    with pytest.raises(ValueError):
        save_late_growth_review(
            database, "A7", "T3", "maybe", "tester", "", "now"
        )


def test_manual_missed_objects_persist_across_inference_rounds(tmp_path):
    database = tmp_path / "annotations.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE quick_missed_objects (
              candidate_id TEXT, round_id TEXT, well TEXT, timepoint TEXT,
              x_px REAL, y_px REAL, diameter_px REAL, reviewed_label TEXT,
              updated_at TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO quick_missed_objects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("A2:T0:manual:1", "old-round", "A2", "T0", 10, 20, 12, "single", "now"),
        )

    manual = _manual_missed_objects(database, "new-round")

    assert list(manual["candidate_id"]) == ["A2:T0:manual:1"]
    assert bool(manual.iloc[0]["is_manual_missed"]) is True


def test_cell_total_recalculates_from_labels_until_human_override(tmp_path):
    artifact = tmp_path / "artifacts"
    predictions = artifact / "predictions"
    manifests = artifact / "manifests"
    predictions.mkdir(parents=True)
    manifests.mkdir(parents=True)
    rows = []
    for index, timepoint in enumerate(("T0", "T1", "T2"), start=1):
        rows.append({
            "candidate_id": f"A1:{timepoint}:1",
            "well": "A1",
            "timepoint": timepoint,
            "x_px": 100.0 * index,
            "y_px": 100.0 * index,
            "diameter_px": 12.0,
            "integrated_label": "single",
            "integrated_confidence": 0.95,
            "integrated_round_id": "round-1",
        })
    pd.DataFrame(rows).to_csv(
        predictions / "latest_integrated_predictions.csv", index=False
    )
    pd.DataFrame([
        {"well": "A1", "timepoint": timepoint, "decode_status": "ok"}
        for timepoint in ("T0", "T1", "T2")
    ]).to_csv(manifests / "images.csv", index=False)
    config = {"paths": {"artifact_root": str(artifact)}}
    database = artifact / "annotations" / "annotations.db"
    database.parent.mkdir(parents=True)

    build_well_screening(config, database)
    initial = pd.read_csv(predictions / "latest_well_screening.csv").iloc[0]
    assert initial["t1_cell_units"] == 1
    assert initial["t1_cell_units_source"] == "automatic"

    ensure_integrated_review_table(database)
    save_integrated_reviews(database, "round-1", [{
        "candidate_id": "A1:T1:1",
        "predicted_label": "single",
        "reviewed_label": "touching_doublet",
    }], "tester")
    build_well_screening(config, database, selected_wells={"A1"})
    recalculated = pd.read_csv(predictions / "latest_well_screening.csv").iloc[0]
    assert recalculated["t1_cell_units"] == 2
    assert recalculated["t1_cell_units_source"] == "automatic"

    ensure_well_timepoint_cell_count_review_table(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO well_timepoint_cell_count_reviews VALUES (?, ?, ?, ?, ?)",
            ("A1", "T1", 7, "tester", "now"),
        )
    save_integrated_reviews(database, "round-1", [{
        "candidate_id": "A1:T1:1",
        "predicted_label": "single",
        "reviewed_label": "cluster_3plus",
    }], "tester")
    build_well_screening(config, database, selected_wells={"A1"})
    locked = pd.read_csv(predictions / "latest_well_screening.csv").iloc[0]
    assert locked["t1_cell_units"] == 7
    assert locked["t1_cell_units_source"] == "human"

    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM well_timepoint_cell_count_reviews WHERE well='A1' AND timepoint='T1'"
        )
    build_well_screening(config, database, selected_wells={"A1"})
    restored = pd.read_csv(predictions / "latest_well_screening.csv").iloc[0]
    assert restored["t1_cell_units"] == 3
    assert restored["t1_cell_units_source"] == "automatic"

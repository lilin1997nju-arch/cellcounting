from __future__ import annotations

import json

import numpy as np
import pandas as pd
from PIL import Image

from cellvision.gated_screening import (
    build_gated_plate_report,
    classify_gated_well,
    locate_day7_dense_regions,
)


def test_day14_negative_skips_all_early_computation() -> None:
    result = classify_gated_well(
        day14_available=True,
        day14_obvious_growth=False,
        early_results_available=False,
    )
    assert result["final_category"] == "no_obvious_growth"
    assert result["skip_early_computation"] is True


def test_single_and_multi_origin_are_mutually_exclusive() -> None:
    single = classify_gated_well(
        day14_available=True,
        day14_obvious_growth=True,
        t0_cell_units=1,
        t1_cell_units=2,
        t2_cell_units=4,
        t0_cell_instances=1,
    )
    touching_doublet = classify_gated_well(
        day14_available=True,
        day14_obvious_growth=True,
        t0_cell_units=2,
        t1_cell_units=4,
        t2_cell_units=8,
        t0_cell_instances=1,
    )
    assert single["final_category"] == "single_cell_origin"
    assert touching_doublet["final_category"] == "multi_cell_origin"


def test_required_undetermined_reasons() -> None:
    missing_t0 = classify_gated_well(
        day14_available=True,
        day14_obvious_growth=True,
        t0_cell_units=0,
        t1_cell_units=1,
        t2_cell_units=2,
    )
    no_division = classify_gated_well(
        day14_available=True,
        day14_obvious_growth=True,
        t0_cell_units=1,
        t1_cell_units=1,
        t2_cell_units=1,
        t0_cell_instances=1,
    )
    assert missing_t0["undetermined_reason"] == "t0_missing_later_detected"
    assert no_division["undetermined_reason"] == "t0_t2_no_division"


def test_day7_locator_prefers_dense_region_over_thin_line(tmp_path) -> None:
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:, 50:53] = 255
    mask[300:390, 320:420] = 255
    path = tmp_path / "A2-cf.tif"
    Image.fromarray(mask).save(path)
    result = locate_day7_dense_regions(path, roi_size_px=128, max_regions=1)
    assert result["available"] is True
    region = result["regions"][0]
    assert region["center_x"] > 300
    assert region["center_y"] > 280


def test_plate_report_has_96_wells_and_positive_queue(tmp_path) -> None:
    day14 = pd.DataFrame([
        {
            "group_id": "plate-1",
            "well": "A1",
            "is_positive_control": True,
            "day14_obvious_sheet_growth": True,
            "screening_decision": "retain",
            "raw_image_path": "control.tif",
            "cf_mask_path": "control-cf.tif",
        },
        {
            "group_id": "plate-1",
            "well": "A2",
            "is_positive_control": False,
            "day14_obvious_sheet_growth": True,
            "screening_decision": "retain",
            "raw_image_path": "A2.tif",
            "cf_mask_path": "A2-cf.tif",
        },
        {
            "group_id": "plate-1",
            "well": "A3",
            "is_positive_control": False,
            "day14_obvious_sheet_growth": False,
            "screening_decision": "exclude",
            "raw_image_path": "A3.tif",
            "cf_mask_path": "A3-cf.tif",
        },
    ])
    early = pd.DataFrame([
        {
            "well": "A2",
            "t0_cell_units": 1,
            "t1_cell_units": 2,
            "t2_cell_units": 4,
            "t0_cell_instances": 1,
            "early_timepoints_complete": True,
        }
    ])
    day14_path = tmp_path / "day14.csv"
    early_path = tmp_path / "early.csv"
    day14.to_csv(day14_path, index=False)
    early.to_csv(early_path, index=False)
    result = build_gated_plate_report(
        day14_path,
        "plate-1",
        tmp_path / "report",
        early_screening_csv=early_path,
        locate_day7=False,
    )
    assert result["well_count"] == 96
    assert result["day14_positive_sample_wells"] == 1
    rows = {row["well"]: row for row in result["wells"]}
    assert rows["A1"]["final_category"] == "positive_control"
    assert rows["A2"]["final_category"] == "single_cell_origin"
    assert rows["A3"]["final_category"] == "no_obvious_growth"
    assert rows["A4"]["undetermined_reason"] == "day14_missing_or_unreadable"
    parsed = json.loads((tmp_path / "report" / "plate_overview.json").read_text(encoding="utf-8"))
    assert parsed["well_count"] == 96


def test_day14_human_no_growth_override_replaces_automatic_positive(tmp_path) -> None:
    day14 = pd.DataFrame([
        {
            "group_id": "plate-1",
            "well": "G3",
            "is_positive_control": False,
            "day14_obvious_sheet_growth": True,
            "screening_decision": "retain",
            "raw_image_path": "G3.tif",
            "cf_mask_path": "G3-cf.tif",
        }
    ])
    day14_path = tmp_path / "day14.csv"
    day14.to_csv(day14_path, index=False)

    result = build_gated_plate_report(
        day14_path,
        "plate-1",
        tmp_path / "report",
        locate_day7=False,
        day14_growth_overrides={"g3": "no_growth"},
    )

    rows = {row["well"]: row for row in result["wells"]}
    assert rows["G3"]["day14_obvious_growth"] is False
    assert rows["G3"]["day14_screening_decision"] == "human_no_growth"
    assert rows["G3"]["final_category"] == "no_obvious_growth"

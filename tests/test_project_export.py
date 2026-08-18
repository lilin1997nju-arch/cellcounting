import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from cellvision.project_server import _export_project_rows, _write_project_result_excel


def _write_plate_results(root: Path, board_id: str, *, include_objects: bool = True) -> dict:
    artifact_root = root / board_id / "artifacts"
    gated_root = artifact_root / "gated"
    predictions_root = artifact_root / "predictions"
    gated_root.mkdir(parents=True)
    predictions_root.mkdir(parents=True)

    pd.DataFrame([
        {
            "well": "A1",
            "final_category": "single_cell_origin",
            "final_category_label": "单细胞来源",
        },
        {
            "well": "A2",
            "final_category": "no_obvious_growth",
            "final_category_label": "无明显生长",
        },
    ]).to_csv(gated_root / "plate_overview.csv", index=False)
    pd.DataFrame([
        {"well": "A1", "t0_cell_units": 6, "t1_cell_units": 2, "t2_cell_units": 2},
        {"well": "A2", "t0_cell_units": 3, "t1_cell_units": 0, "t2_cell_units": 0},
    ]).to_csv(predictions_root / "latest_well_screening.csv", index=False)
    if include_objects:
        pd.DataFrame([
            {"well": "A1", "timepoint": "T0", "screen_label": "single"},
            {"well": "A1", "timepoint": "T0", "screen_label": "touching_doublet"},
            {"well": "A1", "timepoint": "T0", "screen_label": "cluster_3plus"},
            {"well": "A1", "timepoint": "T1", "screen_label": "single"},
            {"well": "A1", "timepoint": "T1", "screen_label": "single"},
            {"well": "A1", "timepoint": "T2", "screen_label": "touching_doublet"},
        ]).to_csv(predictions_root / "latest_screening_objects.csv", index=False)
    return {
        "slug": board_id.lower(),
        "board_id": board_id,
        "artifact_root": str(artifact_root),
        "gated_output_dir": str(gated_root),
    }


def test_project_export_contains_all_boards_and_weighted_counts(tmp_path: Path):
    manifest_path = tmp_path / "project.json"
    manifest = {
        "project_id": "demo",
        "project_name": "Demo project",
        "plates": [
            _write_plate_results(tmp_path, "Board-1"),
            _write_plate_results(tmp_path, "Board-2", include_objects=False),
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows, warnings = _export_project_rows(manifest, manifest_path, task_name="Demo task")

    assert warnings == {}
    assert len(rows) == 4
    first = next(row for row in rows if row["板子名称"] == "Board-1" and row["孔号"] == "A1")
    assert first["任务名称"] == "Demo task"
    assert first["孔结论"] == "单细胞来源"
    assert first["T0单细胞个数"] == 1
    assert first["T0双细胞个数"] == 1
    assert first["T0多细胞个数"] == 1
    assert first["T0推测细胞总数"] == 6
    assert first["T1推测细胞总数"] == 2
    assert first["T2推测细胞总数"] == 2

    fallback = next(row for row in rows if row["板子名称"] == "Board-2" and row["孔号"] == "A2")
    assert fallback["T0单细胞个数"] == 0
    assert fallback["T0推测细胞总数"] == 3


def test_project_export_writes_readable_workbook(tmp_path: Path):
    manifest_path = tmp_path / "project.json"
    manifest = {
        "project_id": "demo",
        "project_name": "Demo project",
        "plates": [_write_plate_results(tmp_path, "Board-1")],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    destination = tmp_path / "exports" / "demo.xlsx"

    summary = _write_project_result_excel(manifest, manifest_path, destination)

    assert summary["row_count"] == 2
    workbook = load_workbook(destination, data_only=True)
    sheet = workbook["检测结果"]
    assert sheet.max_row == 3
    assert sheet.max_column == 16
    assert sheet.freeze_panes == "A2"
    assert sheet.auto_filter.ref == "A1:P3"
    assert [cell.value for cell in sheet[1]] == [
        "任务名称", "板子名称", "孔号", "孔结论",
        "T0单细胞个数", "T0双细胞个数", "T0多细胞个数",
        "T1单细胞个数", "T1双细胞个数", "T1多细胞个数",
        "T2单细胞个数", "T2双细胞个数", "T2多细胞个数",
        "T0推测细胞总数", "T1推测细胞总数", "T2推测细胞总数",
    ]

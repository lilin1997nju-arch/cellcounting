import json
import sqlite3

import numpy as np
import pandas as pd

from cellvision.review_server import initialize_database
from cellvision.v2_instance_inference import _rle
from cellvision.v2_mask_review import (
    MASK_SIZE,
    mask_review_summary,
    round_directory,
    save_mask_review,
)


def _review_config(tmp_path):
    return {"paths": {"artifact_root": str(tmp_path / "artifacts")}}


def test_manual_mask_review_recomputes_derived_fields_and_persists(tmp_path):
    config = _review_config(tmp_path)
    round_id = "p0-mask-review-test"
    folder = round_directory(config, round_id)
    mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    mask[43:53, 43:53] = True
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "B1:T0:test:1",
                "well": "B1",
                "timepoint": "T0",
                "x_px": 50.0,
                "y_px": 50.0,
                "integrated_label": "single",
                "integrated_confidence": 0.9,
                "cell_probability": 0.9,
                "debris_probability": 0.02,
                "invalid_probability": 0.02,
                "v2_mask_valid": True,
                "v2_mask_rle": _rle(mask),
                "v2_mask_origin_x": 2,
                "v2_mask_origin_y": 2,
                "v2_contour_json": "[]",
                "v2_instance_area_px": int(mask.sum()),
                "v2_instance_diameter_px": 2.0,
                "v2_instance_confidence": 0.9,
                "v2_objectness": 0.9,
                "v2_wall_rejected": False,
                "v2_refinement_status": "refined",
                "v2_refinement_area_ratio": 1.0,
                "v2_refinement_iou": 1.0,
            }
        ]
    )
    for name in (
        "latest_v2_predictions.csv",
        "latest_v2_pre_temporal_predictions.csv",
        "reviewed_v2_predictions.csv",
    ):
        frame.to_csv(folder / name, index=False)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "round_id": round_id,
                "mask_size": MASK_SIZE,
                "pre_temporal_predictions": str(folder / "latest_v2_pre_temporal_predictions.csv"),
                "reviewed_predictions": str(folder / "reviewed_v2_predictions.csv"),
                "created_at": "test",
                "algorithm_version": "test",
            }
        ),
        encoding="utf-8",
    )
    database = initialize_database(tmp_path / "annotations.db")
    edited = mask.copy()
    edited[40, 48] = True
    result = save_mask_review(
        config,
        database,
        round_id=round_id,
        candidate_id="B1:T0:test:1",
        decision="edited",
        reviewed_mask_rle=_rle(edited),
        reviewer="tester",
        notes="补齐边界",
    )

    saved = pd.read_csv(folder / "reviewed_v2_predictions.csv")
    assert result["reviewed_area_px"] == int(edited.sum())
    assert int(saved.loc[0, "v2_instance_area_px"]) == int(edited.sum())
    assert saved.loc[0, "v2_mask_review_status"] == "edited"
    assert saved.loc[0, "v2_mask_reviewed_by"] == "tester"
    with sqlite3.connect(database) as connection:
        stored = connection.execute(
            "SELECT decision, reviewed_area_px, notes FROM v2_mask_reviews"
        ).fetchone()
    assert stored == ("edited", int(edited.sum()), "补齐边界")
    assert mask_review_summary(config, database, round_id)["edited_count"] == 1

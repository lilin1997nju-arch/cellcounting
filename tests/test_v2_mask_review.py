import json
import sqlite3

import numpy as np
import pandas as pd

from cellvision.review_server import initialize_database
from cellvision.v2_instance_inference import _rle
from cellvision.v2_mask_review import (
    MASK_SIZE,
    mask_review_candidate,
    mask_review_candidates,
    mask_review_summary,
    round_directory,
    save_mask_review,
)
from cellvision.v2_instance_dataset import _decode_review_mask
from cellvision.train_v2_instance import _instance_sampler_weights


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
        output_frame = frame.copy()
        # Keep the review metadata columns present but empty.  After a CSV
        # round-trip pandas infers these all-empty columns as float64; saving
        # a reviewer string must still work.
        output_frame["v2_mask_review_status"] = "pending"
        output_frame["v2_mask_reviewed_by"] = ""
        output_frame["v2_mask_reviewed_at"] = ""
        output_frame["v2_mask_review_notes"] = ""
        output_frame["v2_reviewed_area_px"] = 0
        output_frame["v2_reviewed_diameter_px"] = 0.0
        output_frame.to_csv(folder / name, index=False)
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
    accepted = save_mask_review(
        config,
        database,
        round_id=round_id,
        candidate_id="B1:T0:test:1",
        decision="accepted",
        reviewed_mask_rle=None,
        reviewer="tester",
        notes="接受模型轮廓",
    )
    assert accepted["reviewed_area_px"] == int(mask.sum())
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


def test_review_mask_rle_decoder_preserves_empty_negative_and_pixels():
    assert not _decode_review_mask("[]", MASK_SIZE).any()
    mask = _decode_review_mask(
        f"[[0, 2], [{MASK_SIZE * MASK_SIZE - 1}, 1]]", MASK_SIZE
    )
    assert int(mask.sum()) == 3
    assert bool(mask[0, 0])
    assert bool(mask[-1, -1])


def test_comparison_round_exposes_old_and_new_masks_and_saves_manual_edit(tmp_path):
    config = _review_config(tmp_path)
    round_id = "model-comparison-test"
    folder = round_directory(config, round_id)
    old_mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    old_mask[44:50, 44:50] = True
    new_mask = old_mask.copy()
    new_mask[43:51, 43:51] = True
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "B1:T0:comparison:1",
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
                "v2_mask_rle": _rle(new_mask),
                "v2_mask_origin_x": 2,
                "v2_mask_origin_y": 2,
                "v2_contour_json": "[]",
                "v2_instance_area_px": int(new_mask.sum()),
                "v2_instance_diameter_px": 2.0,
                "v2_instance_confidence": 0.9,
                "v2_objectness": 0.9,
                "v2_wall_rejected": False,
                "v2_refinement_status": "refined",
                "v2_refinement_area_ratio": 1.0,
                "v2_refinement_iou": 1.0,
                "comparison_old_mask_rle": _rle(old_mask),
                "comparison_old_mask_valid": True,
                "comparison_old_area_px": int(old_mask.sum()),
                "comparison_old_diameter_px": 2.0,
                "comparison_old_confidence": 0.8,
                "comparison_old_refinement_status": "refined",
                "comparison_old_refinement_area_ratio": 1.0,
                "comparison_old_refinement_iou": 1.0,
                "comparison_new_mask_rle": _rle(new_mask),
                "comparison_new_mask_valid": True,
                "comparison_new_area_px": int(new_mask.sum()),
                "comparison_new_diameter_px": 2.0,
                "comparison_new_confidence": 0.9,
                "comparison_new_refinement_status": "refined",
                "comparison_source_config": "configs/generated/holdout.yaml",
                "comparison_old_checkpoint": "old.pt",
                "comparison_new_checkpoint": "new.pt",
            }
        ]
    )
    output = frame.copy()
    output["v2_mask_review_status"] = "pending"
    output["v2_mask_reviewed_by"] = ""
    output["v2_mask_reviewed_at"] = ""
    output["v2_mask_review_notes"] = ""
    output["v2_reviewed_area_px"] = int(new_mask.sum())
    output["v2_reviewed_diameter_px"] = 2.0
    output.to_csv(folder / "reviewed_v2_predictions.csv", index=False)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "round_id": round_id,
                "kind": "comparison",
                "mask_size": MASK_SIZE,
                "reviewed_predictions": str(folder / "reviewed_v2_predictions.csv"),
                "created_at": "test",
                "algorithm_version": "test",
            }
        ),
        encoding="utf-8",
    )
    database = initialize_database(tmp_path / "annotations.db")

    queue = mask_review_candidates(config, database, round_id)
    assert len(queue) == 1
    assert queue[0]["comparison"] is True
    assert queue[0]["old_model_area_px"] == int(old_mask.sum())
    assert queue[0]["new_model_area_px"] == int(new_mask.sum())

    detail = mask_review_candidate(config, database, round_id, queue[0]["candidate_id"])
    assert detail is not None
    assert detail["old_model_mask_rle"] == _rle(old_mask)
    assert detail["new_model_mask_rle"] == _rle(new_mask)

    save_mask_review(
        config,
        database,
        round_id=round_id,
        candidate_id=queue[0]["candidate_id"],
        decision="edited",
        reviewed_mask_rle=_rle(old_mask),
        reviewer="tester",
        notes="采用旧模型轮廓",
    )
    saved_detail = mask_review_candidate(
        config, database, round_id, queue[0]["candidate_id"]
    )
    assert saved_detail is not None
    assert saved_detail["decision"] == "edited"
    assert saved_detail["reviewed_area_px"] == int(old_mask.sum())
    assert mask_review_summary(config, database, round_id)["edited_count"] == 1


def test_review_sample_multipliers_increase_exact_review_sampling_weight():
    weights = _instance_sampler_weights(
        np.asarray(["invalid", "single"]),
        np.asarray(["old", "old"]),
        np.asarray(["v2_mask_review_rejected", "human_review"]),
        {"v2_mask_review_rejected": 8.0},
    )
    assert weights[0] > weights[1]

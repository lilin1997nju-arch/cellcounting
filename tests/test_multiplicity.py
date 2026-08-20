import sqlite3

import pandas as pd

from cellvision.multiplicity import (
    carry_forward_integrated_reviews,
    _filter_unconfirmed_wall_queue_candidates,
    _multiplicity_targets_for_source,
    multiplicity_stats,
    read_multiplicity_labels,
    save_categorized_review_labels,
    save_integrated_reviews,
    save_multiplicity_labels,
)
from cellvision.teaching import read_teaching_labels


def test_targeted_queue_drops_unconfirmed_wall_residuals_but_keeps_clear_buffer_cells(tmp_path):
    database = tmp_path / "annotations.db"
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "wall-residual",
                "candidate_zone": "wall_residual",
                "candidate_source": "wall_residual_peak",
                "cell_probability": 0.30,
                "predicted_label": "invalid",
            },
            {
                "candidate_id": "wall-buffer-clear-cell",
                "candidate_zone": "wall_cell_buffer",
                "candidate_source": "cf_component",
                "cell_probability": 0.80,
                "predicted_label": "cell",
            },
            {
                "candidate_id": "wall-rescue-weak",
                "candidate_zone": "wall_cell_rescue",
                "candidate_source": "wall_cell_rescue_peak",
                "cell_probability": 0.20,
                "wall_rescue_blobness": 0.10,
                "dense_response": 5.0,
                "predicted_label": "invalid",
            },
            {
                "candidate_id": "interior-cell",
                "candidate_zone": "well_interior",
                "candidate_source": "multiscale_dense_peak",
                "cell_probability": 0.70,
                "predicted_label": "cell",
            },
        ]
    )
    filtered = _filter_unconfirmed_wall_queue_candidates(
        {"multiplicity_review_queue": {"wall_filter_enabled": True}},
        database,
        frame,
    )
    assert set(filtered["candidate_id"]) == {
        "wall-buffer-clear-cell",
        "interior-cell",
    }


def test_multiplicity_labels_upsert_and_stats(tmp_path):
    database = tmp_path / "annotations.db"
    item = {
        "candidate_id": "H6:T1:cf:154",
        "well": "H6",
        "timepoint": "T1",
        "x_px": 100.0,
        "y_px": 200.0,
        "label": "single",
        "source": "test",
    }
    assert save_multiplicity_labels(database, [item], "tester") == 1
    item["label"] = "touching_doublet"
    assert save_multiplicity_labels(database, [item], "tester") == 1
    labels = read_multiplicity_labels(database)
    assert len(labels) == 1
    assert labels.iloc[0]["label"] == "touching_doublet"
    stats = multiplicity_stats(database)
    assert stats["total"] == 1
    assert stats["counts"]["touching_doublet"] == 1


def test_multiplicity_review_accepts_debris_and_wall_corrections(tmp_path):
    database = tmp_path / "annotations.db"
    items = [
        {
            "candidate_id": "debris-correction",
            "well": "A1",
            "timepoint": "T0",
            "x_px": 10.0,
            "y_px": 20.0,
            "label": "debris",
        },
        {
            "candidate_id": "wall-correction",
            "well": "A2",
            "timepoint": "T1",
            "x_px": 30.0,
            "y_px": 40.0,
            "label": "invalid",
        },
    ]
    assert save_multiplicity_labels(database, items, "tester") == 2
    stats = multiplicity_stats(database)
    assert stats["counts"]["debris"] == 1
    assert stats["counts"]["invalid"] == 1


def test_categorized_review_confirmation_feeds_both_training_heads(tmp_path):
    database = tmp_path / "annotations.db"
    items = [
        {
            "candidate_id": "confirmed-single",
            "well": "A1",
            "timepoint": "T0",
            "x_px": 10.0,
            "y_px": 20.0,
            "label": "single",
            "source": "categorized_batch_confirmed",
        },
        {
            "candidate_id": "confirmed-debris",
            "well": "A2",
            "timepoint": "T1",
            "x_px": 30.0,
            "y_px": 40.0,
            "label": "debris",
            "source": "categorized_batch_confirmed",
        },
    ]
    assert save_categorized_review_labels(database, items, "tester") == 2
    multiplicity = read_multiplicity_labels(database)
    assert set(multiplicity["label"]) == {"single", "debris"}
    teaching = read_teaching_labels(database)
    assert dict(zip(teaching["candidate_id"], teaching["label"])) == {
        "confirmed-single": "cell",
        "confirmed-debris": "debris",
    }


def test_integrated_review_records_approval_and_correction(tmp_path):
    database = tmp_path / "annotations.db"
    items = [
        {
            "candidate_id": "H6:T1:cf:154",
            "predicted_label": "touching_doublet",
            "reviewed_label": "touching_doublet",
        },
        {
            "candidate_id": "C5:T2:cf:60",
            "predicted_label": "debris",
            "reviewed_label": "invalid",
        },
    ]
    assert (
        save_integrated_reviews(
            database, "integrated-round-test", items, "tester"
        )
        == 2
    )
    with sqlite3.connect(database) as connection:
        decisions = dict(
            connection.execute(
                """
                SELECT candidate_id, decision
                FROM integrated_training_reviews
                """
            ).fetchall()
        )
    assert decisions == {
        "H6:T1:cf:154": "approved",
        "C5:T2:cf:60": "corrected",
    }


def test_corrected_single_doublet_swap_gets_extra_training_weight(tmp_path):
    database = tmp_path / "annotations.db"
    save_integrated_reviews(
        database,
        "round-test",
        [
            {
                "candidate_id": "swap",
                "predicted_label": "single",
                "reviewed_label": "touching_doublet",
            },
            {
                "candidate_id": "approved",
                "predicted_label": "single",
                "reviewed_label": "single",
            },
        ],
        "tester",
    )
    metadata = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "well": "A1",
                "timepoint": "T1",
                "x_px": x_px,
                "y_px": 20.0,
                "diameter_px": 12.0,
            }
            for candidate_id, x_px in (("swap", 10.0), ("approved", 50.0))
        ]
    )

    targets = _multiplicity_targets_for_source(
        {"multiplicity": {"corrected_single_doublet_weight": 4.0}},
        database,
        metadata,
    ).set_index("candidate_id")

    assert targets.loc["swap", "sample_weight"] == 4.0
    assert targets.loc["approved", "sample_weight"] == 1.0


def test_integrated_reviews_carry_forward_for_stable_candidates(tmp_path):
    database = tmp_path / "annotations.db"
    save_integrated_reviews(
        database,
        "old-round",
        [
            {
                "candidate_id": "H6:T1:cf:154",
                "predicted_label": "single",
                "reviewed_label": "touching_doublet",
            },
            {
                "candidate_id": "removed-candidate",
                "predicted_label": "debris",
                "reviewed_label": "debris",
            },
        ],
        "tester",
    )
    predictions = pd.DataFrame(
        [
            {
                "candidate_id": "H6:T1:cf:154",
                "integrated_label": "touching_doublet",
            },
            {
                "candidate_id": "new-candidate",
                "integrated_label": "single",
            },
        ]
    )
    assert (
        carry_forward_integrated_reviews(
            database, "new-round", predictions
        )
        == 1
    )
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """
            SELECT predicted_label, reviewed_label, decision
            FROM integrated_training_reviews
            WHERE round_id = 'new-round'
            """
        ).fetchone()
    assert row == ("touching_doublet", "touching_doublet", "approved")

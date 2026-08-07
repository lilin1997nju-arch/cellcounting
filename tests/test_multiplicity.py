import sqlite3

import pandas as pd

from cellvision.multiplicity import (
    carry_forward_integrated_reviews,
    multiplicity_stats,
    read_multiplicity_labels,
    save_integrated_reviews,
    save_multiplicity_labels,
)


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

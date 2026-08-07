from __future__ import annotations

from cellvision.review_server import initialize_database
from cellvision.teaching import (
    read_teaching_labels,
    save_auto_annotation_reviews,
    save_teaching_labels,
    teaching_stats,
)


def test_teaching_labels_upsert_and_stats(tmp_path) -> None:
    database = initialize_database(tmp_path / "annotations.db")
    item = {
        "candidate_id": "H6:T0:cf:1",
        "well": "H6",
        "timepoint": "T0",
        "x_px": 100.0,
        "y_px": 200.0,
        "label": "cell",
        "source": "quick_teaching_seed",
    }
    assert save_teaching_labels(database, [item], "tester") == 1
    item["label"] = "debris"
    assert save_teaching_labels(database, [item], "tester") == 1
    labels = read_teaching_labels(database)
    assert len(labels) == 1
    assert labels.iloc[0]["label"] == "debris"
    stats = teaching_stats(database)
    assert stats["total"] == 1
    assert stats["counts"]["debris"] == 1
    assert stats["counts"]["cell"] == 0


def test_auto_review_records_approval_and_correction(tmp_path) -> None:
    database = initialize_database(tmp_path / "annotations.db")
    saved = save_auto_annotation_reviews(
        database,
        "round-1",
        [
            {
                "candidate_id": "A2:T0:cf:1",
                "predicted_label": "cell",
                "reviewed_label": "cell",
            },
            {
                "candidate_id": "A2:T0:cf:2",
                "predicted_label": "cell",
                "reviewed_label": "invalid",
            },
        ],
        "tester",
    )
    assert saved == 2


def test_dead_like_is_not_a_static_teaching_label(tmp_path) -> None:
    database = initialize_database(tmp_path / "annotations.db")
    item = {
        "candidate_id": "H6:T0:cf:2",
        "well": "H6",
        "timepoint": "T0",
        "x_px": 100.0,
        "y_px": 200.0,
        "label": "dead_like",
        "source": "quick_teaching",
    }
    try:
        save_teaching_labels(database, [item], "tester")
    except ValueError as exc:
        assert "Invalid teaching label" in str(exc)
    else:
        raise AssertionError("dead_like must be deferred to temporal review")

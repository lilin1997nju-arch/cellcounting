from __future__ import annotations

import pandas as pd

from cellvision.temporal_shadow_review import _track_record, _well_summaries


def _track_rows(
    *,
    behavior: str = "no_decisive_temporal_evidence",
    conclusion: str = "",
    v2_labels: tuple[str, str, str] = ("single", "single", "single"),
    v3_labels: tuple[str, str, str] = ("single", "single", "single"),
    identity: float = 0.90,
    static: float = 0.90,
    shape: float = 0.95,
    foreground: float = 0.90,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": f"A1:T{index}:x",
                "well": "A1",
                "timepoint": f"T{index}",
                "integrated_label": v2_labels[index],
                "v3_proposed_label": v3_labels[index],
                "v3_track_behavior": behavior,
                "v3_track_conclusion": conclusion,
                "v3_identity_score": identity,
                "v3_static_similarity": static,
                "v3_shape_similarity": shape,
                "v3_foreground_quality": foreground,
                "v3_track_frame_count": 3,
                "v3_track_pair_count": 2,
                "v3_track_id": "A1:track-1",
            }
            for index in range(3)
        ]
    )


def test_shadow_review_marks_true_v2_v3_frame_difference():
    track = _track_record(
        "QL2603_T1-1",
        _track_rows(v2_labels=("single", "single", "single"), v3_labels=("debris", "debris", "debris")),
        None,
    )

    assert track["v2_v3_mismatch"] is True
    assert "frame_label_difference" in track["v2_v3_mismatch_reasons"]
    assert track["low_match_confidence"] is False


def test_shadow_review_includes_unified_dead_cell_conclusion():
    track = _track_record(
        "QL2603_T1-1",
        _track_rows(behavior="cell_to_debris", conclusion="dead_cell"),
        None,
    )

    assert track["v2_v3_mismatch"] is True
    assert track["v2_v3_mismatch_reasons"] == ["v3_unified_track_conclusion"]


def test_shadow_review_marks_low_confidence_three_frame_match():
    track = _track_record(
        "QL2603_T1-1",
        _track_rows(identity=0.66, static=0.74, shape=0.84, foreground=0.80),
        None,
    )

    assert track["matched_three_frames"] is True
    assert track["low_match_confidence"] is True
    assert track["low_match_reasons"] == ["identity", "shape_similarity"]


def test_shadow_review_well_summary_deduplicates_tracks():
    first = _track_record("QL2603_T1-1", _track_rows(), None)
    second = _track_record(
        "QL2603_T1-1",
        _track_rows(identity=0.60, static=0.60, shape=0.80),
        None,
    )

    summary = _well_summaries([first, second])

    assert len(summary) == 1
    assert summary[0]["track_count"] == 2
    assert summary[0]["low_match_track_count"] == 1

import pandas as pd

from cellvision.temporal_completion import complete_temporal_candidates


def _config() -> dict:
    return {
        "candidate_filter": {"hard_wall_exclusion_fraction": 0.44},
        "temporal_completion": {
            "anchor_cell_probability": 0.82,
            "anchor_integrated_confidence": 0.82,
            "search_radii_px": [96, 192, 384, 768],
            "existing_match_radius_px": 640,
            "automatic_promotion_score": 0.78,
            "review_score": 0.58,
            "ambiguity_margin": 0.10,
            "strong_debris_probability": 0.72,
            "passes": 1,
        },
    }


def _row(
    candidate_id: str,
    timepoint: str,
    x: float,
    *,
    label: str,
    cell_probability: float,
    debris_probability: float = 0.02,
    confidence: float = 0.95,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "well": "B3",
        "timepoint": timepoint,
        "x_px": x,
        "y_px": 100.0,
        "aligned_x_px": x,
        "aligned_y_px": 100.0,
        "area_px": 48.0,
        "diameter_px": 7.8,
        "circularity": 0.82,
        "eccentricity": 0.35,
        "solidity": 0.90,
        "radial_fraction": 0.2,
        "candidate_source": "raw_dense_peak",
        "auto_status": "needs_review",
        "integrated_label": label,
        "integrated_confidence": confidence,
        "cell_probability": cell_probability,
        "debris_probability": debris_probability,
        "invalid_probability": 0.02,
        "predicted_multiplicity": "single",
        "multiplicity_confidence": 0.80,
    }


def test_later_high_confidence_cell_recovers_t0_uncertain() -> None:
    frame = pd.DataFrame(
        [
            _row(
                "B3:T0:uncertain",
                "T0",
                105,
                label="uncertain",
                cell_probability=0.75,
                confidence=0.40,
            ),
            _row(
                "B3:T1:cell",
                "T1",
                100,
                label="single",
                cell_probability=0.96,
            ),
        ]
    )

    completed, suggestions, report = complete_temporal_candidates(
        _config(), frame
    )

    recovered = completed[completed["candidate_id"] == "B3:T0:uncertain"].iloc[0]
    assert recovered["integrated_label"] == "single"
    assert recovered["temporal_completion_status"] == "auto_promoted"
    assert report["promoted_candidate_count"] == 1
    assert "auto_promoted" in set(suggestions["decision"])


def test_strong_debris_is_not_retroactively_promoted() -> None:
    frame = pd.DataFrame(
        [
            _row(
                "B3:T0:debris",
                "T0",
                105,
                label="debris",
                cell_probability=0.05,
                debris_probability=0.90,
                confidence=0.90,
            ),
            _row(
                "B3:T1:cell",
                "T1",
                100,
                label="single",
                cell_probability=0.96,
            ),
        ]
    )

    completed, _, report = complete_temporal_candidates(_config(), frame)

    candidate = completed[completed["candidate_id"] == "B3:T0:debris"].iloc[0]
    assert candidate["integrated_label"] == "debris"
    assert report["promoted_candidate_count"] == 0


def test_similar_competing_candidates_require_review() -> None:
    frame = pd.DataFrame(
        [
            _row(
                "B3:T0:a",
                "T0",
                95,
                label="uncertain",
                cell_probability=0.74,
                confidence=0.40,
            ),
            _row(
                "B3:T0:b",
                "T0",
                105,
                label="uncertain",
                cell_probability=0.74,
                confidence=0.40,
            ),
            _row(
                "B3:T1:cell",
                "T1",
                100,
                label="single",
                cell_probability=0.96,
            ),
        ]
    )

    completed, suggestions, report = complete_temporal_candidates(
        _config(), frame
    )

    assert set(completed["integrated_label"]) == {"uncertain", "single"}
    assert report["promoted_candidate_count"] == 0
    assert "ambiguous_temporal_candidate" in set(suggestions["decision"])


def test_t0_cell_can_recover_later_uncertain_candidate() -> None:
    frame = pd.DataFrame(
        [
            _row(
                "B3:T0:cell",
                "T0",
                100,
                label="single",
                cell_probability=0.96,
            ),
            _row(
                "B3:T1:uncertain",
                "T1",
                110,
                label="uncertain",
                cell_probability=0.76,
                confidence=0.40,
            ),
        ]
    )

    completed, _, report = complete_temporal_candidates(_config(), frame)

    recovered = completed[
        completed["candidate_id"] == "B3:T1:uncertain"
    ].iloc[0]
    assert recovered["integrated_label"] == "single"
    assert report["promoted_candidate_count"] == 1


def test_missing_candidate_does_not_create_a_search_hint() -> None:
    frame = pd.DataFrame(
        [
            _row(
                "B3:T1:cell",
                "T1",
                100,
                label="single",
                cell_probability=0.96,
            )
        ]
    )

    _, suggestions, report = complete_temporal_candidates(_config(), frame)

    assert suggestions.empty
    assert report["promoted_candidate_count"] == 0
    assert report["evaluated_gap_count"] >= 1

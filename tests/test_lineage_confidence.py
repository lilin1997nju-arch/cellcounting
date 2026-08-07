import pandas as pd

from cellvision.lineage_engine import _link_confidence


def _root(x: float, y: float) -> dict:
    return {
        "current_aligned": [x, y],
        "current_area": 48.0,
        "circularity": 0.82,
        "eccentricity": 0.35,
        "solidity": 0.90,
    }


def _candidate(x: float, y: float) -> pd.Series:
    return pd.Series(
        {
            "aligned_x_px": x,
            "aligned_y_px": y,
            "area_px": 48.0,
            "circularity": 0.82,
            "eccentricity": 0.35,
            "solidity": 0.90,
            "integrated_confidence": 0.95,
        }
    )


def test_link_confidence_rejects_candidate_closer_to_another_root() -> None:
    near_root = _root(0, 0)
    far_root = _root(900, 0)
    candidate = _candidate(30, 0)

    correct = _link_confidence(
        near_root,
        candidate,
        [near_root, far_root],
        distance_scale=1000,
        ambiguity_scale=140,
    )
    crossed = _link_confidence(
        far_root,
        candidate,
        [near_root, far_root],
        distance_scale=1000,
        ambiguity_scale=140,
    )

    assert correct["link_confidence"] >= 0.58
    assert crossed["link_confidence"] < 0.58
    assert crossed["ambiguity_score"] < 0.1


def test_unambiguous_long_motion_can_still_pass() -> None:
    root = _root(0, 0)
    evidence = _link_confidence(
        root,
        _candidate(900, 0),
        [root],
        distance_scale=1000,
        ambiguity_scale=140,
    )

    assert evidence["link_confidence"] >= 0.58
    assert evidence["link_distance_px"] == 900


def test_previous_motion_predicts_next_position() -> None:
    moving_root = _root(100, 0)
    moving_root["last_motion_vector"] = [100.0, 0.0]
    evidence = _link_confidence(
        moving_root,
        _candidate(200, 0),
        [moving_root],
        distance_scale=1000,
        ambiguity_scale=140,
    )

    assert evidence["link_distance_px"] == 0
    assert evidence["predicted_x_px"] == 200

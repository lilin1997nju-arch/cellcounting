from __future__ import annotations

import pandas as pd

from cellvision.late_full_inference import (
    _dense_late_group_mask,
    _dense_v2_passthrough,
    _selected_late_timepoints,
)


def test_late_cell_inference_uses_only_configured_endpoint() -> None:
    images = pd.DataFrame({"timepoint": ["T3", "T4"]})

    assert _selected_late_timepoints(
        {"late_growth": {"endpoint_timepoint": "T4"}}, images
    ) == ("T4",)


def test_late_cell_inference_falls_back_to_day7_when_it_is_the_only_endpoint() -> None:
    images = pd.DataFrame({"timepoint": ["T3"]})

    assert _selected_late_timepoints(
        {"late_growth": {"endpoint_timepoint": "T4"}}, images
    ) == ("T3",)


def test_dense_late_mask_selects_whole_image_and_ignores_suppressed_rows() -> None:
    frame = pd.DataFrame(
        {
            "well": ["B2"] * 5 + ["C3"] * 2,
            "timepoint": ["T1"] * 7,
            "candidate_id": [f"candidate-{index}" for index in range(7)],
            "integrated_label": ["cluster_3plus"] * 5 + ["single", "single"],
            "integrated_confidence": [0.9] * 7,
            "is_counting_instance": [True, True, False, True, True, True, False],
            "area_px": [40] * 7,
        }
    )

    selected = _dense_late_group_mask(
        frame,
        minimum_instances=10,
        minimum_units=12,
    )

    assert selected.tolist() == [True, True, True, True, True, False, False]
    passthrough = _dense_v2_passthrough(frame[selected])
    assert int(passthrough["v2_is_counting_instance"].sum()) == 4
    assert not bool(passthrough.loc[2, "v2_is_counting_instance"])
    assert set(passthrough["v2_processing_mode"]) == {
        "dense_integrated_passthrough"
    }


def test_sparse_late_image_keeps_precise_v2_path() -> None:
    frame = pd.DataFrame(
        {
            "well": ["D4", "D4"],
            "timepoint": ["T2", "T2"],
            "candidate_id": ["one", "two"],
            "integrated_label": ["single", "debris"],
            "is_counting_instance": [True, True],
        }
    )

    selected = _dense_late_group_mask(
        frame,
        minimum_instances=5,
        minimum_units=8,
    )

    assert not selected.any()

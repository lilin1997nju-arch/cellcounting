import json
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, disk

from cellvision.v2_instance_inference import (
    _contour,
    _rle,
    _selected_component,
    consolidate_v2_masks,
    decode_rle,
    finalize_v2_instances,
    refine_instance_mask,
)


def _row(candidate_id, x, mask, label="single", confidence=.9):
    return {
        "candidate_id": candidate_id, "well": "B1", "timepoint": "T0", "x_px": x, "y_px": 50,
        "integrated_label": label, "integrated_confidence": confidence,
        "v2_mask_valid": True, "v2_wall_rejected": False, "v2_instance_confidence": confidence,
        "v2_mask_rle": _rle(mask), "v2_mask_origin_x": int(x - mask.shape[1] // 2), "v2_mask_origin_y": 2,
    }


def test_rle_round_trip():
    mask = np.zeros((8, 8), bool); mask[2:5, 3:7] = True
    assert np.array_equal(mask, decode_rle(_rle(mask), 8))


def test_mask_overlap_suppresses_duplicate():
    mask = np.zeros((96, 96), bool); mask[42:55, 42:55] = True
    frame = pd.DataFrame([_row("first", 50, mask, confidence=.95), _row("second", 52, mask, confidence=.8)])
    result = consolidate_v2_masks(frame, 96)
    assert result["v2_is_suppressed"].sum() == 1
    assert result.loc[result.candidate_id == "second", "v2_suppressed_by"].iloc[0] == "first"


def test_separated_instances_are_kept():
    left = np.zeros((96, 96), bool); left[42:52, 25:35] = True
    right = np.zeros((96, 96), bool); right[42:52, 61:71] = True
    frame = pd.DataFrame([_row("left", 50, left), _row("right", 50, right)])
    result = consolidate_v2_masks(frame, 96)
    assert not result["v2_is_suppressed"].any()


def test_refinement_is_narrow_band_and_contour_is_subpixel():
    yy, xx = np.mgrid[:96, :96]
    raw = np.full((96, 96), .55, np.float32)
    raw[((xx - 48) ** 2 + (yy - 48) ** 2) <= 11**2] = .18
    mask = ((xx - 48) ** 2 + (yy - 48) ** 2) <= 9**2
    mask[47:49, 47:49] = False
    refined = refine_instance_mask(mask, raw, np.zeros_like(raw))
    assert refined.any()
    assert not np.logical_and(refined, ((xx - 48) ** 2 + (yy - 48) ** 2) > 13**2).any()
    contour = json.loads(_contour(refined, 0, 0))
    assert len(contour) >= 12
    assert any(float(x) != round(float(x)) or float(y) != round(float(y)) for x, y in contour)


def test_group_component_selection_retains_nearby_lobes():
    yy, xx = np.mgrid[:32, :32]
    left = (xx - 12) ** 2 + (yy - 16) ** 2 <= 4**2
    right = (xx - 23) ** 2 + (yy - 16) ** 2 <= 4**2
    binary = left | right
    seed = np.exp(-((xx - 15.5) ** 2 + (yy - 15.5) ** 2) / (2 * 3**2))

    single = _selected_component(binary, seed)
    group = _selected_component(binary, seed, retain_nearby=True, nearby_distance=3)

    assert single[16, 12]
    assert not single[16, 23]
    assert group[16, 12]
    assert group[16, 23]
    assert group.sum() > single.sum() * 1.7


def test_refinement_falls_back_when_active_contour_cuts_through_cell():
    yy, xx = np.mgrid[:64, :64]
    mask = (xx - 32) ** 2 + (yy - 32) ** 2 <= 10**2
    mask[29:35, 29:35] = False
    raw = np.full((64, 64), 0.5, dtype=np.float32)
    wall = np.zeros_like(raw)
    cut_mask = mask & (xx <= 25)
    diagnostics = {}

    with patch(
        "cellvision.v2_instance_inference.morphological_geodesic_active_contour",
        return_value=cut_mask,
    ):
        refined = refine_instance_mask(
            mask,
            raw,
            wall,
            diagnostics=diagnostics,
        )

    expected = np.asarray(binary_fill_holes(closing(mask, disk(1))), dtype=bool)
    assert np.array_equal(refined, expected)
    assert diagnostics["status"] == "fallback_area_shrink"
    assert refined[32, 32]


def test_strong_independent_cell_evidence_rescues_conservative_boundary_confidence():
    mask = np.zeros((96, 96), bool)
    mask[43:53, 43:53] = True
    row = _row("strong-cell", 50, mask, label="invalid", confidence=0.50)
    row.update(
        {
            "cell_probability": 0.97,
            "debris_probability": 0.02,
            "invalid_probability": 0.01,
            "v2_objectness": 0.95,
            "v2_wall_overlap": 0.0,
            "predicted_multiplicity": "single",
        }
    )

    result = finalize_v2_instances(pd.DataFrame([row]), 96)

    assert result.loc[0, "integrated_label"] == "single"
    assert bool(result.loc[0, "v2_rescued_from_invalid"])

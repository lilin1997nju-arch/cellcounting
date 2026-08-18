import json
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, binary_fill_holes, label as ndi_label
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


def _strong_single_row(candidate_id, x, mask, raw_path):
    row = _row(candidate_id, x, mask, confidence=.88)
    row.update(
        {
            "raw_image_path": str(raw_path),
            "cell_probability": .99,
            "single_probability": .91,
            "invalid_probability": .01,
            "v2_objectness": .99,
            "v2_instance_diameter_px": float(2 * np.sqrt(mask.sum() / np.pi)),
        }
    )
    return row


def _global_pair_masks(result):
    masks = []
    for row in result.itertuples(index=False):
        local = decode_rle(row.v2_mask_rle, 96)
        global_mask = np.zeros((128, 128), dtype=bool)
        x0, y0 = int(row.v2_mask_origin_x), int(row.v2_mask_origin_y)
        global_mask[y0 : y0 + 96, x0 : x0 + 96] = local
        masks.append(global_mask)
    return masks


def test_bright_gap_overlapping_single_masks_are_split_and_kept(tmp_path):
    yy, xx = np.mgrid[:128, :128]
    first_cell = (xx - 50) ** 2 + (yy - 50) ** 2 <= 6**2
    second_cell = (xx - 66) ** 2 + (yy - 50) ** 2 <= 6**2
    leaked_union = first_cell | second_cell
    leaked_union[48:53, 56:61] = True
    raw = np.full((128, 128), 150, dtype=np.uint8)
    raw[first_cell | second_cell] = 18
    raw_path = tmp_path / "separate_cells.png"
    Image.fromarray(raw).save(raw_path)

    rows = []
    for candidate_id, x in (("left", 50), ("right", 66)):
        x0 = x - 48
        local = leaked_union[2:98, x0 : x0 + 96]
        rows.append(_strong_single_row(candidate_id, x, local, raw_path))
    result = consolidate_v2_masks(pd.DataFrame(rows), 96)

    assert not result["v2_is_suppressed"].any()
    assert result["v2_competing_seed_split"].all()
    assert result["v2_instance_id"].nunique() == 2
    first, second = _global_pair_masks(result)
    assert not np.logical_and(first, second).any()
    assert not np.logical_and(first, binary_dilation(second, structure=np.ones((3, 3)))).any()
    assert ndi_label(first | second, structure=np.ones((3, 3), dtype=np.uint8))[1] == 2


def test_dark_path_duplicate_proposals_remain_suppressed(tmp_path):
    yy, xx = np.mgrid[:128, :128]
    one_object = ((xx - 58) / 13) ** 2 + ((yy - 50) / 7) ** 2 <= 1
    raw = np.full((128, 128), 150, dtype=np.uint8)
    raw[one_object] = 18
    raw_path = tmp_path / "one_object.png"
    Image.fromarray(raw).save(raw_path)

    rows = []
    for candidate_id, x in (("first", 50), ("second", 66)):
        x0 = x - 48
        local = one_object[2:98, x0 : x0 + 96]
        rows.append(_strong_single_row(candidate_id, x, local, raw_path))
    result = consolidate_v2_masks(pd.DataFrame(rows), 96)

    assert result["v2_is_suppressed"].sum() == 1
    assert not result["v2_competing_seed_split"].any()


def test_same_core_alias_does_not_create_a_third_instance(tmp_path):
    yy, xx = np.mgrid[:128, :128]
    first_cell = (xx - 50) ** 2 + (yy - 50) ** 2 <= 6**2
    second_cell = (xx - 66) ** 2 + (yy - 50) ** 2 <= 6**2
    leaked_union = first_cell | second_cell
    leaked_union[48:53, 56:61] = True
    raw = np.full((128, 128), 150, dtype=np.uint8)
    raw[first_cell | second_cell] = 18
    raw_path = tmp_path / "two_cells_three_seeds.png"
    Image.fromarray(raw).save(raw_path)

    rows = []
    for candidate_id, x in (("left-raw", 50), ("left-cf", 52), ("right", 66)):
        x0 = x - 48
        local = leaked_union[2:98, x0 : x0 + 96]
        rows.append(_strong_single_row(candidate_id, x, local, raw_path))
    result = consolidate_v2_masks(pd.DataFrame(rows), 96)

    assert result["v2_competing_seed_split"].all()
    assert result["v2_is_suppressed"].sum() == 1
    assert result.loc[~result["v2_is_suppressed"], "v2_instance_id"].nunique() == 2
    alias_rows = result[result["v2_competing_seed_aliases"].astype(bool)]
    assert len(alias_rows) == 2


def test_human_confirmed_pair_bypasses_only_the_area_ratio_guard(tmp_path):
    yy, xx = np.mgrid[:128, :128]
    small_cell = (xx - 50) ** 2 + (yy - 50) ** 2 <= 3**2
    large_cell = (xx - 66) ** 2 + (yy - 50) ** 2 <= 7**2
    leaked_union = small_cell | large_cell
    leaked_union[49:52, 53:60] = True
    raw = np.full((128, 128), 150, dtype=np.uint8)
    raw[small_cell | large_cell] = 18
    raw_path = tmp_path / "human_confirmed_unequal_cells.png"
    Image.fromarray(raw).save(raw_path)

    rows = []
    for candidate_id, x in (("small", 50), ("large", 66)):
        x0 = x - 48
        local = leaked_union[2:98, x0 : x0 + 96]
        rows.append(_strong_single_row(candidate_id, x, local, raw_path))
    frame = pd.DataFrame(rows)
    frame.loc[frame.candidate_id == "small", "cell_probability"] = 0.70
    frame.loc[frame.candidate_id == "small", "single_probability"] = 0.45
    frame.loc[frame.candidate_id == "small", "v2_objectness"] = 0.55
    guarded = consolidate_v2_masks(
        frame,
        96,
        {"competing_single_minimum_area_ratio": 0.80},
    )
    confirmed = consolidate_v2_masks(
        frame,
        96,
        {
            "competing_single_minimum_area_ratio": 0.80,
            "competing_single_confirmed_split_groups": [["small", "large"]],
        },
    )

    assert not guarded["v2_competing_seed_split"].any()
    assert confirmed["v2_competing_seed_split"].all()
    assert confirmed["v2_competing_seed_human_confirmed"].all()
    assert not confirmed["v2_is_suppressed"].any()


def test_human_excluded_candidate_cannot_trigger_competing_seed_split(tmp_path):
    yy, xx = np.mgrid[:128, :128]
    first_cell = (xx - 50) ** 2 + (yy - 50) ** 2 <= 6**2
    second_cell = (xx - 66) ** 2 + (yy - 50) ** 2 <= 6**2
    leaked_union = first_cell | second_cell
    leaked_union[48:53, 56:61] = True
    raw = np.full((128, 128), 150, dtype=np.uint8)
    raw[first_cell | second_cell] = 18
    raw_path = tmp_path / "excluded_candidate.png"
    Image.fromarray(raw).save(raw_path)
    rows = []
    for candidate_id, x in (("excluded", 50), ("other", 66)):
        x0 = x - 48
        local = leaked_union[2:98, x0 : x0 + 96]
        rows.append(_strong_single_row(candidate_id, x, local, raw_path))

    result = consolidate_v2_masks(
        pd.DataFrame(rows),
        96,
        {"competing_single_excluded_candidate_ids": ["excluded"]},
    )

    assert not result["v2_competing_seed_split"].any()
    assert result["v2_is_suppressed"].sum() == 1


def test_human_confirmed_children_suppress_overlapping_coarse_cluster(tmp_path):
    yy, xx = np.mgrid[:128, :128]
    left_cell = (xx - 50) ** 2 + (yy - 50) ** 2 <= 6**2
    right_cell = (xx - 66) ** 2 + (yy - 50) ** 2 <= 6**2
    leaked_union = left_cell | right_cell
    leaked_union[48:53, 56:61] = True
    raw = np.full((128, 128), 150, dtype=np.uint8)
    raw[left_cell | right_cell] = 18
    raw_path = tmp_path / "confirmed_children_with_cluster.png"
    Image.fromarray(raw).save(raw_path)
    rows = []
    for candidate_id, x in (("left", 50), ("right", 66)):
        x0 = x - 48
        local = leaked_union[2:98, x0 : x0 + 96]
        rows.append(_strong_single_row(candidate_id, x, local, raw_path))
    coarse = _row(
        "coarse-cluster",
        58,
        leaked_union[2:98, 10:106],
        label="cluster_3plus",
        confidence=.99,
    )
    rows.append(coarse)

    result = consolidate_v2_masks(
        pd.DataFrame(rows),
        96,
        {"competing_single_confirmed_split_groups": [["left", "right"]]},
    )

    assert result.loc[result.candidate_id == "coarse-cluster", "v2_is_suppressed"].iloc[0]
    assert (~result["v2_is_suppressed"]).sum() == 2


def test_human_confirmed_parent_suppression_is_explicit():
    first = np.zeros((96, 96), bool)
    first[40:50, 25:35] = True
    second = np.zeros((96, 96), bool)
    second[40:50, 60:70] = True
    parent = np.zeros((96, 96), bool)
    parent[10:15, 10:15] = True
    frame = pd.DataFrame(
        [
            _row("first", 50, first),
            _row("second", 50, second),
            _row("parent", 50, parent, label="cluster_3plus"),
        ]
    )

    result = consolidate_v2_masks(
        frame,
        96,
        {"competing_single_confirmed_parent_suppressions": {"parent": "first"}},
    )

    parent_row = result[result.candidate_id == "parent"].iloc[0]
    assert parent_row.v2_is_suppressed
    assert parent_row.v2_suppressed_by == "first"
    assert parent_row.v2_suppression_reason == "human_confirmed_split_parent"


def test_manual_candidate_label_override_wins_over_model_label():
    mask = np.zeros((96, 96), bool)
    mask[43:53, 43:53] = True
    row = _row("reviewed-noncell", 50, mask, label="single", confidence=.90)
    row.update(
        {
            "cell_probability": .99,
            "debris_probability": .005,
            "invalid_probability": .005,
            "v2_objectness": .99,
            "predicted_multiplicity": "single",
        }
    )

    result = finalize_v2_instances(
        pd.DataFrame([row]),
        96,
        inference_settings={
            "manual_candidate_label_overrides": {"reviewed-noncell": "debris"}
        },
    )

    assert result.loc[0, "integrated_label"] == "debris"
    assert result.loc[0, "v2_manual_label_override"]


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

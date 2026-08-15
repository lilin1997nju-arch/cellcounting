import pandas as pd
import numpy as np

from cellvision.temporal_training_cache import _review_triplet_static_target
from cellvision.v2_temporal_inference import (
    _v3_proposal_is_decisive,
    _adjust_cell_debris_probabilities,
    _component_temporal_outputs,
    _detect_cross_track_division_rescues,
    _is_static_wall_artifact,
    _local_patch_similarity,
    _track_fused_probabilities,
    preserve_temporal_multiplicity_labels,
    resolve_low_cell_noncell_labels,
)
from cellvision.multiplicity_identity import candidate_multiplicity_label
from cellvision.temporal_objects import (
    TemporalObjectDescriptor,
    TemporalPairEvidence,
    compare_object_descriptors,
)


def _row(label: str, x: float, y: float, area: float) -> pd.Series:
    return pd.Series(
        {
            "reviewed_label": label,
            "aligned_x_px": x,
            "aligned_y_px": y,
            "v2_instance_area_px": area,
        }
    )


def test_v3_active_fusion_only_overwrites_v2_for_decisive_behavior():
    assert not _v3_proposal_is_decisive(
        {"v3_track_behavior": "no_decisive_temporal_evidence"}
    )
    assert not _v3_proposal_is_decisive(
        {"v3_track_behavior": "wall_uncertain"}
    )
    assert not _v3_proposal_is_decisive(
        {"v3_track_behavior": "decline_without_morphology_evidence"}
    )
    assert _v3_proposal_is_decisive(
        {"v3_track_behavior": "wall_structure_invalid"}
    )
    assert _v3_proposal_is_decisive(
        {"v3_track_behavior": "division_or_growth"}
    )


def test_stable_reviewed_single_triplet_is_static_similarity_evidence():
    rows = [
        _row("single", 100, 100, 129),
        _row("single", 102, 100, 140),
        _row("single", 106, 98, 133),
    ]

    assert _review_triplet_static_target(rows) == 1.0


def test_reviewed_division_triplet_is_dynamic_similarity_evidence():
    rows = [
        _row("single", 100, 100, 100),
        _row("touching_doublet", 103, 101, 160),
        _row("cluster_3plus", 106, 102, 220),
    ]

    assert _review_triplet_static_target(rows) == 0.0


def test_reviewed_cell_to_debris_triplet_is_dynamic_similarity_evidence():
    rows = [
        _row("single", 100, 100, 140),
        _row("uncertain", 101, 100, 136),
        _row("debris", 102, 101, 130),
    ]

    assert _review_triplet_static_target(rows) == 0.0


def test_moving_single_triplet_is_not_pseudo_labelled_dead():
    rows = [
        _row("single", 100, 100, 100),
        _row("single", 120, 100, 105),
        _row("single", 140, 100, 110),
    ]

    assert _review_triplet_static_target(rows) is None


def test_static_similarity_moves_only_ambiguous_probability_toward_debris():
    cell, debris, boost, applied, reason = _adjust_cell_debris_probabilities(
        0.55, 0.40, 0.05, 0.95, 0.95, 3
    )

    assert applied is True
    assert reason == "applied"
    assert boost > 0
    assert cell < 0.55
    assert debris > 0.40


def test_high_confidence_cell_is_protected_from_temporal_adjustment():
    cell, debris, boost, applied, reason = _adjust_cell_debris_probabilities(
        0.95, 0.03, 0.02, 0.98, 0.98, 3
    )

    assert (cell, debris, boost, applied) == (0.95, 0.03, 0.0, False)
    assert reason == "high_confidence_base"


def test_single_candidate_preserves_original_probabilities():
    cell, debris, boost, applied, reason = _adjust_cell_debris_probabilities(
        0.68, 0.31, 0.01, 0.99, 0.99, 1
    )

    assert (cell, debris, boost, applied) == (0.68, 0.31, 0.0, False)
    assert reason == "insufficient_parallel_candidates"


def test_two_real_candidates_can_apply_reduced_temporal_adjustment():
    cell, debris, boost, applied, reason = _adjust_cell_debris_probabilities(
        0.55, 0.40, 0.05, 0.95, 0.95, 2
    )

    assert applied is True
    assert reason == "applied"
    assert 0 < boost < 0.25
    assert cell < 0.55
    assert debris > 0.40


def test_invalid_probability_is_included_in_temporal_cell_evidence():
    cell, debris, boost, applied, reason = _adjust_cell_debris_probabilities(
        0.56, 0.01, 0.43, 0.95, 0.95, 3
    )

    assert applied is True
    assert reason == "applied"
    assert cell < 0.56
    assert debris > 0.01
    assert abs(cell + debris + 0.43 - 1.0) < 1e-6


def test_static_wall_artifact_requires_three_stable_wall_overlapping_frames():
    assert _is_static_wall_artifact(
        [1.0, 1.0, 1.0],
        [(100.0, 100.0), (102.0, 101.0), (103.0, 100.0)],
        [86.0, 49.0, 44.0],
        [(0.52, 0.43, 0.05, 1.0)] * 3,
        0.95,
        0.90,
    )


def test_static_wall_artifact_is_vetoed_by_repeated_strong_cell_evidence():
    assert not _is_static_wall_artifact(
        [1.0, 1.0, 1.0],
        [(100.0, 100.0), (102.0, 101.0), (103.0, 100.0)],
        [86.0, 49.0, 44.0],
        [(0.93, 0.04, 0.03, 1.0)] * 3,
        0.95,
        0.90,
    )


def test_track_fusion_keeps_low_confidence_unmarked_observation():
    cell, debris, invalid = _track_fused_probabilities(
        [
            (0.537, 0.452, 0.011, 0.0),
            (0.605, 0.387, 0.008, 0.0),
            (0.469, 0.518, 0.013, 0.0),
        ],
        [1.0, 1.0, 1.0],
    )

    assert 0.52 < cell < 0.56
    assert 0.43 < debris < 0.48
    assert invalid < 0.02


def test_local_patch_similarity_tolerates_small_registration_shift():
    patch = np.zeros((64, 64), np.float32)
    patch[26:38, 27:39] = 1.0
    shifted = np.roll(np.roll(patch, 3, axis=0), -2, axis=1)

    assert _local_patch_similarity(patch, shifted) > 0.95


def test_low_cell_uncertain_resolves_between_debris_and_invalid():
    frame = pd.DataFrame(
        [
            {"integrated_label": "uncertain", "cell_probability": 0.10, "debris_probability": 0.60, "invalid_probability": 0.30},
            {"integrated_label": "uncertain", "cell_probability": 0.20, "debris_probability": 0.25, "invalid_probability": 0.55},
            {"integrated_label": "uncertain", "cell_probability": 0.40, "debris_probability": 0.35, "invalid_probability": 0.25},
        ]
    )

    result, resolved = resolve_low_cell_noncell_labels(frame)

    assert resolved == 2
    assert result["integrated_label"].tolist() == ["debris", "invalid", "uncertain"]


def test_candidate_multiplicity_preserves_its_own_pre_temporal_label():
    candidate = pd.Series(
        {
            "integrated_label": "cluster_3plus",
            "v2_pre_temporal_integrated_label": "single",
            "predicted_multiplicity": "touching_doublet",
            "manual_override": True,
        }
    )

    assert candidate_multiplicity_label(candidate) == "single"


def test_temporal_cell_decision_does_not_copy_track_max_multiplicity():
    frame = pd.DataFrame(
        [
            {
                "integrated_label": "cluster_3plus",
                "v2_temporal_adjusted_label": "cluster_3plus",
                "v2_pre_temporal_integrated_label": "single",
                "predicted_multiplicity": "single",
            },
            {
                "integrated_label": "cluster_3plus",
                "v2_temporal_adjusted_label": "cluster_3plus",
                "v2_pre_temporal_integrated_label": "touching_doublet",
                "predicted_multiplicity": "touching_doublet",
            },
            {
                "integrated_label": "debris",
                "v2_temporal_adjusted_label": "debris",
                "v2_pre_temporal_integrated_label": "cluster_3plus",
                "predicted_multiplicity": "cluster_3plus",
            },
        ]
    )

    result, changed = preserve_temporal_multiplicity_labels(frame)

    assert changed == 2
    assert result["integrated_label"].tolist() == [
        "single",
        "touching_doublet",
        "debris",
    ]


def test_temporally_promoted_candidate_uses_own_probability_argmax():
    candidate = pd.Series(
        {
            "integrated_label": "uncertain",
            "predicted_multiplicity": "",
            "single_probability": 0.15,
            "touching_doublet_probability": 0.70,
            "cluster_3plus_probability": 0.15,
        }
    )

    assert candidate_multiplicity_label(candidate) == "touching_doublet"


def _descriptor(index: int, raw: np.ndarray, mask: np.ndarray, x: float = 100.0):
    gy, gx = np.gradient(raw.astype(np.float32))
    return TemporalObjectDescriptor(
        index=index,
        timepoint=f"T{index}",
        aligned_x=x,
        aligned_y=100.0,
        area=float(mask.sum()),
        equivalent_diameter=10.0,
        raw=raw.astype(np.float32),
        gradient=np.hypot(gx, gy).astype(np.float32),
        soft_mask=mask.astype(np.float32),
        binary_mask=mask.astype(bool),
        quality=0.9,
    )


def test_object_similarity_tolerates_mask_jitter_without_using_background():
    yy, xx = np.mgrid[:48, :48]
    first_mask = (xx - 24) ** 2 + (yy - 24) ** 2 <= 6**2
    second_mask = (xx - 25) ** 2 + (yy - 23) ** 2 <= 7**2
    first_raw = np.full((48, 48), 0.75, np.float32)
    second_raw = np.full((48, 48), 0.15, np.float32)  # deliberately different background
    object_pattern = np.exp(-((xx - 24) ** 2 + (yy - 24) ** 2) / 16.0)
    first_raw[first_mask] = object_pattern[first_mask]
    shifted_pattern = np.exp(-((xx - 25) ** 2 + (yy - 23) ** 2) / 16.0)
    second_raw[second_mask] = shifted_pattern[second_mask]

    evidence = compare_object_descriptors(
        _descriptor(0, first_raw, first_mask),
        _descriptor(1, second_raw, second_mask, x=102.0),
    )

    assert evidence.intensity > 0.85
    assert evidence.tolerant_shape > 0.85
    assert evidence.static > 0.80


def test_same_background_cannot_make_different_objects_static():
    yy, xx = np.mgrid[:48, :48]
    mask = (xx - 24) ** 2 + (yy - 24) ** 2 <= 7**2
    first_raw = np.full((48, 48), 0.5, np.float32)
    second_raw = first_raw.copy()
    first_raw[mask] = (xx[mask] - 17) / 14.0
    second_raw[mask] = (yy[mask] - 17) / 14.0

    evidence = compare_object_descriptors(
        _descriptor(0, first_raw, mask),
        _descriptor(1, second_raw, mask),
    )

    assert evidence.intensity < 0.70
    assert evidence.static < 0.80


def _temporal_frame(probabilities):
    rows = []
    for index, (cell, debris, label) in enumerate(probabilities):
        rows.append(
            {
                "candidate_id": f"candidate-{index}",
                "timepoint": f"T{index}",
                "integrated_label": label,
                "v2_pre_temporal_integrated_label": label,
                "predicted_multiplicity": "single",
                "cell_probability": cell,
                "debris_probability": debris,
                "invalid_probability": max(0.0, 1.0 - cell - debris),
                "v2_instance_area_px": 100.0,
                "v2_wall_overlap": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _edge(left, right, *, kind="continuation", static=0.94):
    return TemporalPairEvidence(
        left=left,
        right=right,
        identity=0.95,
        static=static,
        intensity=static,
        gradient=static,
        tolerant_shape=0.93,
        area_similarity=0.96,
        distance_px=3.0,
        foreground_quality=0.9,
        kind=kind,
    )


def _split_descriptor(index, timepoint, x, y, area=48.0):
    raw = np.full((48, 48), 0.5, np.float32)
    mask = np.zeros((48, 48), dtype=bool)
    yy, xx = np.mgrid[:48, :48]
    mask[(xx - 24) ** 2 + (yy - 24) ** 2 <= 4**2] = True
    return TemporalObjectDescriptor(
        index=index,
        timepoint=timepoint,
        aligned_x=float(x),
        aligned_y=float(y),
        area=float(area),
        equivalent_diameter=8.0,
        raw=raw,
        gradient=np.zeros_like(raw),
        soft_mask=mask.astype(np.float32),
        binary_mask=mask,
        quality=0.9,
    )


def _textured_split_descriptor(index, timepoint, x, y, area=48.0):
    descriptor = _split_descriptor(index, timepoint, x, y, area)
    yy, xx = np.mgrid[:48, :48]
    descriptor.raw[descriptor.binary_mask] = np.exp(
        -(
            (xx[descriptor.binary_mask] - 24) ** 2
            + (yy[descriptor.binary_mask] - 24) ** 2
        )
        / 8.0
    )
    return descriptor


def test_cross_track_division_rescue_recovers_a_child_consumed_by_neighbor():
    frame = pd.DataFrame(
        [
            {
                "candidate_id": f"split-{index}",
                "timepoint": timepoint,
                "integrated_label": "single",
                "v2_pre_temporal_integrated_label": "single",
                "cell_probability": 0.95,
                "debris_probability": 0.03,
                "invalid_probability": 0.02,
                "v2_is_unique_instance": True,
                "v2_is_counting_instance": True,
                "v2_wall_overlap": 0.0,
            }
            for index, timepoint in enumerate(("T0", "T1", "T1", "T2", "T2"))
        ]
    )
    descriptors = {
        0: _split_descriptor(0, "T0", 100, 100),
        1: _split_descriptor(1, "T1", 100, 100),
        2: _split_descriptor(2, "T1", 125, 85),
        3: _split_descriptor(3, "T2", 102, 100),
        4: _split_descriptor(4, "T2", 102, 85),
    }
    pair_evidence = {
        (1, 4): _edge(1, 4),
        (2, 4): _edge(2, 4),
    }
    rescues = _detect_cross_track_division_rescues(
        frame,
        list(frame.index),
        {"T0": [descriptors[0]], "T1": [descriptors[1], descriptors[2]], "T2": [descriptors[3], descriptors[4]]},
        descriptors,
        [_edge(0, 1), _edge(1, 3), _edge(2, 4)],
        pair_evidence,
        {0: 1, 1: 1, 3: 1, 2: 2, 4: 2},
        {1: [0, 1, 3], 2: [2, 4]},
        {index: f"W:O{component:03d}" for component, nodes in {1: [0, 1, 3], 2: [2, 4]}.items() for index in nodes},
        {},
        {
            "cross_track_division_rescue_enabled": True,
            "cross_track_division_rescue_radius_px": 88,
            "cross_track_division_rescue_minimum_identity": 0.62,
            "cross_track_division_rescue_minimum_foreground_quality": 0.30,
            "cross_track_division_rescue_minimum_child_separation_px": 5,
        },
    )

    assert len(rescues) == 1
    assert rescues[0]["parent_index"] == 1
    assert rescues[0]["secondary_child_index"] == 4
    assert rescues[0]["interval"] == "T1->T2"


def test_cross_track_division_rescue_ignores_suppressed_duplicate_child():
    frame = pd.DataFrame(
        [
            {
                "candidate_id": f"duplicate-{index}",
                "timepoint": timepoint,
                "integrated_label": "single",
                "v2_pre_temporal_integrated_label": "single",
                "cell_probability": 0.95,
                "debris_probability": 0.03,
                "invalid_probability": 0.02,
                "v2_is_unique_instance": index != 4,
                "v2_is_counting_instance": index != 4,
                "v2_wall_overlap": 0.0,
            }
            for index, timepoint in enumerate(("T0", "T1", "T1", "T2", "T2"))
        ]
    )
    descriptors = {
        0: _split_descriptor(0, "T0", 100, 100),
        1: _split_descriptor(1, "T1", 100, 100),
        2: _split_descriptor(2, "T1", 125, 85),
        3: _split_descriptor(3, "T2", 102, 100),
        4: _split_descriptor(4, "T2", 102, 85),
    }
    rescues = _detect_cross_track_division_rescues(
        frame,
        list(frame.index),
        {"T0": [descriptors[0]], "T1": [descriptors[1], descriptors[2]], "T2": [descriptors[3], descriptors[4]]},
        descriptors,
        [_edge(0, 1), _edge(1, 3), _edge(2, 4)],
        {(1, 4): _edge(1, 4), (2, 4): _edge(2, 4)},
        {0: 1, 1: 1, 3: 1, 2: 2, 4: 2},
        {1: [0, 1, 3], 2: [2, 4]},
        {index: f"W:O{component:03d}" for component, nodes in {1: [0, 1, 3], 2: [2, 4]}.items() for index in nodes},
        {},
        {"cross_track_division_rescue_enabled": True},
    )

    assert rescues == []


def test_cross_track_division_rescue_supports_strong_wall_attached_parent():
    values = [
        ("T0", 0.927, 0.045, 0.027, False, 1.0),
        ("T1", 0.952, 0.030, 0.018, False, 1.0),
        ("T1", 0.864, 0.080, 0.056, True, 0.0),
        ("T2", 0.889, 0.070, 0.041, False, 1.0),
        ("T2", 0.967, 0.020, 0.013, True, 0.0),
    ]
    frame = pd.DataFrame(
        [
            {
                "candidate_id": f"wall-split-{index}",
                "timepoint": timepoint,
                "integrated_label": "single",
                "v2_pre_temporal_integrated_label": "single",
                "cell_probability": cell,
                "debris_probability": debris,
                "invalid_probability": invalid,
                "v2_is_unique_instance": True,
                "v2_is_counting_instance": counting,
                "v2_mask_valid": True,
                "v2_wall_rejected": False,
                "v2_is_suppressed": False,
                "v2_instance_confidence": 0.89,
                "v2_objectness": 0.95,
                "v2_wall_overlap": wall_overlap,
            }
            for index, (
                timepoint,
                cell,
                debris,
                invalid,
                counting,
                wall_overlap,
            ) in enumerate(values)
        ]
    )
    descriptors = {
        0: _textured_split_descriptor(0, "T0", 2912, 2168, area=55),
        1: _textured_split_descriptor(1, "T1", 2918, 2167, area=42),
        2: _textured_split_descriptor(2, "T1", 2749, 2105, area=245),
        3: _textured_split_descriptor(3, "T2", 2916, 2163, area=52),
        4: _textured_split_descriptor(4, "T2", 2741, 2034, area=300),
    }
    component_nodes = {1: [0, 1, 3], 2: [2, 4]}
    rescues = _detect_cross_track_division_rescues(
        frame,
        list(frame.index),
        {
            "T0": [descriptors[0]],
            "T1": [descriptors[1], descriptors[2]],
            "T2": [descriptors[3], descriptors[4]],
        },
        descriptors,
        [_edge(0, 1), _edge(1, 3), _edge(2, 4)],
        {(2, 4): _edge(2, 4)},
        {0: 1, 1: 1, 3: 1, 2: 2, 4: 2},
        component_nodes,
        {
            index: f"H9:O{component:03d}"
            for component, nodes in component_nodes.items()
            for index in nodes
        },
        {
            index: {"v3_track_behavior": "wall_structure_invalid"}
            for index in frame.index
        },
        {
            "cross_track_division_rescue_enabled": True,
            "cross_track_division_rescue_radius_px": 88,
            "cross_track_division_rescue_wall_radius_px": 224,
            "cross_track_division_rescue_minimum_identity": 0.62,
            "cross_track_division_rescue_wall_minimum_identity": 0.25,
            "cross_track_division_rescue_wall_minimum_shape_similarity": 0.70,
            "cross_track_division_rescue_wall_minimum_secondary_strong_frames": 2,
            "cross_track_division_rescue_wall_maximum_area_ratio": 10.0,
            "cross_track_division_rescue_minimum_foreground_quality": 0.30,
            "cross_track_division_rescue_minimum_child_separation_px": 5,
        },
    )

    assert {rescue["interval"] for rescue in rescues} == {"T0->T1", "T1->T2"}
    assert {rescue["secondary_child_index"] for rescue in rescues} == {2, 4}


def test_component_static_wall_rule_preserves_repeated_strong_cell_track():
    frame = _temporal_frame([(0.93, 0.04, "single")] * 3)
    frame["v2_wall_overlap"] = 1.0
    frame["v2_instance_confidence"] = 0.89
    frame["v2_objectness"] = 0.95
    frame["v2_mask_valid"] = True
    frame["v2_is_unique_instance"] = True
    frame["v2_is_suppressed"] = False
    frame["v2_wall_rejected"] = False
    outputs = _component_temporal_outputs(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        {},
        "H9:wall-cell",
    )

    assert all(output["static_wall_cell_veto"] for output in outputs.values())
    assert all(output["strong_cell_frame_count"] == 3 for output in outputs.values())
    assert all(not output["static_wall"] for output in outputs.values())
    assert all(not output["suspected_dead_cell"] for output in outputs.values())
    assert {output["label"] for output in outputs.values()} == {"single"}


def test_three_frame_foreground_stability_is_strong_debris_evidence():
    frame = _temporal_frame([(0.55, 0.40, "uncertain")] * 3)
    outputs = _component_temporal_outputs(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        {},
        "A1:object",
    )

    assert all(output["three_frame_static"] for output in outputs.values())
    assert all(output["applied"] for output in outputs.values())
    assert all(output["label"] == "debris" for output in outputs.values())
    assert all(
        output["reason"] == "three_frame_static_debris_consensus"
        for output in outputs.values()
    )


def test_triplet_mean_and_shape_tolerate_one_focus_shift_for_static_debris():
    frame = _temporal_frame(
        [
            (0.52, 0.45, "single"),
            (0.70, 0.27, "single"),
            (0.77, 0.11, "single"),
        ]
    )
    first = _edge(0, 1, static=0.95)
    second = TemporalPairEvidence(
        **{
            **_edge(1, 2, static=0.76).__dict__,
            "tolerant_shape": 0.96,
        }
    )

    outputs = _component_temporal_outputs(
        frame, [0, 1, 2], [first, second], {}, "B6:static"
    )

    assert all(output["three_frame_static"] for output in outputs.values())
    assert all(output["label"] == "debris" for output in outputs.values())
    assert all(
        output["reason"] == "three_frame_static_debris_consensus"
        for output in outputs.values()
    )


def test_morphology_stable_triplet_tolerates_photometric_variation():
    frame = _temporal_frame(
        [
            (0.589, 0.390, "touching_doublet"),
            (0.674, 0.306, "touching_doublet"),
            (0.530, 0.432, "touching_doublet"),
        ]
    )
    edges = [
        TemporalPairEvidence(
            **{**_edge(0, 1, static=0.74).__dict__, "tolerant_shape": 0.89}
        ),
        TemporalPairEvidence(
            **{**_edge(1, 2, static=0.74).__dict__, "tolerant_shape": 0.89}
        ),
    ]

    outputs = _component_temporal_outputs(
        frame, [0, 1, 2], edges, {}, "A4:morphology-stable"
    )

    assert all(not output["three_frame_static"] for output in outputs.values())
    assert all(
        output["morphology_stable_three_frame"]
        for output in outputs.values()
    )
    assert all(output["label"] == "debris" for output in outputs.values())
    assert all(
        output["reason"] == "three_frame_morphology_stable_debris_consensus"
        for output in outputs.values()
    )


def test_static_cell_like_frames_do_not_emit_v2_suspected_dead_cell():
    frame = _temporal_frame(
        [
            (0.68, 0.28, "single"),
            (0.93, 0.06, "single"),
            (0.79, 0.20, "single"),
        ]
    )
    outputs = _component_temporal_outputs(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        {},
        "B10:legacy-dead-cell-case",
    )

    assert all(not output["suspected_dead_cell"] for output in outputs.values())
    assert all(output["suspected_dead_cell_score"] == 0.0 for output in outputs.values())
    assert all(output["label"] == "single" for output in outputs.values())
    assert all(output["reason"] != "suspected_dead_cell" for output in outputs.values())


def test_static_two_debris_frames_override_one_confident_cell_frame():
    frame = _temporal_frame(
        [
            (0.10, 0.89, "debris"),
            (0.02, 0.98, "debris"),
            (0.81, 0.19, "single"),
        ]
    )
    outputs = _component_temporal_outputs(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        {},
        "D8:static-debris-consensus",
    )

    assert outputs[2]["label"] == "debris"
    assert outputs[2]["cell"] < outputs[2]["debris"]
    assert outputs[2]["reason"] == "three_frame_static_debris_consensus"


def test_debris_dominant_unary_candidate_cannot_be_restored_as_single():
    frame = _temporal_frame([(0.395, 0.511, "single")])
    outputs = _component_temporal_outputs(
        frame, [0], [], {}, "B9:unary"
    )

    assert outputs[0]["label"] == "debris"
    assert outputs[0]["reason"] == "unary_debris_probability_dominant"


def test_debris_multiplicity_head_cannot_create_growth_evidence():
    frame = _temporal_frame(
        [
            (0.41, 0.53, "single"),
            (0.22, 0.53, "debris"),
            (0.26, 0.66, "debris"),
        ]
    )
    frame["predicted_multiplicity"] = [
        "single",
        "single",
        "touching_doublet",
    ]
    outputs = _component_temporal_outputs(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        {},
        "H11:false-growth",
    )

    assert all(output["growth"] == 0 for output in outputs.values())
    assert all(
        output["reason"] != "division_or_growth_cell_evidence"
        for output in outputs.values()
    )
    assert all(output["label"] == "debris" for output in outputs.values())


def test_rising_debris_probability_across_three_matched_frames_shifts_track_to_debris():
    frame = _temporal_frame(
        [
            (0.58, 0.39, "single"),
            (0.41, 0.55, "uncertain"),
            (0.18, 0.81, "debris"),
        ]
    )
    outputs = _component_temporal_outputs(
        frame,
        [0, 1, 2],
        [_edge(0, 1, static=0.78), _edge(1, 2, static=0.76)],
        {},
        "C2:rising-debris",
    )

    assert outputs[0]["applied"] is True
    assert outputs[0]["debris"] > frame.loc[0, "debris_probability"]
    assert outputs[0]["reason"] == "multi_frame_debris_probability_trend"
    assert outputs[0]["proposal_count"] == 3


def test_static_temporal_evidence_does_not_revive_invalid_candidate():
    frame = _temporal_frame(
        [
            (0.01, 0.45, "invalid"),
            (0.02, 0.70, "debris"),
            (0.01, 0.68, "debris"),
        ]
    )
    frame.loc[0, "invalid_probability"] = 0.54
    outputs = _component_temporal_outputs(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        {},
        "invalid:static",
    )

    assert outputs[0]["label"] == "invalid"
    assert outputs[0]["reason"] == "invalid_candidate"


def test_two_frame_stability_is_weaker_than_three_frame_stability():
    two_frame = _temporal_frame([(0.55, 0.40, "uncertain")] * 2)
    three_frame = _temporal_frame([(0.55, 0.40, "uncertain")] * 3)

    two_outputs = _component_temporal_outputs(
        two_frame,
        [0, 1],
        [_edge(0, 1)],
        {},
        "A1:two-frame",
    )
    three_outputs = _component_temporal_outputs(
        three_frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        {},
        "A1:three-frame",
    )

    assert two_outputs[0]["applied"] is True
    assert two_outputs[0]["reason"] == "two_frame_static_object"
    assert three_outputs[0]["debris"] > two_outputs[0]["debris"]


def test_division_and_later_cell_anchor_correct_ambiguous_parent_to_cell():
    frame = _temporal_frame(
        [
            (0.52, 0.43, "single"),
            (0.94, 0.04, "single"),
            (0.95, 0.03, "single"),
        ]
    )
    outputs = _component_temporal_outputs(
        frame,
        [0, 1, 2],
        [_edge(0, 1, kind="division", static=0.55), _edge(0, 2, kind="division", static=0.52)],
        {},
        "A1:growth",
    )

    assert outputs[0]["applied"] is True
    assert outputs[0]["label"] == "single"
    assert outputs[0]["reason"] == "division_or_growth_cell_evidence"
    assert outputs[1]["label"] == "single"


def test_high_confidence_debris_anchor_keeps_its_exact_unary_label():
    frame = _temporal_frame(
        [
            (0.55, 0.40, "uncertain"),
            (0.03, 0.95, "debris"),
            (0.04, 0.94, "debris"),
        ]
    )
    outputs = _component_temporal_outputs(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        {},
        "A1:debris",
    )

    assert outputs[1]["label"] == "debris"
    assert outputs[2]["label"] == "debris"
    assert outputs[1]["debris"] >= frame.loc[1, "debris_probability"]
    assert outputs[1]["reason"] == "three_frame_static_debris_consensus"


def test_clean_eighty_percent_cell_anchor_corrects_matched_ambiguous_parent():
    frame = _temporal_frame(
        [
            (0.255858, 0.642381, "debris"),
            (0.814744, 0.162519, "single"),
        ]
    )
    frame["v2_instance_confidence"] = [0.73, 0.79]
    frame["v2_objectness"] = [0.99, 0.99]
    frame["v2_is_unique_instance"] = True
    frame["v2_is_counting_instance"] = True
    frame.loc[0, "predicted_multiplicity"] = "touching_doublet"
    edge = TemporalPairEvidence(
        **{
            **_edge(0, 1, static=0.841).__dict__,
            "identity": 0.85,
            "tolerant_shape": 0.978,
            "foreground_quality": 0.88,
        }
    )

    outputs = _component_temporal_outputs(frame, [0, 1], [edge], {}, "A11:object")

    assert outputs[0]["applied"] is True
    assert outputs[0]["cell"] > frame.loc[0, "cell_probability"]
    assert outputs[0]["debris"] < frame.loc[0, "debris_probability"]
    assert outputs[0]["label"] == "touching_doublet"
    assert outputs[0]["reason"] == "high_confidence_cell_anchor"
    assert outputs[1]["applied"] is False
    assert outputs[1]["label"] == "single"


def test_low_quality_eighty_percent_cell_candidate_is_not_an_anchor():
    frame = _temporal_frame(
        [
            (0.45, 0.50, "uncertain"),
            (0.82, 0.16, "single"),
        ]
    )
    frame["v2_instance_confidence"] = [0.80, 0.30]
    frame["v2_objectness"] = [0.99, 0.99]
    frame["v2_is_unique_instance"] = True
    frame["v2_is_counting_instance"] = True

    outputs = _component_temporal_outputs(
        frame,
        [0, 1],
        [_edge(0, 1, static=0.90)],
        {},
        "A11:low-quality",
    )

    # The shared morphology consensus may still contribute weak cell evidence,
    # but the low-quality T1 observation must not be treated as an anchor.
    assert outputs[0]["reason"] in {
        "multi_frame_cell_consensus",
        "conflicting_temporal_evidence",
    }
    assert outputs[0]["reason"] != "high_confidence_cell_anchor"

from __future__ import annotations

import pandas as pd

from cellvision.temporal_behavior import evaluate_temporal_behavior
from cellvision.temporal_objects import TemporalPairEvidence, conditional_cell_probability


def _row(
    index: int,
    timepoint: str,
    cell: float,
    debris: float,
    *,
    label: str = "uncertain",
    radial: float = 0.20,
    wall_overlap: float = 0.0,
    neighbours: int = 0,
    anisotropy: float = 0.10,
    blobness: float = 0.0,
    dense_response: float = 0.0,
    source: str = "cf_component",
) -> dict[str, object]:
    return {
        "index": index,
        "timepoint": timepoint,
        "cell_probability": cell,
        "debris_probability": debris,
        "invalid_probability": 0.05,
        "integrated_label": label,
        "v2_pre_temporal_integrated_label": label,
        "v2_mask_valid": True,
        "v2_wall_rejected": False,
        "v2_is_suppressed": False,
        "radial_fraction": radial,
        "v2_wall_overlap": wall_overlap,
        "wall_neighbor_count": neighbours,
        "background_anisotropy": anisotropy,
        "wall_rescue_blobness": blobness,
        "dense_response": dense_response,
        "candidate_source": source,
    }


def _edge(
    left: int,
    right: int,
    *,
    identity: float = 0.93,
    static: float = 0.95,
    shape: float = 0.96,
    area: float = 0.98,
    kind: str = "continuation",
) -> TemporalPairEvidence:
    return TemporalPairEvidence(
        left=left,
        right=right,
        identity=identity,
        static=static,
        intensity=static,
        gradient=static,
        tolerant_shape=shape,
        area_similarity=area,
        distance_px=2.0,
        foreground_quality=0.95,
        kind=kind,
    )


def _settings() -> dict[str, object]:
    return {
        "minimum_identity": 0.58,
        "minimum_foreground_quality": 0.30,
        "stable_static_similarity": 0.82,
        "stable_shape_similarity": 0.90,
    }


def test_three_frame_stability_promotes_ambiguous_interior_track_to_debris():
    frame = pd.DataFrame(
        [
            _row(0, "T0", 0.52, 0.43),
            _row(1, "T1", 0.48, 0.47),
            _row(2, "T2", 0.50, 0.45),
        ]
    ).set_index("index")
    outputs = evaluate_temporal_behavior(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        _settings(),
        "W:track",
    )

    assert {value["v3_track_behavior"] for value in outputs.values()} == {"stable_debris"}
    assert {value["v3_proposed_label"] for value in outputs.values()} == {"debris"}


def test_morphology_stability_tolerates_photometric_variation():
    frame = pd.DataFrame(
        [
            _row(0, "T0", 0.589, 0.390, label="touching_doublet"),
            _row(1, "T1", 0.674, 0.306, label="touching_doublet"),
            _row(2, "T2", 0.530, 0.432, label="touching_doublet"),
        ]
    ).set_index("index")
    outputs = evaluate_temporal_behavior(
        frame,
        [0, 1, 2],
        [
            _edge(0, 1, identity=0.81, static=0.74, shape=0.89),
            _edge(1, 2, identity=0.81, static=0.74, shape=0.89),
        ],
        _settings(),
        "A4:morphology-stable",
    )

    assert {value["v3_track_behavior"] for value in outputs.values()} == {
        "stable_debris"
    }
    assert {value["v3_proposed_label"] for value in outputs.values()} == {
        "debris"
    }
    assert {
        value["v3_reason"] for value in outputs.values()
    } == {
        "three_frame_morphology_stable_noncell_without_persistent_cell_evidence"
    }


def test_division_vetoes_static_debris_and_dead_cell_logic_in_both_intervals():
    frame = pd.DataFrame(
        [
            _row(0, "T0", 0.86, 0.08, label="single"),
            _row(1, "T1", 0.78, 0.13, label="single"),
            _row(2, "T1", 0.76, 0.14, label="single"),
        ]
    ).set_index("index")
    outputs = evaluate_temporal_behavior(
        frame,
        [0, 1, 2],
        [_edge(0, 1, kind="division"), _edge(0, 2, kind="division")],
        _settings(),
        "W:division",
    )

    assert outputs[0]["v3_track_behavior"] == "division_or_growth"
    assert outputs[0]["v3_division_veto"] is True
    assert outputs[0]["v3_division_interval"] == "T0->T1"
    assert all(value["v3_proposed_label"] in {"single", "uncertain"} for value in outputs.values())
    assert all(value["v3_proposed_label"] != "debris" for value in outputs.values())


def test_cell_to_debris_accepts_debris_leaning_t2_without_requiring_strong_t2_debris():
    frame = pd.DataFrame(
        [
            _row(0, "T0", 0.85, 0.10, label="single"),
            _row(1, "T1", 0.64, 0.20, label="uncertain"),
            _row(2, "T2", 0.40, 0.60, label="uncertain"),
        ]
    ).set_index("index")
    outputs = evaluate_temporal_behavior(
        frame,
        [0, 1, 2],
        [
            _edge(0, 1, static=0.42, shape=0.34, area=0.72),
            _edge(1, 2, static=0.38, shape=0.30, area=0.68),
        ],
        _settings(),
        "W:death",
    )

    assert {value["v3_track_behavior"] for value in outputs.values()} == {"cell_to_debris"}
    assert {value["v3_track_conclusion"] for value in outputs.values()} == {"dead_cell"}
    assert {value["v3_label_mode"] for value in outputs.values()} == {"unified_track"}
    assert outputs[0]["v3_proposed_label"] == "single"
    assert outputs[1]["v3_frame_state"] == "degenerating"
    assert outputs[2]["v3_proposed_label"] == "debris"
    assert outputs[2]["v3_proposed_cell_probability"] <= 0.40


def test_probability_decline_without_morphology_degradation_remains_uncertain():
    frame = pd.DataFrame(
        [
            _row(0, "T0", 0.85, 0.10, label="single"),
            _row(1, "T1", 0.75, 0.15, label="single"),
            _row(2, "T2", 0.49, 0.51, label="single"),
        ]
    ).set_index("index")
    outputs = evaluate_temporal_behavior(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        _settings(),
        "W:decline",
    )

    assert {value["v3_track_behavior"] for value in outputs.values()} == {
        "decline_without_morphology_evidence"
    }
    assert outputs[2]["v3_proposed_label"] != "debris"


def test_semantic_cell_to_debris_transition_does_not_require_mask_shape_change():
    frame = pd.DataFrame(
        [
            _row(0, "T0", 0.85, 0.10, label="single"),
            _row(1, "T1", 0.64, 0.20, label="single"),
            _row(2, "T2", 0.40, 0.60, label="debris"),
        ]
    ).set_index("index")
    outputs = evaluate_temporal_behavior(
        frame,
        [0, 1, 2],
        # Static geometry can coexist with loss of internal cell texture.
        [_edge(0, 1), _edge(1, 2)],
        _settings(),
        "W:semantic-death",
    )

    assert {value["v3_track_behavior"] for value in outputs.values()} == {
        "cell_to_debris"
    }
    assert {value["v3_track_conclusion"] for value in outputs.values()} == {"dead_cell"}
    assert all(value["v3_semantic_degradation"] for value in outputs.values())
    assert outputs[2]["v3_proposed_label"] == "debris"


def test_stable_wall_texture_is_invalid_only_with_positive_structure_evidence():
    frame = pd.DataFrame(
        [
            _row(0, "T0", 0.52, 0.43, radial=0.48, wall_overlap=0.96, neighbours=3, anisotropy=0.80),
            _row(1, "T1", 0.48, 0.47, radial=0.48, wall_overlap=0.95, neighbours=3, anisotropy=0.82),
            _row(2, "T2", 0.50, 0.45, radial=0.48, wall_overlap=0.96, neighbours=3, anisotropy=0.79),
        ]
    ).set_index("index")
    outputs = evaluate_temporal_behavior(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        _settings(),
        "W:wall-texture",
    )

    assert {value["v3_wall_origin"] for value in outputs.values()} == {"wall_structure"}
    assert {value["v3_track_behavior"] for value in outputs.values()} == {"wall_structure_invalid"}
    assert {value["v3_proposed_label"] for value in outputs.values()} == {"invalid"}


def test_compact_wall_rescue_is_not_forced_to_wall_structure_invalid():
    frame = pd.DataFrame(
        [
            _row(0, "T0", 0.84, 0.10, label="single", radial=0.48, wall_overlap=0.55, blobness=0.60, source="wall_cell_rescue_peak"),
            _row(1, "T1", 0.82, 0.12, label="single", radial=0.48, wall_overlap=0.52, blobness=0.58, source="wall_cell_rescue_peak"),
            _row(2, "T2", 0.83, 0.11, label="single", radial=0.48, wall_overlap=0.54, blobness=0.59, source="wall_cell_rescue_peak"),
        ]
    ).set_index("index")
    outputs = evaluate_temporal_behavior(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        _settings(),
        "W:wall-cell",
    )

    assert {value["v3_wall_origin"] for value in outputs.values()} == {"wall_independent_object"}
    assert {value["v3_proposed_label"] for value in outputs.values()} == {"single"}


def test_persistent_high_confidence_wall_cell_overrides_wall_structure():
    frame = pd.DataFrame(
        [
            _row(0, "T0", 0.82, 0.10, label="single", radial=0.48, wall_overlap=0.96, neighbours=3, anisotropy=0.80, dense_response=48.0, source="wall_residual_peak"),
            _row(1, "T1", 0.81, 0.11, label="single", radial=0.48, wall_overlap=0.95, neighbours=3, anisotropy=0.82, dense_response=50.0, source="wall_residual_peak"),
            _row(2, "T2", 0.80, 0.12, label="single", radial=0.48, wall_overlap=0.96, neighbours=3, anisotropy=0.79, dense_response=52.0, source="wall_residual_peak"),
        ]
    ).set_index("index")
    outputs = evaluate_temporal_behavior(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        _settings(),
        "W:wall-residual-cell",
    )

    assert {value["v3_wall_origin"] for value in outputs.values()} == {
        "wall_independent_object"
    }
    assert {value["v3_track_behavior"] for value in outputs.values()} == {
        "stable_cell_or_conflict"
    }
    assert {value["v3_proposed_label"] for value in outputs.values()} == {"single"}
    assert all(value["v3_wall_cell_veto"] for value in outputs.values())
    assert {
        value["v3_wall_strong_cell_frame_count"] for value in outputs.values()
    } == {3}


def test_b8_like_stable_low_blobness_wall_residual_is_invalid():
    frame = pd.DataFrame(
        [
            _row(0, "T0", 0.421, 0.082, radial=0.441, wall_overlap=1.0, anisotropy=0.789, blobness=0.219, dense_response=43.2, source="wall_residual_peak"),
            _row(1, "T1", 0.863, 0.003, label="single", radial=0.441, wall_overlap=1.0, anisotropy=0.847, blobness=0.272, dense_response=53.5, source="wall_residual_peak"),
            _row(2, "T2", 0.675, 0.014, label="single", radial=0.441, wall_overlap=1.0, anisotropy=0.810, blobness=0.250, dense_response=62.8, source="wall_residual_peak"),
        ]
    ).set_index("index")
    outputs = evaluate_temporal_behavior(
        frame,
        [0, 1, 2],
        [
            _edge(0, 1, identity=0.873, static=0.887, shape=0.903),
            _edge(1, 2, identity=0.873, static=0.887, shape=0.903),
        ],
        _settings(),
        "B8:wall-rescue:264:2070",
    )

    assert {value["v3_wall_origin"] for value in outputs.values()} == {
        "wall_structure"
    }
    assert {value["v3_reason"] for value in outputs.values()} == {
        "stable_wall_site_structure"
    }
    assert {value["v3_proposed_label"] for value in outputs.values()} == {"invalid"}


def test_invalid_probability_is_not_removed_from_persistent_cell_evidence():
    frame = pd.DataFrame(
        [
            _row(0, "T0", 0.56, 0.01, label="single"),
            _row(1, "T1", 0.56, 0.01, label="single"),
            _row(2, "T2", 0.56, 0.01, label="single"),
        ]
    ).set_index("index")
    frame["invalid_probability"] = 0.43

    assert conditional_cell_probability(frame.loc[2]) == 0.56
    outputs = evaluate_temporal_behavior(
        frame,
        [0, 1, 2],
        [_edge(0, 1), _edge(1, 2)],
        _settings(),
        "D2:invalid-aware",
    )

    assert all(value["v3_persistent_cell_evidence"] is False for value in outputs.values())
    assert {value["v3_track_behavior"] for value in outputs.values()} == {"stable_debris"}
    assert {value["v3_proposed_label"] for value in outputs.values()} == {"debris"}

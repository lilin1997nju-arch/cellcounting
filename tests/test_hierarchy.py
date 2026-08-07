import pandas as pd

from cellvision.hierarchy import suppress_nested_single_candidates


def _config() -> dict:
    return {
        "hierarchical_suppression": {
            "minimum_group_confidence": 0.72,
            "single_diameter_reference_px": 8.0,
            "doublet_radius_scale": 2.5,
            "cluster_radius_scale": 3.0,
            "maximum_radius_px": 96.0,
        }
    }


def test_single_inside_reviewed_doublet_is_suppressed() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "group",
                "well": "H4",
                "timepoint": "T2",
                "x_px": 100.0,
                "y_px": 100.0,
                "diameter_px": 14.0,
                "label": "touching_doublet",
                "confidence": 0.4,
                "reviewed_label": "touching_doublet",
            },
            {
                "candidate_id": "nested",
                "well": "H4",
                "timepoint": "T2",
                "x_px": 112.0,
                "y_px": 100.0,
                "diameter_px": 8.0,
                "label": "single",
                "confidence": 0.99,
                "reviewed_label": "single",
            },
        ]
    )
    result = suppress_nested_single_candidates(
        frame, _config(), label_column="label", confidence_column="confidence"
    )
    nested = result[result["candidate_id"] == "nested"].iloc[0]
    assert bool(nested["is_hierarchy_suppressed"])
    assert nested["suppressed_by_candidate_id"] == "group"


def test_nearby_independent_single_is_preserved() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "group",
                "well": "H4",
                "timepoint": "T2",
                "x_px": 100.0,
                "y_px": 100.0,
                "diameter_px": 14.0,
                "label": "touching_doublet",
                "confidence": 0.95,
            },
            {
                "candidate_id": "independent",
                "well": "H4",
                "timepoint": "T2",
                "x_px": 151.0,
                "y_px": 100.0,
                "diameter_px": 8.0,
                "label": "single",
                "confidence": 0.99,
            },
        ]
    )
    result = suppress_nested_single_candidates(
        frame, _config(), label_column="label", confidence_column="confidence"
    )
    independent = result[result["candidate_id"] == "independent"].iloc[0]
    assert not bool(independent["is_hierarchy_suppressed"])


def test_low_confidence_unreviewed_group_does_not_hide_single() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "weak-group",
                "well": "H4",
                "timepoint": "T2",
                "x_px": 100.0,
                "y_px": 100.0,
                "diameter_px": 14.0,
                "label": "cluster_3plus",
                "confidence": 0.5,
            },
            {
                "candidate_id": "single",
                "well": "H4",
                "timepoint": "T2",
                "x_px": 108.0,
                "y_px": 100.0,
                "diameter_px": 8.0,
                "label": "single",
                "confidence": 0.99,
            },
        ]
    )
    result = suppress_nested_single_candidates(
        frame, _config(), label_column="label", confidence_column="confidence"
    )
    single = result[result["candidate_id"] == "single"].iloc[0]
    assert not bool(single["is_hierarchy_suppressed"])


def test_cross_source_duplicate_single_is_counted_once() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "cf",
                "well": "C3",
                "timepoint": "T1",
                "x_px": 100.0,
                "y_px": 100.0,
                "diameter_px": 12.0,
                "area_px": 110.0,
                "candidate_source": "cf_component",
                "label": "single",
                "confidence": 0.91,
            },
            {
                "candidate_id": "dense",
                "well": "C3",
                "timepoint": "T1",
                "x_px": 105.0,
                "y_px": 101.0,
                "diameter_px": 11.0,
                "area_px": 96.0,
                "candidate_source": "multiscale_dense_peak",
                "label": "single",
                "confidence": 0.96,
            },
        ]
    )
    result = suppress_nested_single_candidates(
        frame, _config(), label_column="label", confidence_column="confidence"
    )
    assert int(result["is_counting_instance"].sum()) == 1
    assert result.loc[result["candidate_id"] == "dense", "is_duplicate_suppressed"].item()


def test_empty_review_columns_do_not_protect_automatic_duplicates() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "well": "C3",
                "timepoint": "T1",
                "x_px": 100.0,
                "y_px": 100.0,
                "diameter_px": 12.0,
                "area_px": 110.0,
                "candidate_source": "cf_component",
                "label": "single",
                "confidence": 0.91,
                "reviewed_label": None,
            },
            {
                "candidate_id": "b",
                "well": "C3",
                "timepoint": "T1",
                "x_px": 104.0,
                "y_px": 100.0,
                "diameter_px": 12.0,
                "area_px": 110.0,
                "candidate_source": "multiscale_dense_peak",
                "label": "single",
                "confidence": 0.93,
                "reviewed_label": None,
            },
        ]
    )
    result = suppress_nested_single_candidates(
        frame, _config(), label_column="label", confidence_column="confidence"
    )
    assert int(result["is_counting_instance"].sum()) == 1


def test_two_adjacent_true_singles_are_not_merged() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "left",
                "well": "C3",
                "timepoint": "T1",
                "x_px": 100.0,
                "y_px": 100.0,
                "diameter_px": 12.0,
                "area_px": 110.0,
                "candidate_source": "cf_component",
                "label": "single",
                "confidence": 0.91,
            },
            {
                "candidate_id": "right",
                "well": "C3",
                "timepoint": "T1",
                "x_px": 112.0,
                "y_px": 100.0,
                "diameter_px": 12.0,
                "area_px": 110.0,
                "candidate_source": "cf_component",
                "label": "single",
                "confidence": 0.90,
            },
        ]
    )
    result = suppress_nested_single_candidates(
        frame, _config(), label_column="label", confidence_column="confidence"
    )
    assert int(result["is_counting_instance"].sum()) == 2


def test_group_duplicates_and_internal_single_have_one_counting_parent() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "group-a",
                "well": "C3",
                "timepoint": "T1",
                "x_px": 100.0,
                "y_px": 100.0,
                "diameter_px": 22.0,
                "area_px": 330.0,
                "candidate_source": "cf_component",
                "label": "touching_doublet",
                "confidence": 0.84,
            },
            {
                "candidate_id": "group-b",
                "well": "C3",
                "timepoint": "T1",
                "x_px": 105.0,
                "y_px": 101.0,
                "diameter_px": 20.0,
                "area_px": 300.0,
                "candidate_source": "multiscale_dense_peak",
                "label": "cluster_3plus",
                "confidence": 0.76,
            },
            {
                "candidate_id": "internal-core",
                "well": "C3",
                "timepoint": "T1",
                "x_px": 108.0,
                "y_px": 100.0,
                "diameter_px": 9.0,
                "area_px": 64.0,
                "candidate_source": "multiscale_dense_peak",
                "label": "single",
                "confidence": 0.97,
            },
        ]
    )
    result = suppress_nested_single_candidates(
        frame, _config(), label_column="label", confidence_column="confidence"
    )
    assert int(result["is_counting_instance"].sum()) == 1
    core = result[result["candidate_id"] == "internal-core"].iloc[0]
    assert bool(
        core["is_hierarchy_suppressed"]
        or core["is_duplicate_suppressed"]
    )
    assert core["parent_candidate_id"] or core["duplicate_of_candidate_id"]


def test_local_peak_inherits_complete_cf_instance_beyond_nms_radius() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "B1:T0:cf:76",
                "well": "B1",
                "timepoint": "T0",
                "x_px": 320.91,
                "y_px": 1420.06,
                "diameter_px": 6.58,
                "area_px": 34.0,
                "candidate_source": "cf_component",
                "label": "single",
                "confidence": 0.86,
            },
            {
                "candidate_id": "B1:T0:raw:327:1421",
                "well": "B1",
                "timepoint": "T0",
                "x_px": 327.0,
                "y_px": 1421.0,
                "diameter_px": 8.29,
                "area_px": 54.0,
                "candidate_source": "wall_residual_peak",
                "label": "single",
                "confidence": 0.86,
            },
        ]
    )
    result = suppress_nested_single_candidates(
        frame, _config(), label_column="label", confidence_column="confidence"
    )
    assert int(result["is_counting_instance"].sum()) == 1
    raw = result[result["candidate_source"] == "wall_residual_peak"].iloc[0]
    assert raw["instance_component_id"] == "B1:T0:cf:76"


def test_single_peak_inside_doublet_cf_component_is_not_counted_twice() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "B1:T2:cf:23",
                "well": "B1",
                "timepoint": "T2",
                "x_px": 781.51,
                "y_px": 472.22,
                "diameter_px": 10.28,
                "area_px": 83.0,
                "candidate_source": "cf_component",
                "label": "touching_doublet",
                "confidence": 0.46,
            },
            {
                "candidate_id": "B1:T2:raw:787:476",
                "well": "B1",
                "timepoint": "T2",
                "x_px": 787.0,
                "y_px": 476.0,
                "diameter_px": 8.74,
                "area_px": 60.0,
                "candidate_source": "wall_residual_peak",
                "label": "single",
                "confidence": 0.52,
            },
        ]
    )
    result = suppress_nested_single_candidates(
        frame, _config(), label_column="label", confidence_column="confidence"
    )
    assert int(result["is_counting_instance"].sum()) == 1
    assert result.loc[
        result["candidate_id"] == "B1:T2:raw:787:476",
        "is_duplicate_suppressed",
    ].item()

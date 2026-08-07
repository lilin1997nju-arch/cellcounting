import pandas as pd

from cellvision.active_learning import (
    filter_review_candidates,
    model_lineage_candidates,
    morphology_review_candidates,
)


def test_blank_candidates_and_positive_control_are_excluded():
    frame = pd.DataFrame(
        [
            {
                "well": "A1",
                "candidate_source": "cf_component",
                "foreground_fraction": 0.2,
                "confidence": 0.99,
            },
            {
                "well": "H6",
                "candidate_source": "instrument_csv",
                "foreground_fraction": 0.0,
                "confidence": 0.59,
            },
            {
                "well": "H6",
                "candidate_source": "cf_component",
                "foreground_fraction": 0.005,
                "confidence": 0.84,
            },
        ]
    )
    filtered = filter_review_candidates(
        frame,
        {
            "excluded_wells": ["A1"],
            "minimum_foreground_fraction": 0.001,
            "minimum_confidence": 0.65,
            "prefer_candidate_sources": ["cf_component", "instrument_csv"],
        },
    )
    assert len(filtered) == 1
    assert filtered.iloc[0]["well"] == "H6"
    assert filtered.iloc[0]["candidate_source"] == "cf_component"


def test_morphology_queue_covers_wells_and_limits_targets(tmp_path):
    artifact_root = tmp_path / "artifacts"
    source = artifact_root / "pseudo_labels" / "morphology_candidates.csv"
    source.parent.mkdir(parents=True)
    rows = []
    for well in ("A1", "A2", "H6"):
        for index in range(5):
            rows.append(
                {
                    "candidate_id": f"{well}:T0:cf:{index}",
                    "well": well,
                    "timepoint": "T0",
                    "x_px": 100 + index,
                    "y_px": 200 + index,
                    "area_px": 50,
                    "radial_fraction": 0.3,
                    "circularity": 0.8,
                    "solidity": 0.9,
                    "extent": 0.7,
                    "temporal_support": 1,
                    "background_anisotropy": 0.2,
                    "pseudo_label": "uncertain",
                }
            )
    pd.DataFrame(rows).to_csv(source, index=False)
    queue = morphology_review_candidates(
        {
            "paths": {"artifact_root": str(artifact_root)},
            "review_queue": {
                "excluded_wells": ["A1"],
                "max_candidates_per_well": 3,
            },
        }
    )
    assert queue is not None
    assert set(queue["well"]) == {"A2", "H6"}
    assert queue.groupby("well").size().to_dict() == {"A2": 3, "H6": 3}


def test_model_queue_keeps_cells_and_cell_like_uncertain_candidates(tmp_path):
    artifact_root = tmp_path / "artifacts"
    source = (
        artifact_root
        / "predictions"
        / "latest_auto_annotations.csv"
    )
    source.parent.mkdir(parents=True)
    rows = [
        ("A1:T0:cf:1", "A1", "cell", "auto_high_confidence", 0.99),
        ("A2:T0:cf:1", "A2", "cell", "auto_high_confidence", 0.98),
        ("A2:T0:cf:2", "A2", "debris", "needs_review", 0.31),
        ("A2:T0:cf:3", "A2", "invalid", "needs_review", 0.24),
        ("A3:T0:cf:1", "A3", "invalid", "auto_high_confidence", 0.40),
    ]
    pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "well": well,
                "timepoint": "T0",
                "x_px": 100,
                "y_px": 200,
                "area_px": 50,
                "auto_label": label,
                "auto_status": status,
                "cell_probability": cell_probability,
                "debris_probability": 0.2,
                "invalid_probability": 1 - cell_probability,
                "confidence": max(cell_probability, 1 - cell_probability),
            }
            for candidate_id, well, label, status, cell_probability in rows
        ]
    ).to_csv(source, index=False)
    queue = model_lineage_candidates(
        {
            "paths": {"artifact_root": str(artifact_root)},
            "review_queue": {
                "excluded_wells": ["A1"],
                "model_cell_probability_min": 0.25,
                "model_max_candidates_per_well": 4,
            },
        }
    )
    assert queue is not None
    assert set(queue["candidate_id"]) == {
        "A2:T0:cf:1",
        "A2:T0:cf:2",
    }
    assert dict(zip(queue["candidate_id"], queue["candidate_source"])) == {
        "A2:T0:cf:1": "trained_model_cell",
        "A2:T0:cf:2": "trained_model_debris",
    }

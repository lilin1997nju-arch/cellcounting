from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cellvision.models.morphology_classifier import MorphologyClassifier
from PIL import Image

from cellvision.pseudo_labels import (
    _assign_pseudo_labels,
    _component_rows,
    _crop_with_padding,
)


def test_morphology_classifier_output_shape() -> None:
    logits = MorphologyClassifier()(torch.zeros(3, 1, 64, 64))
    assert logits.shape == (3, 2)


def test_crop_with_padding_preserves_requested_size() -> None:
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch = _crop_with_padding(image, 0, 0, 8)
    assert patch.shape == (8, 8)


def test_crop_with_padding_returns_exact_interior_pixels() -> None:
    image = np.arange(400).reshape(20, 20).astype(np.uint8)
    patch = _crop_with_padding(image, 10, 10, 8)
    np.testing.assert_array_equal(patch, image[6:14, 6:14])


def test_large_cf_component_uses_noncell_solidity_sentinel(tmp_path) -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:56, 8:56] = 255
    path = tmp_path / "large-cf.png"
    Image.fromarray(mask).save(path)

    rows = _component_rows(path, "A1", "T0", {})

    assert len(rows) == 1
    assert rows[0]["area_px"] > 1200
    assert rows[0]["solidity"] == 0.0


def test_pseudo_label_rules_keep_wall_position_as_auxiliary_only() -> None:
    rows = pd.DataFrame(
        [
            {
                "area_px": 80,
                "circularity": 0.9,
                "eccentricity": 0.3,
                "solidity": 0.95,
                "extent": 0.8,
                "radial_fraction": 0.44,
                "wall_neighbor_count": 0,
                "temporal_support": 1,
                "background_anisotropy": 0.1,
            },
            {
                "area_px": 80,
                "circularity": 0.2,
                "eccentricity": 0.99,
                "solidity": 0.5,
                "extent": 0.2,
                "radial_fraction": 0.48,
                "wall_neighbor_count": 3,
                "temporal_support": 1,
                "background_anisotropy": 0.9,
            },
        ]
    )
    labelled = _assign_pseudo_labels(rows)
    assert labelled.loc[0, "pseudo_label"] == "cell"
    assert labelled.loc[1, "pseudo_label"] == "debris_artifact"


def test_outer_wall_cell_like_object_is_deferred_not_forced_negative() -> None:
    rows = pd.DataFrame(
        [
            {
                "area_px": 80,
                "circularity": 0.9,
                "eccentricity": 0.3,
                "solidity": 0.95,
                "extent": 0.8,
                "radial_fraction": 0.48,
                "wall_neighbor_count": 0,
                "temporal_support": 1,
                "background_anisotropy": 0.1,
            }
        ]
    )
    labelled = _assign_pseudo_labels(rows)
    assert labelled.loc[0, "pseudo_label"] == "uncertain"

from __future__ import annotations

import torch

from cellvision.models.temporal_pairwise import TemporalPairwiseNet


def test_pairwise_model_returns_identity_and_static_logits():
    model = TemporalPairwiseNet(numeric_features=5, embedding_dim=16)
    images = torch.zeros(3, 2, 2, 48, 48)
    numeric = torch.zeros(3, 5)
    result = model(images, numeric)

    assert result["same_object"].shape == (3,)
    assert result["static_similarity"].shape == (3,)


def test_pairwise_representation_is_symmetric_for_swapped_frames():
    torch.manual_seed(7)
    model = TemporalPairwiseNet(numeric_features=5, embedding_dim=16).eval()
    images = torch.randn(2, 2, 2, 48, 48)
    numeric = torch.randn(2, 5)
    forward = model(images, numeric)
    swapped = model(images[:, [1, 0]], numeric)

    torch.testing.assert_close(forward["same_object"], swapped["same_object"])
    torch.testing.assert_close(
        forward["static_similarity"], swapped["static_similarity"]
    )


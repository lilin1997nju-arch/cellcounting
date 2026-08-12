from __future__ import annotations

import torch
from torch import nn


class TemporalPairwiseNet(nn.Module):
    """Small symmetric pair encoder for temporal identity/static evidence.

    The pair representation is intentionally symmetric: absolute embedding
    difference, elementwise product, and embedding mean are used instead of
    concatenating an ordered left/right identity vector.  The numeric branch
    contains only geometry/area features; cell/debris probabilities are not
    accepted as identity inputs.
    """

    def __init__(self, numeric_features: int = 5, embedding_dim: int = 64):
        super().__init__()
        self.numeric_features = int(numeric_features)
        self.embedding_dim = int(embedding_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 24, 5, stride=2, padding=2),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 72, 3, stride=2, padding=1),
            nn.GroupNorm(8, 72),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(72, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3 + self.numeric_features, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(160, 64),
            nn.SiLU(),
        )
        self.same_object_head = nn.Linear(64, 1)
        self.static_similarity_head = nn.Linear(64, 1)

    def forward(
        self,
        images: torch.Tensor,
        numeric: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 5 or images.shape[1] != 2 or images.shape[2] != 2:
            raise ValueError(
                "TemporalPairwiseNet expects images shaped [batch, 2 frames, 2 channels, height, width]"
            )
        left = self.encoder(images[:, 0])
        right = self.encoder(images[:, 1])
        symmetric = torch.cat(
            [torch.abs(left - right), left * right, 0.5 * (left + right), numeric],
            dim=1,
        )
        fused = self.fusion(symmetric)
        return {
            "same_object": self.same_object_head(fused).squeeze(1),
            "static_similarity": self.static_similarity_head(fused).squeeze(1),
        }


from __future__ import annotations

import torch
from torch import nn


class TemporalEvidenceNet(nn.Module):
    """Learned triplet correspondence and static-similarity model."""

    def __init__(
        self,
        numeric_features: int = 12,
        embedding_dim: int = 96,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.SiLU(),
            nn.Conv2d(48, 72, 3, stride=2, padding=1), nn.BatchNorm2d(72), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(72, embedding_dim), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3 + numeric_features + 3, 192),
            nn.SiLU(), nn.Dropout(0.2), nn.Linear(192, 96), nn.SiLU(),
        )
        self.same_object_head = nn.Linear(96, 1)
        self.static_similarity_head = nn.Linear(96, 1)

    def forward(self, images: torch.Tensor, numeric: torch.Tensor, present: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = [self.encoder(images[:, index]) for index in range(3)]
        fused = self.fusion(torch.cat(embeddings + [numeric, present], dim=1))
        return {
            "same_object": self.same_object_head(fused).squeeze(1),
            "static_similarity": self.static_similarity_head(fused).squeeze(1),
        }

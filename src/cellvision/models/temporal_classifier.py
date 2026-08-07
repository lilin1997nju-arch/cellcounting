from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class TemporalMultiTaskClassifier(nn.Module):
    """T0/T1/T2 shared encoder scaffold for the human-labelled round."""

    def __init__(self, numeric_features: int = 16, embedding_dim: int = 128):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.fusion = nn.Sequential(
            nn.Linear(512 * 3 + numeric_features + 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.object_type_head = nn.Linear(512, 3)
        self.viability_head = nn.Linear(512, 3)
        self.division_head = nn.Linear(512, 2)
        self.embedding_head = nn.Linear(512, embedding_dim)

    def forward(
        self,
        timepoint_images: torch.Tensor,
        numeric_features: torch.Tensor,
        timepoint_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embeddings = [self.encoder(timepoint_images[:, index]) for index in range(3)]
        fused = self.fusion(torch.cat(embeddings + [numeric_features, timepoint_mask], dim=1))
        return {
            "object_type": self.object_type_head(fused),
            "viability": self.viability_head(fused),
            "division": self.division_head(fused),
            "embedding": nn.functional.normalize(self.embedding_head(fused), dim=1),
        }


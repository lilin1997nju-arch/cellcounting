from __future__ import annotations

import torch
import torch.nn.functional as functional


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    numerator = 2 * (probabilities * targets).sum(dim=(1, 2, 3)) + epsilon
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + epsilon
    return 1 - (numerator / denominator).mean()


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    losses = functional.cross_entropy(logits, targets, reduction="none")
    valid = valid_mask.to(losses.dtype)
    return (losses * valid).sum() / valid.sum().clamp_min(1)


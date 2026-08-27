"""Strictly offline loading for third-party pretrained feature extractors."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from torchvision.models import resnet18

from .config import PROJECT_ROOT, artifact_path


RESNET18_IMAGENET_FILENAME = "resnet18-f37072fd.pth"
RESNET18_IMAGENET_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_resnet18_imagenet_checkpoint(config: dict[str, Any]) -> Path:
    candidates = [artifact_path(config, "models", RESNET18_IMAGENET_FILENAME)]
    model_root = str(config.get("paths", {}).get("model_root", "")).strip()
    if model_root:
        candidates.append(
            Path(os.path.expandvars(model_root)).expanduser()
            / "models"
            / RESNET18_IMAGENET_FILENAME
        )
    shared_root = str(
        config.get("late_growth", {}).get("shared_model_root", "")
    ).strip()
    if shared_root:
        candidates.append(
            Path(os.path.expandvars(shared_root)).expanduser()
            / "models"
            / RESNET18_IMAGENET_FILENAME
        )
    candidates.append(
        PROJECT_ROOT / "artifacts" / "models" / RESNET18_IMAGENET_FILENAME
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Offline ResNet18 ImageNet weight is missing. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def load_resnet18_imagenet_extractor(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    expected_sha256: str = RESNET18_IMAGENET_SHA256,
) -> torch.nn.Module:
    """Load the torchvision ResNet18 backbone without any network fallback."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Offline ResNet18 ImageNet weight is missing: {path}"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256.casefold() != expected_sha256.casefold():
        raise RuntimeError(
            f"Offline ResNet18 ImageNet weight checksum mismatch: {path}; "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    extractor = resnet18(weights=None)
    extractor.load_state_dict(state, strict=True)
    extractor.fc = torch.nn.Identity()
    return extractor.eval().to(device)

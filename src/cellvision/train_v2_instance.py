from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import artifact_path
from .models.losses import dice_loss
from .models.v2_instance_segmenter import SeededInstanceUNet
from .v2_instance_dataset import V2InstanceDataset, build_v2_instance_cache


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("runtime", {}).get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_v2_instance_segmenter(config: dict[str, Any]) -> Path:
    settings = config["v2_instance_segmentation"]
    seed = int(settings.get("seed", 20260802))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cache_path = build_v2_instance_cache(config)
    dataset = V2InstanceDataset(cache_path, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=int(settings.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(config.get("runtime", {}).get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    device = _device(config)
    model = SeededInstanceUNet(int(settings.get("base_channels", 16))).to(device)
    initial_checkpoint = settings.get("initial_checkpoint")
    if initial_checkpoint:
        checkpoint_path = Path(str(initial_checkpoint))
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if checkpoint_path.exists():
            initial = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(initial["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("learning_rate", 1e-3)),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )
    use_amp = device.type == "cuda" and bool(config.get("runtime", {}).get("mixed_precision", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []
    epochs = int(settings.get("epochs", 18))
    model.train()
    for epoch in range(epochs):
        totals = [0.0, 0.0, 0.0]
        seen = 0
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(inputs)
                instance_loss = F.binary_cross_entropy_with_logits(logits[:, :1], targets[:, :1])
                instance_loss = instance_loss + dice_loss(logits[:, :1], targets[:, :1])
                wall_loss = F.binary_cross_entropy_with_logits(logits[:, 1:2], targets[:, 1:2])
                wall_loss = wall_loss + dice_loss(logits[:, 1:2], targets[:, 1:2])
                presence_targets = (targets[:, :1].sum(dim=(2, 3), keepdim=True) > 0).float().expand_as(logits[:, 2:3])
                presence_loss = F.binary_cross_entropy_with_logits(logits[:, 2:3], presence_targets)
                loss = (
                    instance_loss
                    + float(settings.get("wall_loss_weight", 0.45)) * wall_loss
                    + float(settings.get("presence_loss_weight", 0.70)) * presence_loss
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch = len(inputs)
            totals[0] += float(loss.detach()) * batch
            totals[1] += float(instance_loss.detach()) * batch
            totals[2] += float(wall_loss.detach()) * batch
            seen += batch
        history.append(
            {"epoch": epoch + 1, "loss": totals[0] / seen, "instance_loss": totals[1] / seen, "wall_loss": totals[2] / seen}
        )

    run_name = datetime.now().strftime("v2-instance-%Y%m%d-%H%M%S")
    run_dir = artifact_path(config, "v2", "runs", run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "base_channels": int(settings.get("base_channels", 16)),
            "patch_size_px": int(settings.get("patch_size_px", 96)),
            "seed_sigma_px": float(settings.get("seed_sigma_px", 4.0)),
            "threshold": float(settings.get("threshold", 0.5)),
            "wall_threshold": float(settings.get("wall_threshold", 0.5)),
            "presence_threshold": float(settings.get("presence_threshold", 0.55)),
            "output_channels": 3,
            "training_sources": settings.get("training_sources", []),
            "A12_22_excluded": True,
            "initial_checkpoint": str(initial_checkpoint or ""),
        },
        checkpoint,
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"device": str(device), "sample_count": len(dataset), "fit_history": history, "external_validation": "A12-22 only; not used for weights"}, indent=2),
        encoding="utf-8",
    )
    latest = artifact_path(config, "v2", "models", "latest_instance_segmenter.pt")
    latest.write_bytes(checkpoint.read_bytes())
    return run_dir

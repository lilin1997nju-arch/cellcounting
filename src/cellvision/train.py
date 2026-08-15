from __future__ import annotations

import csv
import json
import os
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from PIL import Image, ImageDraw

from .config import artifact_path
from .datasets import WeakMaskDataset
from .models.instance_segmenter import TinyUNet
from .models.losses import dice_loss
from .patches import build_weak_patch_cache
from .runtime import ensure_training_allowed


def _metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float]:
    predictions = torch.sigmoid(logits) >= threshold
    truth = targets >= 0.5
    tp = (predictions & truth).sum().item()
    fp = (predictions & ~truth).sum().item()
    fn = (~predictions & truth).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {"precision": precision, "recall": recall, "dice": dice, "iou": iou}


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, threshold: float
) -> dict[str, float]:
    model.eval()
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    losses: list[float] = []
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            losses.append(float(dice_loss(logits, masks).item()))
            predictions = torch.sigmoid(logits) >= threshold
            truth = masks >= 0.5
            totals["tp"] += int((predictions & truth).sum().item())
            totals["fp"] += int((predictions & ~truth).sum().item())
            totals["fn"] += int((~predictions & truth).sum().item())
            totals["tn"] += int((~predictions & ~truth).sum().item())
    tp, fp, fn, tn = totals["tp"], totals["fp"], totals["fn"], totals["tn"]
    return {
        "dice_loss": float(np.mean(losses)),
        "pixel_precision": tp / max(tp + fp, 1),
        "pixel_recall": tp / max(tp + fn, 1),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "true_negative_pixels": tn,
    }


def _write_diagnostic_images(
    run_dir: Path, threshold_rows: list[dict[str, float]], calibrated: dict[str, float]
) -> None:
    chart = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(chart)
    draw.text((24, 15), "Validation threshold sweep (CF weak-mask agreement)", fill="black")
    left, top, right, bottom = 70, 55, 760, 440
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    for key, color in (("dice", "blue"), ("pixel_precision", "green"), ("pixel_recall", "red")):
        points = []
        for row in threshold_rows:
            x = left + (row["threshold"] - 0.30) / 0.65 * (right - left)
            y = bottom - row[key] * (bottom - top)
            points.append((x, y))
        draw.line(points, fill=color, width=3)
    draw.text((580, 15), "Dice", fill="blue")
    draw.text((640, 15), "Precision", fill="green")
    draw.text((715, 15), "Recall", fill="red")
    chart.save(run_dir / "calibration.png")

    matrix = Image.new("RGB", (640, 480), "white")
    matrix_draw = ImageDraw.Draw(matrix)
    matrix_draw.text((24, 18), "Pixel confusion matrix at calibrated threshold", fill="black")
    values = [
        [int(calibrated["true_negative_pixels"]), int(calibrated["false_positive_pixels"])],
        [int(calibrated["false_negative_pixels"]), int(calibrated["true_positive_pixels"])],
    ]
    labels = [["TN", "FP"], ["FN", "TP"]]
    maximum = max(max(row) for row in values) or 1
    for row in range(2):
        for column in range(2):
            x0, y0 = 130 + column * 220, 90 + row * 160
            intensity = int(245 - 170 * values[row][column] / maximum)
            matrix_draw.rectangle(
                (x0, y0, x0 + 190, y0 + 130),
                fill=(intensity, intensity, 255),
                outline="black",
                width=2,
            )
            matrix_draw.text(
                (x0 + 18, y0 + 45),
                f"{labels[row][column]}: {values[row][column]:,}",
                fill="black",
            )
    matrix.save(run_dir / "confusion_matrix.png")


def train_weak_segmenter(config: dict[str, Any]) -> Path:
    ensure_training_allowed()
    model_config = config["weak_segmenter"]
    seed = int(model_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    train_cache = build_weak_patch_cache(
        config,
        "train",
        int(model_config["patch_size_px"]),
        int(model_config["patches_per_image"]),
        int(model_config["max_train_wells"]),
        seed,
    )
    validation_cache = build_weak_patch_cache(
        config,
        "validation",
        int(model_config["patch_size_px"]),
        int(model_config["patches_per_image"]),
        int(model_config["max_validation_wells"]),
        seed,
    )
    train_data = WeakMaskDataset(train_cache, augment=True, seed=seed)
    validation_data = WeakMaskDataset(validation_cache, augment=False, seed=seed)
    train_loader = DataLoader(
        train_data,
        batch_size=int(model_config["batch_size"]),
        shuffle=True,
        num_workers=int(model_config["num_workers"]),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=int(model_config["batch_size"]),
        shuffle=False,
        num_workers=int(model_config["num_workers"]),
    )

    requested = config["runtime"]["device"]
    use_cuda = torch.cuda.is_available() and requested in ("auto", "cuda")
    device = torch.device("cuda" if use_cuda else "cpu")
    model = TinyUNet().to(device)
    positive_fraction = float(train_data.masks.mean())
    positive_weight = min(20.0, max(1.0, (1 - positive_fraction) / max(positive_fraction, 1e-6)))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]))

    run_id = datetime.now().strftime("weak-seg-%Y%m%d-%H%M%S")
    run_dir = Path(config["paths"]["artifact_root"]) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "training_label_source": "instrument_cf_weak_mask",
        "human_ground_truth": False,
        "train_patch_count": len(train_data),
        "validation_patch_count": len(validation_data),
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    split_source = artifact_path(config, "splits", "split_manifest.json")
    (run_dir / "split_manifest.json").write_text(split_source.read_text(encoding="utf-8"), encoding="utf-8")

    logs: list[dict[str, float]] = []
    best_dice = -1.0
    threshold = float(model_config["threshold"])
    for epoch in range(1, int(model_config["epochs"]) + 1):
        model.train()
        train_losses: list[float] = []
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        validation = _evaluate(model, validation_loader, device, threshold)
        log = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **validation}
        logs.append(log)
        checkpoint = {
            "model_state": model.state_dict(),
            "model_name": "TinyUNet",
            "patch_size_px": int(model_config["patch_size_px"]),
            "threshold": threshold,
            "label_source": "instrument_cf_weak_mask",
        }
        torch.save(checkpoint, run_dir / "last_model.pt")
        if validation["dice"] > best_dice:
            best_dice = validation["dice"]
            torch.save(checkpoint, run_dir / "best_model.pt")

    with (run_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)
    best_checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state"])
    threshold_rows: list[dict[str, float]] = []
    for candidate_threshold in np.arange(0.30, 0.96, 0.05):
        metrics = _evaluate(model, validation_loader, device, float(candidate_threshold))
        threshold_rows.append({"threshold": float(candidate_threshold), **metrics})
    calibrated = max(threshold_rows, key=lambda item: item["dice"])
    best_checkpoint["threshold"] = calibrated["threshold"]
    torch.save(best_checkpoint, run_dir / "best_model.pt")
    with (run_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=threshold_rows[0].keys())
        writer.writeheader()
        writer.writerows(threshold_rows)
    _write_diagnostic_images(run_dir, threshold_rows, calibrated)
    final_metrics = {
        **calibrated,
        "last_epoch_train_loss": logs[-1]["train_loss"],
        "best_epoch_dice_at_initial_threshold": best_dice,
        "calibrated_validation_dice": calibrated["dice"],
        "calibrated_threshold": calibrated["threshold"],
        "metric_scope": "pixel agreement with instrument CF weak masks",
        "not_valid_for": ["live/dead/debris classification", "human-ground-truth segmentation claims"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (run_dir / "error_cases.csv").write_text("source,error_type,notes\n", encoding="utf-8")
    (run_dir / "report.md").write_text(
        "# First weak-segmentation baseline\n\n"
        "This run learns the instrument `-cf` masks. Metrics are weak-label agreement, not human-ground-truth "
        "cell segmentation or viability classification metrics.\n\n"
        f"- Device: `{device}`\n- Train patches: {len(train_data)}\n"
        f"- Validation patches: {len(validation_data)}\n"
        f"- Calibrated threshold: {calibrated['threshold']:.2f}\n"
        f"- Calibrated validation Dice: {calibrated['dice']:.4f}\n",
        encoding="utf-8",
    )
    return run_dir

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
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .models.morphology_classifier import MorphologyClassifier
from .pseudo_labels import build_morphology_pseudo_labels


CLASS_NAMES = ["debris_artifact", "cell"]


class PseudoMorphologyDataset(Dataset):
    def __init__(self, cache_path: str | Path, augment: bool, seed: int) -> None:
        cache = np.load(cache_path)
        self.images = cache["images"]
        self.labels = cache["labels"].astype(np.int64)
        self.candidate_ids = cache["candidate_ids"]
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image = self.images[index]
        if self.augment:
            generator = np.random.default_rng(self.seed + index + random.randrange(10_000_000))
            if generator.random() < 0.5:
                image = np.fliplr(image)
            if generator.random() < 0.5:
                image = np.flipud(image)
            image = np.rot90(image, int(generator.integers(0, 4)))
        image = np.ascontiguousarray(image, dtype=np.float32)
        median = float(np.median(image))
        scale = max(float(np.std(image)), 8.0)
        image = np.clip((image - median) / scale, -3.0, 3.0) / 3.0
        return (
            torch.from_numpy(image[None]),
            torch.tensor(self.labels[index], dtype=torch.long),
            str(self.candidate_ids[index]),
        )


def _training_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrix = np.zeros((2, 2), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for images, labels, candidate_ids in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predicted = probabilities.argmax(axis=1)
            truth = labels.numpy()
            for true_index, predicted_index in zip(truth, predicted, strict=True):
                matrix[int(true_index), int(predicted_index)] += 1
            for candidate_id, true_index, predicted_index, scores in zip(
                candidate_ids, truth, predicted, probabilities, strict=True
            ):
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "pseudo_label": CLASS_NAMES[int(true_index)],
                        "predicted_label": CLASS_NAMES[int(predicted_index)],
                        "cell_probability": float(scores[1]),
                        "confidence": float(scores[int(predicted_index)]),
                        "correct_on_pseudo_label": bool(true_index == predicted_index),
                    }
                )
    accuracy = float(np.trace(matrix) / max(matrix.sum(), 1))
    return pd.DataFrame(rows), {
        "training_accuracy_on_pseudo_labels": accuracy,
        "training_confusion_matrix": matrix.tolist(),
    }


def train_morphology_classifier(config: dict[str, Any]) -> Path:
    settings = config.get("morphology_classifier", {})
    seed = int(settings.get("seed", 20260729))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    cache_path = build_morphology_pseudo_labels(config)
    dataset = PseudoMorphologyDataset(cache_path, augment=True, seed=seed)
    evaluation_dataset = PseudoMorphologyDataset(cache_path, augment=False, seed=seed)
    batch_size = int(settings.get("batch_size", 256))
    num_workers = int(settings.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    requested = config.get("runtime", {}).get("device", "auto")
    use_cuda = torch.cuda.is_available() and requested in ("auto", "cuda")
    device = torch.device("cuda" if use_cuda else "cpu")
    model = MorphologyClassifier().to(device)
    counts = np.bincount(dataset.labels, minlength=2)
    class_weights = counts.sum() / np.maximum(counts, 1)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("learning_rate", 0.001)),
        weight_decay=float(settings.get("weight_decay", 0.0001)),
    )
    epochs = int(settings.get("epochs", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = bool(config.get("runtime", {}).get("mixed_precision", True) and use_cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_id = datetime.now().strftime("morphology-%Y%m%d-%H%M%S")
    run_dir = Path(config["paths"]["artifact_root"]) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if use_cuda else None,
        "cuda_available": torch.cuda.is_available(),
        "mixed_precision": use_amp,
        "training_label_source": "high_confidence_temporal_morphology_pseudo_labels",
        "human_ground_truth": False,
        "internal_validation_split": False,
        "external_validation_required": True,
        "training_patch_count": len(dataset),
        "class_counts": {
            CLASS_NAMES[index]: int(value) for index, value in enumerate(counts)
        },
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logs: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(labels)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_items += len(labels)
        scheduler.step()
        epoch_log = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_items, 1),
            "train_accuracy": total_correct / max(total_items, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        logs.append(epoch_log)
        print(
            f"epoch {epoch:02d}/{epochs}: loss={epoch_log['train_loss']:.4f}, "
            f"pseudo-label accuracy={epoch_log['train_accuracy']:.4f}",
            flush=True,
        )

    checkpoint = {
        "model_state": model.state_dict(),
        "model_name": "MorphologyClassifier",
        "class_names": CLASS_NAMES,
        "patch_size_px": int(settings.get("patch_size_px", 64)),
        "label_source": "high_confidence_temporal_morphology_pseudo_labels",
        "internal_validation_split": False,
    }
    torch.save(checkpoint, run_dir / "model.pt")
    with (run_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)

    predictions, training_metrics = _training_predictions(
        model, evaluation_loader, device
    )
    predictions.to_csv(
        run_dir / "training_predictions.csv", index=False, encoding="utf-8"
    )
    uncertain = predictions.nsmallest(
        min(1000, len(predictions)), "confidence"
    )
    uncertain.to_csv(
        run_dir / "uncertain_training_examples.csv", index=False, encoding="utf-8"
    )
    metrics = {
        **training_metrics,
        "final_epoch_loss": float(logs[-1]["train_loss"]),
        "final_epoch_augmented_accuracy": float(logs[-1]["train_accuracy"]),
        "metric_scope": "fit to automatically generated QL11111 pseudo-labels",
        "generalization_metric_available": False,
        "external_queue_validation_required": True,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(
        "# QL11111 morphology classifier\n\n"
        "All eligible QL11111 T0/T1/T2 images were used to create and train on "
        "high-confidence morphology/temporal pseudo-labels. No internal validation "
        "split was made, as requested. T3/T4 are reserved as later growth signals rather "
        "than being treated as single-cell morphology.\n\n"
        f"- Device: `{environment['device_name'] or device}`\n"
        f"- Training patches: {len(dataset):,}\n"
        f"- Debris/artifact patches: {int(counts[0]):,}\n"
        f"- Cell patches: {int(counts[1]):,}\n"
        f"- Final training loss: {logs[-1]['train_loss']:.4f}\n"
        f"- Training accuracy on pseudo-labels: "
        f"{training_metrics['training_accuracy_on_pseudo_labels']:.4f}\n\n"
        "These metrics measure pseudo-label fitting only. Generalization must be "
        "measured on an independently labelled external queue.\n",
        encoding="utf-8",
    )
    return run_dir

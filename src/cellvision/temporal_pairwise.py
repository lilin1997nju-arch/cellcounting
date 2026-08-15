from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from .config import artifact_path
from .models.temporal_pairwise import TemporalPairwiseNet
from .temporal_objects import TemporalObjectDescriptor, TemporalPairEvidence
from .runtime import ensure_training_allowed


PAIR_SPECS = ((0, 1), (1, 2), (0, 2))
PAIR_NUMERIC_FEATURES = 5


def _pair_numeric_from_triplet(
    numeric: np.ndarray,
    left: int,
    right: int,
) -> np.ndarray:
    """Convert the legacy 24-feature triplet row to symmetric pair geometry."""

    offset_by_pair = {(0, 1): 0, (1, 2): 2, (0, 2): 4}
    offset = offset_by_pair[(left, right)]
    delta = np.asarray(numeric[offset : offset + 2], dtype=np.float32)
    delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    if (left, right) == (0, 1):
        area_log = float(numeric[6])
    elif (left, right) == (1, 2):
        area_log = float(numeric[7])
    else:
        # The cache stores adjacent area changes, so the T0-T2 log ratio is
        # their sum.  A small absolute-area feature is not needed by the
        # identity head and would introduce plate-scale dependence.
        area_log = float(numeric[6]) + float(numeric[7])
    area_log = float(np.nan_to_num(area_log, nan=0.0, posinf=0.0, neginf=0.0))
    distance_norm = float(np.clip(2.0 * np.linalg.norm(delta), 0.0, 4.0))
    area_similarity = float(np.exp(-abs(area_log) / 0.85))
    return np.asarray(
        [
            abs(float(delta[0])),
            abs(float(delta[1])),
            abs(area_log),
            distance_norm,
            area_similarity,
        ],
        dtype=np.float32,
    )


def _descriptor_pair_numeric(
    left: TemporalObjectDescriptor,
    right: TemporalObjectDescriptor,
) -> np.ndarray:
    dx = abs(float(right.aligned_x - left.aligned_x)) / 256.0
    dy = abs(float(right.aligned_y - left.aligned_y)) / 256.0
    area_log = float(np.log((right.area + 1.0) / (left.area + 1.0)))
    distance_norm = float(
        np.clip(
            np.hypot(right.aligned_x - left.aligned_x, right.aligned_y - left.aligned_y)
            / 128.0,
            0.0,
            4.0,
        )
    )
    area_similarity = float(np.exp(-abs(area_log) / 0.85))
    return np.asarray(
        [dx, dy, abs(area_log), distance_norm, area_similarity],
        dtype=np.float32,
    )


def build_temporal_pairwise_training_cache(config: dict[str, Any]) -> Path:
    """Expand reviewed triplets into adjacent/direct pair samples.

    This reuses the existing reviewed temporal cache, so no additional image
    decoding is needed.  Pair labels are only emitted when both frames are
    present; missing proposals remain represented by the existing matcher
    dummy/unmatched path instead of becoming a fabricated negative pair.
    """

    settings = config.get("v3_temporal_behavior", {}).get("pairwise_training", {})
    patch_size = int(config.get("v2_temporal_model", {}).get("patch_size_px", 64))
    cache_path = artifact_path(
        config,
        "v2",
        "cache",
        f"temporal_pairwise_training_{patch_size}.npz",
    )
    if cache_path.exists() and bool(settings.get("reuse_training_cache", True)):
        with np.load(cache_path) as cached:
            if "groups" in cached.files:
                return cache_path

    # Local import avoids importing the legacy trainer during normal inference.
    from .train_v2_temporal import build_temporal_training_cache

    triplet_path = build_temporal_training_cache(config)
    triplets = np.load(triplet_path)
    images = np.asarray(triplets["images"], dtype=np.float32)
    numeric = np.asarray(triplets["numeric"], dtype=np.float32)
    present = np.asarray(triplets["present"], dtype=np.float32)
    same = np.asarray(triplets["same"], dtype=np.float32)
    static = np.asarray(triplets["static"], dtype=np.float32)
    static_valid = np.asarray(triplets["static_valid"], dtype=np.float32)
    triplet_groups = (
        np.asarray(triplets["groups"]).astype(str)
        if "groups" in triplets.files
        else np.full(len(images), "legacy_cache", dtype=str)
    )

    pair_images: list[np.ndarray] = []
    pair_numeric: list[np.ndarray] = []
    pair_present: list[np.ndarray] = []
    pair_same: list[float] = []
    pair_static: list[float] = []
    pair_static_valid: list[float] = []
    pair_groups: list[str] = []
    for sample_index in range(len(images)):
        for left, right in PAIR_SPECS:
            left_present = bool(present[sample_index, left] > 0.5)
            right_present = bool(present[sample_index, right] > 0.5)
            if not (left_present and right_present):
                continue
            pair_images.append(images[sample_index, [left, right]])
            pair_numeric.append(
                _pair_numeric_from_triplet(numeric[sample_index], left, right)
            )
            pair_present.append(np.asarray([1.0, 1.0], dtype=np.float32))
            pair_same.append(float(same[sample_index] > 0.5))
            pair_static.append(float(static[sample_index]))
            pair_static_valid.append(float(static_valid[sample_index] > 0.5))
            pair_groups.append(str(triplet_groups[sample_index]))
    if not pair_images:
        raise RuntimeError("No complete pair samples are available for V3 temporal training.")
    np.savez_compressed(
        cache_path,
        images=np.asarray(pair_images, dtype=np.float32),
        numeric=np.asarray(pair_numeric, dtype=np.float32),
        present=np.asarray(pair_present, dtype=np.float32),
        same=np.asarray(pair_same, dtype=np.float32),
        static=np.asarray(pair_static, dtype=np.float32),
        static_valid=np.asarray(pair_static_valid, dtype=np.float32),
        groups=np.asarray(pair_groups, dtype=str),
    )
    return cache_path


def _binary_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = logits.sigmoid() >= 0.5
    expected = target >= 0.5
    tp = float((predicted & expected).sum())
    fp = float((predicted & ~expected).sum())
    fn = float((~predicted & expected).sum())
    tn = float((~predicted & ~expected).sum())
    return {
        "accuracy": float((predicted == expected).float().mean()),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "specificity": tn / max(tn + fp, 1.0),
    }


def train_temporal_pairwise_model(config: dict[str, Any]) -> Path:
    """Train the optional low-cost V3 pairwise checkpoint."""

    ensure_training_allowed()

    settings = config.get("v3_temporal_behavior", {}).get("pairwise_training", {})
    cache = np.load(build_temporal_pairwise_training_cache(config))
    dataset = TensorDataset(
        *(torch.from_numpy(cache[key]) for key in ("images", "numeric", "present", "same", "static", "static_valid"))
    )
    if len(dataset) < 4:
        raise RuntimeError("At least four pair samples are required for a train/validation split.")
    seed = int(settings.get("seed", 20260809))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(dataset))
    validation_fraction = float(settings.get("validation_fraction", 0.20))
    groups = np.asarray(cache["groups"]).astype(str)
    unique_groups = np.asarray(sorted(set(groups.tolist())), dtype=str)
    if len(unique_groups) >= 2:
        group_order = rng.permutation(len(unique_groups))
        validation_group_count = max(
            1,
            min(
                len(unique_groups) - 1,
                int(round(len(unique_groups) * validation_fraction)),
            ),
        )
        validation_groups = set(unique_groups[group_order[:validation_group_count]].tolist())
        validation_indices = np.flatnonzero(np.isin(groups, list(validation_groups))).tolist()
        training_indices = np.flatnonzero(~np.isin(groups, list(validation_groups))).tolist()
    else:
        # A one-plate cache cannot support a true plate-level holdout.  Keep a
        # deterministic sample split and report the limitation in metrics.
        validation_count = max(1, min(len(dataset) - 1, int(round(len(dataset) * validation_fraction))))
        validation_indices = permutation[:validation_count].tolist()
        training_indices = permutation[validation_count:].tolist()
    train_loader = DataLoader(
        Subset(dataset, training_indices),
        batch_size=int(settings.get("batch_size", 32)),
        shuffle=True,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=int(settings.get("batch_size", 32)),
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalPairwiseNet(
        numeric_features=PAIR_NUMERIC_FEATURES,
        embedding_dim=int(settings.get("embedding_dim", 64)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("learning_rate", 8e-4)),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )
    same_values = cache["same"]
    same_counts = np.bincount((same_values > 0.5).astype(np.int64), minlength=2)
    same_weights = np.ones(2, dtype=np.float32)
    present_classes = same_counts > 0
    same_weights[present_classes] = len(same_values) / (2 * same_counts[present_classes])
    same_weights_tensor = torch.from_numpy(same_weights).to(device)
    static_values = cache["static"]
    static_valid_values = cache["static_valid"] > 0.5
    static_counts = np.bincount(static_values[static_valid_values].astype(np.int64), minlength=2)
    static_weights = np.ones(2, dtype=np.float32)
    static_classes = static_counts > 0
    static_weights[static_classes] = max(int(static_valid_values.sum()), 1) / (2 * static_counts[static_classes])
    static_weights_tensor = torch.from_numpy(static_weights).to(device)

    history: list[dict[str, float]] = []
    epochs = int(settings.get("epochs", 40))
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, numeric, _present, same_target, static_target, static_valid_target in train_loader:
            images = images.float().to(device)
            numeric = numeric.float().to(device)
            same_target = same_target.float().to(device)
            static_target = static_target.float().to(device)
            static_valid_target = static_valid_target.float().to(device)
            result = model(images, numeric)
            same_raw = F.binary_cross_entropy_with_logits(
                result["same_object"], same_target, reduction="none"
            )
            same_loss = (same_raw * same_weights_tensor[same_target.long()]).mean()
            static_raw = F.binary_cross_entropy_with_logits(
                result["static_similarity"], static_target, reduction="none"
            )
            static_loss = (
                static_raw
                * static_weights_tensor[static_target.long()]
                * static_valid_target
            ).sum() / static_valid_target.sum().clamp_min(1.0)
            loss = same_loss + static_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)

        model.eval()
        same_logits: list[torch.Tensor] = []
        same_targets: list[torch.Tensor] = []
        static_logits: list[torch.Tensor] = []
        static_targets: list[torch.Tensor] = []
        with torch.no_grad():
            for images, numeric, _present, same_target, static_target, static_valid_target in validation_loader:
                result = model(images.float().to(device), numeric.float().to(device))
                same_logits.append(result["same_object"].cpu())
                same_targets.append(same_target.float())
                valid_mask = static_valid_target > 0.5
                if bool(valid_mask.any()):
                    static_logits.append(result["static_similarity"].cpu()[valid_mask])
                    static_targets.append(static_target.float()[valid_mask])
        validation_same_metrics = _binary_metrics(
            torch.cat(same_logits), torch.cat(same_targets)
        )
        validation_static_metrics = (
            _binary_metrics(torch.cat(static_logits), torch.cat(static_targets))
            if static_logits
            else {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0}
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": total_loss / max(len(training_indices), 1),
                "val_same_accuracy": validation_same_metrics["accuracy"],
                "val_same_precision": validation_same_metrics["precision"],
                "val_same_recall": validation_same_metrics["recall"],
                "val_static_accuracy": validation_static_metrics["accuracy"],
                "val_static_precision": validation_static_metrics["precision"],
                "val_static_recall": validation_static_metrics["recall"],
            }
        )

    run = artifact_path(
        config,
        "v2",
        "runs",
        datetime.now().strftime("v3-temporal-pairwise-%Y%m%d-%H%M%S"),
    )
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "model.pt"
    payload = {
        "algorithm_version": "v3-temporal-pairwise-v1",
        "model_state": model.state_dict(),
        "numeric_features": PAIR_NUMERIC_FEATURES,
        "embedding_dim": int(settings.get("embedding_dim", 64)),
        "same_object_threshold": float(settings.get("same_object_threshold", 0.58)),
        "static_similarity_threshold": float(settings.get("static_similarity_threshold", 0.75)),
        "training_cache": str(build_temporal_pairwise_training_cache(config)),
    }
    torch.save(payload, checkpoint)
    metrics = {
        "device": str(device),
        "sample_count": len(dataset),
        "training_count": len(training_indices),
        "validation_count": len(validation_indices),
        "group_count": int(len(unique_groups)),
        "group_split": "plate_or_source_group" if len(unique_groups) >= 2 else "sample_fallback_single_group",
        "same_object_positive": int((cache["same"] > 0.5).sum()),
        "same_object_negative": int((cache["same"] <= 0.5).sum()),
        "static_valid": int((cache["static_valid"] > 0.5).sum()),
        "policy": "symmetric pairwise identity/static evidence; cell/debris probabilities are excluded from identity features",
        "fit_history": history,
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    latest = artifact_path(config, "v2", "models", "latest_temporal_pairwise.pt")
    latest.write_bytes(checkpoint.read_bytes())
    return run


class TemporalPairwiseScorer:
    def __init__(self, model: TemporalPairwiseNet, device: torch.device):
        self.model = model.eval()
        self.device = device

    def __call__(
        self,
        left: TemporalObjectDescriptor,
        right: TemporalObjectDescriptor,
        base: TemporalPairEvidence,
    ) -> TemporalPairEvidence:
        images = np.asarray(
            [
                np.stack([left.raw, left.soft_mask]),
                np.stack([right.raw, right.soft_mask]),
            ],
            dtype=np.float32,
        )
        numeric = _descriptor_pair_numeric(left, right)
        with torch.no_grad():
            result = self.model(
                torch.from_numpy(images).unsqueeze(0).to(self.device),
                torch.from_numpy(numeric).unsqueeze(0).to(self.device),
            )
        identity = float(result["same_object"].sigmoid().item())
        static = float(result["static_similarity"].sigmoid().item())
        return replace(base, identity=identity, static=static)


def load_temporal_pairwise_scorer(
    checkpoint_path: str | Path,
    *,
    device: torch.device | None = None,
) -> TemporalPairwiseScorer:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    version = str(checkpoint.get("algorithm_version", ""))
    if not version.startswith("v3-temporal-pairwise"):
        raise ValueError(f"Unsupported temporal pairwise checkpoint: {version or 'missing version'}")
    model = TemporalPairwiseNet(
        numeric_features=int(checkpoint.get("numeric_features", PAIR_NUMERIC_FEATURES)),
        embedding_dim=int(checkpoint.get("embedding_dim", 64)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return TemporalPairwiseScorer(model, device)

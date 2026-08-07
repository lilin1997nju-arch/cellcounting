from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import artifact_path
from .multiplicity import MULTIPLICITY_CLASSES
from .teaching import MORPHOLOGY_CLASSES, ensure_teaching_features


def _weighted_knn(
    query: np.ndarray,
    reference: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    *,
    neighbours: int,
    scale: float,
) -> np.ndarray:
    output = np.zeros((len(query), class_count), dtype=np.float32)
    if not len(reference):
        return output
    k = min(neighbours, len(reference))
    for start in range(0, len(query), 2048):
        batch = query[start : start + 2048]
        similarities = batch @ reference.T
        indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        selected = np.take_along_axis(similarities, indices, axis=1)
        weights = np.exp(np.clip((selected - 0.45) * scale, -8, 8))
        local = np.zeros((len(batch), class_count), dtype=np.float32)
        for class_index in range(class_count):
            local[:, class_index] = (
                weights * (targets[indices] == class_index)
            ).sum(axis=1)
        output[start : start + len(batch)] = local / np.maximum(
            local.sum(axis=1, keepdims=True), 1e-8
        )
    return output


def _linear_probabilities(
    model: nn.Module, features: np.ndarray, device: torch.device
) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 2048):
            tensor = torch.from_numpy(features[start : start + 2048]).float().to(device)
            rows.append(torch.softmax(model(tensor), dim=1).cpu().numpy())
    return np.concatenate(rows)


def predict_teaching_checkpoint(
    config: dict[str, Any], checkpoint_path: str | Path
) -> dict[str, Any]:
    metadata, features = ensure_teaching_features(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "training_features" not in checkpoint:
        raise ValueError("Retrain the teaching classifier to export inference exemplars.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.LayerNorm(int(checkpoint["input_dimensions"])),
        nn.Linear(int(checkpoint["input_dimensions"]), len(MORPHOLOGY_CLASSES)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    linear = _linear_probabilities(model, features, device)
    reference = checkpoint["training_features"].cpu().numpy().astype(np.float32)
    targets = checkpoint["target_classes"].cpu().numpy().astype(np.int64)
    global_probabilities = _weighted_knn(
        features, reference, targets, len(MORPHOLOGY_CLASSES),
        neighbours=7, scale=8.0,
    )
    global_blend = float(checkpoint.get("global_knn_blend", 0.30))
    probabilities = (1.0 - global_blend) * linear + global_blend * global_probabilities
    target_timepoints = np.asarray(checkpoint.get("target_timepoints", []), dtype=str)
    timepoint_blend = float(checkpoint.get("timepoint_knn_blend", 0.40))
    if len(target_timepoints) == len(reference):
        for timepoint in ("T0", "T1", "T2"):
            query_indices = np.flatnonzero(metadata["timepoint"].astype(str).eq(timepoint))
            reference_mask = target_timepoints == timepoint
            if len(query_indices) == 0 or int(reference_mask.sum()) < 3:
                continue
            local = _weighted_knn(
                features[query_indices], reference[reference_mask], targets[reference_mask],
                len(MORPHOLOGY_CLASSES), neighbours=7, scale=8.0,
            )
            probabilities[query_indices] = (
                (1.0 - timepoint_blend) * linear[query_indices]
                + timepoint_blend * local
            )
    predictions = metadata.copy()
    for index, name in enumerate(MORPHOLOGY_CLASSES):
        predictions[f"{name}_probability"] = probabilities[:, index]
    predictions["predicted_label"] = [
        MORPHOLOGY_CLASSES[index] for index in probabilities.argmax(axis=1)
    ]
    predictions["confidence"] = probabilities.max(axis=1)
    predictions["uncertainty"] = 1.0 - predictions["confidence"]
    output = artifact_path(config, "predictions", "teaching_classifier_predictions.csv")
    predictions.to_csv(output, index=False, encoding="utf-8")
    return {"prediction_count": int(len(predictions)), "predictions": str(output)}


def predict_multiplicity_checkpoint(
    config: dict[str, Any], checkpoint_path: str | Path
) -> dict[str, Any]:
    metadata, features = ensure_teaching_features(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "training_features" not in checkpoint:
        raise ValueError("Retrain the multiplicity classifier to export inference exemplars.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dimensions = int(checkpoint["input_dimensions"])
    model = nn.Sequential(
        nn.LayerNorm(dimensions), nn.Linear(dimensions, 48), nn.GELU(),
        nn.Dropout(0.12), nn.Linear(48, len(MULTIPLICITY_CLASSES)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    linear = _linear_probabilities(model, features, device)
    reference = checkpoint["training_features"].cpu().numpy().astype(np.float32)
    targets = checkpoint["target_classes"].cpu().numpy().astype(np.int64)
    neighbours = _weighted_knn(
        features, reference, targets, len(MULTIPLICITY_CLASSES),
        neighbours=5, scale=9.0,
    )
    blend = float(checkpoint.get("knn_blend", 0.45))
    probabilities = (1.0 - blend) * linear + blend * neighbours
    predictions = metadata.copy()
    for index, name in enumerate(MULTIPLICITY_CLASSES):
        predictions[f"{name}_probability"] = probabilities[:, index]
    predictions["predicted_multiplicity"] = [
        MULTIPLICITY_CLASSES[index] for index in probabilities.argmax(axis=1)
    ]
    predictions["multiplicity_confidence"] = probabilities.max(axis=1)
    predictions["multiplicity_uncertainty"] = 1.0 - predictions["multiplicity_confidence"]
    output = artifact_path(config, "predictions", "multiplicity_predictions.csv")
    predictions.to_csv(output, index=False, encoding="utf-8")
    return {"prediction_count": int(len(predictions)), "predictions": str(output)}

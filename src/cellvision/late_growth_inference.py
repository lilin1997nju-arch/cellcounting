from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from torchvision.models import ResNet18_Weights, resnet18

from .config import artifact_path
from .dense_candidates import detect_dynamic_wall_inner_fraction
from .decode import read_mask
from .registration import phase_correlation_shift
from .v2_instance_dataset import _crop, _seed_heatmap, _wall_prior
from .models.v2_instance_segmenter import SeededInstanceUNet


CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}
LATE_TIMEPOINTS = ("T3", "T4")


@dataclass(frozen=True)
class GrowthThresholds:
    minimum_added_cells: int = 2
    minimum_ratio: float = 1.5
    minimum_cell_probability: float = 0.78
    minimum_instance_confidence: float = 0.58


@dataclass(frozen=True)
class DenseGrowthMetrics:
    candidate_count: int
    foreground_fraction: float
    maximum_tile_foreground_fraction: float
    maximum_tile_candidate_count: int
    center_x: float
    center_y: float


def dense_growth_decision(
    baseline: DenseGrowthMetrics,
    late: DenseGrowthMetrics,
    *,
    minimum_foreground_fraction: float = 0.008,
    minimum_foreground_delta: float = 0.006,
    minimum_foreground_ratio: float = 1.8,
    minimum_candidate_ratio: float = 1.5,
    minimum_tile_foreground_fraction: float = 0.025,
) -> bool:
    """Detect a cell-rich region without requiring separable instances."""
    foreground_ratio = late.foreground_fraction / max(
        baseline.foreground_fraction, 1e-5
    )
    candidate_ratio = late.candidate_count / max(baseline.candidate_count, 1)
    foreground_growth = (
        late.foreground_fraction >= minimum_foreground_fraction
        and late.foreground_fraction - baseline.foreground_fraction
        >= minimum_foreground_delta
        and foreground_ratio >= minimum_foreground_ratio
    )
    locally_dense = (
        late.maximum_tile_foreground_fraction
        >= minimum_tile_foreground_fraction
        and late.maximum_tile_foreground_fraction
        >= baseline.maximum_tile_foreground_fraction + minimum_foreground_delta
    )
    return bool(foreground_growth and locally_dense and candidate_ratio >= minimum_candidate_ratio)


def late_growth_decision(
    baseline_units: int,
    detected_units: int,
    *,
    final_stage: bool,
    thresholds: GrowthThresholds = GrowthThresholds(),
) -> str:
    """Classify growth without turning a small count fluctuation into growth."""
    baseline = max(1, int(baseline_units))
    obvious = (
        detected_units >= baseline + thresholds.minimum_added_cells
        and detected_units / baseline >= thresholds.minimum_ratio
    )
    if obvious:
        return "obvious_growth"
    if not final_stage:
        return "continue_search"
    if detected_units <= baseline + 1:
        return "no_growth"
    return "uncertain"


def _instrument_candidates(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["x_px", "y_px", "response", "radius_px"])
    with Path(path).open("r", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                rows.append({
                    "x_px": float(row[0]),
                    "y_px": float(row[1]),
                    "response": float(row[2]) if len(row) > 2 else 0.0,
                    "radius_px": float(row[3]) if len(row) > 3 else 4.0,
                })
            except ValueError:
                continue
    return pd.DataFrame(rows)


def _dense_growth_metrics(
    raw: np.ndarray,
    cf_path: str | Path,
    candidates: pd.DataFrame,
    wall_inner_fraction: float,
    tile_size: int,
) -> DenseGrowthMetrics:
    height, width = raw.shape
    yy, xx = np.ogrid[:height, :width]
    radial = np.hypot(xx - width / 2, yy - height / 2) / min(height, width)
    # The strict interior removes the continuous physical wall. A colony that
    # touches the wall is still detected through the part extending inward.
    strict_interior = radial <= max(0.38, wall_inner_fraction - 0.01)
    cf = np.asarray(read_mask(cf_path), dtype=bool) & strict_interior
    points = candidates.copy()
    if not points.empty:
        point_radius = np.hypot(
            points["x_px"].to_numpy(float) - width / 2,
            points["y_px"].to_numpy(float) - height / 2,
        ) / min(height, width)
        points = points[point_radius <= max(0.38, wall_inner_fraction - 0.01)]
    maximum_fraction = 0.0
    maximum_candidates = 0
    best_center = (width / 2, height / 2)
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            y1, x1 = min(height, y0 + tile_size), min(width, x0 + tile_size)
            usable = strict_interior[y0:y1, x0:x1]
            usable_area = int(usable.sum())
            if usable_area < 0.20 * (y1 - y0) * (x1 - x0):
                continue
            fraction = float(cf[y0:y1, x0:x1].sum() / max(usable_area, 1))
            local_candidates = points[
                points["x_px"].between(x0, x1, inclusive="left")
                & points["y_px"].between(y0, y1, inclusive="left")
            ]
            count = int(len(local_candidates))
            # Foreground coverage is primary; candidate density breaks ties.
            if (fraction, count) > (maximum_fraction, maximum_candidates):
                maximum_fraction = fraction
                maximum_candidates = count
                best_center = ((x0 + x1) / 2, (y0 + y1) / 2)
    return DenseGrowthMetrics(
        candidate_count=int(len(points)),
        foreground_fraction=float(cf.sum() / max(strict_interior.sum(), 1)),
        maximum_tile_foreground_fraction=maximum_fraction,
        maximum_tile_candidate_count=maximum_candidates,
        center_x=float(best_center[0]),
        center_y=float(best_center[1]),
    )


def _crop_with_padding(raw: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    half = size // 2
    cx, cy = int(round(x)), int(round(y))
    padded = np.pad(raw, half, mode="reflect")
    return padded[cy : cy + size, cx : cx + size]


def _registration_shift(reference_path: str, moving_path: str) -> tuple[float, float]:
    with Image.open(reference_path) as image:
        reference = np.asarray(image.convert("L").resize((512, 512)), dtype=np.float32)
    with Image.open(moving_path) as image:
        moving = np.asarray(image.convert("L").resize((512, 512)), dtype=np.float32)
        moving_size = image.size
    reference = gaussian_filter(reference, 1.2)
    moving = gaussian_filter(moving, 1.2)
    shift_x, shift_y = phase_correlation_shift(reference, moving)
    # phase_correlation_shift returns the translation applied to moving in
    # order to align it with reference. Coordinates therefore use -shift.
    return (
        -shift_x * moving_size[0] / 512.0,
        -shift_y * moving_size[1] / 512.0,
    )


class _MorphologyPredictor:
    def __init__(self, checkpoint_path: Path, crop_size: int, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.device = device
        self.crop_size = crop_size
        self.classes = list(checkpoint["classes"])
        self.model = torch.nn.Sequential(
            torch.nn.LayerNorm(int(checkpoint["input_dimensions"])),
            torch.nn.Linear(int(checkpoint["input_dimensions"]), len(self.classes)),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.training_features = checkpoint["training_features"].float().to(device)
        self.target_classes = checkpoint["target_classes"].long().to(device)
        self.knn_blend = float(checkpoint.get("global_knn_blend", 0.30))
        extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        extractor.fc = torch.nn.Identity()
        self.extractor = extractor.eval().to(device)
        self.means = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        self.stds = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    def predict(self, raw: np.ndarray, candidates: pd.DataFrame) -> np.ndarray:
        if candidates.empty:
            return np.empty((0, len(self.classes)), dtype=np.float32)
        patches = np.stack([
            _crop_with_padding(raw, row.x_px, row.y_px, self.crop_size)
            for row in candidates.itertuples(index=False)
        ]).astype(np.float32) / 255.0
        outputs: list[np.ndarray] = []
        for start in range(0, len(patches), 128):
            tensor = torch.from_numpy(patches[start : start + 128, None]).to(self.device)
            tensor = torch.nn.functional.interpolate(
                tensor, size=(224, 224), mode="bilinear", align_corners=False
            ).repeat(1, 3, 1, 1)
            tensor = (tensor - self.means) / self.stds
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, enabled=self.device.type == "cuda"
            ):
                features = self.extractor(tensor).float()
                features /= torch.linalg.vector_norm(features, dim=1, keepdim=True).clamp_min(1e-8)
                linear = torch.softmax(self.model(features), dim=1)
                similarities = features @ self.training_features.T
                k = min(7, self.training_features.shape[0])
                values, indices = torch.topk(similarities, k=k, dim=1)
                weights = torch.exp(torch.clamp((values - 0.45) * 8.0, -8, 8))
                neighbour = torch.zeros_like(linear)
                neighbour.scatter_add_(
                    1,
                    self.target_classes[indices],
                    weights,
                )
                neighbour /= neighbour.sum(dim=1, keepdim=True).clamp_min(1e-8)
                probability = (1.0 - self.knn_blend) * linear + self.knn_blend * neighbour
            outputs.append(probability.cpu().numpy())
        return np.concatenate(outputs)


class _InstanceValidator:
    def __init__(self, checkpoint_path: Path, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.device = device
        self.size = int(checkpoint["patch_size_px"])
        self.threshold = float(checkpoint.get("threshold", 0.5))
        self.wall_threshold = float(checkpoint.get("wall_threshold", 0.5))
        self.seed = _seed_heatmap(self.size, float(checkpoint["seed_sigma_px"]))
        self.model = SeededInstanceUNet(int(checkpoint["base_channels"]))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(device).eval()

    def validate(
        self,
        raw: np.ndarray,
        candidates: pd.DataFrame,
        wall_inner_fraction: float,
        minimum_confidence: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if candidates.empty:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32)
        values = []
        for row in candidates.itertuples(index=False):
            crop = _crop(raw, row.x_px, row.y_px, self.size, int(np.median(raw)))
            wall = _wall_prior(raw.shape, row.x_px, row.y_px, self.size, wall_inner_fraction)
            lo, hi = np.percentile(crop, [2, 98])
            normalized = np.clip((crop.astype(np.float32) - lo) / max(float(hi - lo), 1.0), 0, 1)
            values.append(np.stack([normalized, self.seed, wall]))
        valid_values: list[bool] = []
        confidences: list[float] = []
        for start in range(0, len(values), 128):
            tensor = torch.from_numpy(np.asarray(values[start : start + 128], np.float32)).to(self.device)
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, enabled=self.device.type == "cuda"
            ):
                probabilities = torch.sigmoid(self.model(tensor)).float().cpu().numpy()
            for object_probability, wall_probability, presence_probability in probabilities:
                mask = object_probability >= self.threshold
                mask &= ~((wall_probability >= max(self.wall_threshold, 0.82)) & (object_probability < 0.72))
                area = int(mask.sum())
                confidence = float(object_probability[mask].mean()) if area else 0.0
                wall_overlap = float((wall_probability[mask] >= self.wall_threshold).mean()) if area else 1.0
                valid_values.append(
                    3 <= area <= 2200
                    and confidence >= minimum_confidence
                    and not (wall_overlap > 0.90 and confidence < 0.88)
                )
                confidences.append(confidence)
        return np.asarray(valid_values), np.asarray(confidences, dtype=np.float32)


def _model_path(config: dict[str, Any], *parts: str) -> Path:
    local = artifact_path(config, *parts)
    if local.exists():
        return local
    shared_root = Path(config.get("late_growth", {}).get(
        "shared_model_root", Path(__file__).resolve().parents[2] / "artifacts"
    ))
    return shared_root.joinpath(*parts)


def _anchor_rows(config: dict[str, Any]) -> pd.DataFrame:
    for name in ("latest_v2_predictions.csv", "latest_integrated_predictions.csv"):
        path = artifact_path(config, "predictions", name)
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            break
    else:
        raise ValueError("T0-T2 predictions are unavailable.")
    label = frame.get("integrated_label", pd.Series("", index=frame.index)).astype(str)
    confidence = frame.get("integrated_confidence", pd.Series(0.0, index=frame.index)).astype(float)
    if "v2_is_counting_instance" in frame:
        unique = frame["v2_is_counting_instance"].fillna(False).astype(bool)
    else:
        unique = pd.Series(True, index=frame.index)
    threshold = float(config.get("late_growth", {}).get("anchor_confidence", 0.82))
    return frame[
        frame["timepoint"].isin(["T0", "T1", "T2"])
        & label.isin(CELL_LABELS)
        & confidence.ge(threshold)
        & unique
    ].copy()


def _deduplicate(candidates: pd.DataFrame, radius: float = 12.0) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    ordered = candidates.sort_values(["cell_probability", "instance_confidence"], ascending=False)
    kept: list[int] = []
    points: list[tuple[float, float]] = []
    for index, row in ordered.iterrows():
        point = (float(row.x_px), float(row.y_px))
        if points and cKDTree(points).query(point, k=1)[0] <= radius:
            continue
        kept.append(index)
        points.append(point)
    return ordered.loc[kept]


def _infer_late_growth_legacy(
    config: dict[str, Any], selected_wells: set[str] | None = None
) -> Path:
    settings = config.get("late_growth", {})
    thresholds = GrowthThresholds(
        minimum_added_cells=int(settings.get("minimum_added_cells", 2)),
        minimum_ratio=float(settings.get("minimum_growth_ratio", 1.5)),
        minimum_cell_probability=float(settings.get("minimum_cell_probability", 0.78)),
        minimum_instance_confidence=float(settings.get("minimum_instance_confidence", 0.58)),
    )
    stage_radii = [float(value) for value in settings.get("stage_radii_px", [450, 850])]
    stage_candidate_quotas = [
        int(value) for value in settings.get("stage_candidate_quotas", [160, 280, 480])
    ]
    dense_tile_size = int(settings.get("dense_tile_size_px", 384))
    images = pd.read_csv(artifact_path(config, "manifests", "images.csv"))
    anchors = _anchor_rows(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"late growth runtime device={device}; loading morphology model", flush=True)
    morphology = _MorphologyPredictor(
        _model_path(config, "models", "teaching_classifier.pt"),
        int(config.get("teaching", {}).get("crop_size_px", 128)),
        device,
    )
    instance = _InstanceValidator(
        _model_path(config, "v2", "models", "latest_instance_segmenter.pt"),
        device,
    )
    print("late growth models ready", flush=True)
    rows: list[dict[str, Any]] = []
    object_rows: list[pd.DataFrame] = []
    excluded_wells = {
        str(value).upper()
        for value in config.get("review_queue", {}).get("excluded_wells", [])
    }
    wells_to_process = [
        value for value in images["well"].astype(str).unique()
        if value.upper() not in excluded_wells
        and (selected_wells is None or value.upper() in selected_wells)
    ]
    for well in sorted(wells_to_process, key=lambda value: (value[0], int(value[1:]))):
        well_anchors = anchors[anchors["well"].astype(str) == well]
        baseline_units = 0
        for timepoint in ("T0", "T1", "T2"):
            local = well_anchors[well_anchors["timepoint"] == timepoint]
            units = local["integrated_label"].map(
                {"single": 1, "touching_doublet": 2, "cluster_3plus": 3}
            ).fillna(0).sum()
            baseline_units = max(baseline_units, int(units))
        baseline_image = images[
            (images["well"].astype(str) == well)
            & (images["timepoint"].astype(str) == "T2")
            & (images["decode_status"].astype(str) == "ok")
        ]
        baseline_dense = None
        if not baseline_image.empty:
            baseline_row = baseline_image.iloc[0]
            with Image.open(baseline_row.raw_image_path) as opened:
                baseline_raw = np.asarray(opened.convert("L"), dtype=np.uint8)
            baseline_candidates = _instrument_candidates(baseline_row.cells_csv_path)
            baseline_wall_inner = detect_dynamic_wall_inner_fraction(
                baseline_raw, config.get("dense_detection", {})
            )
            baseline_dense = _dense_growth_metrics(
                baseline_raw,
                baseline_row.cf_image_path,
                baseline_candidates,
                baseline_wall_inner,
                dense_tile_size,
            )
        for timepoint in LATE_TIMEPOINTS:
            late_row = images[
                (images["well"].astype(str) == well)
                & (images["timepoint"].astype(str) == timepoint)
                & (images["decode_status"].astype(str) == "ok")
            ]
            if late_row.empty:
                continue
            image_row = late_row.iloc[0]
            with Image.open(image_row.raw_image_path) as opened:
                raw = np.asarray(opened.convert("L"), dtype=np.uint8)
            candidates = _instrument_candidates(image_row.cells_csv_path)
            height, width = raw.shape
            yy = candidates["y_px"].to_numpy(float) if not candidates.empty else np.zeros(0)
            xx = candidates["x_px"].to_numpy(float) if not candidates.empty else np.zeros(0)
            radial = np.hypot(xx - width / 2, yy - height / 2) / min(height, width)
            candidates = candidates[radial <= 0.485].copy()
            wall_inner = detect_dynamic_wall_inner_fraction(raw, config.get("dense_detection", {}))
            late_dense = _dense_growth_metrics(
                raw,
                image_row.cf_image_path,
                candidates,
                wall_inner,
                dense_tile_size,
            )
            mapped_anchors: list[tuple[float, float]] = []
            for source_timepoint, source_group in well_anchors.groupby("timepoint"):
                source_image = images[
                    (images["well"].astype(str) == well)
                    & (images["timepoint"].astype(str) == source_timepoint)
                    & (images["decode_status"].astype(str) == "ok")
                ]
                if source_image.empty:
                    continue
                shift_x, shift_y = _registration_shift(
                    str(source_image.iloc[0].raw_image_path), str(image_row.raw_image_path)
                )
                mapped_anchors.extend([
                    (float(row.x_px) + shift_x, float(row.y_px) + shift_y)
                    for row in source_group.itertuples(index=False)
                ])
            evaluated: dict[int, dict[str, float]] = {}
            dense_positive = bool(
                baseline_dense is not None
                and dense_growth_decision(
                    baseline_dense,
                    late_dense,
                    minimum_foreground_fraction=float(
                        settings.get("dense_minimum_foreground_fraction", 0.008)
                    ),
                    minimum_foreground_delta=float(
                        settings.get("dense_minimum_foreground_delta", 0.006)
                    ),
                    minimum_foreground_ratio=float(
                        settings.get("dense_minimum_foreground_ratio", 1.8)
                    ),
                    minimum_candidate_ratio=float(
                        settings.get("dense_minimum_candidate_ratio", 1.5)
                    ),
                    minimum_tile_foreground_fraction=float(
                        settings.get("dense_minimum_tile_foreground_fraction", 0.025)
                    ),
                )
            )
            if dense_positive:
                rows.append({
                    "well": well,
                    "timepoint": timepoint,
                    "automatic_decision": "obvious_growth",
                    "search_stage": "dense_growth_region",
                    "decision_channel": "dense_region",
                    "baseline_cell_units": baseline_units,
                    "detected_late_cell_units": 0,
                    "anchor_count": len(mapped_anchors),
                    "evaluated_candidate_count": 0,
                    "available_candidate_count": len(candidates),
                    "dense_baseline_candidate_count": baseline_dense.candidate_count,
                    "dense_late_candidate_count": late_dense.candidate_count,
                    "dense_baseline_foreground_fraction": baseline_dense.foreground_fraction,
                    "dense_late_foreground_fraction": late_dense.foreground_fraction,
                    "dense_maximum_tile_foreground_fraction": late_dense.maximum_tile_foreground_fraction,
                    "dense_center_x": late_dense.center_x,
                    "dense_center_y": late_dense.center_y,
                    "device": str(device),
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                })
                print(
                    f"late growth {well} {timepoint}: obvious_growth "
                    f"stage=dense_growth_region foreground={late_dense.foreground_fraction:.4f}",
                    flush=True,
                )
                continue
            final_decision = "no_growth"
            final_stage = "whole_well_no_anchor" if not mapped_anchors else "whole_well"
            stages: list[tuple[str, float | None]] = (
                [
                    ("anchor_neighbourhood", stage_radii[0]),
                    ("expanded_neighbourhood", stage_radii[1]),
                    ("whole_well", None),
                ]
                if mapped_anchors
                else [("whole_well_no_anchor", None)]
            )
            for stage_index, (stage_name, radius) in enumerate(stages):
                if radius is None:
                    selected_indices = candidates.index
                else:
                    points = candidates[["x_px", "y_px"]].to_numpy(float)
                    selected = np.zeros(len(candidates), dtype=bool)
                    for anchor_x, anchor_y in mapped_anchors:
                        selected |= np.hypot(points[:, 0] - anchor_x, points[:, 1] - anchor_y) <= radius
                    selected_indices = candidates.index[selected]
                quota = stage_candidate_quotas[min(stage_index, len(stage_candidate_quotas) - 1)]
                if len(selected_indices) > quota:
                    selected_indices = (
                        candidates.loc[selected_indices]
                        .nlargest(quota, "response")
                        .index
                    )
                new_indices = [index for index in selected_indices if int(index) not in evaluated]
                if new_indices:
                    new = candidates.loc[new_indices].copy()
                    probability = morphology.predict(raw, new)
                    cell_index = morphology.classes.index("cell")
                    new["cell_probability"] = probability[:, cell_index]
                    morphological_cells = new[
                        new["cell_probability"] >= thresholds.minimum_cell_probability
                    ].copy()
                    valid, instance_confidence = instance.validate(
                        raw, morphological_cells, wall_inner, thresholds.minimum_instance_confidence
                    )
                    morphological_cells["instance_confidence"] = instance_confidence
                    valid_lookup = dict(zip(morphological_cells.index, valid))
                    confidence_lookup = dict(zip(morphological_cells.index, instance_confidence))
                    for index, row in new.iterrows():
                        evaluated[int(index)] = {
                            "cell_probability": float(row.cell_probability),
                            "instance_confidence": float(confidence_lookup.get(index, 0.0)),
                            "is_cell": bool(valid_lookup.get(index, False)),
                        }
                visible = candidates.loc[list(evaluated)].copy() if evaluated else candidates.iloc[0:0].copy()
                if not visible.empty:
                    visible["cell_probability"] = [evaluated[int(index)]["cell_probability"] for index in visible.index]
                    visible["instance_confidence"] = [evaluated[int(index)]["instance_confidence"] for index in visible.index]
                    visible["is_cell"] = [evaluated[int(index)]["is_cell"] for index in visible.index]
                    cells = _deduplicate(visible[visible["is_cell"]].copy())
                else:
                    cells = visible
                decision = late_growth_decision(
                    baseline_units,
                    len(cells),
                    final_stage=stage_index == len(stages) - 1,
                    thresholds=thresholds,
                )
                if decision == "obvious_growth" or stage_index == len(stages) - 1:
                    final_decision = decision
                    final_stage = stage_name
                    if not cells.empty:
                        object_rows.append(cells.assign(
                            well=well,
                            timepoint=timepoint,
                            detection_stage=stage_name,
                        ))
                    break
            rows.append({
                "well": well,
                "timepoint": timepoint,
                "automatic_decision": final_decision,
                "search_stage": final_stage,
                "decision_channel": "sparse_instance",
                "baseline_cell_units": baseline_units,
                "detected_late_cell_units": int(len(cells)) if "cells" in locals() else 0,
                "anchor_count": len(mapped_anchors),
                "evaluated_candidate_count": len(evaluated),
                "available_candidate_count": len(candidates),
                "dense_baseline_candidate_count": baseline_dense.candidate_count if baseline_dense else 0,
                "dense_late_candidate_count": late_dense.candidate_count,
                "dense_baseline_foreground_fraction": baseline_dense.foreground_fraction if baseline_dense else 0.0,
                "dense_late_foreground_fraction": late_dense.foreground_fraction,
                "dense_maximum_tile_foreground_fraction": late_dense.maximum_tile_foreground_fraction,
                "dense_center_x": late_dense.center_x,
                "dense_center_y": late_dense.center_y,
                "device": str(device),
                "computed_at": datetime.now(timezone.utc).isoformat(),
            })
            print(
                f"late growth {well} {timepoint}: {final_decision} "
                f"stage={final_stage} cells={rows[-1]['detected_late_cell_units']}",
                flush=True,
            )
    output = artifact_path(config, "predictions", "latest_late_growth_predictions.csv")
    result = pd.DataFrame(rows)
    if selected_wells is not None and output.exists():
        previous = pd.read_csv(output)
        previous = previous[
            ~previous["well"].astype(str).str.upper().isin(selected_wells)
        ]
        result = pd.concat([previous, result], ignore_index=True, sort=False)
        result = result.sort_values(["well", "timepoint"]).reset_index(drop=True)
    result.to_csv(output, index=False, encoding="utf-8")
    objects_output = artifact_path(config, "predictions", "latest_late_growth_objects.csv")
    if object_rows:
        pd.concat(object_rows, ignore_index=True).to_csv(objects_output, index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(objects_output, index=False, encoding="utf-8")
    summary = {
        "algorithm": "staged_sparse_plus_dense_late_growth_v2",
        "device": str(device),
        "well_timepoint_count": len(result),
        "decision_counts": result["automatic_decision"].value_counts().to_dict() if len(result) else {},
        "stage_counts": result["search_stage"].value_counts().to_dict() if len(result) else {},
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _boolean_values(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _representative_instance_center(
    cells: pd.DataFrame, radius: float = 550.0
) -> tuple[float, float, int, int] | None:
    if cells.empty:
        return None
    coordinates = cells[["x_px", "y_px"]].to_numpy(float)
    unit_weights = cells["integrated_label"].map(
        {"single": 1, "touching_doublet": 2, "cluster_3plus": 3}
    ).fillna(1).to_numpy(float)
    best_indices = np.asarray([0], dtype=int)
    best_score = (-1.0, -1)
    for coordinate in coordinates:
        indices = np.flatnonzero(
            np.hypot(
                coordinates[:, 0] - coordinate[0],
                coordinates[:, 1] - coordinate[1],
            )
            <= radius
        )
        score = (float(unit_weights[indices].sum()), int(len(indices)))
        if score > best_score:
            best_score = score
            best_indices = indices
    weights = unit_weights[best_indices]
    center = np.average(coordinates[best_indices], axis=0, weights=weights)
    return (
        float(center[0]),
        float(center[1]),
        int(len(best_indices)),
        int(round(float(weights.sum()))),
    )


def infer_late_growth(
    config: dict[str, Any],
    selected_wells: set[str] | None = None,
    *,
    reuse_existing_integrated: bool = False,
) -> Path:
    """Detect T3/T4 growth with the full T0-T2 instance stack.

    Discrete late cells use the same multiscale proposals, morphology model,
    multiplicity model, and V2 instance masks as T0-T2.  Confluent sheets are
    handled as a region-level event so one colony is not forced into hundreds
    of unreliable single-cell instances.
    """

    from .late_full_inference import infer_late_full_instances

    full_path = infer_late_full_instances(
        config,
        selected_wells,
        reuse_existing_integrated=reuse_existing_integrated,
    )
    full = pd.read_csv(full_path, low_memory=False)
    images = pd.read_csv(artifact_path(config, "manifests", "images.csv"))
    anchors = _anchor_rows(config)
    settings = config.get("late_growth", {})
    thresholds = GrowthThresholds(
        minimum_added_cells=int(settings.get("minimum_added_cells", 2)),
        minimum_ratio=float(settings.get("minimum_growth_ratio", 1.5)),
        minimum_cell_probability=float(
            settings.get("minimum_cell_probability", 0.78)
        ),
        minimum_instance_confidence=float(
            settings.get("minimum_instance_confidence", 0.58)
        ),
    )
    dense_tile_size = int(settings.get("dense_tile_size_px", 384))
    excluded_wells = {
        str(value).upper()
        for value in config.get("review_queue", {}).get("excluded_wells", [])
    }
    wells = [
        str(value)
        for value in images["well"].astype(str).unique()
        if str(value).upper() not in excluded_wells
        and (selected_wells is None or str(value).upper() in selected_wells)
    ]
    rows: list[dict[str, Any]] = []
    object_rows: list[pd.DataFrame] = []
    for well in sorted(wells, key=lambda value: (value[0], int(value[1:]))):
        well_anchors = anchors[anchors["well"].astype(str) == well]
        baseline_units = 0
        for timepoint in ("T0", "T1", "T2"):
            local = well_anchors[well_anchors["timepoint"] == timepoint]
            units = local["integrated_label"].map(
                {"single": 1, "touching_doublet": 2, "cluster_3plus": 3}
            ).fillna(0).sum()
            baseline_units = max(baseline_units, int(units))

        baseline_image = images[
            (images["well"].astype(str) == well)
            & (images["timepoint"].astype(str) == "T2")
            & (images["decode_status"].astype(str) == "ok")
        ]
        baseline_dense = None
        if not baseline_image.empty:
            baseline_row = baseline_image.iloc[0]
            with Image.open(baseline_row.raw_image_path) as opened:
                baseline_raw = np.asarray(opened.convert("L"), dtype=np.uint8)
            baseline_candidates = _instrument_candidates(
                baseline_row.cells_csv_path
            )
            baseline_wall_inner = detect_dynamic_wall_inner_fraction(
                baseline_raw, config.get("dense_detection", {})
            )
            baseline_dense = _dense_growth_metrics(
                baseline_raw,
                baseline_row.cf_image_path,
                baseline_candidates,
                baseline_wall_inner,
                dense_tile_size,
            )

        for timepoint in LATE_TIMEPOINTS:
            image_frame = images[
                (images["well"].astype(str) == well)
                & (images["timepoint"].astype(str) == timepoint)
                & (images["decode_status"].astype(str) == "ok")
            ]
            if image_frame.empty:
                continue
            image_row = image_frame.iloc[0]
            with Image.open(image_row.raw_image_path) as opened:
                raw = np.asarray(opened.convert("L"), dtype=np.uint8)
            instrument = _instrument_candidates(image_row.cells_csv_path)
            height, width = raw.shape
            if not instrument.empty:
                radial = np.hypot(
                    instrument["x_px"].to_numpy(float) - width / 2,
                    instrument["y_px"].to_numpy(float) - height / 2,
                ) / min(height, width)
                instrument = instrument[radial <= 0.485].copy()
            wall_inner = detect_dynamic_wall_inner_fraction(
                raw, config.get("dense_detection", {})
            )
            late_dense = _dense_growth_metrics(
                raw,
                image_row.cf_image_path,
                instrument,
                wall_inner,
                dense_tile_size,
            )

            proposals = full[
                (full["well"].astype(str) == well)
                & (full["timepoint"].astype(str) == timepoint)
            ].copy()
            if not proposals.empty and "v2_is_counting_instance" in proposals:
                counting = _boolean_values(proposals["v2_is_counting_instance"])
            else:
                counting = pd.Series(False, index=proposals.index)
            cells = proposals[
                proposals.get(
                    "integrated_label", pd.Series("", index=proposals.index)
                ).astype(str).isin(CELL_LABELS)
                & counting
            ].copy()
            detected_units = int(
                cells.get(
                    "integrated_label", pd.Series("", index=cells.index)
                ).map(
                    {"single": 1, "touching_doublet": 2, "cluster_3plus": 3}
                ).fillna(0).sum()
            )
            representative = _representative_instance_center(cells)
            local_instances = representative[2] if representative else 0
            local_units = representative[3] if representative else 0
            local_cluster_count = 0
            if representative and not cells.empty:
                distance = np.hypot(
                    cells["x_px"].to_numpy(float) - representative[0],
                    cells["y_px"].to_numpy(float) - representative[1],
                )
                local_cluster_count = int(
                    (
                        cells.loc[distance <= 550, "integrated_label"].astype(str)
                        == "cluster_3plus"
                    ).sum()
                )

            cf_dense_positive = bool(
                baseline_dense is not None
                and dense_growth_decision(
                    baseline_dense,
                    late_dense,
                    minimum_foreground_fraction=float(
                        settings.get("dense_minimum_foreground_fraction", 0.008)
                    ),
                    minimum_foreground_delta=float(
                        settings.get("dense_minimum_foreground_delta", 0.006)
                    ),
                    minimum_foreground_ratio=float(
                        settings.get("dense_minimum_foreground_ratio", 1.8)
                    ),
                    minimum_candidate_ratio=float(
                        settings.get("dense_minimum_candidate_ratio", 1.5)
                    ),
                    minimum_tile_foreground_fraction=float(
                        settings.get(
                            "dense_minimum_tile_foreground_fraction", 0.025
                        )
                    ),
                )
            )
            instance_dense_positive = bool(
                local_instances >= int(settings.get("dense_minimum_local_instances", 8))
                and local_units >= int(settings.get("dense_minimum_local_units", 12))
                and (
                    local_cluster_count
                    >= int(settings.get("dense_minimum_local_clusters", 2))
                    or local_units >= max(18, baseline_units * 2 + 2)
                )
            )

            if cf_dense_positive:
                decision = "obvious_growth"
                channel = "dense_region"
                stage = "dense_growth_region"
                center_x, center_y = late_dense.center_x, late_dense.center_y
            elif instance_dense_positive:
                decision = "obvious_growth"
                channel = "dense_instance_region"
                stage = "dense_instance_region"
                center_x, center_y = representative[0], representative[1]
            else:
                decision = late_growth_decision(
                    baseline_units,
                    detected_units,
                    final_stage=True,
                    thresholds=thresholds,
                )
                channel = "full_v2_instance"
                stage = "whole_well_v2"
                center_x = representative[0] if representative else late_dense.center_x
                center_y = representative[1] if representative else late_dense.center_y

            if not cells.empty:
                object_rows.append(
                    cells.assign(
                        well=well,
                        timepoint=timepoint,
                        detection_stage=stage,
                    )
                )
            rows.append(
                {
                    "well": well,
                    "timepoint": timepoint,
                    "automatic_decision": decision,
                    "search_stage": stage,
                    "decision_channel": channel,
                    "baseline_cell_units": baseline_units,
                    "detected_late_cell_units": detected_units,
                    "detected_late_instance_count": int(len(cells)),
                    "anchor_count": int(len(well_anchors)),
                    "evaluated_candidate_count": int(len(proposals)),
                    "available_candidate_count": int(len(proposals)),
                    "dense_baseline_candidate_count": baseline_dense.candidate_count if baseline_dense else 0,
                    "dense_late_candidate_count": late_dense.candidate_count,
                    "dense_baseline_foreground_fraction": baseline_dense.foreground_fraction if baseline_dense else 0.0,
                    "dense_late_foreground_fraction": late_dense.foreground_fraction,
                    "dense_maximum_tile_foreground_fraction": late_dense.maximum_tile_foreground_fraction,
                    "dense_local_instance_count": local_instances,
                    "dense_local_cell_units": local_units,
                    "dense_local_cluster_count": local_cluster_count,
                    "dense_center_x": float(center_x),
                    "dense_center_y": float(center_y),
                    "device": str(
                        full.get("device", pd.Series("cuda", index=full.index)).iloc[0]
                        if len(full)
                        else "unknown"
                    ),
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(
                f"late growth {well} {timepoint}: {decision} channel={channel} "
                f"instances={len(cells)} units={detected_units}",
                flush=True,
            )

    output = artifact_path(
        config, "predictions", "latest_late_growth_predictions.csv"
    )
    result = pd.DataFrame(rows)
    if selected_wells is not None and output.exists():
        previous = pd.read_csv(output)
        previous = previous[
            ~previous["well"].astype(str).str.upper().isin(selected_wells)
        ]
        result = pd.concat([previous, result], ignore_index=True, sort=False)
    result = result.sort_values(["well", "timepoint"]).reset_index(drop=True)
    result.to_csv(output, index=False, encoding="utf-8")

    objects_output = artifact_path(
        config, "predictions", "latest_late_growth_objects.csv"
    )
    objects = (
        pd.concat(object_rows, ignore_index=True)
        if object_rows
        else pd.DataFrame()
    )
    if selected_wells is not None and objects_output.exists():
        previous_objects = pd.read_csv(objects_output, low_memory=False)
        previous_objects = previous_objects[
            ~previous_objects["well"].astype(str).str.upper().isin(selected_wells)
        ]
        objects = pd.concat(
            [previous_objects, objects], ignore_index=True, sort=False
        )
    objects.to_csv(objects_output, index=False, encoding="utf-8")
    summary = {
        "algorithm": "full_t0_t2_v2_instances_plus_dense_regions_v3",
        "well_timepoint_count": int(len(result)),
        "decision_counts": result["automatic_decision"].value_counts().to_dict(),
        "channel_counts": result["decision_channel"].value_counts().to_dict(),
        "stage_counts": result["search_stage"].value_counts().to_dict(),
        "output": str(output),
        "full_instance_predictions": str(full_path),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from torch.utils.data import Dataset

from .config import artifact_path, load_config


OBJECT_LABELS = {"single", "touching_doublet", "cluster_3plus", "debris"}


def _crop(array: np.ndarray, x: float, y: float, size: int, fill: float = 0) -> np.ndarray:
    half = size // 2
    left, top = int(round(x)) - half, int(round(y)) - half
    output = np.full((size, size), fill, dtype=array.dtype)
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(array.shape[1], left + size), min(array.shape[0], top + size)
    if x0 < x1 and y0 < y1:
        output[y0 - top : y1 - top, x0 - left : x1 - left] = array[y0:y1, x0:x1]
    return output


def _latest_reviews(database: Path) -> pd.DataFrame:
    with sqlite3.connect(database) as connection:
        frame = pd.read_sql_query(
            """
            SELECT integrated_review_id, candidate_id, reviewed_label, updated_at
            FROM integrated_training_reviews
            ORDER BY updated_at, integrated_review_id
            """,
            connection,
        )
    return frame.drop_duplicates("candidate_id", keep="last")


def _seed_heatmap(size: int, sigma: float) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2
    return np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma**2)).astype(np.float32)


def _wall_prior(
    image_shape: tuple[int, int], x: float, y: float, size: int, inner_fraction: float
) -> np.ndarray:
    half = size // 2
    yy, xx = np.mgrid[:size, :size]
    global_x = xx + int(round(x)) - half
    global_y = yy + int(round(y)) - half
    height, width = image_shape
    radial = np.hypot(global_x - width / 2, global_y - height / 2) / min(width, height)
    return (radial >= inner_fraction - 0.012).astype(np.float32)


def _nearest_compact_component(mask: np.ndarray, maximum_area: int) -> np.ndarray:
    labelled = label(mask, connectivity=2)
    center = np.asarray([(mask.shape[0] - 1) / 2, (mask.shape[1] - 1) / 2])
    choices = []
    for region in regionprops(labelled):
        if 3 <= region.area <= maximum_area:
            distance = float(np.linalg.norm(np.asarray(region.centroid) - center))
            choices.append((distance, -float(region.area), region.label))
    if not choices:
        return np.zeros_like(mask, dtype=np.uint8)
    distance, _, chosen = min(choices)
    if distance > mask.shape[0] * 0.22:
        return np.zeros_like(mask, dtype=np.uint8)
    return (labelled == chosen).astype(np.uint8)


def _residual_pseudo_mask(raw: np.ndarray, maximum_area: int) -> np.ndarray:
    smooth = gaussian_filter(raw.astype(np.float32), sigma=5.0)
    residual = np.abs(raw.astype(np.float32) - smooth)
    threshold = max(float(np.percentile(residual, 88)), float(residual.mean() + residual.std()))
    yy, xx = np.ogrid[: raw.shape[0], : raw.shape[1]]
    central = (xx - raw.shape[1] / 2) ** 2 + (yy - raw.shape[0] / 2) ** 2 <= 20**2
    return _nearest_compact_component((residual >= threshold) & central, maximum_area)


def build_v2_instance_cache(config: dict[str, Any]) -> Path:
    settings = config["v2_instance_segmentation"]
    size = int(settings.get("patch_size_px", 96))
    maximum_area = int(settings.get("maximum_instance_area_px", 1800))
    maximum_samples = int(settings.get("maximum_training_samples", 6000))
    seed = int(settings.get("seed", 20260802))
    output = artifact_path(config, "cache", f"v2_instance_training_{size}.npz")
    if bool(settings.get("reuse_training_cache", False)) and output.exists():
        return output

    inputs: list[np.ndarray] = []
    instances: list[np.ndarray] = []
    walls: list[np.ndarray] = []
    labels_out: list[str] = []
    sources_out: list[str] = []
    origins_out: list[str] = []
    wells_out: list[str] = []
    timepoints_out: list[str] = []
    source_configs = settings.get(
        "training_sources", ["configs/default.yaml", "configs/ql2202_validation.yaml"]
    )
    for source_config_path in source_configs:
        source_config = load_config(source_config_path)
        predictions_path = artifact_path(
            source_config, "predictions", "latest_integrated_predictions.csv"
        )
        database = artifact_path(source_config, "annotations", "annotations.db")
        if not predictions_path.exists() or not database.exists():
            continue
        predictions = pd.read_csv(predictions_path, low_memory=False)
        reviews = _latest_reviews(database)
        reviewed = predictions.merge(
            reviews[["candidate_id", "reviewed_label"]], on="candidate_id", how="inner"
        )
        reviewed = reviewed[
            reviewed["timepoint"].isin(["T0", "T1", "T2"])
            & reviewed["reviewed_label"].isin(list(OBJECT_LABELS | {"invalid"}))
        ].copy()
        reviewed["v2_label_origin"] = "human_review"
        reviewed_ids = set(reviewed["candidate_id"].astype(str))
        inner_series = predictions["detected_wall_inner_fraction"].fillna(
            source_config.get("candidate_filter", {}).get("hard_wall_exclusion_fraction", 0.44)
        )
        review_anisotropy = predictions["review_anisotropy"].fillna(0) if "review_anisotropy" in predictions else pd.Series(0, index=predictions.index)
        background_anisotropy = predictions["background_anisotropy"].fillna(0) if "background_anisotropy" in predictions else pd.Series(0, index=predictions.index)
        wall_neighbors = predictions["wall_neighbor_count"].fillna(0) if "wall_neighbor_count" in predictions else pd.Series(0, index=predictions.index)
        structural_wall = predictions[
            (predictions["integrated_label"].astype(str) == "invalid")
            & (predictions["radial_fraction"].astype(float) >= inner_series.astype(float) - 0.025)
            & (
                (review_anisotropy.astype(float) >= 0.55)
                | (background_anisotropy.astype(float) >= 0.60)
                | (wall_neighbors.astype(float) >= 3)
                | predictions["candidate_source"].astype(str).str.contains("wall", case=False, na=False)
            )
            & ~predictions["candidate_id"].astype(str).isin(reviewed_ids)
        ].copy()
        if len(structural_wall) > 800:
            structural_wall = structural_wall.sample(800, random_state=seed)
        structural_wall["reviewed_label"] = "invalid"
        structural_wall["v2_label_origin"] = "structural_wall_hard_negative"
        texture_negative = predictions[
            (predictions["invalid_probability"].fillna(0).astype(float) >= 0.60)
            & predictions["integrated_label"].astype(str).isin(["invalid", "uncertain", "unmarked"])
            & (predictions["radial_fraction"].astype(float) < inner_series.astype(float) - 0.02)
            & ~predictions["candidate_id"].astype(str).isin(reviewed_ids)
            & ~predictions["candidate_id"].astype(str).isin(set(structural_wall["candidate_id"].astype(str)))
        ].copy()
        if len(texture_negative) > 600:
            texture_negative = texture_negative.sample(600, random_state=seed + 1)
        texture_negative["reviewed_label"] = "invalid"
        texture_negative["v2_label_origin"] = "high_invalid_probability_texture_negative"
        reviewed = pd.concat([reviewed, structural_wall, texture_negative], ignore_index=True, sort=False)
        for raw_path, group in reviewed.groupby("raw_image_path", sort=False):
            with Image.open(raw_path) as image:
                raw_full = np.asarray(image.convert("L"), dtype=np.uint8)
            cf_path = str(group.iloc[0]["cf_image_path"])
            with Image.open(cf_path) as image:
                cf_full = np.asarray(image.convert("1"), dtype=bool)
            for row in group.itertuples(index=False):
                raw = _crop(raw_full, row.x_px, row.y_px, size, int(np.median(raw_full)))
                cf = _crop(cf_full, row.x_px, row.y_px, size, False)
                inner = float(
                    getattr(row, "detected_wall_inner_fraction", np.nan)
                    if pd.notna(getattr(row, "detected_wall_inner_fraction", np.nan))
                    else source_config.get("candidate_filter", {}).get(
                        "hard_wall_exclusion_fraction", 0.44
                    )
                )
                wall = _wall_prior(raw_full.shape, row.x_px, row.y_px, size, inner)
                instance = np.zeros((size, size), dtype=np.uint8)
                if str(row.reviewed_label) in OBJECT_LABELS:
                    instance = _nearest_compact_component(cf, maximum_area)
                    if not instance.any():
                        instance = _residual_pseudo_mask(raw, maximum_area)
                    if not instance.any():
                        continue
                normalized = raw.astype(np.float32)
                lo, hi = np.percentile(normalized, [2, 98])
                normalized = np.clip((normalized - lo) / max(hi - lo, 1.0), 0, 1)
                value = np.stack(
                    [normalized, _seed_heatmap(size, float(settings.get("seed_sigma_px", 4.0))), wall]
                ).astype(np.float32)
                inputs.append(value)
                instances.append(instance.astype(np.uint8))
                walls.append(wall.astype(np.uint8))
                labels_out.append(str(row.reviewed_label))
                sources_out.append(f"{source_config_path}|{row.candidate_id}")
                origins_out.append(str(getattr(row, "v2_label_origin", "human_review")))
                wells_out.append(str(row.well))
                timepoints_out.append(str(row.timepoint))

    if not inputs:
        raise RuntimeError("No reviewed V2 instance samples are available.")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(inputs))
    if len(indices) > maximum_samples:
        indices = rng.choice(indices, maximum_samples, replace=False)
    np.savez_compressed(
        output,
        inputs=np.stack(inputs)[indices],
        instance_masks=np.stack(instances)[indices],
        wall_masks=np.stack(walls)[indices],
        labels=np.asarray(labels_out)[indices],
        sources=np.asarray(sources_out)[indices],
        wells=np.asarray(wells_out)[indices],
        timepoints=np.asarray(timepoints_out)[indices],
        label_origins=np.asarray(origins_out)[indices],
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "sample_count": int(len(indices)),
                "source_configs": source_configs,
                "patch_size_px": size,
                "label_counts": {
                    str(key): int(value)
                    for key, value in pd.Series(np.asarray(labels_out)[indices]).value_counts().items()
                },
                "A12_22_excluded": True,
                "label_source": "reviewed_candidate_seed_plus_CF_or_residual_pseudo_mask",
                "label_origin_counts": {
                    str(key): int(value)
                    for key, value in pd.Series(np.asarray(origins_out)[indices]).value_counts().items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


class V2InstanceDataset(Dataset):
    """Seed-conditioned training patches with synchronized geometric augmentation."""

    def __init__(self, cache_path: str | Path, augment: bool = False):
        payload = np.load(cache_path, allow_pickle=False)
        self.inputs = payload["inputs"]
        self.instances = payload["instance_masks"]
        self.walls = payload["wall_masks"]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.inputs[index].copy()
        target = np.stack([self.instances[index], self.walls[index]]).astype(np.float32)
        if self.augment:
            turns = int(np.random.randint(0, 4))
            value = np.rot90(value, turns, axes=(1, 2)).copy()
            target = np.rot90(target, turns, axes=(1, 2)).copy()
            if bool(np.random.randint(0, 2)):
                value = value[:, :, ::-1].copy()
                target = target[:, :, ::-1].copy()
            if bool(np.random.randint(0, 2)):
                value = value[:, ::-1, :].copy()
                target = target[:, ::-1, :].copy()
        return torch.from_numpy(value), torch.from_numpy(target)

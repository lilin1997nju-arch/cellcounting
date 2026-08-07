from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import artifact_path


def _crop(array: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    half = size // 2
    return array[y - half:y + half, x - half:x + half]


def build_weak_patch_cache(
    config: dict[str, Any],
    split_name: str,
    patch_size: int,
    patches_per_image: int,
    max_wells: int,
    seed: int,
) -> Path:
    destination = artifact_path(config, "cache", f"weak_{split_name}_{patch_size}.npz")
    images = pd.read_csv(artifact_path(config, "manifests", "images.csv"))
    split_file = artifact_path(config, "splits", f"{split_name}_sequences.csv")
    wells = pd.read_csv(split_file)["well"].tolist()[:max_wells]
    selected = images[
        images["well"].isin(wells)
        & images["timepoint"].isin(["T0", "T1", "T2"])
        & (images["decode_status"] == "ok")
    ].sort_values(["well", "timepoint"])
    rng = np.random.default_rng(seed + sum(ord(ch) for ch in split_name))
    image_patches: list[np.ndarray] = []
    mask_patches: list[np.ndarray] = []
    sources: list[str] = []
    half = patch_size // 2

    for row in selected.itertuples(index=False):
        with Image.open(row.raw_image_path) as image:
            raw = np.asarray(image.convert("L"), dtype=np.uint8)
        with Image.open(row.cf_image_path) as image:
            mask = np.asarray(image.convert("1"), dtype=bool).copy()

        height, width = raw.shape
        yy, xx = np.ogrid[:height, :width]
        inner_well = (
            (xx - width / 2) ** 2 + (yy - height / 2) ** 2
            <= (min(height, width) * 0.42) ** 2
        )
        mask &= inner_well
        valid_mask = mask.copy()
        valid_mask[:half, :] = False
        valid_mask[-half:, :] = False
        valid_mask[:, :half] = False
        valid_mask[:, -half:] = False
        positive_indices = np.flatnonzero(valid_mask)

        count_positive = patches_per_image // 2
        count_negative = patches_per_image - count_positive
        centers: list[tuple[int, int, str]] = []
        if positive_indices.size:
            chosen = rng.choice(positive_indices, size=count_positive, replace=positive_indices.size < count_positive)
            for flat in chosen:
                y, x = np.unravel_index(int(flat), mask.shape)
                centers.append((int(x), int(y), "positive"))
        attempts = 0
        while sum(source == "negative" for _, _, source in centers) < count_negative and attempts < 1000:
            x = int(rng.integers(half, width - half))
            y = int(rng.integers(half, height - half))
            if (
                inner_well[y, x]
                and not mask[y, x]
                and mask[y - 4:y + 5, x - 4:x + 5].sum() == 0
            ):
                centers.append((x, y, "negative"))
            attempts += 1

        for x, y, kind in centers:
            image_patch = _crop(raw, x, y, patch_size)
            mask_patch = _crop(mask, x, y, patch_size)
            if image_patch.shape != (patch_size, patch_size):
                continue
            image_patches.append(image_patch)
            mask_patches.append(mask_patch.astype(np.uint8))
            sources.append(f"{row.well}|{row.timepoint}|{x}|{y}|{kind}")

    np.savez_compressed(
        destination,
        images=np.stack(image_patches),
        masks=np.stack(mask_patches),
        sources=np.asarray(sources),
    )
    metadata = {
        "split": split_name,
        "wells": wells,
        "patch_count": len(image_patches),
        "patch_size": patch_size,
        "patches_per_image": patches_per_image,
        "label_source": "instrument_cf_weak_mask",
        "seed": seed,
    }
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination

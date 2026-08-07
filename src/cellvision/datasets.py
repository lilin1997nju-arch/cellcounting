from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WeakMaskDataset(Dataset):
    def __init__(self, path: str | Path, augment: bool = False, seed: int = 0):
        payload = np.load(path)
        self.images = payload["images"]
        self.masks = payload["masks"]
        self.sources = payload["sources"]
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index].astype(np.float32) / 255.0
        mask = self.masks[index].astype(np.float32)
        if self.augment:
            rng = np.random.default_rng(self.seed + index)
            rotation = int(rng.integers(0, 4))
            image = np.rot90(image, rotation).copy()
            mask = np.rot90(mask, rotation).copy()
            if bool(rng.integers(0, 2)):
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
        return (
            torch.from_numpy(image[None]),
            torch.from_numpy(mask[None]),
            str(self.sources[index]),
        )


def synchronous_transform(
    images: list[np.ndarray], masks: list[np.ndarray], rotation: int, flip_horizontal: bool
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    transformed_images = [np.rot90(image, rotation) for image in images]
    transformed_masks = [np.rot90(mask, rotation) for mask in masks]
    if flip_horizontal:
        transformed_images = [np.fliplr(image) for image in transformed_images]
        transformed_masks = [np.fliplr(mask) for mask in transformed_masks]
    return transformed_images, transformed_masks


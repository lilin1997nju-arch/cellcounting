"""Build and merge the reviewed QL2603 instance cache with the old cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cellvision.config import artifact_path, load_config
from cellvision.v2_instance_dataset import (
    _balanced_training_indices,
    build_v2_instance_cache,
)


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_QL2603_SOURCES = [
    "configs/generated/ql2603-t1-1.yaml",
    "configs/generated/ql2603-t1-3.yaml",
    "configs/generated/ql2603-t1-4.yaml",
    "configs/generated/ql2603-t1-5.yaml",
    "configs/generated/ql2603-t2-1.yaml",
    "configs/generated/ql2603-t2-2.yaml",
    "configs/generated/ql2603-t2-3.yaml",
    "configs/generated/ql2603-t3-3.yaml",
    "configs/generated/ql2603-t4-3.yaml",
]


def main() -> None:
    config = load_config("configs/v2_training.yaml")
    settings = config["v2_instance_segmentation"]
    settings["training_sources"] = ACTIVE_QL2603_SOURCES
    settings["cache_name"] = "v2_instance_training_96_ql2603_addon"
    # Keep the reviewed add-on compact: the older cache already carries the
    # broad replay set, while these rows provide fresh 2603 morphology.
    settings["per_source_cap"] = 100
    settings["maximum_training_samples"] = 6000
    addon_path = build_v2_instance_cache(config)

    old_path = ROOT / "artifacts/backups/ql2603_training_round_20260811_pre/v2/v2_instance_training_96_old.npz"
    merged_path = artifact_path(config, "cache", "v2_instance_training_96.npz")
    old = np.load(old_path, allow_pickle=False)
    addon = np.load(addon_path, allow_pickle=False)
    arrays = {
        key: np.concatenate([old[key], addon[key]], axis=0)
        for key in (
            "inputs",
            "instance_masks",
            "wall_masks",
            "labels",
            "sources",
            "wells",
            "timepoints",
            "label_origins",
        )
    }
    settings["cache_name"] = "v2_instance_training_96"
    indices = _balanced_training_indices(
        arrays["labels"],
        arrays["sources"],
        int(settings.get("maximum_training_samples", 6000)),
        int(settings.get("seed", 20260802)),
    )
    np.savez_compressed(
        merged_path,
        **{key: value[indices] for key, value in arrays.items()},
    )
    metadata = {
        "sample_count": int(len(indices)),
        "source_configs": ["configs/default.yaml", "configs/ql2202_validation.yaml"]
        + ACTIVE_QL2603_SOURCES,
        "patch_size_px": int(settings.get("patch_size_px", 96)),
        "label_counts": {
            str(key): int(value)
            for key, value in zip(
                *np.unique(arrays["labels"][indices], return_counts=True)
            )
        },
        "balanced_by_plate_and_class": True,
        "validation_holdouts": config.get("validation_holdout_sources", []),
        "label_origin_counts": {
            str(key): int(value)
            for key, value in zip(
                *np.unique(arrays["label_origins"][indices], return_counts=True)
            )
        },
    }
    merged_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"addon": str(addon_path), "merged": str(merged_path), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from torch.utils.data import Dataset

from .config import artifact_path, is_validation_holdout, load_config


OBJECT_LABELS = {"single", "touching_doublet", "cluster_3plus", "debris"}
MASK_REVIEW_DECISIONS = {"accepted", "edited", "rejected"}


def _balanced_training_indices(
    labels: np.ndarray,
    sources: np.ndarray,
    maximum_samples: int,
    seed: int,
) -> np.ndarray:
    """Select a reproducible, plate/class-balanced cache subset.

    The reviewed 2603 material is much larger than the older sources and is
    dominated by debris.  Uniform random truncation can therefore discard
    most of the older plates or the rare multiplicity classes.  We allocate a
    quota to each source plate, stratify that quota by reviewed class, and use
    a deterministic fill pass for any unused capacity.
    """

    total = len(labels)
    if total <= maximum_samples:
        return np.random.default_rng(seed).permutation(total).astype(np.int64)

    rng = np.random.default_rng(seed)
    source_groups = np.asarray(
        [str(value).split("|", 1)[0] for value in sources], dtype=str
    )
    group_values = sorted(set(source_groups.tolist()))
    selected: list[int] = []

    # Start with an equal plate quota.  A later fill pass gives the unused
    # slots back to larger plates without changing the initial balance.
    quota = max(1, maximum_samples // max(len(group_values), 1))
    for group in group_values:
        group_indices = np.flatnonzero(source_groups == group)
        if len(group_indices) <= quota:
            chosen = group_indices
            remainder = np.empty(0, dtype=np.int64)
        else:
            class_values = sorted(set(labels[group_indices].astype(str).tolist()))
            per_class = max(1, quota // max(len(class_values), 1))
            chosen_parts: list[int] = []
            for label in class_values:
                label_indices = group_indices[labels[group_indices].astype(str) == label]
                take = min(len(label_indices), per_class)
                if take:
                    chosen_parts.extend(
                        rng.choice(label_indices, take, replace=False).tolist()
                    )
            chosen_set = set(chosen_parts)
            unselected = np.asarray(
                [index for index in group_indices if int(index) not in chosen_set],
                dtype=np.int64,
            )
            remaining = max(0, quota - len(chosen_parts))
            if remaining and len(unselected):
                extra = rng.choice(
                    unselected, min(remaining, len(unselected)), replace=False
                )
                chosen_parts.extend(extra.tolist())
                chosen_set.update(int(value) for value in extra)
            chosen = np.asarray(chosen_parts, dtype=np.int64)
            remainder = np.asarray(
                [index for index in unselected if int(index) not in chosen_set],
                dtype=np.int64,
            )
        selected.extend(chosen.tolist())

    selected_set = set(selected)
    remaining = np.asarray(
        [index for index in range(total) if index not in selected_set],
        dtype=np.int64,
    )
    if len(selected) < maximum_samples and len(remaining):
        remaining = rng.permutation(remaining)
        selected.extend(
            remaining[: maximum_samples - len(selected)].astype(np.int64).tolist()
        )
    return rng.permutation(np.asarray(selected[:maximum_samples], dtype=np.int64))


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


def _resolve_review_file(value: str | Path, base: Path) -> Path:
    candidate = Path(str(value)).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _load_mask_review_sources(
    settings: dict[str, Any],
) -> list[tuple[dict[str, Any], str, pd.DataFrame]]:
    """Load completed pixel-level review rounds as exact training sources.

    The historical V2 cache is built from ``integrated_training_reviews`` and
    reconstructs a compact pseudo-mask from the CF channel.  A mask review
    round is different: its ``reviewed_v2_predictions.csv`` already contains
    the reviewer-approved RLE, including an empty RLE for an explicitly
    rejected non-cell proposal.  Keep this source separate so exact human
    masks are not replaced by a weaker pseudo-mask during cache construction.
    """

    sources: list[tuple[dict[str, Any], str, pd.DataFrame]] = []
    for entry in settings.get("mask_review_rounds", []) or []:
        if isinstance(entry, (str, Path)):
            raise ValueError(
                "mask_review_rounds entries must include both config and round_id"
            )
        if not isinstance(entry, dict):
            raise ValueError("mask_review_rounds entries must be mappings")
        source_config_path = entry.get("config") or entry.get("source_config")
        round_id = str(entry.get("round_id", "")).strip()
        if not source_config_path or not round_id:
            raise ValueError("each mask review round requires config and round_id")
        source_config = load_config(source_config_path)
        source_config_id = str(Path(str(source_config_path)).expanduser().resolve())
        explicit_round_dir = entry.get("round_dir")
        if explicit_round_dir:
            round_dir = Path(str(explicit_round_dir)).expanduser()
            if not round_dir.is_absolute():
                round_dir = (Path.cwd() / round_dir).resolve()
            else:
                round_dir = round_dir.resolve()
        else:
            round_dir = (
                Path(source_config["paths"]["artifact_root"])
                / "v2"
                / "mask_review"
                / round_id
            ).resolve()
        manifest_path = round_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"mask review manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reviewed_value = manifest.get("reviewed_predictions")
        if not reviewed_value:
            raise ValueError(f"mask review manifest is missing reviewed_predictions: {manifest_path}")
        reviewed_path = _resolve_review_file(reviewed_value, round_dir)
        if not reviewed_path.exists():
            raise FileNotFoundError(f"reviewed mask predictions are missing: {reviewed_path}")
        frame = pd.read_csv(reviewed_path, low_memory=False)
        required = {
            "candidate_id",
            "well",
            "timepoint",
            "x_px",
            "y_px",
            "raw_image_path",
            "v2_mask_rle",
            "v2_mask_review_status",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"reviewed mask predictions are missing columns {missing}: {reviewed_path}"
            )
        frame["v2_mask_review_status"] = frame["v2_mask_review_status"].fillna("").astype(str).str.lower()
        frame = frame[frame["v2_mask_review_status"].isin(MASK_REVIEW_DECISIONS)].copy()
        frame = frame.drop_duplicates("candidate_id", keep="last").reset_index(drop=True)
        frame["v2_mask_review_round"] = round_id
        frame["v2_mask_size"] = int(manifest.get("mask_size", 96))
        frame["v2_mask_review_source_config"] = source_config_id
        frame["v2_label_origin"] = frame["v2_mask_review_status"].map(
            lambda status: f"v2_mask_review_{status}"
        )
        source_id = f"mask_review:{Path(str(source_config_path)).as_posix()}:{round_id}"
        sources.append((source_config, source_id, frame))
    return sources


def _decode_review_mask(value: Any, size: int) -> np.ndarray:
    """Decode and validate a review-round RLE into a local binary patch."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        runs: Any = []
    else:
        try:
            runs = json.loads(str(value))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("reviewed mask RLE must be valid JSON") from exc
    if not isinstance(runs, list):
        raise ValueError("reviewed mask RLE must be a list")
    total = int(size) * int(size)
    mask = np.zeros(total, dtype=np.uint8)
    for run in runs:
        if not isinstance(run, (list, tuple)) or len(run) != 2:
            raise ValueError("each reviewed mask RLE run must contain start and length")
        start, length = int(run[0]), int(run[1])
        if start < 0 or length < 0 or start + length > total:
            raise ValueError("reviewed mask RLE run is outside the patch")
        mask[start : start + length] = 1
    return mask.reshape((int(size), int(size)))


def _review_training_arrays(
    settings: dict[str, Any],
    size: int,
    mask_review_sources: list[tuple[dict[str, Any], str, pd.DataFrame]],
) -> dict[str, np.ndarray]:
    inputs: list[np.ndarray] = []
    instances: list[np.ndarray] = []
    walls: list[np.ndarray] = []
    labels: list[str] = []
    sources: list[str] = []
    wells: list[str] = []
    timepoints: list[str] = []
    origins: list[str] = []
    for source_config, source_id, reviewed in mask_review_sources:
        source_started = time.perf_counter()
        size_from_round = int(reviewed["v2_mask_size"].iloc[0]) if not reviewed.empty else size
        if size_from_round != size:
            raise ValueError(
                f"mask review size {size_from_round} does not match training size {size}"
            )
        for raw_path, group in reviewed.groupby("raw_image_path", sort=False):
            with Image.open(raw_path) as image:
                raw_full = np.asarray(image.convert("L"), dtype=np.uint8)
            for row in group.itertuples(index=False):
                raw = _crop(raw_full, row.x_px, row.y_px, size, int(np.median(raw_full)))
                inner = float(
                    getattr(row, "detected_wall_inner_fraction", np.nan)
                    if pd.notna(getattr(row, "detected_wall_inner_fraction", np.nan))
                    else source_config.get("candidate_filter", {}).get(
                        "hard_wall_exclusion_fraction", 0.44
                    )
                )
                wall = _wall_prior(raw_full.shape, row.x_px, row.y_px, size, inner)
                instance = _decode_review_mask(row.v2_mask_rle, size)
                status = str(row.v2_mask_review_status).lower()
                if status == "rejected":
                    instance.fill(0)
                    label_value = "invalid"
                else:
                    label_value = str(getattr(row, "integrated_label", "single"))
                    if label_value not in OBJECT_LABELS:
                        label_value = "single"
                normalized = raw.astype(np.float32)
                lo, hi = np.percentile(normalized, [2, 98])
                normalized = np.clip(
                    (normalized - lo) / max(hi - lo, 1.0), 0, 1
                )
                inputs.append(
                    np.stack(
                        [
                            normalized,
                            _seed_heatmap(
                                size, float(settings.get("seed_sigma_px", 4.0))
                            ),
                            wall,
                        ]
                    ).astype(np.float32)
                )
                instances.append(instance.astype(np.uint8))
                walls.append(wall.astype(np.uint8))
                labels.append(label_value)
                sources.append(f"{source_id}|{row.candidate_id}")
                wells.append(str(row.well))
                timepoints.append(str(row.timepoint))
                origins.append(str(row.v2_label_origin))
        print(
            f"v2 mask review source {source_id}: {len(reviewed)} rows, "
            f"completed in {time.perf_counter() - source_started:.1f}s",
            flush=True,
        )
    if not inputs:
        raise RuntimeError("No completed V2 mask review samples are available.")
    return {
        "inputs": np.stack(inputs),
        "instance_masks": np.stack(instances),
        "wall_masks": np.stack(walls),
        "labels": np.asarray(labels),
        "sources": np.asarray(sources),
        "wells": np.asarray(wells),
        "timepoints": np.asarray(timepoints),
        "label_origins": np.asarray(origins),
    }


def build_v2_mask_review_replay_cache(config: dict[str, Any]) -> Path:
    """Build a replay cache without rescanning the historical image sources."""

    settings = config["v2_instance_segmentation"]
    size = int(settings.get("patch_size_px", 96))
    output = artifact_path(
        config,
        "cache",
        f"{str(settings.get('cache_name', f'v2_instance_training_{size}')).strip()}.npz",
    )
    replay_value = settings.get("replay_cache_path")
    if not replay_value:
        raise ValueError("replay_cache_path is required for mask review replay cache")
    replay_path = Path(str(replay_value)).expanduser()
    if not replay_path.is_absolute():
        replay_path = (Path.cwd() / replay_path).resolve()
    if not replay_path.exists():
        raise FileNotFoundError(f"replay cache is missing: {replay_path}")
    replay = np.load(replay_path, allow_pickle=False)
    keys = (
        "inputs",
        "instance_masks",
        "wall_masks",
        "labels",
        "sources",
        "wells",
        "timepoints",
        "label_origins",
    )
    missing = [key for key in keys if key not in replay.files]
    if missing:
        raise ValueError(f"replay cache is missing arrays: {missing}")
    if replay["inputs"].shape[1:] != (3, size, size):
        raise ValueError(f"replay cache has incompatible input shape: {replay['inputs'].shape}")
    review = _review_training_arrays(
        settings, size, _load_mask_review_sources(settings)
    )
    replay_count = len(replay["inputs"])
    review_count = len(review["inputs"])
    maximum_samples = int(settings.get("maximum_training_samples", replay_count + review_count))
    if maximum_samples < review_count:
        raise ValueError(
            f"maximum_training_samples={maximum_samples} cannot retain {review_count} reviewed samples"
        )
    replay_take = min(replay_count, maximum_samples - review_count)
    if replay_take < replay_count:
        replay_indices = _balanced_training_indices(
            replay["labels"], replay["sources"], replay_take, int(settings.get("seed", 20260802))
        )
    else:
        replay_indices = np.arange(replay_count, dtype=np.int64)
    arrays = {
        key: np.concatenate([replay[key][replay_indices], review[key]], axis=0)
        for key in keys
    }
    rng = np.random.default_rng(int(settings.get("seed", 20260802)))
    order = rng.permutation(len(arrays["inputs"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{key: value[order] for key, value in arrays.items()})
    metadata = {
        "sample_count": int(len(order)),
        "replay_cache": str(replay_path),
        "replay_sample_count": int(len(replay_indices)),
        "mask_review_sample_count": int(review_count),
        "source_configs": settings.get("training_sources", []),
        "mask_review_rounds": settings.get("mask_review_rounds", []),
        "patch_size_px": size,
        "label_counts": {
            str(key): int(value)
            for key, value in zip(*np.unique(arrays["labels"][order], return_counts=True))
        },
        "label_origin_counts": {
            str(key): int(value)
            for key, value in zip(
                *np.unique(arrays["label_origins"][order], return_counts=True)
            )
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


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
    if bool(settings.get("replay_mask_review_cache", False)):
        return build_v2_mask_review_replay_cache(config)
    size = int(settings.get("patch_size_px", 96))
    maximum_area = int(settings.get("maximum_instance_area_px", 1800))
    maximum_samples = int(settings.get("maximum_training_samples", 6000))
    seed = int(settings.get("seed", 20260802))
    cache_name = str(
        settings.get("cache_name", f"v2_instance_training_{size}")
    ).strip()
    output = artifact_path(config, "cache", f"{cache_name}.npz")
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
    mask_review_sources = _load_mask_review_sources(settings)
    mask_review_ids_by_config: dict[str, set[str]] = {}
    for _, _, frame in mask_review_sources:
        if frame.empty:
            continue
        source_config_id = str(frame["v2_mask_review_source_config"].iloc[0])
        mask_review_ids_by_config[source_config_id] = set(
            frame["candidate_id"].astype(str)
        )
    for source_config_path in source_configs:
        if is_validation_holdout(config, source_config_path):
            continue
        source_started = time.perf_counter()
        source_config = load_config(source_config_path)
        predictions_path = artifact_path(
            source_config, "predictions", "latest_integrated_predictions.csv"
        )
        database = artifact_path(source_config, "annotations", "annotations.db")
        if not predictions_path.exists() or not database.exists():
            continue
        predictions = pd.read_csv(predictions_path, low_memory=False)
        reviews = _latest_reviews(database)
        if reviews.empty and not bool(
            settings.get("include_unreviewed_hard_negatives", True)
        ):
            continue
        reviewed = predictions.merge(
            reviews[["candidate_id", "reviewed_label"]], on="candidate_id", how="inner"
        )
        reviewed = reviewed[
            reviewed["timepoint"].isin(["T0", "T1", "T2"])
            & reviewed["reviewed_label"].isin(list(OBJECT_LABELS | {"invalid"}))
        ].copy()
        source_key = str(Path(str(source_config_path)).expanduser().resolve())
        if source_key in mask_review_ids_by_config:
            reviewed = reviewed[
                ~reviewed["candidate_id"].astype(str).isin(
                    mask_review_ids_by_config[source_key]
                )
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
        per_source_cap = int(settings.get("per_source_cap", 0) or 0)
        if per_source_cap > 0 and len(reviewed) > per_source_cap:
            local_indices = _balanced_training_indices(
                reviewed["reviewed_label"].astype(str).to_numpy(),
                np.full(len(reviewed), str(source_config_path), dtype=str),
                per_source_cap,
                seed,
            )
            reviewed = reviewed.iloc[local_indices].reset_index(drop=True)
        print(
            f"v2 instance source {source_config_path}: {len(reviewed)} rows",
            flush=True,
        )
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
        print(
            f"v2 instance source {source_config_path}: completed in "
            f"{time.perf_counter() - source_started:.1f}s",
            flush=True,
        )

    # Pixel-level review rounds are appended after the legacy source pass.
    # Their exact RLE replaces any same-candidate pseudo-mask above, while
    # rejected non-cell proposals remain as all-zero hard negatives.
    for source_config, source_id, reviewed in mask_review_sources:
        source_started = time.perf_counter()
        size_from_round = int(reviewed["v2_mask_size"].iloc[0]) if not reviewed.empty else size
        if size_from_round != size:
            raise ValueError(
                f"mask review size {size_from_round} does not match training size {size}"
            )
        for raw_path, group in reviewed.groupby("raw_image_path", sort=False):
            with Image.open(raw_path) as image:
                raw_full = np.asarray(image.convert("L"), dtype=np.uint8)
            for row in group.itertuples(index=False):
                raw = _crop(raw_full, row.x_px, row.y_px, size, int(np.median(raw_full)))
                inner = float(
                    getattr(row, "detected_wall_inner_fraction", np.nan)
                    if pd.notna(getattr(row, "detected_wall_inner_fraction", np.nan))
                    else source_config.get("candidate_filter", {}).get(
                        "hard_wall_exclusion_fraction", 0.44
                    )
                )
                wall = _wall_prior(raw_full.shape, row.x_px, row.y_px, size, inner)
                instance = _decode_review_mask(row.v2_mask_rle, size)
                status = str(row.v2_mask_review_status).lower()
                if status == "rejected":
                    instance.fill(0)
                    label_value = "invalid"
                else:
                    label_value = str(getattr(row, "integrated_label", "single"))
                    if label_value not in OBJECT_LABELS:
                        label_value = "single"
                normalized = raw.astype(np.float32)
                lo, hi = np.percentile(normalized, [2, 98])
                normalized = np.clip(
                    (normalized - lo) / max(hi - lo, 1.0), 0, 1
                )
                value = np.stack(
                    [
                        normalized,
                        _seed_heatmap(size, float(settings.get("seed_sigma_px", 4.0))),
                        wall,
                    ]
                ).astype(np.float32)
                inputs.append(value)
                instances.append(instance.astype(np.uint8))
                walls.append(wall.astype(np.uint8))
                labels_out.append(label_value)
                sources_out.append(f"{source_id}|{row.candidate_id}")
                origins_out.append(str(row.v2_label_origin))
                wells_out.append(str(row.well))
                timepoints_out.append(str(row.timepoint))
        print(
            f"v2 mask review source {source_id}: {len(reviewed)} rows, "
            f"completed in {time.perf_counter() - source_started:.1f}s",
            flush=True,
        )

    if not inputs:
        raise RuntimeError("No reviewed V2 instance samples are available.")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(inputs))
    if len(indices) > maximum_samples:
        if bool(settings.get("balance_by_plate_and_class", True)):
            indices = _balanced_training_indices(
                np.asarray(labels_out),
                np.asarray(sources_out),
                maximum_samples,
                seed,
            )
        else:
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
                "balanced_by_plate_and_class": bool(
                    settings.get("balance_by_plate_and_class", True)
                ),
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

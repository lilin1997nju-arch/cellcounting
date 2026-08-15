from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label as ndi_label

from .config import artifact_path, is_validation_holdout, load_config
from .v2_instance_dataset import _crop
from .v2_instance_inference import decode_rle


CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}


def _lineage_static_target(row: pd.Series) -> float | None:
    if row.object_type == "debris":
        return 1.0
    if row.object_type == "cell" and row.viability == "live":
        return 0.0
    if row.object_type == "cell" and row.viability == "dead":
        return 1.0
    return None


def _point_mask(cf: np.ndarray, size: int) -> np.ndarray:
    labelled, _ = ndi_label(cf > 0)
    center = size // 2
    value = int(labelled[center, center])
    if value:
        return (labelled == value).astype(np.float32)
    yy, xx = np.ogrid[:size, :size]
    return (((xx - center) ** 2 + (yy - center) ** 2) <= 5**2).astype(np.float32)


def _review_triplet_static_target(rows: list[pd.Series]) -> float | None:
    """Derive whether a reviewed triplet is static, without assigning viability."""

    if len(rows) != 3:
        return None
    labels = [str(row.reviewed_label) for row in rows]
    if all(label == "debris" for label in labels):
        return 1.0
    # A reviewed cell that progressively loses its cell morphology is a
    # dynamic track, even when it does not divide.  Keeping this sequence out
    # of the static class was allowing the temporal model to learn that a
    # stable-looking T0 cell could remain a static object while it died.
    if (
        labels[0] in CELL_LABELS
        and labels[-1] in {"uncertain", "debris"}
        and any(label not in CELL_LABELS for label in labels[1:])
    ):
        return 0.0
    if not all(label in CELL_LABELS for label in labels):
        return None
    areas = [max(float(row.v2_instance_area_px), 1.0) for row in rows]
    x0, y0 = float(rows[0].aligned_x_px), float(rows[0].aligned_y_px)
    maximum_displacement = max(
        float(np.hypot(float(row.aligned_x_px) - x0, float(row.aligned_y_px) - y0))
        for row in rows[1:]
    )
    area_ratio = max(areas) / min(areas)
    if all(label == "single" for label in labels) and maximum_displacement <= 12.0 and area_ratio <= 1.60:
        return 1.0
    if labels[0] == "single" and any(
        label in {"touching_doublet", "cluster_3plus"} for label in labels[1:]
    ):
        return 0.0
    return None


def _append_integrated_review_triplets(
    source: dict[str, Any],
    size: int,
    images_out: list[np.ndarray],
    numeric_out: list[np.ndarray],
    present_out: list[list[float]],
    same_out: list[float],
    static_out: list[float],
    static_valid_out: list[float],
    groups_out: list[str],
    group_id: str,
) -> None:
    """Use the current rapid-review labels as temporal supervision.

    The rapid review UI stores per-timepoint morphology in
    integrated_training_reviews rather than lineage_reviews.  Reciprocal
    nearest-neighbour matching in registered coordinates turns conservative
    three-frame patterns into learned temporal examples without using the
    external A12-22 validation plate.
    """

    database = artifact_path(source, "annotations", "annotations.db")
    v2_path = artifact_path(source, "predictions", "latest_v2_predictions.csv")
    if not database.exists() or not v2_path.exists():
        return
    with sqlite3.connect(database) as connection:
        reviews = pd.read_sql_query(
            "SELECT candidate_id, reviewed_label, updated_at, integrated_review_id "
            "FROM integrated_training_reviews ORDER BY updated_at, integrated_review_id",
            connection,
        ).drop_duplicates("candidate_id", keep="last")
    if reviews.empty:
        return
    frame = pd.read_csv(v2_path, low_memory=False)
    latest_labels = dict(zip(reviews["candidate_id"].astype(str), reviews["reviewed_label"].astype(str)))
    frame["reviewed_label"] = frame["candidate_id"].astype(str).map(latest_labels)
    frame = frame[
        frame["v2_mask_valid"].fillna(False).astype(bool)
        & ~frame["v2_is_suppressed"].fillna(False).astype(bool)
        & frame["reviewed_label"].notna()
        & frame["timepoint"].isin(["T0", "T1", "T2"])
        & (frame["well"].astype(str).str.upper() != "A1")
    ].copy()
    if frame.empty:
        return
    x_column = "aligned_x_px" if "aligned_x_px" in frame else "x_px"
    y_column = "aligned_y_px" if "aligned_y_px" in frame else "y_px"
    frame["aligned_x_px"] = frame[x_column].astype(float)
    frame["aligned_y_px"] = frame[y_column].astype(float)

    for _, group in frame.groupby("well", sort=False):
        by_time = {timepoint: local for timepoint, local in group.groupby("timepoint")}
        if not all(timepoint in by_time for timepoint in ("T0", "T1", "T2")):
            continue
        image_cache: dict[str, np.ndarray] = {}
        used_triplets: set[tuple[str, str, str]] = set()
        t0_group = by_time["T0"]
        for _, anchor in t0_group.iterrows():
            selected = [anchor]
            valid_match = True
            for timepoint in ("T1", "T2"):
                local = by_time[timepoint]
                distances = np.hypot(
                    local["aligned_x_px"].astype(float) - float(anchor.aligned_x_px),
                    local["aligned_y_px"].astype(float) - float(anchor.aligned_y_px),
                )
                nearest_index = distances.idxmin()
                if float(distances.loc[nearest_index]) > 20.0:
                    valid_match = False
                    break
                candidate = local.loc[nearest_index]
                reverse_distances = np.hypot(
                    t0_group["aligned_x_px"].astype(float) - float(candidate.aligned_x_px),
                    t0_group["aligned_y_px"].astype(float) - float(candidate.aligned_y_px),
                )
                if str(t0_group.loc[reverse_distances.idxmin(), "candidate_id"]) != str(anchor.candidate_id):
                    valid_match = False
                    break
                selected.append(candidate)
            if not valid_match:
                continue
            triplet_key = tuple(str(row.candidate_id) for row in selected)
            if triplet_key in used_triplets:
                continue
            used_triplets.add(triplet_key)
            static_target = _review_triplet_static_target(selected)
            if static_target is None:
                continue

            triplet, coords, areas, multiplicities = [], [], [], []
            for row in selected:
                raw_path = str(row.raw_image_path)
                if raw_path not in image_cache:
                    with Image.open(raw_path) as image:
                        image_cache[raw_path] = np.asarray(image.convert("L"), dtype=np.uint8)
                raw_full = image_cache[raw_path]
                raw = _crop(
                    raw_full, float(row.x_px), float(row.y_px), size, int(np.median(raw_full))
                ).astype(np.float32)
                lo, hi = np.percentile(raw, [2, 98])
                raw = np.clip((raw - lo) / max(float(hi - lo), 1.0), 0, 1)
                full_mask = decode_rle(str(row.v2_mask_rle), 96).astype(np.float32)
                offset = (96 - size) // 2
                mask = full_mask[offset : offset + size, offset : offset + size]
                triplet.append(np.stack([raw, mask]))
                coords.append((float(row.aligned_x_px), float(row.aligned_y_px)))
                areas.append(float(row.v2_instance_area_px))
                label = str(row.reviewed_label)
                multiplicities.append(2 if label == "touching_doublet" else 3 if label == "cluster_3plus" else 1)

            numeric = np.zeros(24, np.float32)
            for pair_index, (left, right) in enumerate(((0, 1), (1, 2), (0, 2))):
                numeric[pair_index * 2 : pair_index * 2 + 2] = (
                    np.asarray(coords[right]) - np.asarray(coords[left])
                ) / 256.0
            numeric[6:9] = [
                np.log((areas[1] + 1) / (areas[0] + 1)),
                np.log((areas[2] + 1) / (areas[1] + 1)),
                max(areas) / 500.0,
            ]
            numeric[9:12] = np.asarray(multiplicities) / 4.0
            numeric[12:15] = [float(row.cell_probability) for row in selected]
            numeric[15:18] = [float(row.debris_probability) for row in selected]
            numeric[18:21] = [float(row.invalid_probability) for row in selected]
            numeric[21:24] = [float(row.v2_wall_overlap) for row in selected]
            images_out.append(np.asarray(triplet, np.float32))
            numeric_out.append(numeric)
            present_out.append([1.0, 1.0, 1.0])
            same_out.append(1.0)
            static_out.append(static_target)
            static_valid_out.append(1.0)
            groups_out.append(group_id)


def build_temporal_training_cache(config: dict[str, Any]) -> Path:
    settings = config["v2_temporal_model"]
    size = int(settings.get("patch_size_px", 64))
    output = artifact_path(config, "v2", "cache", f"temporal_training_{size}.npz")
    if bool(settings.get("reuse_training_cache", False)) and output.exists():
        return output
    images_out, numeric_out, present_out, same_out, static_out, static_valid_out, groups_out = [], [], [], [], [], [], []
    source_configs = settings.get("training_sources", ["configs/default.yaml", "configs/ql2202_validation.yaml"])
    for source_path in source_configs:
        if is_validation_holdout(config, source_path):
            continue
        source = load_config(source_path)
        experiment = source.get("experiment", {})
        group_id = f"{Path(source_path).stem}:{experiment.get('plate_id', '')}"
        database = artifact_path(source, "annotations", "annotations.db")
        manifest_path = artifact_path(source, "manifests", "images.csv")
        if not database.exists() or not manifest_path.exists():
            continue
        with sqlite3.connect(database) as connection:
            reviews = pd.read_sql_query("SELECT * FROM lineage_reviews ORDER BY updated_at", connection)
        reviews = reviews.drop_duplicates(["well", "canonical_target_id"], keep="last")
        manifest = pd.read_csv(manifest_path)
        lookup = {(row.well, row.timepoint): row for row in manifest.itertuples(index=False)}
        v2_path = artifact_path(source, "predictions", "latest_v2_predictions.csv")
        v2_instances = pd.read_csv(v2_path, low_memory=False) if v2_path.exists() else pd.DataFrame()
        for review in reviews.itertuples(index=False):
            points = json.loads(review.timepoint_points_json)
            triplet, masks, coords, areas, multiplicities, availability = [], [], [], [], [], []
            morphology_features: list[tuple[float, float, float, float]] = []
            for timepoint in ("T0", "T1", "T2"):
                point = points.get(timepoint, {})
                key = (review.well, timepoint)
                available = bool(point.get("present", False) and key in lookup)
                availability.append(float(available))
                if available:
                    record = lookup[key]
                    with Image.open(record.raw_image_path) as image:
                        raw_full = np.asarray(image.convert("L"), dtype=np.uint8)
                    with Image.open(record.cf_image_path) as image:
                        cf_full = np.asarray(image.convert("1"), dtype=bool)
                    x, y = float(point["x_px"]), float(point["y_px"])
                    raw = _crop(raw_full, x, y, size, int(np.median(raw_full))).astype(np.float32)
                    lo, hi = np.percentile(raw, [2, 98])
                    raw = np.clip((raw - lo) / max(float(hi - lo), 1.0), 0, 1)
                    cf = _crop(cf_full, x, y, size, False)
                    mask = None
                    morphology = (0.0, 0.0, 0.0, 0.0)
                    if not v2_instances.empty:
                        temporal_column = (
                            "v2_is_temporal_candidate"
                            if "v2_is_temporal_candidate" in v2_instances
                            else "v2_is_counting_instance"
                        )
                        local_instances = v2_instances[
                            (v2_instances["well"] == review.well)
                            & (v2_instances["timepoint"] == timepoint)
                            & v2_instances[temporal_column].fillna(False).astype(bool)
                        ]
                        if not local_instances.empty:
                            distances = np.hypot(local_instances["x_px"].astype(float) - x, local_instances["y_px"].astype(float) - y)
                            nearest = distances.idxmin()
                            if float(distances.loc[nearest]) <= 20:
                                instance = v2_instances.loc[nearest]
                                full_mask = decode_rle(str(instance["v2_mask_rle"]), 96).astype(np.float32)
                                offset = (96 - size) // 2
                                mask = full_mask[offset : offset + size, offset : offset + size]
                                morphology = (
                                    float(instance.get("cell_probability", 0.0)),
                                    float(instance.get("debris_probability", 0.0)),
                                    float(instance.get("invalid_probability", 0.0)),
                                    float(instance.get("v2_wall_overlap", 0.0)),
                                )
                    if mask is None:
                        mask = _point_mask(cf, size)
                    coords.append((x, y)); areas.append(float(point.get("area_px") or mask.sum()))
                    multiplicities.append(1 + len(point.get("additional_points", [])))
                else:
                    raw = np.zeros((size, size), np.float32); mask = np.zeros_like(raw)
                    coords.append((np.nan, np.nan)); areas.append(np.nan); multiplicities.append(0)
                    morphology = (0.0, 0.0, 0.0, 0.0)
                morphology_features.append(morphology)
                triplet.append(np.stack([raw, mask])); masks.append(mask)
            numeric = np.zeros(24, np.float32)
            pairs = [(0, 1), (1, 2), (0, 2)]
            for pair_index, (left, right) in enumerate(pairs):
                if availability[left] and availability[right]:
                    numeric[pair_index * 2 : pair_index * 2 + 2] = (np.asarray(coords[right]) - np.asarray(coords[left])) / 256.0
            numeric[6:9] = [np.log((areas[i + 1] + 1) / (areas[i] + 1)) if np.isfinite(areas[i]) and np.isfinite(areas[i + 1]) else 0 for i in range(2)] + [np.nanmax(np.nan_to_num(areas)) / 500.0]
            numeric[9:12] = np.asarray(multiplicities) / 4.0
            numeric[12:15] = [item[0] for item in morphology_features]
            numeric[15:18] = [item[1] for item in morphology_features]
            numeric[18:21] = [item[2] for item in morphology_features]
            numeric[21:24] = [item[3] for item in morphology_features]
            images_out.append(np.asarray(triplet, np.float32)); numeric_out.append(numeric); present_out.append(availability)
            same_out.append(float(sum(availability) >= 2 and review.object_type in {"cell", "debris"}))
            static_target = _lineage_static_target(pd.Series(review._asdict()))
            static_out.append(float(static_target) if static_target is not None else 0.0)
            static_valid_out.append(float(static_target is not None))
            groups_out.append(group_id)
        _append_integrated_review_triplets(
            source,
            size,
            images_out,
            numeric_out,
            present_out,
            same_out,
            static_out,
            static_valid_out,
            groups_out,
            group_id,
        )
    if not images_out:
        raise RuntimeError("No lineage reviews are available for V2 temporal training.")
    # Missing-proposal augmentation: keep the real raw crop but remove one
    # proposal mask and its morphology features.  This teaches the temporal
    # network that an absent detector proposal is not the same as an absent
    # object, provided the other timepoints and pixels remain consistent.
    original_count = len(images_out)
    for sample_index in range(original_count):
        if same_out[sample_index] < 0.5 or sum(present_out[sample_index]) < 3:
            continue
        missing_index = sample_index % 3
        augmented_image = np.asarray(images_out[sample_index], np.float32).copy()
        augmented_image[missing_index, 1] = 0.0
        augmented_numeric = np.asarray(numeric_out[sample_index], np.float32).copy()
        augmented_numeric[6:9] = 0.0
        augmented_numeric[9 + missing_index] = 0.0
        if len(augmented_numeric) >= 24:
            for start in (12, 15, 18, 21):
                augmented_numeric[start + missing_index] = 0.0
        images_out.append(augmented_image)
        numeric_out.append(augmented_numeric)
        present_out.append(list(present_out[sample_index]))
        same_out.append(same_out[sample_index])
        static_out.append(static_out[sample_index])
        static_valid_out.append(static_valid_out[sample_index])
        groups_out.append(f"{groups_out[sample_index]}:missing")
    # Rolled mismatches teach the correspondence head not to force a link.
    positive_images = np.asarray(images_out)
    if len(positive_images) > 1:
        negatives = positive_images.copy()
        negatives[:, 1:] = np.roll(negatives[:, 1:], 1, axis=0)
        images_out.extend(list(negatives))
        numeric_out.extend([np.zeros(len(numeric_out[0]), np.float32) for _ in negatives])
        present_out.extend([[1.0, 1.0, 1.0] for _ in negatives])
        same_out.extend([0.0] * len(negatives))
        static_out.extend([0.0] * len(negatives))
        static_valid_out.extend([0.0] * len(negatives))
        groups_out.extend([f"{group}:rolled" for group in groups_out[: len(negatives)]])
    np.savez_compressed(
        output,
        images=np.asarray(images_out, dtype=np.float32),
        numeric=np.asarray(numeric_out, dtype=np.float32),
        present=np.asarray(present_out, dtype=np.float32),
        same=np.asarray(same_out, dtype=np.float32),
        static=np.asarray(static_out, dtype=np.float32),
        static_valid=np.asarray(static_valid_out, dtype=np.float32),
        groups=np.asarray(groups_out, dtype=str),
    )
    return output

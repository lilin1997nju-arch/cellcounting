from __future__ import annotations

import json
import hashlib
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes, gaussian_filter, gaussian_filter1d, label as ndi_label, zoom
from skimage.morphology import closing, dilation, disk, erosion, remove_small_holes, remove_small_objects
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour, watershed
from skimage.measure import find_contours, regionprops

from .config import artifact_path
from .models.v2_instance_segmenter import SeededInstanceUNet
from .v2_instance_dataset import _crop, _seed_heatmap, _wall_prior


LOW_CELL_NONCELL_RESOLUTION_THRESHOLD = 0.30


def _normalize(raw: np.ndarray) -> np.ndarray:
    value = raw.astype(np.float32)
    lo, hi = np.percentile(value, [2, 98])
    return np.clip((value - lo) / max(float(hi - lo), 1.0), 0, 1)


def _selected_component(
    binary: np.ndarray,
    seed_weights: np.ndarray | None = None,
    *,
    retain_nearby: bool = False,
    nearby_distance: int = 3,
) -> np.ndarray:
    """Select the component supported by the seed, optionally retaining group lobes.

    Instance crops are seed-conditioned, so component ownership should follow
    the seed heatmap rather than only the component centroid.  Reviewed
    doublets and clusters may contain two thresholded lobes separated by a
    narrow one- or two-pixel gap; those nearby lobes are reconnected before
    contour extraction instead of being silently discarded.
    """

    binary = np.asarray(binary, dtype=bool)
    labelled, _ = ndi_label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    regions = regionprops(labelled)
    if not regions:
        return np.zeros_like(binary, dtype=bool)
    center = np.asarray([(binary.shape[0] - 1) / 2, (binary.shape[1] - 1) / 2])
    eligible = [region for region in regions if region.area >= 3]
    if not eligible:
        return np.zeros_like(binary, dtype=bool)
    if seed_weights is not None:
        weights = np.asarray(seed_weights, dtype=np.float32)
        if weights.shape != binary.shape:
            raise ValueError("seed_weights must have the same shape as binary")
        chosen = max(
            eligible,
            key=lambda region: (
                float(weights[labelled == region.label].sum()),
                -float(np.linalg.norm(np.asarray(region.centroid) - center)),
                float(region.area),
            ),
        )
    else:
        chosen = min(
            eligible,
            key=lambda region: np.linalg.norm(np.asarray(region.centroid) - center),
        )
    selected = labelled == chosen.label
    if not retain_nearby:
        return selected

    remaining = [region for region in eligible if region.label != chosen.label]
    # Grow iteratively because a three-cell cluster may form a short chain.
    while remaining:
        neighborhood = dilation(selected, disk(max(int(nearby_distance), 1)))
        attached = [region for region in remaining if np.any(neighborhood & (labelled == region.label))]
        if not attached:
            break
        for region in attached:
            selected |= labelled == region.label
            remaining.remove(region)
    if selected.any():
        selected = closing(selected, disk(min(max(int(nearby_distance), 1), 2)))
    return selected


def _rle(mask: np.ndarray) -> str:
    values = mask.astype(np.uint8).ravel()
    padded = np.pad(values, (1, 1))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    runs = changes.reshape(-1, 2)
    return json.dumps([[int(start), int(end - start)] for start, end in runs], separators=(",", ":"))


def decode_rle(value: str, size: int) -> np.ndarray:
    mask = np.zeros(size * size, dtype=bool)
    for start, length in json.loads(value or "[]"):
        mask[int(start) : int(start) + int(length)] = True
    return mask.reshape(size, size)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / max(int(union), 1))


def refine_instance_mask(
    mask: np.ndarray,
    raw: np.ndarray,
    wall_probability: np.ndarray,
    seed_weights: np.ndarray | None = None,
    *,
    retain_nearby: bool = False,
    nearby_distance: int = 3,
    minimum_area_ratio: float = 0.80,
    minimum_iou: float = 0.70,
    diagnostics: dict[str, Any] | None = None,
) -> np.ndarray:
    """Snap a coarse mask to nearby image edges without allowing large shape drift."""
    original = remove_small_objects(mask.astype(bool), max_size=2)
    original = closing(original, disk(1))
    # Cell interiors often contain a dark nucleus or phase halo.  A closed
    # cavity is therefore foreground for whole-cell feature extraction, not a
    # second boundary to be preserved.
    original = binary_fill_holes(original)
    if not original.any():
        if diagnostics is not None:
            diagnostics.update(status="empty", area_ratio=0.0, iou=0.0)
        return original
    edge_map = inverse_gaussian_gradient(raw.astype(np.float32), alpha=80.0, sigma=1.0)
    evolved = morphological_geodesic_active_contour(
        edge_map, 7, init_level_set=original, smoothing=1, threshold="auto", balloon=0
    ).astype(bool)
    allowed = dilation(original, disk(3))
    protected_core = erosion(original, disk(2))
    evolved = (evolved & allowed) | protected_core
    # A real cell may overlap the wall, but new contour growth is not allowed
    # to run longitudinally into very high-confidence physical wall pixels.
    evolved &= ~((wall_probability >= 0.92) & ~dilation(original, disk(1)))
    # Closing repairs one-pixel notches; avoid opening here because it can erase
    # the narrow bridge that is diagnostically important for touching doublets.
    evolved = closing(evolved, disk(1))
    evolved = binary_fill_holes(evolved)
    evolved = remove_small_objects(evolved, max_size=2)
    evolved = _selected_component(
        evolved,
        seed_weights,
        retain_nearby=retain_nearby,
        nearby_distance=nearby_distance,
    )
    evolved = binary_fill_holes(evolved)

    original_area = int(original.sum())
    evolved_area = int(evolved.sum())
    area_ratio = float(evolved_area / max(original_area, 1))
    overlap = _mask_iou(original, evolved)
    center = tuple(int(round((axis - 1) / 2)) for axis in original.shape)
    lost_center_seed = bool(original[center] and not evolved[center])
    status = "refined"
    if not evolved.any():
        status = "fallback_empty"
    elif area_ratio < float(minimum_area_ratio):
        status = "fallback_area_shrink"
    elif overlap < float(minimum_iou):
        status = "fallback_low_iou"
    elif lost_center_seed:
        status = "fallback_seed_lost"
    if diagnostics is not None:
        diagnostics.update(status=status, area_ratio=area_ratio, iou=overlap)
    return original if status.startswith("fallback_") else evolved


def lightweight_refine_instance_mask(mask: np.ndarray) -> np.ndarray:
    """Cheap contour cleanup for confident non-cell objects.

    Debris existence matters to the plate conclusion, but an expensive active
    contour is unnecessary unless the candidate could be a cell.
    """

    output = remove_small_objects(mask.astype(bool), max_size=2)
    output = closing(output, disk(1))
    output = remove_small_holes(output, max_size=9)
    return _selected_component(output)


def _path_signature(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.exists():
        return {"path": str(value.resolve()), "missing": True}
    stat = value.stat()
    return {
        "path": str(value.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _inference_fingerprint(
    config: dict[str, Any],
    predictions_path: Path,
    checkpoint_path: str | Path,
) -> str:
    database = artifact_path(config, "annotations", "annotations.db")
    manual_overrides = config.get("review_queue", {}).get(
        "apply_manual_point_overrides", True
    )
    payload = {
        "algorithm": "v2-instance-competing-seed-split-20260818",
        "predictions": _path_signature(predictions_path),
        "checkpoint": _path_signature(checkpoint_path),
        "annotations": _path_signature(database) if manual_overrides else {"ignored": True},
        "settings": config.get("v2_inference", {}),
        "manual_overrides": manual_overrides,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contour(mask: np.ndarray, origin_x: int, origin_y: int) -> str:
    # Smooth in probability space, extract at 4x resolution and then apply a
    # light periodic coordinate filter. This removes pixel stairs while keeping
    # the contour within roughly one source pixel of the refined mask.
    surface = zoom(gaussian_filter(mask.astype(np.float32), sigma=0.65), 4, order=3)
    contours = find_contours(surface, 0.5)
    if not contours:
        return "[]"
    # Prefer the enclosing boundary by polygon area.  Internal texture holes
    # can occasionally have more samples than a compact outer boundary after
    # interpolation, so point count alone is not a safe selector.
    def polygon_area(points: np.ndarray) -> float:
        y, x = points[:, 0], points[:, 1]
        return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)

    points = max(contours, key=polygon_area) / 4.0
    points[:, 0] = gaussian_filter1d(points[:, 0], sigma=1.1, mode="wrap")
    points[:, 1] = gaussian_filter1d(points[:, 1], sigma=1.1, mode="wrap")
    stride = max(1, int(np.ceil(len(points) / 120)))
    output = [[round(float(x + origin_x), 2), round(float(y + origin_y), 2)] for y, x in points[::stride]]
    return json.dumps(output, separators=(",", ":"))


def _global_overlap(first: pd.Series, second: pd.Series, size: int) -> tuple[float, float]:
    first_mask = decode_rle(str(first.v2_mask_rle), size)
    second_mask = decode_rle(str(second.v2_mask_rle), size)
    ax0, ay0 = int(first.v2_mask_origin_x), int(first.v2_mask_origin_y)
    bx0, by0 = int(second.v2_mask_origin_x), int(second.v2_mask_origin_y)
    left, top = max(ax0, bx0), max(ay0, by0)
    right, bottom = min(ax0 + size, bx0 + size), min(ay0 + size, by0 + size)
    if left >= right or top >= bottom:
        return 0.0, 0.0
    a = first_mask[top - ay0 : bottom - ay0, left - ax0 : right - ax0]
    b = second_mask[top - by0 : bottom - by0, left - bx0 : right - bx0]
    intersection = int(np.logical_and(a, b).sum())
    if intersection == 0:
        return 0.0, 0.0
    area_a, area_b = int(first_mask.sum()), int(second_mask.sum())
    return intersection / max(area_a + area_b - intersection, 1), intersection / max(min(area_a, area_b), 1)


def _row_float(row: pd.Series, name: str, default: float) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _dark_core_gap_score(
    raw: np.ndarray,
    first: pd.Series,
    second: pd.Series,
) -> float:
    """Measure whether two dark candidate cores have a background-like gap.

    Duplicate proposals on one cell normally have a dark path between their
    seeds. Separate phase-contrast cells instead have two dark cores with a
    bright saddle between them. The score is normalized to local contrast so
    it remains useful across plates and exposure levels.
    """

    first_x = _row_float(first, "x_px", np.nan)
    first_y = _row_float(first, "y_px", np.nan)
    second_x = _row_float(second, "x_px", np.nan)
    second_y = _row_float(second, "y_px", np.nan)
    if not all(np.isfinite(value) for value in (first_x, first_y, second_x, second_y)):
        return 0.0
    distance = float(np.hypot(first_x - second_x, first_y - second_y))
    if distance < 4.0:
        return 0.0

    height, width = raw.shape
    pad = max(6, int(np.ceil(distance * 0.75)))
    left = max(0, int(np.floor(min(first_x, second_x))) - pad)
    right = min(width, int(np.ceil(max(first_x, second_x))) + pad + 1)
    top = max(0, int(np.floor(min(first_y, second_y))) - pad)
    bottom = min(height, int(np.ceil(max(first_y, second_y))) + pad + 1)
    local = raw[top:bottom, left:right].astype(np.float32)
    if local.size < 16:
        return 0.0
    low, high = np.percentile(local, [5, 90])
    scale = max(float(high - low), 8.0)

    def core_level(x: float, y: float) -> float:
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(0, cx - 2), min(width, cx + 3)
        y0, y1 = max(0, cy - 2), min(height, cy + 3)
        patch = raw[y0:y1, x0:x1].astype(np.float32)
        if not patch.size:
            return float(high)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        values = patch[(xx - x) ** 2 + (yy - y) ** 2 <= 2.25**2]
        return float(np.median(values if values.size else patch))

    sample_count = max(9, int(np.ceil(distance * 1.5)))
    fractions = np.linspace(0.28, 0.72, sample_count)
    line_values = []
    for fraction in fractions:
        x = int(round(first_x * (1.0 - fraction) + second_x * fraction))
        y = int(round(first_y * (1.0 - fraction) + second_y * fraction))
        if 0 <= x < width and 0 <= y < height:
            line_values.append(float(raw[y, x]))
    if len(line_values) < 3:
        return 0.0
    gap_level = float(np.percentile(line_values, 75))
    darker_core_ceiling = max(core_level(first_x, first_y), core_level(second_x, second_y))
    contrast = (gap_level - darker_core_ceiling) / scale
    background_fraction = (gap_level - float(low)) / scale
    if background_fraction < 0.55:
        return 0.0
    return max(float(contrast), 0.0)


def _strong_single_candidate(row: pd.Series, settings: dict[str, Any]) -> bool:
    return bool(
        str(row.get("integrated_label", "")) == "single"
        and _row_float(row, "cell_probability", 0.0)
        >= float(settings.get("competing_single_minimum_cell_probability", 0.90))
        and _row_float(row, "single_probability", 0.0)
        >= float(settings.get("competing_single_minimum_single_probability", 0.65))
        and _row_float(row, "invalid_probability", 1.0)
        <= float(settings.get("competing_single_maximum_invalid_probability", 0.10))
        and _row_float(row, "v2_objectness", 0.0)
        >= float(settings.get("competing_single_minimum_objectness", 0.75))
        and _row_float(row, "v2_instance_confidence", 0.0)
        >= float(settings.get("competing_single_minimum_instance_confidence", 0.35))
    )


def _split_competing_single_masks(
    frame: pd.DataFrame,
    patch_size: int,
    settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Split leaked masks when two strong single-cell seeds have a real gap."""

    output = frame.copy()
    output["v2_competing_seed_split"] = False
    output["v2_competing_seed_rivals"] = ""
    output["v2_competing_seed_aliases"] = ""
    output["v2_competing_seed_gap_score"] = 0.0
    output["v2_competing_seed_human_confirmed"] = False
    settings = settings or {}
    if not bool(settings.get("competing_single_split_enabled", True)):
        return output
    required = {
        "raw_image_path", "cell_probability", "single_probability",
        "invalid_probability", "v2_objectness", "v2_instance_diameter_px",
    }
    if not required.issubset(output.columns):
        return output

    image_cache: dict[str, np.ndarray] = {}
    minimum_ratio = float(settings.get("competing_single_minimum_center_diameter_ratio", 0.65))
    maximum_ratio = float(settings.get("competing_single_maximum_center_diameter_ratio", 1.60))
    minimum_gap = float(settings.get("competing_single_minimum_gap_contrast", 0.22))
    minimum_area_ratio = float(settings.get("competing_single_minimum_area_ratio", 0.25))
    confirmed_groups = [
        {str(candidate_id) for candidate_id in group if str(candidate_id)}
        for group in settings.get("competing_single_confirmed_split_groups", [])
        if isinstance(group, (list, tuple, set)) and len(group) >= 2
    ]
    excluded_candidate_ids = {
        str(candidate_id)
        for candidate_id in settings.get("competing_single_excluded_candidate_ids", [])
        if str(candidate_id)
    }

    valid = output[output["v2_mask_valid"] & ~output["v2_wall_rejected"]]
    for (_, _), group in valid.groupby(["well", "timepoint"], sort=False):
        group_ids = set(group["candidate_id"].astype(str))
        group_confirmed_groups = [
            confirmed_group
            for confirmed_group in confirmed_groups
            if confirmed_group.issubset(group_ids)
        ]
        group_confirmed_ids = (
            set().union(*group_confirmed_groups) if group_confirmed_groups else set()
        )
        strong_indices = [
            index for index, row in group.iterrows()
            if str(row.get("candidate_id", "")) not in excluded_candidate_ids
            and (
                _strong_single_candidate(row, settings)
                or str(row.get("candidate_id", "")) in group_confirmed_ids
            )
        ]
        if len(strong_indices) < 2:
            continue
        edges: dict[int, set[int]] = {index: set() for index in strong_indices}
        pair_scores: dict[tuple[int, int], float] = {}
        for position, first_index in enumerate(strong_indices):
            first = output.loc[first_index]
            for second_index in strong_indices[position + 1 :]:
                second = output.loc[second_index]
                first_id = str(first.get("candidate_id", ""))
                second_id = str(second.get("candidate_id", ""))
                human_confirmed_pair = any(
                    {first_id, second_id}.issubset(confirmed_group)
                    for confirmed_group in group_confirmed_groups
                )
                distance = float(np.hypot(
                    _row_float(first, "x_px", 0.0) - _row_float(second, "x_px", 0.0),
                    _row_float(first, "y_px", 0.0) - _row_float(second, "y_px", 0.0),
                ))
                diameter = max(
                    _row_float(first, "v2_instance_diameter_px", 0.0),
                    _row_float(second, "v2_instance_diameter_px", 0.0),
                    1.0,
                )
                ratio = distance / diameter
                if not human_confirmed_pair and not minimum_ratio <= ratio <= maximum_ratio:
                    continue
                iou, containment = _global_overlap(first, second, patch_size)
                if (
                    not human_confirmed_pair
                    and iou < 0.35
                    and containment < 0.65
                ):
                    continue
                first_path = str(first.get("raw_image_path", ""))
                second_path = str(second.get("raw_image_path", ""))
                if not first_path or first_path != second_path:
                    continue
                if first_path not in image_cache:
                    path = Path(first_path)
                    if not path.exists():
                        continue
                    with Image.open(path) as image:
                        image_cache[first_path] = np.asarray(image.convert("L"), dtype=np.uint8)
                score = _dark_core_gap_score(image_cache[first_path], first, second)
                if not human_confirmed_pair and score < minimum_gap:
                    continue
                edges[first_index].add(second_index)
                edges[second_index].add(first_index)
                pair_scores[(min(first_index, second_index), max(first_index, second_index))] = score

        pending = {index for index, rivals in edges.items() if rivals}
        while pending:
            start = pending.pop()
            component = {start}
            frontier = [start]
            while frontier:
                current = frontier.pop()
                for neighbor in edges[current]:
                    if neighbor not in component:
                        component.add(neighbor)
                        pending.discard(neighbor)
                        frontier.append(neighbor)
            if len(component) < 2:
                continue
            indices = sorted(component)
            rows = [output.loc[index] for index in indices]
            paths = {str(row.get("raw_image_path", "")) for row in rows}
            if len(paths) != 1 or not next(iter(paths)):
                continue
            raw = image_cache[next(iter(paths))]
            left = min(int(row.v2_mask_origin_x) for row in rows)
            top = min(int(row.v2_mask_origin_y) for row in rows)
            right = max(int(row.v2_mask_origin_x) + patch_size for row in rows)
            bottom = max(int(row.v2_mask_origin_y) + patch_size for row in rows)
            union = np.zeros((bottom - top, right - left), dtype=bool)
            originals: dict[int, np.ndarray] = {}
            for index, row in zip(indices, rows):
                mask = decode_rle(str(row.v2_mask_rle), patch_size)
                originals[index] = mask
                x0 = int(row.v2_mask_origin_x) - left
                y0 = int(row.v2_mask_origin_y) - top
                union[y0 : y0 + patch_size, x0 : x0 + patch_size] |= mask
            if not union.any():
                continue

            # Candidate generation can place both a raw and a CF seed on the
            # same dark core. If each proposal became a watershed marker, a
            # two-cell scene could be split into three instances. Collapse
            # near-coincident seeds first, then give every alias the same
            # refined region so ordinary duplicate suppression keeps one.
            seed_clusters: list[list[int]] = []
            for index in indices:
                row = output.loc[index]
                assigned_cluster: list[int] | None = None
                for cluster in seed_clusters:
                    representative = output.loc[cluster[0]]
                    distance = float(np.hypot(
                        _row_float(row, "x_px", 0.0)
                        - _row_float(representative, "x_px", 0.0),
                        _row_float(row, "y_px", 0.0)
                        - _row_float(representative, "y_px", 0.0),
                    ))
                    diameter = min(
                        _row_float(row, "v2_instance_diameter_px", 1.0),
                        _row_float(representative, "v2_instance_diameter_px", 1.0),
                    )
                    if distance <= max(4.0, 0.30 * diameter):
                        assigned_cluster = cluster
                        break
                if assigned_cluster is None:
                    seed_clusters.append([index])
                else:
                    assigned_cluster.append(index)
            if len(seed_clusters) < 2:
                continue

            def seed_quality(index: int) -> tuple[float, float]:
                row = output.loc[index]
                evidence = (
                    _row_float(row, "cell_probability", 0.0)
                    * _row_float(row, "single_probability", 0.0)
                    * _row_float(row, "v2_objectness", 0.0)
                    * _row_float(row, "v2_instance_confidence", 0.0)
                )
                return evidence, _row_float(row, "v2_instance_area_px", 0.0)

            representatives = [max(cluster, key=seed_quality) for cluster in seed_clusters]
            cluster_by_index = {
                index: cluster_number
                for cluster_number, cluster in enumerate(seed_clusters)
                for index in cluster
            }
            component_ids = {
                str(output.at[index, "candidate_id"]): index for index in indices
            }
            matched_confirmed_groups = [
                group for group in confirmed_groups if group.issubset(component_ids)
            ]
            confirmed_ids = set().union(*matched_confirmed_groups) if matched_confirmed_groups else set()

            raw_canvas = np.full(union.shape, float(np.median(raw)), dtype=np.float32)
            source_left, source_top = max(left, 0), max(top, 0)
            source_right, source_bottom = min(right, raw.shape[1]), min(bottom, raw.shape[0])
            if source_left >= source_right or source_top >= source_bottom:
                continue
            raw_canvas[
                source_top - top : source_bottom - top,
                source_left - left : source_right - left,
            ] = raw[source_top:source_bottom, source_left:source_right]
            markers = np.zeros(union.shape, dtype=np.int32)
            marker_ids: dict[int, int] = {}
            used_positions: set[tuple[int, int]] = set()
            union_points = np.argwhere(union)
            for marker_id, index in enumerate(representatives, start=1):
                row = output.loc[index]
                x = int(round(_row_float(row, "x_px", 0.0))) - left
                y = int(round(_row_float(row, "y_px", 0.0))) - top
                if not (0 <= x < union.shape[1] and 0 <= y < union.shape[0]) or not union[y, x]:
                    nearest = union_points[
                        np.argmin((union_points[:, 0] - y) ** 2 + (union_points[:, 1] - x) ** 2)
                    ]
                    y, x = int(nearest[0]), int(nearest[1])
                if (y, x) in used_positions:
                    continue
                used_positions.add((y, x))
                markers[y, x] = marker_id
                marker_ids[index] = marker_id
            if len(marker_ids) != len(representatives):
                continue
            # A watershed line is intentionally retained as background. The
            # pair passed a bright-gap gate, so preserving a one-pixel divider
            # prevents the two reviewed contours from touching again merely
            # because the original leaked union contained a thin bridge.
            labels = watershed(
                raw_canvas,
                markers=markers,
                mask=union,
                watershed_line=True,
            )
            separator_radius = int(
                settings.get("competing_single_separator_radius_px", 1)
            )
            watershed_line = union & (labels == 0)
            if separator_radius > 0 and watershed_line.any():
                labels[dilation(watershed_line, disk(separator_radius))] = 0
            replacements: dict[int, np.ndarray] = {}
            acceptable = True
            assigned_by_cluster: dict[int, np.ndarray] = {}
            for cluster_number, representative_index in enumerate(representatives):
                row = output.loc[representative_index]
                assigned = labels == marker_ids[representative_index]
                x0 = int(row.v2_mask_origin_x) - left
                y0 = int(row.v2_mask_origin_y) - top
                local = assigned[y0 : y0 + patch_size, x0 : x0 + patch_size]
                yy, xx = np.mgrid[:patch_size, :patch_size]
                seed_x = _row_float(row, "x_px", patch_size / 2) - int(row.v2_mask_origin_x)
                seed_y = _row_float(row, "y_px", patch_size / 2) - int(row.v2_mask_origin_y)
                seed_weights = np.exp(-((xx - seed_x) ** 2 + (yy - seed_y) ** 2) / (2 * 4.0**2))
                local = _selected_component(local, seed_weights)
                cluster_ids = {
                    str(output.at[index, "candidate_id"])
                    for index in seed_clusters[cluster_number]
                }
                area_ratio = 0.0 if cluster_ids & confirmed_ids else minimum_area_ratio
                if int(local.sum()) < max(
                    3, int(originals[representative_index].sum() * area_ratio)
                ):
                    acceptable = False
                    break
                assigned_by_cluster[cluster_number] = assigned
            if not acceptable:
                continue
            for index in indices:
                row = output.loc[index]
                assigned = assigned_by_cluster[cluster_by_index[index]]
                x0 = int(row.v2_mask_origin_x) - left
                y0 = int(row.v2_mask_origin_y) - top
                local = assigned[y0 : y0 + patch_size, x0 : x0 + patch_size]
                seed_x = _row_float(row, "x_px", patch_size / 2) - int(row.v2_mask_origin_x)
                seed_y = _row_float(row, "y_px", patch_size / 2) - int(row.v2_mask_origin_y)
                yy, xx = np.mgrid[:patch_size, :patch_size]
                seed_weights = np.exp(-((xx - seed_x) ** 2 + (yy - seed_y) ** 2) / (2 * 4.0**2))
                replacements[index] = _selected_component(local, seed_weights)
            for index, row in zip(indices, rows):
                mask = replacements[index]
                area = int(mask.sum())
                own_cluster = seed_clusters[cluster_by_index[index]]
                aliases = sorted(
                    str(output.at[alias, "candidate_id"])
                    for alias in own_cluster
                    if alias != index
                )
                rivals = sorted(
                    str(output.at[rival, "candidate_id"])
                    for rival in indices
                    if cluster_by_index[rival] != cluster_by_index[index]
                )
                scores = [
                    score
                    for pair, score in pair_scores.items()
                    if (
                        pair[0] in own_cluster
                        and pair[1] in component
                        and cluster_by_index[pair[1]] != cluster_by_index[index]
                    )
                    or (
                        pair[1] in own_cluster
                        and pair[0] in component
                        and cluster_by_index[pair[0]] != cluster_by_index[index]
                    )
                ]
                output.at[index, "v2_mask_rle"] = _rle(mask)
                output.at[index, "v2_contour_json"] = _contour(
                    mask, int(row.v2_mask_origin_x), int(row.v2_mask_origin_y)
                )
                output.at[index, "v2_instance_area_px"] = area
                output.at[index, "v2_instance_diameter_px"] = float(2.0 * np.sqrt(area / np.pi))
                output.at[index, "v2_competing_seed_split"] = True
                output.at[index, "v2_competing_seed_rivals"] = "|".join(rivals)
                output.at[index, "v2_competing_seed_aliases"] = "|".join(aliases)
                output.at[index, "v2_competing_seed_gap_score"] = max(scores, default=0.0)
                output.at[index, "v2_competing_seed_human_confirmed"] = bool(
                    str(row.get("candidate_id", "")) in confirmed_ids
                )
    return output


def consolidate_v2_masks(
    frame: pd.DataFrame,
    patch_size: int,
    inference_settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    output = _split_competing_single_masks(output, patch_size, inference_settings)
    output["v2_is_suppressed"] = False
    output["v2_suppressed_by"] = ""
    output["v2_suppression_reason"] = ""
    output["v2_instance_id"] = ""
    valid = output[output["v2_mask_valid"] & ~output["v2_wall_rejected"]]
    for (_, _), group in valid.groupby(["well", "timepoint"], sort=False):
        priority = group["integrated_label"].map(
            {"cluster_3plus": 3, "touching_doublet": 2, "single": 1, "debris": 1, "uncertain": 0, "invalid": -1}
        ).fillna(-1)
        # A human-confirmed split is more authoritative than an overlapping
        # unsplit cluster/doublet proposal. Process the reviewed child masks
        # first so the coarse parent is suppressed rather than erasing both.
        if "v2_competing_seed_human_confirmed" in group:
            human_confirmed = group["v2_competing_seed_human_confirmed"].fillna(False).astype(bool)
            priority = priority.where(~human_confirmed, 4)
        ordered = group.assign(_group_priority=priority).sort_values(
            ["_group_priority", "v2_instance_confidence", "integrated_confidence"], ascending=False
        )
        keepers: list[int] = []
        for index, row in ordered.iterrows():
            owner = None
            reason = ""
            for kept_index in keepers:
                kept = output.loc[kept_index]
                iou, containment = _global_overlap(row, kept, patch_size)
                group_labels = {str(row.integrated_label), str(kept.integrated_label)}
                threshold = 0.22 if group_labels & {"touching_doublet", "cluster_3plus"} else 0.35
                if iou >= threshold or containment >= 0.65:
                    owner = kept_index
                    reason = f"mask_overlap:iou={iou:.3f},containment={containment:.3f}"
                    break
            if owner is None:
                keepers.append(index)
                output.at[index, "v2_instance_id"] = f"{row.well}-{row.timepoint}-I{len(keepers):03d}"
            else:
                output.at[index, "v2_is_suppressed"] = True
                output.at[index, "v2_suppressed_by"] = str(output.at[owner, "candidate_id"])
                output.at[index, "v2_suppression_reason"] = reason
                output.at[index, "v2_instance_id"] = str(output.at[owner, "v2_instance_id"])
    parent_suppressions = (inference_settings or {}).get(
        "competing_single_confirmed_parent_suppressions", {}
    )
    if isinstance(parent_suppressions, dict):
        candidate_indices = {
            str(candidate_id): index
            for index, candidate_id in output["candidate_id"].items()
        }
        for parent_id, child_id in parent_suppressions.items():
            parent_index = candidate_indices.get(str(parent_id))
            child_index = candidate_indices.get(str(child_id))
            if parent_index is None or child_index is None:
                continue
            parent = output.loc[parent_index]
            child = output.loc[child_index]
            if (
                str(parent.get("well", "")) != str(child.get("well", ""))
                or str(parent.get("timepoint", "")) != str(child.get("timepoint", ""))
                or bool(child.get("v2_is_suppressed", False))
            ):
                continue
            output.at[parent_index, "v2_is_suppressed"] = True
            output.at[parent_index, "v2_suppressed_by"] = str(child_id)
            output.at[parent_index, "v2_suppression_reason"] = "human_confirmed_split_parent"
            output.at[parent_index, "v2_instance_id"] = str(child.get("v2_instance_id", ""))
    return output


def _reviewed_positive_ids(config: dict[str, Any], frame: pd.DataFrame | None = None) -> set[str]:
    # External validation/review rounds can explicitly require pure model
    # output.  In that mode historical annotations remain archived for
    # evaluation, but must not alter the segmentation gate or final labels.
    if not config.get("review_queue", {}).get("apply_manual_point_overrides", True):
        return set()
    database = artifact_path(config, "annotations", "annotations.db")
    if not database.exists():
        return set()
    try:
        with sqlite3.connect(database) as connection:
            reviews = pd.read_sql_query(
                "SELECT candidate_id, reviewed_label, updated_at FROM integrated_training_reviews ORDER BY updated_at, integrated_review_id",
                connection,
            ).drop_duplicates("candidate_id", keep="last")
            manual = pd.read_sql_query(
                "SELECT candidate_id, well, timepoint, x_px, y_px, reviewed_label, updated_at FROM quick_missed_objects ORDER BY updated_at, quick_missed_id",
                connection,
            ).drop_duplicates("candidate_id", keep="last")
    except Exception:
        return set()
    protected = set(
        reviews.loc[
            reviews["reviewed_label"].isin(["single", "touching_doublet", "cluster_3plus", "debris"]),
            "candidate_id",
        ].astype(str)
    )
    if frame is not None and not manual.empty:
        manual = manual[manual["reviewed_label"].isin(["single", "touching_doublet", "cluster_3plus", "debris"])]
        for point in manual.itertuples(index=False):
            local = frame[(frame["well"] == point.well) & (frame["timepoint"] == point.timepoint)]
            if local.empty:
                continue
            distances = np.hypot(local["x_px"].astype(float) - float(point.x_px), local["y_px"].astype(float) - float(point.y_px))
            protected.update(local.loc[distances <= 20.0, "candidate_id"].astype(str))
    return protected


def finalize_v2_instances(
    enriched: pd.DataFrame,
    patch_size: int,
    protected_positive_ids: set[str] | None = None,
    inference_settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    output = enriched.copy()
    protected_positive_ids = protected_positive_ids or set()
    if "v2_original_integrated_label" not in output:
        output["v2_original_integrated_label"] = output["integrated_label"]
    else:
        # Makes re-finalization deterministic after threshold/label-policy changes.
        output["integrated_label"] = output["v2_original_integrated_label"]
    automatic_invalid = (
        output["invalid_probability"].fillna(0).astype(float).ge(0.60)
        & ~output["candidate_id"].astype(str).isin(protected_positive_ids)
    )
    output.loc[automatic_invalid, "integrated_label"] = "invalid"
    output["v2_auto_invalid_probability_rule"] = automatic_invalid
    recoverable = (
        output["v2_mask_valid"]
        & ~output["v2_wall_rejected"]
        & ~automatic_invalid
        & output["integrated_label"].isin(["invalid", "uncertain", "unmarked"])
        & (output["v2_instance_confidence"] >= 0.72)
    )
    # A compact instance can have a conservative boundary confidence while
    # the independent morphology and presence heads are unequivocally cell.
    # Keep this rescue narrow so tiered inference cannot revive wall texture.
    strong_cell_recoverable = (
        output["v2_mask_valid"]
        & ~output["v2_wall_rejected"]
        & ~automatic_invalid
        & output["integrated_label"].isin(["invalid", "uncertain", "unmarked"])
        & output["cell_probability"].fillna(0).astype(float).ge(0.90)
        & output["invalid_probability"].fillna(1).astype(float).lt(0.10)
        & output["v2_objectness"].fillna(0).astype(float).ge(0.75)
        & output["v2_instance_confidence"].fillna(0).astype(float).ge(0.35)
    )
    recoverable |= strong_cell_recoverable
    cell_probability = output["cell_probability"].fillna(0).astype(float)
    debris_probability = output["debris_probability"].fillna(0).astype(float)
    invalid_probability = output["invalid_probability"].fillna(0).astype(float)
    # A multiplicity head answers only how many cells an object would contain
    # *if it is a cell*.  It must not turn a debris-dominant object into a
    # single/doublet merely because the instance mask is recoverable.
    cell_rescue = (
        recoverable
        & cell_probability.ge(0.28)
        & cell_probability.ge(debris_probability)
        & cell_probability.ge(invalid_probability)
    )
    debris_rescue = (
        recoverable
        & ~cell_rescue
        & debris_probability.ge(cell_probability)
        & debris_probability.ge(invalid_probability)
    )
    if "predicted_multiplicity" in output:
        rescued = output.loc[cell_rescue, "predicted_multiplicity"].fillna("single")
        rescued = rescued.where(rescued.isin(["single", "touching_doublet", "cluster_3plus"]), "single")
        output.loc[cell_rescue, "integrated_label"] = rescued
    else:
        output.loc[cell_rescue, "integrated_label"] = "single"
    output.loc[debris_rescue, "integrated_label"] = "debris"
    output.loc[recoverable & ~cell_rescue & ~debris_rescue, "integrated_label"] = "uncertain"
    # "Uncertain" is reserved for the biologically relevant cell-vs-noncell
    # boundary.  Once cell probability is low, debris-vs-invalid ambiguity is
    # resolved deterministically and should not consume human review time.
    low_cell_noncell = (
        output["integrated_label"].isin(["uncertain", "unmarked"])
        & output["cell_probability"].fillna(0).astype(float).lt(
            LOW_CELL_NONCELL_RESOLUTION_THRESHOLD
        )
    )
    low_cell_debris = low_cell_noncell & (
        output["debris_probability"].fillna(0).astype(float)
        >= output["invalid_probability"].fillna(0).astype(float)
    )
    output.loc[low_cell_debris, "integrated_label"] = "debris"
    output.loc[low_cell_noncell & ~low_cell_debris, "integrated_label"] = "invalid"
    output["v2_low_cell_noncell_resolved"] = low_cell_noncell
    output["v2_noncell_resolution_label"] = np.where(
        low_cell_debris,
        "debris",
        np.where(low_cell_noncell, "invalid", ""),
    )
    manual_label_overrides = (inference_settings or {}).get(
        "manual_candidate_label_overrides", {}
    )
    output["v2_manual_label_override"] = False
    output["v2_manual_label_override_value"] = ""
    if isinstance(manual_label_overrides, dict):
        allowed_labels = {
            "single", "touching_doublet", "cluster_3plus", "debris", "invalid"
        }
        for candidate_id, label in manual_label_overrides.items():
            normalized_label = str(label).strip().lower()
            if normalized_label not in allowed_labels:
                continue
            selected = output["candidate_id"].astype(str) == str(candidate_id)
            output.loc[selected, "integrated_label"] = normalized_label
            output.loc[selected, "v2_manual_label_override"] = True
            output.loc[selected, "v2_manual_label_override_value"] = normalized_label
    output["v2_rescued_from_invalid"] = recoverable & (output["v2_original_integrated_label"] == "invalid")
    output = consolidate_v2_masks(output, patch_size, inference_settings)
    # Keep instance identity, temporal eligibility, review visibility, and
    # final counting as separate states.  A valid low-confidence object must
    # remain available to the temporal model even when it is not yet suitable
    # for formal counting or review.
    output["v2_is_unique_instance"] = (
        output["v2_mask_valid"]
        & ~output["v2_wall_rejected"]
        & ~output["v2_is_suppressed"]
    )
    output["v2_is_temporal_candidate"] = (
        output["v2_is_unique_instance"]
        & ~automatic_invalid
    )
    output["v2_is_reviewable_instance"] = (
        output["v2_is_unique_instance"]
        & output["integrated_label"].isin(
            ["single", "touching_doublet", "cluster_3plus", "debris", "uncertain"]
        )
    )
    output["v2_is_counting_instance"] = (
        output["v2_is_unique_instance"]
        & output["integrated_label"].isin(
            ["single", "touching_doublet", "cluster_3plus", "debris"]
        )
    )
    return output


def refinalize_v2_file(config: dict[str, Any], patch_size: int = 96) -> Path:
    output_path = artifact_path(config, "predictions", "latest_v2_predictions.csv")
    source_frame = pd.read_csv(output_path, low_memory=False)
    frame = finalize_v2_instances(
        source_frame,
        patch_size,
        _reviewed_positive_ids(config, source_frame),
        config.get("v2_inference", {}),
    )
    summary_path = output_path.with_suffix(".json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = Path(summary.get("checkpoint", ""))
        if checkpoint.exists():
            frame["integrated_round_id"] = "v2-round-" + datetime.fromtimestamp(checkpoint.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
    frame.to_csv(output_path, index=False)
    return output_path


def infer_v2_instances(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    predictions_path: str | Path | None = None,
    output_path: str | Path | None = None,
    pre_temporal_output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> Path:
    """Run V2 inference, optionally writing to an isolated versioned output.

    The default paths intentionally remain the historical ``latest_*`` paths
    so existing production callers keep their behavior.  Review workflows can
    pass explicit paths to create a reproducible batch without overwriting the
    active prediction artifacts.
    """

    predictions_path = Path(predictions_path) if predictions_path is not None else artifact_path(
        config, "predictions", "latest_integrated_predictions.csv"
    )
    output = Path(output_path) if output_path is not None else artifact_path(
        config, "predictions", "latest_v2_predictions.csv"
    )
    pre_temporal_output = (
        Path(pre_temporal_output_path)
        if pre_temporal_output_path is not None
        else output.with_name("latest_v2_pre_temporal_predictions.csv")
    )
    summary_path = Path(summary_path) if summary_path is not None else output.with_suffix(".json")
    for destination in (output, pre_temporal_output, summary_path):
        destination.parent.mkdir(parents=True, exist_ok=True)
    inference_settings = config.get("v2_inference", {})
    fingerprint = _inference_fingerprint(config, predictions_path, checkpoint_path)
    if bool(inference_settings.get("reuse_unchanged_stage", True)) and pre_temporal_output.exists() and summary_path.exists():
        try:
            cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached_summary = {}
        if cached_summary.get("inference_fingerprint") == fingerprint:
            shutil.copyfile(pre_temporal_output, output)
            return output

    started = time.perf_counter()
    frame = pd.read_csv(predictions_path, low_memory=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    size = int(checkpoint["patch_size_px"])
    sigma = float(checkpoint["seed_sigma_px"])
    threshold = float(checkpoint.get("threshold", 0.5))
    wall_threshold = float(checkpoint.get("wall_threshold", 0.5))
    presence_threshold = float(checkpoint.get("presence_threshold", 0.55))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeededInstanceUNet(int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    protected_positive_ids = _reviewed_positive_ids(config, frame)
    hard_invalid_threshold = float(inference_settings.get("hard_invalid_skip_threshold", 0.60))
    precise_cell_threshold = float(inference_settings.get("precise_contour_cell_probability", 0.12))
    refinement_minimum_area_ratio = float(
        inference_settings.get("refinement_minimum_area_ratio", 0.80)
    )
    refinement_minimum_iou = float(
        inference_settings.get("refinement_minimum_iou", 0.70)
    )
    group_component_gap_px = int(
        inference_settings.get("group_component_gap_px", 3)
    )
    precise_labels = {
        str(value)
        for value in inference_settings.get(
            "precise_contour_labels",
            ["single", "touching_doublet", "cluster_3plus", "uncertain"],
        )
    }
    precise_sources = {
        str(value)
        for value in inference_settings.get(
            "precise_contour_sources",
            ["manual_cell_anchor", "manual_annotation_anchor", "wall_cell_rescue_peak", "wall_residual_peak"],
        )
    }
    skipped_hard_invalid = 0
    precise_refinement_count = 0
    lightweight_refinement_count = 0

    columns: dict[str, list[Any]] = {
        "v2_mask_valid": [], "v2_mask_rle": [], "v2_mask_origin_x": [], "v2_mask_origin_y": [],
        "v2_contour_json": [], "v2_instance_area_px": [], "v2_instance_diameter_px": [],
        "v2_instance_confidence": [], "v2_objectness": [], "v2_wall_overlap": [], "v2_wall_rejected": [],
        "v2_refinement_status": [], "v2_refinement_area_ratio": [], "v2_refinement_iou": [],
    }
    seed = _seed_heatmap(size, sigma)
    for raw_path, group in frame.groupby("raw_image_path", sort=False):
        with Image.open(raw_path) as image:
            full = np.asarray(image.convert("L"), dtype=np.uint8)
        batch_values: list[np.ndarray] = []
        origins: list[tuple[int, int]] = []
        candidate_presence_thresholds: list[float] = []
        precise_refinement: list[bool] = []
        batch_positions: list[int] = []
        group_results: list[dict[str, Any] | None] = [None] * len(group)
        for position, row in enumerate(group.itertuples(index=False)):
            candidate_id = str(row.candidate_id)
            hard_invalid = (
                float(row.invalid_probability) >= hard_invalid_threshold
                and candidate_id not in protected_positive_ids
            )
            if hard_invalid:
                skipped_hard_invalid += 1
                group_results[position] = {
                    "v2_mask_valid": False,
                    "v2_mask_rle": "[]",
                    "v2_mask_origin_x": int(round(row.x_px)) - size // 2,
                    "v2_mask_origin_y": int(round(row.y_px)) - size // 2,
                    "v2_contour_json": "[]",
                    "v2_instance_area_px": 0,
                    "v2_instance_diameter_px": 0.0,
                    "v2_instance_confidence": 0.0,
                    "v2_objectness": 0.0,
                    "v2_wall_overlap": 0.0,
                    "v2_wall_rejected": True,
                    "v2_refinement_status": "skipped_hard_invalid",
                    "v2_refinement_area_ratio": 0.0,
                    "v2_refinement_iou": 0.0,
                }
                continue
            raw = _crop(full, row.x_px, row.y_px, size, int(np.median(full)))
            inner = getattr(row, "detected_wall_inner_fraction", np.nan)
            if pd.isna(inner):
                inner = config.get("candidate_filter", {}).get("hard_wall_exclusion_fraction", 0.44)
            wall = _wall_prior(full.shape, row.x_px, row.y_px, size, float(inner))
            batch_values.append(np.stack([_normalize(raw), seed, wall]))
            origins.append((int(round(row.x_px)) - size // 2, int(round(row.y_px)) - size // 2))
            # Presence is a proposal-quality gate, not a second morphology
            # classifier.  Near-wall cells can have a low global presence mean
            # even when the object head contains a clean, compact mask.  Keep
            # the strict gate for background proposals, but allow a calibrated
            # low gate when independent morphology or human evidence already
            # says that a real object is present.  The >=0.60 invalid rule is
            # deliberately checked first, so texture negatives are not rescued.
            morphology_positive = (
                str(row.integrated_label)
                in {"single", "touching_doublet", "cluster_3plus", "debris", "uncertain"}
                and float(row.invalid_probability) < 0.60
            )
            has_positive_evidence = candidate_id in protected_positive_ids or morphology_positive
            candidate_presence_thresholds.append(min(presence_threshold, 0.08) if has_positive_evidence else presence_threshold)
            source = str(getattr(row, "candidate_source", ""))
            precise_refinement.append(
                candidate_id in protected_positive_ids
                or float(row.cell_probability) >= precise_cell_threshold
                or str(row.integrated_label) in precise_labels
                or source in precise_sources
            )
            batch_positions.append(position)
        for start in range(0, len(batch_values), 128):
            tensor = torch.from_numpy(np.asarray(batch_values[start : start + 128], dtype=np.float32)).to(device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probabilities = torch.sigmoid(model(tensor)).float().cpu().numpy()
            for offset, probability in enumerate(probabilities):
                batch_index = start + offset
                object_probability, wall_probability, presence_probability = probability
                objectness = float(presence_probability.mean())
                local_presence_threshold = candidate_presence_thresholds[batch_index]
                position = batch_positions[batch_index]
                row = group.iloc[position]
                retain_nearby = str(row.integrated_label) in {
                    "touching_doublet", "cluster_3plus"
                }
                binary = object_probability >= threshold
                binary &= ~((wall_probability >= max(wall_threshold, 0.82)) & (object_probability < 0.72))
                mask = _selected_component(
                    binary,
                    seed,
                    retain_nearby=retain_nearby,
                    nearby_distance=group_component_gap_px,
                )
                refinement_diagnostics: dict[str, Any] = {
                    "status": "not_run",
                    "area_ratio": 1.0 if mask.any() else 0.0,
                    "iou": 1.0 if mask.any() else 0.0,
                }
                if objectness >= local_presence_threshold and mask.any():
                    if precise_refinement[batch_index]:
                        precise_refinement_count += 1
                        mask = refine_instance_mask(
                            mask,
                            batch_values[batch_index][0],
                            wall_probability,
                            seed,
                            retain_nearby=retain_nearby,
                            nearby_distance=group_component_gap_px,
                            minimum_area_ratio=refinement_minimum_area_ratio,
                            minimum_iou=refinement_minimum_iou,
                            diagnostics=refinement_diagnostics,
                        )
                    else:
                        lightweight_refinement_count += 1
                        mask = lightweight_refine_instance_mask(mask)
                        refinement_diagnostics["status"] = "lightweight"
                regions = regionprops(mask.astype(np.uint8))
                origin_x, origin_y = origins[batch_index]
                valid = bool(objectness >= local_presence_threshold and regions and 3 <= regions[0].area <= 2200)
                if valid:
                    region = regions[0]
                    area = int(region.area)
                    diameter = float(region.equivalent_diameter_area)
                    confidence = float(object_probability[mask].mean())
                    overlap = float((wall_probability[mask] >= wall_threshold).mean())
                    elongated = bool(region.eccentricity > 0.97 or region.solidity < 0.25)
                    wall_rejected = bool(overlap > 0.86 and elongated and confidence < 0.88)
                else:
                    area, diameter, confidence, overlap, wall_rejected = 0, 0.0, 0.0, 0.0, True
                group_results[batch_positions[batch_index]] = {
                    "v2_mask_valid": valid,
                    "v2_mask_rle": _rle(mask) if valid else "[]",
                    "v2_mask_origin_x": origin_x,
                    "v2_mask_origin_y": origin_y,
                    "v2_contour_json": _contour(mask, origin_x, origin_y) if valid else "[]",
                    "v2_instance_area_px": area,
                    "v2_instance_diameter_px": diameter,
                    "v2_instance_confidence": confidence,
                    "v2_objectness": objectness,
                    "v2_wall_overlap": overlap,
                    "v2_wall_rejected": wall_rejected,
                    "v2_refinement_status": str(refinement_diagnostics["status"]),
                    "v2_refinement_area_ratio": float(refinement_diagnostics["area_ratio"]),
                    "v2_refinement_iou": float(refinement_diagnostics["iou"]),
                }
        for result in group_results:
            if result is None:
                raise RuntimeError(f"V2 inference did not produce a result for {raw_path}")
            for name in columns:
                columns[name].append(result[name])
    # groupby processing changes row order; align through an explicit grouped index list
    ordered_indices = [index for _, group in frame.groupby("raw_image_path", sort=False) for index in group.index]
    enriched = frame.loc[ordered_indices].copy()
    for name, values in columns.items():
        enriched[name] = values
    enriched = enriched.sort_index()
    enriched = finalize_v2_instances(
        enriched,
        size,
        _reviewed_positive_ids(config, enriched),
        inference_settings,
    )
    enriched["integrated_round_id"] = "v2-round-" + datetime.fromtimestamp(Path(checkpoint_path).stat().st_mtime).strftime("%Y%m%d-%H%M%S")
    enriched.to_csv(pre_temporal_output, index=False)
    enriched.to_csv(output, index=False)
    summary = {
        "algorithm_version": "v2-tiered",
        "source_predictions": str(predictions_path),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "inference_fingerprint": fingerprint,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "candidate_count": int(len(enriched)),
        "valid_instance_count": int(enriched["v2_mask_valid"].sum()),
        "wall_rejected_count": int(enriched["v2_wall_rejected"].sum()),
        "duplicate_or_covered_count": int(enriched["v2_is_suppressed"].sum()),
        "competing_seed_split_count": int(enriched["v2_competing_seed_split"].sum()),
        "unique_instance_count": int(enriched["v2_is_unique_instance"].sum()),
        "temporal_candidate_count": int(enriched["v2_is_temporal_candidate"].sum()),
        "reviewable_instance_count": int(enriched["v2_is_reviewable_instance"].sum()),
        "counting_instance_count": int(enriched["v2_is_counting_instance"].sum()),
        "auto_invalid_probability_count": int(enriched["v2_auto_invalid_probability_rule"].sum()),
        "hard_invalid_segmentation_skipped": skipped_hard_invalid,
        "precise_contour_refinement_count": precise_refinement_count,
        "lightweight_contour_refinement_count": lightweight_refinement_count,
        "contour_refinement_status_counts": {
            str(key): int(value)
            for key, value in enriched["v2_refinement_status"].value_counts().items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output

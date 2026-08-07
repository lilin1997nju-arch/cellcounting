from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from skimage.metrics import structural_similarity

from .multiplicity_identity import candidate_multiplicity_label


CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}


def _crop_with_padding(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    half = size // 2
    left = int(round(x)) - half
    top = int(round(y)) - half
    result = np.full((size, size), float(np.median(image)), dtype=np.float32)
    source_left, source_top = max(0, left), max(0, top)
    source_right = min(image.shape[1], left + size)
    source_bottom = min(image.shape[0], top + size)
    if source_left >= source_right or source_top >= source_bottom:
        return result
    target_left, target_top = source_left - left, source_top - top
    result[
        target_top : target_top + source_bottom - source_top,
        target_left : target_left + source_right - source_left,
    ] = image[source_top:source_bottom, source_left:source_right]
    return result


def _descriptor(image: np.ndarray, x: float, y: float, size: int) -> dict[str, Any]:
    patch = _crop_with_padding(image, x, y, size)
    background = gaussian_filter(patch, sigma=max(3.0, size / 10.0))
    highpass = patch - background
    scale = max(float(np.percentile(np.abs(highpass), 90)), 1.0)
    normalized = np.clip(highpass / scale, -3.0, 3.0).astype(np.float32)

    magnitude = np.abs(highpass)
    threshold = max(float(np.percentile(magnitude, 84)), float(magnitude.mean() + 0.7 * magnitude.std()))
    yy, xx = np.ogrid[:size, :size]
    central = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= (size * 0.34) ** 2
    binary = (magnitude >= threshold) & central
    regions = regionprops(label(binary, connectivity=2))
    if regions:
        centre = np.asarray([size / 2, size / 2])
        chosen = min(
            regions,
            key=lambda region: float(np.linalg.norm(np.asarray(region.centroid) - centre))
            - 0.015 * float(region.area),
        )
        mask = np.zeros_like(binary)
        mask[tuple(chosen.coords.T)] = True
    else:
        mask = binary
    return {
        "patch": normalized,
        "mask": mask,
        "mask_area": int(mask.sum()),
    }


def _translated_overlap(left: np.ndarray, right: np.ndarray, maximum_shift: int = 3) -> float:
    best = 0.0
    for dy in range(-maximum_shift, maximum_shift + 1):
        for dx in range(-maximum_shift, maximum_shift + 1):
            shifted = np.roll(right, (dy, dx), axis=(0, 1))
            if dy > 0:
                shifted[:dy] = False
            elif dy < 0:
                shifted[dy:] = False
            if dx > 0:
                shifted[:, :dx] = False
            elif dx < 0:
                shifted[:, dx:] = False
            union = int(np.logical_or(left, shifted).sum())
            if union:
                best = max(best, float(np.logical_and(left, shifted).sum() / union))
    return best


def _appearance_similarity(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, float]:
    similarity = float(
        np.clip(
            structural_similarity(left["patch"], right["patch"], data_range=6.0),
            -1.0,
            1.0,
        )
    )
    overlap = _translated_overlap(left["mask"], right["mask"])
    return similarity, overlap


def refine_ambiguous_temporal_appearance(
    frame: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Use registered multi-timepoint appearance for ambiguous morphology.

    Only the configured 50--70% morphology band is eligible for automatic
    correction.  Near-identical centred pixels across time support debris;
    coherent shape/area/multiplicity change supports a cell.  Weak or
    conflicting evidence is retained for human review.
    """
    result = frame.copy()
    result["temporal_appearance_status"] = "not_evaluated"
    result["temporal_appearance_confidence"] = 0.0
    result["temporal_appearance_match_count"] = 0
    result["temporal_mean_patch_similarity"] = np.nan
    result["temporal_mean_mask_iou"] = np.nan
    result["temporal_max_area_ratio"] = np.nan
    result["temporal_appearance_matches"] = ""

    settings = config.get("temporal_appearance", {})
    if not bool(settings.get("enabled", True)) or result.empty:
        return result
    required = {"well", "timepoint", "x_px", "y_px", "raw_image_path", "cell_probability"}
    if not required <= set(result.columns):
        return result

    lower = float(settings.get("cell_probability_minimum", 0.50))
    upper = float(settings.get("cell_probability_maximum", 0.70))
    search_radius = float(settings.get("search_radius_px", 96.0))
    division_radius = float(settings.get("division_radius_px", 44.0))
    patch_size = int(settings.get("patch_size_px", 56))
    minimum_match = float(settings.get("minimum_match_score", 0.34))
    static_similarity = float(settings.get("static_patch_similarity", 0.86))
    static_iou = float(settings.get("static_mask_iou", 0.72))
    static_area_tolerance = float(settings.get("static_area_log_tolerance", 0.25))
    dynamic_area_ratio = float(settings.get("dynamic_area_ratio", 1.32))
    dynamic_iou = float(settings.get("dynamic_maximum_mask_iou", 0.68))
    static_debris_floor = float(settings.get("static_debris_probability_minimum", 0.22))

    suppressed = (
        result.get("is_duplicate_suppressed", pd.Series(False, index=result.index)).fillna(False).astype(bool)
        | result.get("is_hierarchy_suppressed", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    )
    labels = result["integrated_label"].astype(str)
    legacy_wall_rescue = result.get(
        "candidate_source", pd.Series("", index=result.index)
    ).astype(str).eq("wall_cell_rescue_peak")
    ambiguous = (
        result["cell_probability"].fillna(0).astype(float).between(lower, upper)
        & ~labels.eq("invalid")
        & ~suppressed
        & ~legacy_wall_rescue
        & result["timepoint"].isin(["T0", "T1", "T2"])
    )
    result.loc[ambiguous, "temporal_appearance_status"] = "insufficient_evidence"
    if not ambiguous.any():
        return result

    image_cache: dict[str, np.ndarray] = {}
    descriptor_cache: dict[int, dict[str, Any]] = {}

    def descriptor(index: int) -> dict[str, Any] | None:
        if index in descriptor_cache:
            return descriptor_cache[index]
        path = str(result.at[index, "raw_image_path"])
        if not path or not Path(path).exists():
            return None
        if path not in image_cache:
            with Image.open(path) as image:
                image_cache[path] = np.asarray(image.convert("L"), dtype=np.float32)
        value = _descriptor(
            image_cache[path], float(result.at[index, "x_px"]), float(result.at[index, "y_px"]), patch_size
        )
        descriptor_cache[index] = value
        return value

    ax = result.get("aligned_x_px", result["x_px"]).fillna(result["x_px"]).astype(float)
    ay = result.get("aligned_y_px", result["y_px"]).fillna(result["y_px"]).astype(float)
    active = result[~labels.eq("invalid") & ~suppressed & result["timepoint"].isin(["T0", "T1", "T2"])]

    for index in result.index[ambiguous]:
        seed_descriptor = descriptor(int(index))
        if seed_descriptor is None:
            continue
        seed_area = max(float(result.at[index, "area_px"] or 1.0), 1.0)
        comparisons: list[dict[str, Any]] = []
        for timepoint in ("T0", "T1", "T2"):
            if timepoint == str(result.at[index, "timepoint"]):
                continue
            local = active[
                active["well"].astype(str).eq(str(result.at[index, "well"]))
                & active["timepoint"].astype(str).eq(timepoint)
            ]
            if local.empty:
                continue
            distances = np.hypot(
                ax.loc[local.index].to_numpy(float) - float(ax.at[index]),
                ay.loc[local.index].to_numpy(float) - float(ay.at[index]),
            )
            nearby = local.index.to_numpy()[distances <= search_radius]
            nearby_distances = distances[distances <= search_radius]
            candidates: list[dict[str, Any]] = []
            for other, distance in zip(nearby, nearby_distances):
                other_descriptor = descriptor(int(other))
                if other_descriptor is None:
                    continue
                similarity, overlap = _appearance_similarity(seed_descriptor, other_descriptor)
                area_ratio = max(float(result.at[other, "area_px"] or 1.0), 1.0) / seed_area
                distance_score = float(np.exp(-float(distance) / max(search_radius * 0.45, 1.0)))
                score = 0.48 * max(similarity, 0.0) + 0.34 * overlap + 0.18 * distance_score
                candidates.append(
                    {
                        "index": int(other),
                        "distance": float(distance),
                        "similarity": similarity,
                        "iou": overlap,
                        "area_ratio": area_ratio,
                        "score": score,
                    }
                )
            if candidates:
                chosen = max(candidates, key=lambda item: item["score"])
                if chosen["score"] >= minimum_match:
                    comparisons.append(chosen)

        if not comparisons:
            continue
        similarities = np.asarray([item["similarity"] for item in comparisons], dtype=float)
        overlaps = np.asarray([item["iou"] for item in comparisons], dtype=float)
        ratios = np.asarray([item["area_ratio"] for item in comparisons], dtype=float)
        result.at[index, "temporal_appearance_match_count"] = len(comparisons)
        result.at[index, "temporal_mean_patch_similarity"] = float(similarities.mean())
        result.at[index, "temporal_mean_mask_iou"] = float(overlaps.mean())
        result.at[index, "temporal_max_area_ratio"] = float(max(ratios.max(), 1.0 / max(ratios.min(), 1e-6)))
        result.at[index, "temporal_appearance_matches"] = ";".join(
            str(result.at[item["index"], "candidate_id"]) for item in comparisons
        )

        maximum_log_area = float(np.max(np.abs(np.log(np.maximum(ratios, 1e-6)))))
        static = (
            len(comparisons) >= 2
            and float(similarities.mean()) >= static_similarity
            and float(overlaps.mean()) >= static_iou
            and maximum_log_area <= static_area_tolerance
        )

        seed_time = int(str(result.at[index, "timepoint"])[1:])
        split = False
        for item in comparisons:
            other = item["index"]
            other_time = int(str(result.at[other, "timepoint"])[1:])
            if other_time <= seed_time:
                continue
            same_time = active[
                active["well"].astype(str).eq(str(result.at[index, "well"]))
                & active["timepoint"].astype(str).eq(str(result.at[other, "timepoint"]))
                & (active["cell_probability"].fillna(0).astype(float) >= lower)
            ]
            count = int(
                (
                    np.hypot(
                        ax.loc[same_time.index].to_numpy(float) - float(ax.at[other]),
                        ay.loc[same_time.index].to_numpy(float) - float(ay.at[other]),
                    )
                    <= division_radius
                ).sum()
            )
            if count >= 2:
                split = True
                break
        coherent_change = (
            max(float(ratios.max()), 1.0 / max(float(ratios.min()), 1e-6)) >= dynamic_area_ratio
            and float(overlaps.mean()) <= dynamic_iou
            and float(np.max([item["score"] for item in comparisons])) >= minimum_match + 0.08
        )
        counterpart_cell = max(
            float(result.at[item["index"], "cell_probability"] or 0.0) for item in comparisons
        )

        if static and float(result.at[index, "debris_probability"] or 0.0) >= static_debris_floor:
            confidence = float(
                np.clip(0.50 + 0.25 * similarities.mean() + 0.20 * overlaps.mean(), 0.0, 0.96)
            )
            result.at[index, "integrated_label"] = "debris"
            result.at[index, "integrated_confidence"] = confidence
            result.at[index, "temporal_appearance_confidence"] = confidence
            result.at[index, "temporal_appearance_status"] = "static_pixel_identity_debris"
        elif (split or coherent_change) and counterpart_cell >= lower:
            label_name = candidate_multiplicity_label(result.loc[index])
            confidence = float(
                np.clip(0.62 + 0.12 * float(split) + 0.12 * float(coherent_change), 0.0, 0.90)
            )
            result.at[index, "integrated_label"] = label_name
            result.at[index, "integrated_confidence"] = confidence
            result.at[index, "temporal_appearance_confidence"] = confidence
            result.at[index, "temporal_appearance_status"] = "changing_shape_or_growth_cell"
        else:
            result.at[index, "temporal_appearance_status"] = "ambiguous_temporal_appearance"

    return result

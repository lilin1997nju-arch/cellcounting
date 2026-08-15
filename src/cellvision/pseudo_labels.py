from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops

from .config import artifact_path
from .review_context import compute_well_registration
from .stage_cache import stage_fingerprint


def _crop_with_padding(
    image: np.ndarray, x: float, y: float, size: int
) -> np.ndarray:
    half = size // 2
    left = int(round(x)) - half
    top = int(round(y)) - half
    if (
        left >= 0
        and top >= 0
        and left + size <= image.shape[1]
        and top + size <= image.shape[0]
    ):
        return image[top : top + size, left : left + size].copy()
    result = np.full((size, size), int(np.median(image)), dtype=np.uint8)
    src_x0, src_y0 = max(0, left), max(0, top)
    src_x1, src_y1 = min(image.shape[1], left + size), min(image.shape[0], top + size)
    dst_x0, dst_y0 = src_x0 - left, src_y0 - top
    result[
        dst_y0:dst_y0 + (src_y1 - src_y0),
        dst_x0:dst_x0 + (src_x1 - src_x0),
    ] = image[src_y0:src_y1, src_x0:src_x1]
    return result


def _component_rows(
    mask_path: str | Path,
    well: str,
    timepoint: str,
    shift: dict[str, float],
) -> list[dict[str, Any]]:
    with Image.open(mask_path) as image:
        width, height = image.size
        mask = np.asarray(image.convert("1"), dtype=bool)
    rows: list[dict[str, Any]] = []
    for component_index, region in enumerate(
        regionprops(label(mask, connectivity=2)), start=1
    ):
        if region.area < 3:
            continue
        y, x = region.centroid
        perimeter = max(float(region.perimeter), 1e-6)
        circularity = float(min(1.0, 4 * np.pi * region.area / perimeter**2))
        radial_fraction = float(
            np.hypot(x - width / 2, y - height / 2) / min(width, height)
        )
        # Components above the pseudo-label hard non-cell area limit are
        # typically the well rim or another large artifact.  Computing their
        # convex hull solely for ``region.solidity`` dominates extraction time
        # on full-resolution CF masks, while the area rule already rejects
        # them unconditionally.  A zero sentinel preserves that rejection and
        # avoids the unnecessary hull calculation.
        solidity = float(region.solidity) if region.area <= 1200 else 0.0
        rows.append(
            {
                "candidate_id": f"{well}:{timepoint}:cf:{component_index}",
                "well": well,
                "timepoint": timepoint,
                "x_px": float(x),
                "y_px": float(y),
                "aligned_x_px": float(x + shift.get("align_shift_x_px", 0.0)),
                "aligned_y_px": float(y + shift.get("align_shift_y_px", 0.0)),
                "area_px": float(region.area),
                "diameter_px": float(region.equivalent_diameter_area),
                "circularity": circularity,
                "eccentricity": float(region.eccentricity),
                "solidity": solidity,
                "extent": float(region.extent),
                "radial_fraction": radial_fraction,
            }
        )
    return rows


def _add_wall_neighbors(frame: pd.DataFrame, radius: float = 110) -> pd.DataFrame:
    frame = frame.copy()
    frame["wall_neighbor_count"] = 0
    for _, indices in frame.groupby(["well", "timepoint"]).groups.items():
        local = frame.loc[list(indices)]
        wall = local[local["radial_fraction"] >= 0.42]
        if wall.empty:
            continue
        coordinates = wall[["x_px", "y_px"]].to_numpy(float)
        radial = wall["radial_fraction"].to_numpy(float)
        distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        radial_delta = np.abs(radial[:, None] - radial[None, :])
        counts = ((distances <= radius) & (radial_delta <= 0.025)).sum(axis=1) - 1
        frame.loc[wall.index, "wall_neighbor_count"] = counts
    return frame


def _add_temporal_support(frame: pd.DataFrame, maximum_distance: float = 24) -> pd.DataFrame:
    frame = frame.copy()
    frame["temporal_support"] = 0
    frame["maximum_temporal_motion_px"] = 0.0
    for well, indices in frame.groupby("well").groups.items():
        local = frame.loc[list(indices)]
        by_timepoint = {
            tp: part for tp, part in local.groupby("timepoint")
        }
        for index, row in local.iterrows():
            support = 0
            motions: list[float] = []
            for timepoint, other in by_timepoint.items():
                if timepoint == row.timepoint or other.empty:
                    continue
                distance = np.hypot(
                    other["aligned_x_px"].to_numpy(float) - float(row.aligned_x_px),
                    other["aligned_y_px"].to_numpy(float) - float(row.aligned_y_px),
                )
                minimum = float(distance.min())
                if minimum <= maximum_distance:
                    support += 1
                    motions.append(minimum)
            frame.at[index, "temporal_support"] = support
            frame.at[index, "maximum_temporal_motion_px"] = max(motions, default=0.0)
    return frame


def _background_anisotropy(
    image: np.ndarray, x: float, y: float, size: int = 96
) -> float:
    patch = _crop_with_padding(image, x, y, size).astype(np.float32)
    gradient_y, gradient_x = np.gradient(patch)
    yy, xx = np.ogrid[:size, :size]
    background = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 > 14**2
    gx = gradient_x[background]
    gy = gradient_y[background]
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    return float(
        np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + 1e-6)
    )


def _add_background_anisotropy(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # A value near one is deliberately conservative: unless a plausible cell
    # seed is visually isotropic, it cannot become a positive pseudo-label.
    result["background_anisotropy"] = 1.0
    plausible_shape = (
        result["area_px"].between(12, 800)
        & (result["circularity"] >= 0.52)
        & (result["eccentricity"] <= 0.96)
        & (result["solidity"] >= 0.74)
        & (result["extent"] >= 0.38)
        & (result["temporal_support"] >= 1)
        & (result["radial_fraction"] < 0.45)
    )
    plausible = result[plausible_shape]
    for raw_path, group in plausible.groupby("raw_image_path", sort=False):
        with Image.open(raw_path) as image:
            raw = np.asarray(image.convert("L"), dtype=np.uint8)
        for row in group.itertuples():
            result.at[row.Index, "background_anisotropy"] = _background_anisotropy(
                raw, row.x_px, row.y_px
            )
    return result


def _assign_pseudo_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    cell_shape = (
        result["area_px"].between(12, 800)
        & (result["circularity"] >= 0.52)
        & (result["eccentricity"] <= 0.96)
        & (result["solidity"] >= 0.74)
        & (result["extent"] >= 0.38)
    )
    obvious_noncell = (
        (result["area_px"] < 7)
        | (result["area_px"] > 1200)
        | (result["circularity"] < 0.30)
        | (result["solidity"] < 0.60)
        | (result["extent"] < 0.22)
    )
    wall_chain = (
        (result["radial_fraction"] >= 0.42)
        & (result["wall_neighbor_count"] >= 2)
    )
    temporal_cell = result["temporal_support"] >= 1
    high_confidence_cell = (
        cell_shape
        & temporal_cell
        & (result["radial_fraction"] < 0.45)
        & (result["background_anisotropy"] <= 0.25)
    )
    directional_wall = wall_chain & (result["background_anisotropy"] >= 0.55)
    high_confidence_noncell = (
        obvious_noncell | directional_wall
    ) & ~high_confidence_cell
    result["pseudo_label"] = "uncertain"
    result.loc[high_confidence_cell, "pseudo_label"] = "cell"
    result.loc[high_confidence_noncell, "pseudo_label"] = "debris_artifact"
    result["pseudo_confidence"] = 0.0
    result.loc[high_confidence_cell, "pseudo_confidence"] = (
        0.65
        + 0.10 * result.loc[high_confidence_cell, "temporal_support"].clip(upper=2)
        + 0.10 * result.loc[high_confidence_cell, "circularity"]
    ).clip(upper=0.98)
    result.loc[high_confidence_noncell, "pseudo_confidence"] = (
        0.72
        + 0.08 * obvious_noncell[high_confidence_noncell].astype(float)
        + 0.08 * wall_chain[high_confidence_noncell].astype(float)
    ).clip(upper=0.96)
    return result


def build_morphology_pseudo_labels(config: dict[str, Any]) -> Path:
    started = time.perf_counter()
    timings: dict[str, float] = {}

    def record_timing(name: str, section_started: float) -> None:
        timings[name] = timings.get(name, 0.0) + (
            time.perf_counter() - section_started
        )

    section_started = time.perf_counter()
    settings = config.get("morphology_classifier", {})
    patch_size = int(settings.get("patch_size_px", 64))
    max_per_class = int(settings.get("max_patches_per_class", 12000))
    max_per_well = int(settings.get("max_patches_per_well_per_class", 220))
    seed = int(settings.get("seed", 20260729))
    excluded_wells = {
        str(well).upper() for well in settings.get("excluded_wells", [])
    }
    reuse_candidates = bool(settings.get("reuse_candidate_manifest", True))
    images = pd.read_csv(artifact_path(config, "manifests", "images.csv"))
    selected = images[
        images["timepoint"].isin(["T0", "T1", "T2"])
        & (images["decode_status"] == "ok")
        & ~images["well"].isin(excluded_wells)
    ].copy()

    manifest_path = artifact_path(
        config, "pseudo_labels", "morphology_candidates.csv"
    )
    source_metadata_path = manifest_path.with_suffix(".source.json")
    source_fingerprint = stage_fingerprint(
        "morphology-candidate-extraction",
        list(selected["raw_image_path"].astype(str))
        + list(selected["cf_image_path"].astype(str)),
        {
            "excluded_wells": sorted(excluded_wells),
            "timepoints": ["T0", "T1", "T2"],
        },
        version="20260804-fingerprinted",
    )
    record_timing("input_and_fingerprint", section_started)
    can_reuse = False
    if reuse_candidates and manifest_path.exists():
        cached = pd.read_csv(manifest_path)
        cached_wells = set(cached["well"].astype(str).str.upper().unique())
        expected_wells = set(selected["well"].astype(str).str.upper().unique())
        try:
            source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            source_metadata = {}
        can_reuse = (
            cached_wells == expected_wells
            and source_metadata.get("fingerprint") == source_fingerprint
        )
    if can_reuse:
        candidates = cached.drop(
            columns=["pseudo_label", "pseudo_confidence", "background_anisotropy"],
            errors="ignore",
        )
        print(
            f"reusing {len(candidates):,} extracted candidates from "
            f"{len(cached_wells)} wells",
            flush=True,
        )
    else:
        section_started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        grouped_wells = list(selected.groupby("well"))

        def extract_well(
            item: tuple[str, pd.DataFrame],
        ) -> tuple[list[dict[str, Any]], float, float]:
            well, well_rows = item
            registration_started = time.perf_counter()
            registration = compute_well_registration(config, images, well)
            registration_elapsed = time.perf_counter() - registration_started
            component_started = time.perf_counter()
            local_rows: list[dict[str, Any]] = []
            for image_row in well_rows.itertuples(index=False):
                shift = registration.get(
                    image_row.timepoint,
                    {"align_shift_x_px": 0.0, "align_shift_y_px": 0.0},
                )
                for component in _component_rows(
                    image_row.cf_image_path, well, image_row.timepoint, shift
                ):
                    component["raw_image_path"] = image_row.raw_image_path
                    component["cf_image_path"] = image_row.cf_image_path
                    local_rows.append(component)
            component_elapsed = time.perf_counter() - component_started
            return local_rows, registration_elapsed, component_elapsed

        worker_count = max(1, int(settings.get("candidate_extraction_workers", 4)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            extracted = executor.map(extract_well, grouped_wells)
            for well_index, extraction in enumerate(extracted, start=1):
                local_rows, registration_elapsed, component_elapsed = extraction
                rows.extend(local_rows)
                timings["registration"] = timings.get("registration", 0.0) + (
                    registration_elapsed
                )
                timings["component_extraction"] = timings.get(
                    "component_extraction", 0.0
                ) + component_elapsed
                if well_index % 8 == 0 or well_index == len(grouped_wells):
                    print(
                        f"pseudo-label extraction: {well_index}/{len(grouped_wells)} wells, "
                        f"{len(rows):,} candidates",
                        flush=True,
                    )
        if not rows:
            raise RuntimeError("No CF-mask candidates were found in T0/T1/T2.")
        record_timing("registration_and_component_extraction", section_started)
        section_started = time.perf_counter()
        candidates = _add_wall_neighbors(pd.DataFrame(rows))
        record_timing("wall_neighbor_features", section_started)
        section_started = time.perf_counter()
        candidates = _add_temporal_support(candidates)
        record_timing("temporal_support_features", section_started)

    section_started = time.perf_counter()
    candidates = _add_background_anisotropy(candidates)
    record_timing("background_anisotropy", section_started)
    section_started = time.perf_counter()
    candidates = _assign_pseudo_labels(candidates)
    record_timing("pseudo_label_rules", section_started)
    section_started = time.perf_counter()
    candidates.to_csv(manifest_path, index=False, encoding="utf-8")
    source_metadata_path.write_text(
        json.dumps(
            {
                "fingerprint": source_fingerprint,
                "candidate_count": int(len(candidates)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    record_timing("candidate_manifest_serialization", section_started)

    section_started = time.perf_counter()
    labelled = candidates[candidates["pseudo_label"] != "uncertain"].copy()
    sampled_parts: list[pd.DataFrame] = []
    for _, group in labelled.groupby(["pseudo_label", "well"]):
        sampled_parts.append(
            group.sample(
                n=min(len(group), max_per_well),
                random_state=seed,
            )
        )
    if not sampled_parts:
        raise RuntimeError("Pseudo-label rules produced no high-confidence samples.")
    sampled = pd.concat(sampled_parts, ignore_index=True)
    class_sizes = sampled["pseudo_label"].value_counts()
    balanced_count = min(max_per_class, int(class_sizes.min()))
    balanced_parts: list[pd.DataFrame] = []
    for label_name, group in sampled.groupby("pseudo_label"):
        balanced_parts.append(
            group.sample(
                n=balanced_count,
                random_state=seed,
            )
        )
    sampled = pd.concat(balanced_parts, ignore_index=True).sample(
        frac=1, random_state=seed
    )
    record_timing("training_sample_selection", section_started)

    section_started = time.perf_counter()
    patches: list[np.ndarray] = []
    patch_rows: list[dict[str, Any]] = []
    for raw_path, group in sampled.groupby("raw_image_path", sort=False):
        with Image.open(raw_path) as image:
            raw = np.asarray(image.convert("L"), dtype=np.uint8)
        for row in group.itertuples(index=False):
            patches.append(_crop_with_padding(raw, row.x_px, row.y_px, patch_size))
            patch_rows.append(row._asdict())
    record_timing("training_patch_extraction", section_started)
    # Keep metadata in exactly the same order as the extracted patch tensor.
    sampled = pd.DataFrame(patch_rows)
    sampled["class_index"] = sampled["pseudo_label"].map(
        {"debris_artifact": 0, "cell": 1}
    )
    cache_path = artifact_path(config, "cache", "morphology_pseudo_labels.npz")
    section_started = time.perf_counter()
    np.savez_compressed(
        cache_path,
        images=np.stack(patches),
        labels=sampled["class_index"].to_numpy(np.int64),
        candidate_ids=sampled["candidate_id"].to_numpy(str),
    )
    sampled.to_csv(
        artifact_path(config, "pseudo_labels", "morphology_training_samples.csv"),
        index=False,
        encoding="utf-8",
    )
    record_timing("training_cache_serialization", section_started)
    measured_seconds = sum(timings.values())
    timings["unattributed_overhead"] = max(
        0.0, time.perf_counter() - started - measured_seconds
    )
    summary = {
        "image_count": int(len(selected)),
        "well_count": int(selected["well"].nunique()),
        "excluded_wells": sorted(excluded_wells),
        "candidate_count": int(len(candidates)),
        "label_counts": candidates["pseudo_label"].value_counts().to_dict(),
        "training_patch_counts": sampled["pseudo_label"].value_counts().to_dict(),
        "patch_size_px": patch_size,
        "validation_source": "external_queue_required",
        "timing_seconds": {
            name: round(seconds, 3) for name, seconds in timings.items()
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    cache_path.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return cache_path

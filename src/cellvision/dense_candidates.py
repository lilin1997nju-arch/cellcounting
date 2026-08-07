from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    gaussian_filter1d,
    label as connected_components,
    map_coordinates,
    maximum_filter,
)
from scipy.spatial import cKDTree
from skimage.measure import regionprops

from .config import artifact_path
from .stage_cache import stage_fingerprint


def detect_dynamic_wall_inner_fraction(
    raw: np.ndarray, settings: dict[str, Any]
) -> float:
    """Locate the first sustained radial edge belonging to the physical wall."""
    target_size = int(settings.get("wall_profile_target_size_px", 768))
    step = max(1, int(np.ceil(max(raw.shape) / target_size)))
    sample = raw[::step, ::step].astype(np.float32, copy=False)
    height, width = sample.shape
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    yy, xx = np.indices(sample.shape)
    radius = np.hypot(xx - center_x, yy - center_y)
    scale = float(min(width, height))
    smoothed = gaussian_filter(sample, 1.2)
    gradient_y, gradient_x = np.gradient(smoothed)
    radial_gradient = np.abs(
        gradient_x * (xx - center_x) / (radius + 1.0)
        + gradient_y * (yy - center_y) / (radius + 1.0)
    )
    search_start = float(settings.get("wall_search_start_fraction", 0.38))
    search_end = float(settings.get("wall_search_end_fraction", 0.49))
    bins = np.arange(
        max(1, int(search_start * scale)),
        max(2, int(search_end * scale) + 1),
    )
    profile = []
    for lower in bins[:-1]:
        values = radial_gradient[(radius >= lower) & (radius < lower + 1)]
        profile.append(float(np.percentile(values, 80)) if len(values) else 0.0)
    if len(profile) < 5:
        return float(settings.get("wall_fallback_inner_fraction", 0.44))
    profile_array = gaussian_filter1d(np.asarray(profile), 2.0)
    baseline = float(np.percentile(profile_array, 20))
    peak = float(profile_array.max())
    minimum_ratio = float(settings.get("wall_profile_minimum_peak_ratio", 3.0))
    if peak / (baseline + 1e-6) < minimum_ratio:
        return float(settings.get("wall_fallback_inner_fraction", 0.44))
    threshold = baseline + float(
        settings.get("wall_profile_rise_fraction", 0.28)
    ) * (peak - baseline)
    above = profile_array >= threshold
    sustained = np.flatnonzero(above[:-2] & above[1:-1] & above[2:])
    if not len(sustained):
        return float(settings.get("wall_fallback_inner_fraction", 0.44))
    detected = float(bins[int(sustained[0])] / scale)
    return float(
        np.clip(
            detected,
            float(settings.get("wall_minimum_inner_fraction", 0.405)),
            float(settings.get("wall_maximum_inner_fraction", 0.46)),
        )
    )


def _zone_peak_indices(
    response: np.ndarray,
    maxima: np.ndarray,
    zone: np.ndarray,
    *,
    percentile: float,
    minimum_response: float,
    quota: int,
    grid_divisions: int,
    coverage_per_tile: int,
) -> list[tuple[int, int]]:
    values = response[zone]
    if quota <= 0 or not len(values):
        return []
    # A percentile-only threshold always admits a fixed share of an empty,
    # textured field.  The grid coverage pass would then promote those weak
    # texture maxima merely to fill tiles.  Keep the coverage mechanism for
    # spatial recall, but only after a peak clears an absolute salience floor.
    threshold = max(
        float(np.percentile(values, percentile)),
        float(minimum_response),
    )
    ys, xs = np.nonzero(maxima & zone & (response >= threshold))
    if not len(xs):
        return []
    order = np.argsort(response[ys, xs])[::-1]
    height, width = response.shape
    groups: dict[tuple[int, int], list[int]] = {}
    for index in order:
        tile = (
            min(grid_divisions - 1, int(ys[index] * grid_divisions / height)),
            min(grid_divisions - 1, int(xs[index] * grid_divisions / width)),
        )
        groups.setdefault(tile, []).append(int(index))
    selected: list[int] = []
    selected_set: set[int] = set()
    for depth in range(max(0, coverage_per_tile)):
        for tile in sorted(groups):
            if depth < len(groups[tile]):
                index = groups[tile][depth]
                selected.append(index)
                selected_set.add(index)
                if len(selected) >= quota:
                    break
        if len(selected) >= quota:
            break
    if len(selected) < quota:
        selected.extend(
            int(index)
            for index in order
            if int(index) not in selected_set
        )
    return [
        (int(ys[index]), int(xs[index])) for index in selected[:quota]
    ]


def _peak_blobness(surface: np.ndarray, y: int, x: int) -> float:
    """Return 1 for a locally round peak and near 0 for a line-like rim."""
    if y < 1 or x < 1 or y >= surface.shape[0] - 1 or x >= surface.shape[1] - 1:
        return 0.0
    dxx = float(surface[y, x + 1] - 2 * surface[y, x] + surface[y, x - 1])
    dyy = float(surface[y + 1, x] - 2 * surface[y, x] + surface[y - 1, x])
    dxy = float(
        surface[y + 1, x + 1]
        - surface[y + 1, x - 1]
        - surface[y - 1, x + 1]
        + surface[y - 1, x - 1]
    ) / 4.0
    eigenvalues = np.linalg.eigvalsh([[dxx, dxy], [dxy, dyy]])
    absolute = np.abs(eigenvalues)
    return float(absolute.min() / (absolute.max() + 1e-6))


def _polar_component_shape(
    response: np.ndarray,
    y: int,
    x: int,
    *,
    tangential_pixel_scale: float,
) -> dict[str, float]:
    """Measure the actual compact residual around one wall-band peak."""
    radial_radius = 18
    angular_radius = 12
    y0 = max(0, y - radial_radius)
    y1 = min(response.shape[0], y + radial_radius + 1)
    angular_indices = np.arange(
        x - angular_radius, x + angular_radius + 1
    ) % response.shape[1]
    patch = response[y0:y1][:, angular_indices]
    local_y = y - y0
    local_x = angular_radius
    threshold = max(10.0, 0.32 * float(response[y, x]))
    mask = patch >= threshold
    labels, _ = connected_components(mask)
    component_id = int(labels[local_y, local_x])
    if component_id <= 0:
        return {
            "area_px": 48.0,
            "diameter_px": float(np.sqrt(4 * 48 / np.pi)),
            "circularity": 0.5,
            "eccentricity": 0.5,
            "solidity": 0.7,
            "extent": 0.5,
        }
    component = labels == component_id
    # Angular samples are denser than physical image pixels.  Expanding that
    # axis before region measurement keeps compactness estimates in Cartesian
    # units instead of making every object appear artificially narrow.
    repeat = max(1, int(round(tangential_pixel_scale)))
    measured = np.repeat(component, repeat, axis=1).astype(np.uint8)
    region = regionprops(measured)[0]
    area = max(1.0, float(region.area))
    perimeter = max(1.0, float(region.perimeter))
    return {
        "area_px": area,
        "diameter_px": float(np.sqrt(4 * area / np.pi)),
        "circularity": float(np.clip(4 * np.pi * area / perimeter**2, 0, 1)),
        "eccentricity": float(np.clip(region.eccentricity, 0, 1)),
        "solidity": float(np.clip(region.solidity, 0, 1)),
        "extent": float(np.clip(region.extent, 0, 1)),
    }


def _polar_wall_residual_peaks(
    raw: np.ndarray,
    wall_inner_fraction: float,
    settings: dict[str, Any],
) -> list[dict[str, float]]:
    """Find compact dark objects after subtracting the tangential wall model.

    The circular wall is unwrapped into a polar strip.  Along that strip the
    physical rim becomes a slowly varying horizontal texture, while a cell is
    a short local interruption.  Tangential background subtraction therefore
    removes the rim without masking a cell whose centre overlaps it.
    """
    if not bool(settings.get("wall_residual_enabled", True)):
        return []
    height, width = raw.shape
    scale = float(min(height, width))
    center_x, center_y = width / 2, height / 2
    inside = float(settings.get("wall_residual_inside_fraction", 0.045))
    outside = float(settings.get("wall_residual_outside_fraction", 0.035))
    radius_start = max(1, int((wall_inner_fraction - inside) * scale))
    radius_end = min(
        int(float(settings.get("wall_residual_maximum_fraction", 0.47)) * scale),
        int((wall_inner_fraction + outside) * scale),
    )
    if radius_end <= radius_start:
        return []
    angular_samples = int(settings.get("wall_residual_angular_samples", 4096))
    radii = np.arange(radius_start, radius_end + 1, dtype=np.float32)
    angles = (
        np.arange(angular_samples, dtype=np.float32)
        * (2 * np.pi / angular_samples)
    )
    cosines = np.cos(angles)[None, :]
    sines = np.sin(angles)[None, :]
    sample_x = center_x + radii[:, None] * cosines
    sample_y = center_y + radii[:, None] * sines
    polar = map_coordinates(
        raw,
        [sample_y, sample_x],
        order=1,
        mode="nearest",
    )
    fine = gaussian_filter(
        polar, sigma=(1.0, 0.55), mode=("nearest", "wrap")
    )
    tangential_background = gaussian_filter1d(
        fine,
        float(settings.get("wall_residual_tangential_sigma", 10.0)),
        axis=1,
        mode="wrap",
    )
    residual = gaussian_filter(
        np.maximum(tangential_background - fine, 0.0),
        sigma=(1.4, 0.7),
        mode=("nearest", "wrap"),
    )
    maxima = residual == maximum_filter(
        residual,
        size=(
            int(settings.get("wall_residual_radial_window", 11)),
            int(settings.get("wall_residual_angular_window", 7)),
        ),
        mode=("nearest", "wrap"),
    )
    minimum_response = float(
        settings.get("wall_residual_minimum_response", 40.0)
    )
    peak_y, peak_x = np.nonzero(maxima & (residual >= minimum_response))
    if not len(peak_x):
        return []
    order = np.argsort(residual[peak_y, peak_x])[::-1]
    quota = int(settings.get("wall_residual_candidate_quota", 24))
    duplicate_radius = float(settings.get("duplicate_radius_px", 7))
    kept_coordinates: list[tuple[float, float]] = []
    output: list[dict[str, float]] = []
    for index in order:
        local_y = int(peak_y[index])
        local_x = int(peak_x[index])
        radius = float(radii[local_y])
        angle = float(angles[local_x])
        x = float(center_x + radius * np.cos(angle))
        y = float(center_y + radius * np.sin(angle))
        if any(
            np.hypot(x - previous_x, y - previous_y) <= duplicate_radius
            for previous_x, previous_y in kept_coordinates
        ):
            continue
        tangential_scale = max(
            1.0, radius * 2 * np.pi / angular_samples
        )
        shape = _polar_component_shape(
            residual,
            local_y,
            local_x,
            tangential_pixel_scale=tangential_scale,
        )
        output.append(
            {
                "x_px": x,
                "y_px": y,
                "response": float(residual[local_y, local_x]),
                "blobness": _peak_blobness(residual, local_y, local_x),
                **shape,
            }
        )
        kept_coordinates.append((x, y))
        if len(output) >= quota:
            break
    return output


def add_wall_neighbor_counts(
    frame: pd.DataFrame,
    *,
    wall_start: float = 0.40,
    radius: float = 110.0,
    maximum_radial_delta: float = 0.025,
) -> pd.DataFrame:
    """Measure chains that follow the well rim without discarding wall cells.

    A real cell may touch the wall, but the optical rim normally produces several
    nearly co-radial peaks.  Keeping this as an explicit feature lets the later
    filter reject the chain while preserving isolated, cell-shaped objects.
    """
    result = frame.copy()
    result["wall_neighbor_count"] = 0
    for _, indices in result.groupby(["well", "timepoint"], sort=False).groups.items():
        local = result.loc[list(indices)]
        wall = local[local["radial_fraction"].astype(float) >= wall_start]
        if len(wall) < 2:
            continue
        coordinates = wall[["x_px", "y_px"]].to_numpy(float)
        radial = wall["radial_fraction"].to_numpy(float)
        tree = cKDTree(coordinates)
        counts = []
        for index, neighbors in enumerate(tree.query_ball_point(coordinates, radius)):
            counts.append(
                sum(
                    other != index
                    and abs(float(radial[other] - radial[index]))
                    <= maximum_radial_delta
                    for other in neighbors
                )
            )
        result.loc[wall.index, "wall_neighbor_count"] = counts
    return result


def _deduplicate_points(
    frame: pd.DataFrame, radius: float = 0.25
) -> pd.DataFrame:
    kept: list[int] = []
    for _, local in frame.groupby(["well", "timepoint"], sort=False):
        coordinates: list[tuple[float, float]] = []
        for index, row in local.iterrows():
            point = (float(row["x_px"]), float(row["y_px"]))
            if any(
                np.hypot(point[0] - x, point[1] - y) <= radius
                for x, y in coordinates
            ):
                continue
            coordinates.append(point)
            kept.append(index)
    return frame.loc[kept].copy()


def _candidate_row(
    *,
    candidate_id: str,
    image: pd.Series,
    x: float,
    y: float,
    shift_x: float,
    shift_y: float,
    source: str,
    response: float,
    wall_rescue_blobness: float = 0.0,
    shape_features: dict[str, float] | None = None,
) -> dict[str, Any]:
    width = float(image["width_px"])
    height = float(image["height_px"])
    radial = float(
        np.hypot(x - width / 2, y - height / 2) / min(width, height)
    )
    shape = shape_features or {
        "area_px": 48.0,
        "diameter_px": float(np.sqrt(4 * 48 / np.pi)),
        "circularity": 0.82,
        "eccentricity": 0.35,
        "solidity": 0.90,
        "extent": 0.70,
    }
    return {
        "candidate_id": candidate_id,
        "well": str(image["well"]),
        "timepoint": str(image["timepoint"]),
        "x_px": float(x),
        "y_px": float(y),
        "aligned_x_px": float(x + shift_x),
        "aligned_y_px": float(y + shift_y),
        "area_px": float(shape["area_px"]),
        "diameter_px": float(shape["diameter_px"]),
        "circularity": float(shape["circularity"]),
        "eccentricity": float(shape["eccentricity"]),
        "solidity": float(shape["solidity"]),
        "extent": float(shape["extent"]),
        "radial_fraction": radial,
        "raw_image_path": str(image["raw_image_path"]),
        "cf_image_path": str(image["cf_image_path"]),
        "wall_neighbor_count": 0,
        "temporal_support": 0,
        "maximum_temporal_motion_px": 0.0,
        "background_anisotropy": 0.25,
        "pseudo_label": "uncertain",
        "pseudo_confidence": 0.0,
        "candidate_source": source,
        "dense_response": float(response),
        "wall_rescue_blobness": float(wall_rescue_blobness),
    }


def augment_candidates_with_dense_raw_proposals(
    config: dict[str, Any],
    database: str | Path,
) -> dict[str, Any]:
    """Add high-recall raw-image peaks and exact manual cell anchors."""
    started = time.perf_counter()
    candidate_path = artifact_path(
        config, "pseudo_labels", "morphology_candidates.csv"
    )
    candidates = pd.read_csv(candidate_path)
    if "candidate_source" not in candidates.columns:
        candidates["candidate_source"] = "cf_component"
    loaded_candidates = candidates.copy()
    settings = config.get("dense_detection", {})
    rebuild_timepoints = {
        str(value).upper()
        for value in settings.get(
            "rebuild_timepoints", ["T0", "T1", "T2"]
        )
    }
    regenerated_dense = (
        candidates["candidate_source"].isin(
            [
                "raw_dense_peak",
                "t0_dense_peak",
                "multiscale_dense_peak",
                "wall_cell_rescue_peak",
                "wall_residual_peak",
            ]
        )
        & candidates["timepoint"].astype(str).str.upper().isin(
            rebuild_timepoints
        )
    )
    regenerated_manual = candidates["candidate_source"].isin(
        ["manual_cell_anchor", "manual_annotation_anchor"]
    )
    candidates = candidates[
        ~(regenerated_dense | regenerated_manual)
    ].copy()
    images = pd.read_csv(
        artifact_path(config, "manifests", "images.csv")
    )
    excluded = {
        str(value).upper()
        for value in config.get("review_queue", {}).get(
            "excluded_wells", []
        )
    }
    images = images[
        images["timepoint"].isin(["T0", "T1", "T2"])
        & (images["decode_status"] == "ok")
        & ~images["well"].astype(str).str.upper().isin(excluded)
    ].copy()
    maximum_per_image = int(settings.get("maximum_peaks_per_image", 120))
    minimum_distance = int(settings.get("minimum_peak_distance_px", 9))
    response_percentile = float(
        settings.get("response_percentile", 99.2)
    )
    duplicate_radius = float(settings.get("duplicate_radius_px", 7))
    wall_mask_enabled = bool(settings.get("dynamic_wall_mask_enabled", True))
    wall_buffer_width = float(
        settings.get("wall_cell_buffer_width_fraction", 0.028)
    )
    buffer_fraction = float(
        settings.get("wall_buffer_candidate_fraction", 0.16)
    )
    rescue_fraction = float(
        settings.get("wall_rescue_candidate_fraction", 0.06)
    )
    rescue_quota = int(round(maximum_per_image * rescue_fraction))
    buffer_quota = int(round(maximum_per_image * buffer_fraction))
    interior_quota = maximum_per_image - buffer_quota - rescue_quota
    grid_divisions = int(settings.get("candidate_grid_divisions", 8))
    coverage_per_tile = int(settings.get("coverage_peaks_per_tile", 2))
    minimum_interior_response = float(
        settings.get("minimum_interior_peak_response", 18.0)
    )
    minimum_wall_buffer_response = float(
        settings.get("minimum_wall_buffer_peak_response", 20.0)
    )
    minimum_wall_rescue_response = float(
        settings.get("minimum_wall_rescue_peak_response", 28.0)
    )

    shift_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for key, local in candidates.groupby(["well", "timepoint"]):
        shift_lookup[(str(key[0]), str(key[1]))] = (
            float(
                np.median(
                    local["aligned_x_px"].to_numpy(float)
                    - local["x_px"].to_numpy(float)
                )
            ),
            float(
                np.median(
                    local["aligned_y_px"].to_numpy(float)
                    - local["y_px"].to_numpy(float)
                )
            ),
        )

    if bool(settings.get("include_manual_anchors", True)):
        with sqlite3.connect(database) as connection:
            manual = pd.read_sql_query(
                """
                SELECT annotation_id, well, timepoint, x_px, y_px, object_type
                FROM annotations
                WHERE object_type IN ('cell', 'debris', 'uncertain')
                  AND timepoint IN ('T0', 'T1', 'T2')
                ORDER BY updated_at DESC
                """,
                connection,
            )
    else:
        manual = pd.DataFrame(
            columns=[
                "annotation_id",
                "well",
                "timepoint",
                "x_px",
                "y_px",
                "object_type",
            ]
        )
    manual = _deduplicate_points(manual)
    base_signature_columns = [
        column
        for column in ("candidate_id", "well", "timepoint", "x_px", "y_px", "candidate_source")
        if column in candidates
    ]
    base_hash = int(
        pd.util.hash_pandas_object(
            candidates[base_signature_columns].sort_values("candidate_id"), index=False
        ).sum()
    ) if len(candidates) else 0
    fingerprint_paths = list(images["raw_image_path"].astype(str))
    if bool(settings.get("include_manual_anchors", True)):
        fingerprint_paths.append(str(database))
    dense_fingerprint = stage_fingerprint(
        "dense-candidate-generation",
        fingerprint_paths,
        {
            "dense_detection": settings,
            "excluded_wells": sorted(excluded),
            "base_candidate_count": int(len(candidates)),
            "base_candidate_hash": base_hash,
        },
        version="20260804-fingerprinted",
    )
    report_path = artifact_path(config, "models", "dense_candidate_report.json")
    loaded_dense = loaded_candidates["candidate_source"].isin(
        [
            "raw_dense_peak",
            "t0_dense_peak",
            "multiscale_dense_peak",
            "wall_cell_rescue_peak",
            "wall_residual_peak",
        ]
    ).any()
    if bool(settings.get("reuse_unchanged_stage", True)) and loaded_dense and report_path.exists():
        try:
            cached_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached_report = {}
        if cached_report.get("stage_fingerprint") == dense_fingerprint:
            return {**cached_report, "cache_hit": True}
    image_lookup = {
        (str(row.well), str(row.timepoint)): row
        for row in images.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for row in manual.itertuples(index=False):
        key = (str(row.well), str(row.timepoint))
        image = image_lookup.get(key)
        if image is None:
            continue
        shift_x, shift_y = shift_lookup.get(key, (0.0, 0.0))
        x, y = float(row.x_px), float(row.y_px)
        rows.append(
            _candidate_row(
                candidate_id=(
                    f"{row.well}:{row.timepoint}:manual:"
                    f"{int(row.annotation_id)}"
                ),
                image=pd.Series(image._asdict()),
                x=x,
                y=y,
                shift_x=shift_x,
                shift_y=shift_y,
                source=(
                    "manual_cell_anchor"
                    if str(row.object_type) == "cell"
                    else "manual_annotation_anchor"
                ),
                response=1.0,
            )
        )

    peak_images = images[
        images["timepoint"].astype(str).str.upper().isin(
            rebuild_timepoints
        )
    ]
    wall_geometry_rows: list[dict[str, Any]] = []
    wall_geometry_lookup: dict[tuple[str, str], float] = {}
    candidate_stage_wall_excluded = 0
    wall_rescue_peaks_tested = 0
    wall_rescue_arc_rejected = 0
    wall_residual_peak_count = 0
    for image_index, image in enumerate(
        peak_images.itertuples(index=False), start=1
    ):
        key = (str(image.well), str(image.timepoint))
        with Image.open(image.raw_image_path) as opened:
            raw_full = np.asarray(opened.convert("L"), dtype=np.float32)
        full_height, full_width = raw_full.shape
        wall_inner_fraction = (
            detect_dynamic_wall_inner_fraction(raw_full, settings)
            if wall_mask_enabled
            else 0.47
        )
        wall_geometry_lookup[key] = wall_inner_fraction
        wall_geometry_rows.append(
            {
                "well": key[0],
                "timepoint": key[1],
                "wall_inner_fraction": wall_inner_fraction,
                "buffer_start_fraction": max(
                    0.0, wall_inner_fraction - wall_buffer_width
                ),
            }
        )
        local_existing_frame = candidates[
            (candidates["well"] == image.well)
            & (candidates["timepoint"] == image.timepoint)
        ]
        local_existing_frame = local_existing_frame[
            local_existing_frame["radial_fraction"].astype(float)
            < wall_inner_fraction
        ]
        local_existing = local_existing_frame[["x_px", "y_px"]].to_numpy(float)
        # Human points do not suppress an independent automatic response.  The
        # occupied set contains only CF proposals that survive the wall gate.
        occupied_tree = cKDTree(local_existing) if len(local_existing) else None
        well_radius = min(full_width, full_height) * wall_inner_fraction
        center_x, center_y = full_width / 2, full_height / 2
        filter_margin = max(
            36,
            int(
                np.ceil(
                    float(settings.get("wall_rescue_width_fraction", 0.03))
                    * min(full_width, full_height)
                )
            )
            + 16,
        )
        roi_x0 = max(0, int(np.floor(center_x - well_radius)) - filter_margin)
        roi_y0 = max(0, int(np.floor(center_y - well_radius)) - filter_margin)
        roi_x1 = min(
            full_width,
            int(np.ceil(center_x + well_radius)) + filter_margin,
        )
        roi_y1 = min(
            full_height,
            int(np.ceil(center_y + well_radius)) + filter_margin,
        )
        raw = raw_full[roi_y0:roi_y1, roi_x0:roi_x1]
        fine = gaussian_filter(raw, 1.2)
        background = gaussian_filter(raw, 12.0)
        response = background - fine
        medium = gaussian_filter(raw, 4.0)
        contrast = np.abs(medium - fine)
        response = np.maximum(
            response,
            float(settings.get("multiscale_contrast_weight", 0.70))
            * contrast,
        )
        height, width = response.shape
        yy, xx = np.ogrid[:height, :width]
        radial_fraction = np.sqrt(
            (xx + roi_x0 - center_x) ** 2
            + (yy + roi_y0 - center_y) ** 2
        ) / min(full_width, full_height)
        wall_buffer_start = max(
            0.0, wall_inner_fraction - wall_buffer_width
        )
        interior = radial_fraction < wall_buffer_start
        wall_buffer = (
            (radial_fraction >= wall_buffer_start)
            & (radial_fraction < wall_inner_fraction)
        )
        wall_rescue_limit = min(
            0.47,
            wall_inner_fraction
            + float(settings.get("wall_rescue_width_fraction", 0.03)),
        )
        wall_rescue_zone = (
            (radial_fraction >= wall_inner_fraction)
            & (radial_fraction < wall_rescue_limit)
        )
        maxima = response == maximum_filter(
            response,
            size=minimum_distance * 2 + 1,
            mode="nearest",
        )
        selected_peaks = _zone_peak_indices(
            response,
            maxima,
            interior,
            percentile=float(
                settings.get("interior_response_percentile", response_percentile)
            ),
            minimum_response=minimum_interior_response,
            quota=interior_quota,
            grid_divisions=grid_divisions,
            coverage_per_tile=coverage_per_tile,
        )
        selected_peaks.extend(
            _zone_peak_indices(
                response,
                maxima,
                wall_buffer,
                percentile=float(
                    settings.get("wall_buffer_response_percentile", 98.0)
                ),
                minimum_response=minimum_wall_buffer_response,
                quota=buffer_quota,
                grid_divisions=grid_divisions,
                coverage_per_tile=1,
            )
        )
        rescue_pool = _zone_peak_indices(
            response,
            maxima,
            wall_rescue_zone,
            percentile=float(
                settings.get("wall_rescue_response_percentile", 98.8)
            ),
            minimum_response=minimum_wall_rescue_response,
            quota=max(rescue_quota * 8, 96),
            grid_divisions=grid_divisions,
            coverage_per_tile=1,
        )
        rescue_surface = gaussian_filter(response, 2.0)
        rescue_threshold = float(
            settings.get("wall_rescue_minimum_blobness", 0.34)
        )
        rescued_peaks: list[tuple[int, int, float]] = []
        wall_rescue_peaks_tested += len(rescue_pool)
        for local_y, local_x in rescue_pool:
            blobness = _peak_blobness(
                rescue_surface, local_y, local_x
            )
            if blobness >= rescue_threshold:
                rescued_peaks.append((local_y, local_x, blobness))
            else:
                wall_rescue_arc_rejected += 1
            if len(rescued_peaks) >= rescue_quota:
                break
        wall_residual_peaks = _polar_wall_residual_peaks(
            raw_full, wall_inner_fraction, settings
        )
        wall_residual_peak_count += len(wall_residual_peaks)
        residual_coordinates = np.asarray(
            [
                [peak["x_px"], peak["y_px"]]
                for peak in wall_residual_peaks
            ],
            dtype=float,
        )
        if selected_peaks or rescued_peaks or wall_residual_peaks:
            selected_count = 0
            shift_x, shift_y = shift_lookup.get(key, (0.0, 0.0))
            for peak in wall_residual_peaks:
                x = float(peak["x_px"])
                y = float(peak["y_px"])
                candidate_id = (
                    f"{image.well}:{image.timepoint}:wall-residual:"
                    f"{int(round(x))}:{int(round(y))}"
                )
                stable_x, stable_y = x, y
                matching_dense = [
                    (
                        float(local_x + roi_x0),
                        float(local_y + roi_y0),
                    )
                    for local_y, local_x in selected_peaks
                ]
                if matching_dense:
                    distances = np.hypot(
                        np.asarray([point[0] for point in matching_dense]) - x,
                        np.asarray([point[1] for point in matching_dense]) - y,
                    )
                    closest = int(np.argmin(distances))
                    if float(distances[closest]) <= duplicate_radius:
                        stable_x, stable_y = matching_dense[closest]
                        candidate_id = (
                            f"{image.well}:{image.timepoint}:raw:"
                            f"{int(stable_x)}:{int(stable_y)}"
                        )
                matching_rescue = [
                    (
                        float(local_x + roi_x0),
                        float(local_y + roi_y0),
                    )
                    for local_y, local_x, _ in rescued_peaks
                ]
                if matching_rescue and candidate_id.endswith(
                    f"wall-residual:{int(round(x))}:{int(round(y))}"
                ):
                    distances = np.hypot(
                        np.asarray([point[0] for point in matching_rescue]) - x,
                        np.asarray([point[1] for point in matching_rescue]) - y,
                    )
                    closest = int(np.argmin(distances))
                    if float(distances[closest]) <= duplicate_radius:
                        stable_x, stable_y = matching_rescue[closest]
                        candidate_id = (
                            f"{image.well}:{image.timepoint}:wall-rescue:"
                            f"{int(stable_x)}:{int(stable_y)}"
                        )
                rows.append(
                    _candidate_row(
                        candidate_id=candidate_id,
                        image=pd.Series(image._asdict()),
                        x=stable_x,
                        y=stable_y,
                        shift_x=shift_x,
                        shift_y=shift_y,
                        source="wall_residual_peak",
                        response=float(peak["response"]),
                        wall_rescue_blobness=float(peak["blobness"]),
                        shape_features={
                            name: float(peak[name])
                            for name in (
                                "area_px",
                                "diameter_px",
                                "circularity",
                                "eccentricity",
                                "solidity",
                                "extent",
                            )
                        },
                    )
                )
            for local_y, local_x in selected_peaks:
                x = float(local_x + roi_x0)
                y = float(local_y + roi_y0)
                if len(residual_coordinates) and float(
                    np.min(
                        np.hypot(
                            residual_coordinates[:, 0] - x,
                            residual_coordinates[:, 1] - y,
                        )
                    )
                ) <= duplicate_radius:
                    continue
                if occupied_tree is not None:
                    distance, _ = occupied_tree.query([x, y], k=1)
                    if float(distance) <= duplicate_radius:
                        continue
                rows.append(
                    _candidate_row(
                        candidate_id=(
                            f"{image.well}:{image.timepoint}:raw:"
                            f"{int(x)}:{int(y)}"
                        ),
                        image=pd.Series(image._asdict()),
                        x=x,
                        y=y,
                        shift_x=shift_x,
                        shift_y=shift_y,
                        source="multiscale_dense_peak",
                        response=float(response[local_y, local_x]),
                    )
                )
                selected_count += 1
                if selected_count >= maximum_per_image:
                    break
            for local_y, local_x, blobness in rescued_peaks:
                x = float(local_x + roi_x0)
                y = float(local_y + roi_y0)
                if len(residual_coordinates) and float(
                    np.min(
                        np.hypot(
                            residual_coordinates[:, 0] - x,
                            residual_coordinates[:, 1] - y,
                        )
                    )
                ) <= duplicate_radius:
                    continue
                if occupied_tree is not None:
                    distance, _ = occupied_tree.query([x, y], k=1)
                    if float(distance) <= duplicate_radius:
                        continue
                rows.append(
                    _candidate_row(
                        candidate_id=(
                            f"{image.well}:{image.timepoint}:wall-rescue:"
                            f"{int(x)}:{int(y)}"
                        ),
                        image=pd.Series(image._asdict()),
                        x=x,
                        y=y,
                        shift_x=shift_x,
                        shift_y=shift_y,
                        source="wall_cell_rescue_peak",
                        response=float(response[local_y, local_x]),
                        wall_rescue_blobness=blobness,
                    )
                )
        if image_index % 20 == 0 or image_index == len(peak_images):
            print(
                f"dense raw candidates: {image_index}/{len(peak_images)}",
                flush=True,
            )

    augmented = pd.concat(
        [candidates, pd.DataFrame(rows)], ignore_index=True
    ).drop_duplicates("candidate_id", keep="last")
    augmented["detected_wall_inner_fraction"] = [
        wall_geometry_lookup.get(
            (str(well), str(timepoint)), 0.47
        )
        for well, timepoint in zip(
            augmented["well"], augmented["timepoint"]
        )
    ]
    manual_anchor = augmented["candidate_source"].isin(
        ["manual_cell_anchor", "manual_annotation_anchor"]
    )
    wall_rescue_anchor = augmented["candidate_source"].astype(str).isin(
        ["wall_cell_rescue_peak", "wall_residual_peak"]
    )
    outside_dynamic_wall = (
        augmented["radial_fraction"].astype(float)
        >= augmented["detected_wall_inner_fraction"].astype(float)
    ) & ~manual_anchor & ~wall_rescue_anchor
    candidate_stage_wall_excluded = int(outside_dynamic_wall.sum())
    augmented = augmented[~outside_dynamic_wall].copy()
    wall_rescue_anchor = augmented["candidate_source"].astype(str).isin(
        ["wall_cell_rescue_peak", "wall_residual_peak"]
    )
    wall_residual_anchor = (
        augmented["candidate_source"].astype(str)
        == "wall_residual_peak"
    )
    augmented["candidate_zone"] = np.select(
        [
            wall_residual_anchor,
            wall_rescue_anchor,
            augmented["radial_fraction"].astype(float)
            >= (
                augmented["detected_wall_inner_fraction"].astype(float)
                - wall_buffer_width
            ),
        ],
        ["wall_residual", "wall_cell_rescue", "wall_cell_buffer"],
        default="well_interior",
    )
    filter_settings = config.get("candidate_filter", {})
    augmented = add_wall_neighbor_counts(
        augmented,
        wall_start=max(
            0.0,
            float(filter_settings.get("wall_band_start_fraction", 0.42))
            - 0.02,
        ),
        radius=float(
            filter_settings.get("wall_chain_neighbor_radius_px", 110)
        ),
    )
    augmented.to_csv(candidate_path, index=False, encoding="utf-8")
    wall_geometry_path = artifact_path(
        config, "cache", "dynamic_wall_geometry.csv"
    )
    pd.DataFrame(wall_geometry_rows).to_csv(
        wall_geometry_path, index=False, encoding="utf-8"
    )
    report = {
        "base_candidate_count": int(len(candidates)),
        "manual_anchor_count": int(
            sum(row["candidate_source"] == "manual_cell_anchor" for row in rows)
        ),
        "dense_peak_count": int(
            sum(
                row["candidate_source"]
                in {
                    "raw_dense_peak",
                    "t0_dense_peak",
                    "multiscale_dense_peak",
                    "wall_cell_rescue_peak",
                    "wall_residual_peak",
                }
                for row in rows
            )
        ),
        "rebuild_timepoints": sorted(rebuild_timepoints),
        "candidate_count": int(len(augmented)),
        "maximum_peaks_per_image": maximum_per_image,
        "candidate_stage_wall_excluded": candidate_stage_wall_excluded,
        "physical_wall_core_search_enabled": False,
        "wall_rescue_peaks_tested": wall_rescue_peaks_tested,
        "wall_rescue_arc_rejected": wall_rescue_arc_rejected,
        "wall_residual_peak_count": wall_residual_peak_count,
        "interior_candidate_quota": interior_quota,
        "wall_buffer_candidate_quota": buffer_quota,
        "wall_rescue_candidate_quota": rescue_quota,
        "minimum_interior_peak_response": minimum_interior_response,
        "minimum_wall_buffer_peak_response": minimum_wall_buffer_response,
        "minimum_wall_rescue_peak_response": minimum_wall_rescue_response,
        "wall_inner_fraction_median": float(
            np.median(
                [row["wall_inner_fraction"] for row in wall_geometry_rows]
            )
        ),
        "wall_geometry": str(wall_geometry_path),
        "candidate_manifest": str(candidate_path),
        "stage_fingerprint": dense_fingerprint,
        "cache_hit": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report

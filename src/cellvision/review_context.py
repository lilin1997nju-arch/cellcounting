from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.measure import label, regionprops
from skimage.registration import phase_cross_correlation

from .config import artifact_path


def _load_registration_image(path: str | Path, size: int = 512) -> np.ndarray:
    with Image.open(path) as image:
        resized = ImageOps.autocontrast(image.convert("L")).resize((size, size))
    array = np.asarray(resized, dtype=np.float32)
    return (array - array.mean()) / max(float(array.std()), 1e-6)


def compute_well_registration(
    config: dict[str, Any], images_manifest: pd.DataFrame, well: str
) -> dict[str, dict[str, float]]:
    cache_path = artifact_path(config, "cache", "registration", f"{well.upper()}.json")
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    selected = images_manifest[
        (images_manifest["well"] == well.upper())
        & images_manifest["timepoint"].isin(["T0", "T1", "T2"])
        & (images_manifest["decode_status"] == "ok")
    ]
    lookup = {row.timepoint: row for row in selected.itertuples(index=False)}
    if "T0" not in lookup:
        raise ValueError(f"T0 unavailable for {well}")
    reference = _load_registration_image(lookup["T0"].raw_image_path)
    original_width = float(lookup["T0"].width_px)
    original_height = float(lookup["T0"].height_px)
    result: dict[str, dict[str, float]] = {
        "T0": {"align_shift_x_px": 0.0, "align_shift_y_px": 0.0}
    }
    for timepoint in ("T1", "T2"):
        if timepoint not in lookup:
            continue
        moving = _load_registration_image(lookup[timepoint].raw_image_path)
        shift_yx, error, _ = phase_cross_correlation(reference, moving, upsample_factor=10)
        result[timepoint] = {
            "align_shift_x_px": float(shift_yx[1] * original_width / reference.shape[1]),
            "align_shift_y_px": float(shift_yx[0] * original_height / reference.shape[0]),
            "registration_error": float(error),
        }
    cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _local_cf_candidates(
    cf_path: str | Path,
    center_x: float,
    center_y: float,
    size: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    half = size // 2
    with Image.open(cf_path) as image:
        image_width, image_height = image.size
        mask = np.asarray(
            image.convert("1").crop(
                (
                    int(round(center_x)) - half,
                    int(round(center_y)) - half,
                    int(round(center_x)) + half,
                    int(round(center_y)) + half,
                )
            ),
            dtype=bool,
        )
    all_candidates: list[dict[str, Any]] = []
    origin_x = center_x - half
    origin_y = center_y - half
    for index, region in enumerate(regionprops(label(mask, connectivity=2)), start=1):
        if region.area < 3:
            continue
        y, x = region.centroid
        perimeter = max(float(region.perimeter), 1e-6)
        global_x = float(origin_x + x)
        global_y = float(origin_y + y)
        all_candidates.append(
            {
                "candidate_id": f"cf-local-{index}",
                "x_px": global_x,
                "y_px": global_y,
                "area_px": float(region.area),
                "equivalent_diameter_px": float(region.equivalent_diameter_area),
                "marker_diameter_px": float(region.equivalent_diameter_area),
                "orientation_rad": float(region.orientation),
                "circularity": float(min(1.0, 4 * np.pi * region.area / perimeter**2)),
                "eccentricity": float(region.eccentricity),
                "solidity": float(region.solidity),
                "extent": float(region.extent),
                "radial_fraction": float(
                    np.hypot(
                        global_x - image_width / 2,
                        global_y - image_height / 2,
                    )
                    / min(image_width, image_height)
                ),
            }
        )

    settings = config.get("candidate_filter", {})
    wall_start = float(settings.get("wall_band_start_fraction", 0.42))
    hard_wall_start = float(
        settings.get("hard_wall_exclusion_fraction", 0.44)
    )
    neighbor_radius = float(settings.get("wall_chain_neighbor_radius_px", 110))
    minimum_neighbors = int(settings.get("wall_chain_min_neighbors", 2))
    candidates: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for candidate in all_candidates:
        neighbors = [
            other
            for other in all_candidates
            if other is not candidate
            and np.hypot(
                float(other["x_px"]) - float(candidate["x_px"]),
                float(other["y_px"]) - float(candidate["y_px"]),
            )
            <= neighbor_radius
            and abs(
                float(other["radial_fraction"])
                - float(candidate["radial_fraction"])
            )
            <= 0.025
        ]
        candidate["wall_neighbor_count"] = len(neighbors)
        high_cell_like_shape = (
            20 <= float(candidate["area_px"]) <= 600
            and float(candidate["circularity"]) >= 0.72
            and float(candidate["eccentricity"]) <= 0.92
            and float(candidate["solidity"]) >= 0.82
            and float(candidate["extent"]) >= 0.5
        )
        obvious_wall_shape = (
            float(candidate["area_px"]) > 800
            or float(candidate["circularity"]) < 0.35
            or float(candidate["solidity"]) < 0.62
            or float(candidate["extent"]) < 0.25
        )
        in_wall_band = float(candidate["radial_fraction"]) >= wall_start
        in_wall_chain = len(neighbors) >= minimum_neighbors
        candidate["cell_like_shape"] = bool(high_cell_like_shape)
        candidate["wall_artifact_score"] = float(
            0.45 * in_wall_band
            + 0.35 * in_wall_chain
            + 0.35 * obvious_wall_shape
            - 0.35 * high_cell_like_shape
        )
        if (
            float(candidate["radial_fraction"]) >= hard_wall_start
            or (
                in_wall_band
                and (
                    obvious_wall_shape
                    or (in_wall_chain and not high_cell_like_shape)
                )
            )
        ):
            candidate["suppression_reason"] = (
                "位于仪器孔壁结构带"
                if float(candidate["radial_fraction"]) >= hard_wall_start
                else (
                    "孔壁连续结构且形态不像完整细胞"
                    if obvious_wall_shape
                    else "多个候选沿孔壁成串排列"
                )
            )
            suppressed.append(candidate)
        else:
            candidates.append(candidate)

    sort_key = lambda item: np.hypot(
        float(item["x_px"]) - center_x,
        float(item["y_px"]) - center_y,
    )
    candidates.sort(
        key=sort_key
    )
    suppressed.sort(key=sort_key)
    # Wall structures are deliberately omitted rather than exposed as
    # low-priority review objects. Human/model selections are rendered through
    # their own layers, so a confirmed near-wall cell remains visible.
    return candidates[:80], []


def build_review_context(
    config: dict[str, Any],
    images_manifest: pd.DataFrame,
    well: str,
    t0_x: float,
    t0_y: float,
    search_size: int = 512,
    timepoint_centers: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    well = well.upper()
    search_size = max(256, min(int(search_size), 1536))
    timepoint_centers = timepoint_centers or {}
    registration = compute_well_registration(config, images_manifest, well)
    selected = images_manifest[
        (images_manifest["well"] == well)
        & images_manifest["timepoint"].isin(["T0", "T1", "T2"])
    ]
    lookup = {row.timepoint: row for row in selected.itertuples(index=False)}
    timepoints: dict[str, Any] = {}
    for timepoint in ("T0", "T1", "T2"):
        if timepoint not in lookup or lookup[timepoint].decode_status != "ok":
            timepoints[timepoint] = {"available": False}
            continue
        shift = registration.get(
            timepoint, {"align_shift_x_px": 0.0, "align_shift_y_px": 0.0}
        )
        # Alignment shift maps the moving image onto T0, so the expected
        # coordinate in the moving image is the T0 coordinate minus that shift.
        if timepoint in timepoint_centers:
            center_x, center_y = (
                float(timepoint_centers[timepoint][0]),
                float(timepoint_centers[timepoint][1]),
            )
        else:
            center_x = float(t0_x - shift["align_shift_x_px"])
            center_y = float(t0_y - shift["align_shift_y_px"])
        candidates, suppressed_candidates = _local_cf_candidates(
            lookup[timepoint].cf_image_path,
            center_x,
            center_y,
            search_size,
            config,
        )
        timepoints[timepoint] = {
            "available": True,
            "center_x_px": center_x,
            "center_y_px": center_y,
            "origin_x_px": center_x - search_size / 2,
            "origin_y_px": center_y - search_size / 2,
            "image_width_px": int(lookup[timepoint].width_px),
            "image_height_px": int(lookup[timepoint].height_px),
            "align_shift_x_px": float(shift["align_shift_x_px"]),
            "align_shift_y_px": float(shift["align_shift_y_px"]),
            "candidates": candidates,
            "suppressed_candidates": suppressed_candidates,
        }

    return {
        "well": well,
        "anchor": {"x_px": float(t0_x), "y_px": float(t0_y)},
        "search_size_px": search_size,
        "resolution_um_per_pixel": float(config["calibration"]["resolution_um_per_pixel"]),
        "significant_displacement_um": float(
            config.get("review_queue", {}).get("significant_displacement_um", 20.0)
        ),
        "review_overrides": config.get("review_overrides", {}).get(well, {}),
        "timepoints": timepoints,
        "interpretation_rules": {
            "debris": "形态与细胞不同，通常不发生明显位移或分裂。",
            "dead_cell": "形态像细胞，但T1/T2不分裂且没有明显位移。",
            "live_cell": "保持细胞形态，并出现可信位移、生长或一对多分裂。",
            "uncertain": "证据不足时保留待定，不把未分裂自动判为死亡。",
        },
    }

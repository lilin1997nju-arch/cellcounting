from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
from skimage.measure import label, regionprops

from .config import artifact_path
from .decode import read_grayscale, read_mask
from .manifest import normalize_well


def _cf_components(mask: np.ndarray, minimum_area: int = 3) -> list[dict[str, Any]]:
    height, width = mask.shape
    yy, xx = np.ogrid[:height, :width]
    usable_radius = min(height, width) * 0.42
    inner_well = (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= usable_radius**2
    components = label(mask & inner_well, connectivity=2)
    objects: list[dict[str, Any]] = []
    for index, region in enumerate(regionprops(components), start=1):
        if region.area < minimum_area:
            continue
        y0, x0, y1, x1 = region.bbox
        y, x = region.centroid
        objects.append(
            {
                "component_index": index,
                "x_px": round(float(x), 2),
                "y_px": round(float(y), 2),
                "bbox_x": int(x0),
                "bbox_y": int(y0),
                "bbox_width": int(x1 - x0),
                "bbox_height": int(y1 - y0),
                "area_px": int(region.area),
            }
        )
    return objects


def generate_baseline(config: dict[str, Any], wells: list[str]) -> pd.DataFrame:
    manifest_path = artifact_path(config, "manifests", "images.csv")
    images = pd.read_csv(manifest_path)
    output_dir = artifact_path(config, "predictions", "baseline", "overlays", "placeholder").parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for well_value in wells:
        well = normalize_well(well_value)
        selected = images[(images["well"] == well) & (images["timepoint"] == "T0")]
        if selected.empty or selected.iloc[0]["decode_status"] != "ok":
            continue
        item = selected.iloc[0]
        raw = read_grayscale(item["raw_image_path"])
        cf = np.asarray(read_mask(item["cf_image_path"]), dtype=bool)
        height, width = cf.shape
        yy, xx = np.ogrid[:height, :width]
        inner_cf = cf & (
            (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= (min(height, width) * 0.42) ** 2
        )
        objects = _cf_components(cf)
        for obj in objects:
            rows.append(
                {
                    "candidate_id": f"{item['plate_id']}:{well}:T0:cf:{obj['component_index']}",
                    "plate_id": item["plate_id"],
                    "well": well,
                    "timepoint": "T0",
                    "source": "cf_component",
                    **obj,
                }
            )

        display_size = 900
        enhanced = ImageEnhance.Contrast(raw).enhance(1.7).convert("RGB")
        thumb = enhanced.resize((display_size, display_size))
        mask_thumb = Image.fromarray((inner_cf.astype(np.uint8) * 255)).resize(
            (display_size, display_size), resample=Image.Resampling.NEAREST
        )
        green = Image.new("RGB", thumb.size, (0, 255, 0))
        overlay = Image.composite(green, thumb, mask_thumb)
        thumb = Image.blend(thumb, overlay, 0.22)
        draw = ImageDraw.Draw(thumb)
        scale = display_size / raw.width
        for obj in objects:
            x0 = obj["bbox_x"] * scale
            y0 = obj["bbox_y"] * scale
            x1 = (obj["bbox_x"] + obj["bbox_width"]) * scale
            y1 = (obj["bbox_y"] + obj["bbox_height"]) * scale
            draw.rectangle((x0, y0, x1, y1), outline=(255, 40, 40), width=1)
        draw.rectangle((0, 0, 260, 34), fill=(255, 255, 255))
        draw.text((8, 8), f"{well} T0 | CF objects: {len(objects)}", fill=(0, 0, 0))
        thumb.save(output_dir / f"{well}_T0_overlay.png")

    frame = pd.DataFrame(rows)
    destination = artifact_path(config, "predictions", "baseline", "candidate_objects.csv")
    frame.to_csv(destination, index=False, encoding="utf-8")
    return frame

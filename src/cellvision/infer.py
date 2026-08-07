from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
import torch

from .config import artifact_path
from .models.instance_segmenter import TinyUNet
from .manifest import normalize_well


def infer_smoke(
    config: dict[str, Any], checkpoint_path: str | Path, wells: list[str], maximum_candidates: int = 40
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TinyUNet()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    patch_size = int(checkpoint["patch_size_px"])
    half = patch_size // 2
    images = pd.read_csv(artifact_path(config, "manifests", "images.csv"))
    baseline_candidates_path = artifact_path(
        config, "predictions", "baseline", "candidate_objects.csv"
    )
    baseline_candidates = (
        pd.read_csv(baseline_candidates_path)
        if baseline_candidates_path.exists()
        else pd.DataFrame()
    )
    output_dir = artifact_path(config, "predictions", checkpoint_path.parent.name, "overlays", "placeholder").parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []

    for well_value in wells:
        well = normalize_well(well_value)
        selected = images[(images["well"] == well) & (images["timepoint"] == "T0")]
        if selected.empty or selected.iloc[0]["decode_status"] != "ok":
            continue
        row = selected.iloc[0]
        with Image.open(row["raw_image_path"]) as image:
            gray = image.convert("L")
            raw = np.asarray(gray, dtype=np.uint8)
        candidates: list[tuple[int, int, str]] = []
        if not baseline_candidates.empty:
            local_cf = baseline_candidates[
                (baseline_candidates["well"] == well)
                & (baseline_candidates["timepoint"] == "T0")
            ]
            candidates.extend(
                (int(item.x_px), int(item.y_px), "cf_component")
                for item in local_cf.head(maximum_candidates).itertuples(index=False)
            )
        if len(candidates) < maximum_candidates:
            with Path(row["cells_csv_path"]).open("r", newline="") as handle:
                for values in csv.reader(handle):
                    if len(values) < 2:
                        continue
                    x, y = int(float(values[0])), int(float(values[1]))
                    radius = np.hypot(x - raw.shape[1] / 2, y - raw.shape[0] / 2)
                    if (
                        half <= x < raw.shape[1] - half
                        and half <= y < raw.shape[0] - half
                        and radius <= min(raw.shape) * 0.42
                        and all(np.hypot(x - px, y - py) > 4 for px, py, _ in candidates)
                    ):
                        candidates.append((x, y, "instrument_csv"))
                    if len(candidates) >= maximum_candidates:
                        break
        display_size = 900
        display = ImageEnhance.Contrast(gray).enhance(1.7).convert("RGB").resize((display_size, display_size))
        draw = ImageDraw.Draw(display)
        scale = display_size / raw.shape[1]
        for index, (x, y, source) in enumerate(candidates):
            patch = raw[y - half:y + half, x - half:x + half].astype(np.float32) / 255.0
            tensor = torch.from_numpy(patch[None, None])
            with torch.no_grad():
                probability = torch.sigmoid(model(tensor))[0, 0].numpy()
            positive_fraction = float((probability >= checkpoint["threshold"]).mean())
            confidence = float(probability.max())
            prediction_rows.append(
                {
                    "candidate_id": f"{well}:T0:csv:{index}",
                    "well": well,
                    "timepoint": "T0",
                    "x_px": x,
                    "y_px": y,
                    "candidate_source": source,
                    "foreground_fraction": positive_fraction,
                    "confidence": confidence,
                    "decision_source": "weak_cf_segmenter",
                }
            )
            color = (0, 255, 0) if positive_fraction > 0.002 else (255, 180, 0)
            radius = max(2, int(half * scale))
            sx, sy = int(x * scale), int(y * scale)
            draw.rectangle((sx - radius, sy - radius, sx + radius, sy + radius), outline=color, width=1)
        display.save(output_dir / f"{well}_T0_weak_inference.png")

    result = pd.DataFrame(prediction_rows)
    destination = output_dir.parent / "candidate_patch_predictions.csv"
    result.to_csv(destination, index=False, encoding="utf-8")
    return destination

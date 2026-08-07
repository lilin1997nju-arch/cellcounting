from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from cellvision.temporal_appearance import refine_ambiguous_temporal_appearance


def _image(path: Path, radius: int, offset: int = 0) -> None:
    image = Image.new("L", (128, 128), 150)
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (64 - radius + offset, 64 - radius, 64 + radius + offset, 64 + radius),
        fill=35,
        outline=230,
        width=2,
    )
    image.save(path)


def _row(path: Path, timepoint: str, area: float) -> dict:
    return {
        "candidate_id": f"x:{timepoint}",
        "well": "B2",
        "timepoint": timepoint,
        "x_px": 64.0,
        "y_px": 64.0,
        "aligned_x_px": 64.0,
        "aligned_y_px": 64.0,
        "area_px": area,
        "raw_image_path": str(path),
        "cell_probability": 0.60,
        "debris_probability": 0.34,
        "predicted_multiplicity": "single",
        "integrated_label": "uncertain",
        "integrated_confidence": 0.60,
        "is_duplicate_suppressed": False,
        "is_hierarchy_suppressed": False,
    }


def test_identical_three_timepoint_object_supports_debris(tmp_path: Path) -> None:
    paths = [tmp_path / f"t{i}.png" for i in range(3)]
    for path in paths:
        _image(path, 6)
    frame = pd.DataFrame(
        [_row(path, timepoint, 110.0) for path, timepoint in zip(paths, ("T0", "T1", "T2"))]
    )
    result = refine_ambiguous_temporal_appearance(frame, {"temporal_appearance": {}})
    assert set(result["integrated_label"]) == {"debris"}
    assert set(result["temporal_appearance_status"]) == {"static_pixel_identity_debris"}


def test_growing_object_supports_cell_without_merging_candidates(tmp_path: Path) -> None:
    paths = [tmp_path / f"t{i}.png" for i in range(3)]
    _image(paths[0], 5)
    _image(paths[1], 8, offset=2)
    _image(paths[2], 10, offset=-2)
    frame = pd.DataFrame(
        [
            _row(paths[0], "T0", 80.0),
            _row(paths[1], "T1", 180.0),
            _row(paths[2], "T2", 280.0),
        ]
    )
    result = refine_ambiguous_temporal_appearance(frame, {"temporal_appearance": {}})
    t0 = result[result["timepoint"] == "T0"].iloc[0]
    assert t0["integrated_label"] == "single"
    assert t0["temporal_appearance_status"] == "changing_shape_or_growth_cell"

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cellvision.review_context import build_review_context


def test_context_searches_registered_neighborhood(tmp_path: Path):
    rows = []
    for index, timepoint in enumerate(("T0", "T1", "T2")):
        image = np.full((320, 320), 120, dtype=np.uint8)
        x, y = 150 + index * 3, 160 + index * 2
        image[y - 3:y + 4, x - 3:x + 4] = 30
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[y - 2:y + 3, x - 2:x + 3] = 255
        for wall_y in (130, 160, 190):
            mask[wall_y - 1:wall_y + 2, 264:273] = 255
        raw_path = tmp_path / f"{timepoint}.tif"
        cf_path = tmp_path / f"{timepoint}-cf.tif"
        Image.fromarray(image).save(raw_path)
        Image.fromarray(mask).convert("1").save(cf_path)
        rows.append(
            {
                "well": "H6",
                "timepoint": timepoint,
                "decode_status": "ok",
                "raw_image_path": str(raw_path),
                "cf_image_path": str(cf_path),
                "width_px": 320,
                "height_px": 320,
            }
        )
    config = {
        "paths": {"artifact_root": str(tmp_path / "artifacts")},
        "calibration": {"resolution_um_per_pixel": 2.08},
        "candidate_filter": {
            "wall_band_start_fraction": 0.33,
            "wall_chain_neighbor_radius_px": 70,
            "wall_chain_min_neighbors": 2,
        },
    }
    context = build_review_context(config, pd.DataFrame(rows), "H6", 150, 160, 256)
    assert all(context["timepoints"][tp]["available"] for tp in ("T0", "T1", "T2"))
    assert context["timepoints"]["T1"]["candidates"]
    candidate = context["timepoints"]["T1"]["candidates"][0]
    assert abs(candidate["x_px"] - 153) < 2
    assert abs(candidate["y_px"] - 162) < 2
    # Instrument-wall components are excluded entirely instead of being
    # exposed as optional low-priority review candidates.
    assert context["timepoints"]["T1"]["suppressed_candidates"] == []
    assert all(
        item["suppression_reason"]
        for item in context["timepoints"]["T1"]["suppressed_candidates"]
    )

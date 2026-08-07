from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from cellvision.config import artifact_path, load_config


def main() -> None:
    config = load_config("configs/default.yaml")
    database = artifact_path(config, "annotations", "annotations.db")
    predictions = pd.read_csv(
        artifact_path(
            config,
            "predictions",
            "teaching_classifier_predictions.csv",
        )
    )
    with sqlite3.connect(database) as connection:
        points = pd.read_sql_query(
            """
            SELECT annotation_id, well, timepoint, x_px, y_px
            FROM annotations
            WHERE object_type = 'cell'
              AND timepoint IN ('T0', 'T1', 'T2')
            ORDER BY annotation_id
            """,
            connection,
        )
    points = points.drop_duplicates(
        ["well", "timepoint", "x_px", "y_px"]
    )
    automatic = predictions[
        ~predictions["candidate_source"].astype(str).isin(
            ["manual_cell_anchor", "manual_annotation_anchor"]
        )
    ]
    report: dict[str, object] = {
        "evaluation": (
            "automatic non-manual candidate recovery around human cell points"
        ),
        "timepoints": {},
    }
    for timepoint in ["T0", "T1", "T2"]:
        local_points = points[points["timepoint"] == timepoint]
        local_automatic = automatic[
            automatic["timepoint"] == timepoint
        ]
        timepoint_report: dict[str, object] = {
            "human_cell_count": int(len(local_points)),
            "automatic_candidate_count": int(len(local_automatic)),
            "radii": {},
        }
        for radius in [8, 12, 16, 24, 32]:
            counts = {
                "candidate": 0,
                "cell_probability_0_50": 0,
                "cell_probability_0_78": 0,
                "cell_probability_0_90": 0,
            }
            for point in local_points.itertuples(index=False):
                well_candidates = local_automatic[
                    local_automatic["well"] == point.well
                ]
                if well_candidates.empty:
                    continue
                distance = np.hypot(
                    well_candidates["x_px"].to_numpy(float)
                    - float(point.x_px),
                    well_candidates["y_px"].to_numpy(float)
                    - float(point.y_px),
                )
                nearby = well_candidates.loc[
                    distance <= radius, "cell_probability"
                ]
                if nearby.empty:
                    continue
                counts["candidate"] += 1
                counts["cell_probability_0_50"] += int(
                    (nearby >= 0.50).any()
                )
                counts["cell_probability_0_78"] += int(
                    (nearby >= 0.78).any()
                )
                counts["cell_probability_0_90"] += int(
                    (nearby >= 0.90).any()
                )
            timepoint_report["radii"][str(radius)] = {
                **counts,
                **{
                    f"{key}_recall": float(
                        value / max(len(local_points), 1)
                    )
                    for key, value in counts.items()
                },
            }
        report["timepoints"][timepoint] = timepoint_report
    output = artifact_path(
        config,
        "diagnostics",
        "latest_timepoint_automatic_recall.json",
    )
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

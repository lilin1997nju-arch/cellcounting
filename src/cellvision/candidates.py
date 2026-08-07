from __future__ import annotations

import csv
from pathlib import Path


def read_instrument_candidates(path: str | Path) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    with Path(path).open("r", newline="") as handle:
        for index, row in enumerate(csv.reader(handle)):
            if len(row) < 2:
                continue
            candidates.append(
                {
                    "instrument_index": index,
                    "x_px": float(row[0]),
                    "y_px": float(row[1]),
                    "response": float(row[2]) if len(row) > 2 else float("nan"),
                    "radius_px": float(row[3]) if len(row) > 3 else float("nan"),
                }
            )
    return candidates


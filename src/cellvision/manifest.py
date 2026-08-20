from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import artifact_path
from .decode import inspect_tiff, quick_hash


WELL_RE = re.compile(r"^([A-H])0?([1-9]|1[0-2])$")
SUPPORTED_TIMEPOINTS = ("T0", "T1", "T2", "T3", "T4")


def all_wells(rows: str = "ABCDEFGH", columns: int = 12) -> list[str]:
    return [f"{row}{column}" for row in rows for column in range(1, columns + 1)]


def normalize_well(value: str) -> str:
    match = WELL_RE.match(value.strip().upper())
    if not match:
        raise ValueError(f"Invalid 96-well position: {value}")
    return f"{match.group(1)}{int(match.group(2))}"


def stable_sequence_id(experiment_id: str, plate_id: str, well: str) -> str:
    key = f"{experiment_id}|{plate_id}|{normalize_well(well)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _timepoint_for_folder(folder: str, aliases: dict[str, list[str]]) -> str | None:
    lowered = folder.casefold()
    for timepoint, values in aliases.items():
        if any(lowered == str(value).casefold() for value in values):
            return timepoint
    return None


def _find_session(day_dir: Path) -> Path | None:
    # ``timepoint_directories`` may point directly at a vendor session folder
    # (the timestamped directory containing A1.tif), not only at a date
    # directory.  Check the supplied path itself before recursing so explicit
    # session manifests cannot be silently dropped.
    def contains_raw_well_image(path: Path) -> bool:
        return any(
            candidate.is_file() and WELL_RE.fullmatch(candidate.stem.upper())
            for candidate in path.glob("*.tif")
        )

    if day_dir.is_dir() and contains_raw_well_image(day_dir):
        return day_dir
    candidates = [
        path
        for path in day_dir.rglob("*")
        if path.is_dir() and contains_raw_well_image(path)
    ]
    return sorted(candidates)[0] if candidates else None


def _acquisition_datetime(session: Path) -> str:
    try:
        return datetime.strptime(session.name, "%Y-%m-%d %H-%M-%S").isoformat()
    except ValueError:
        return datetime.fromtimestamp(session.stat().st_mtime).isoformat()


def build_manifest(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = Path(config["paths"]["data_root"])
    exp = config["experiment"]
    aliases = exp["timepoint_aliases"]
    rows: list[dict[str, Any]] = []

    explicit_directories = exp.get("timepoint_directories", {})
    if explicit_directories:
        timepoint_directories = []
        for timepoint, value in explicit_directories.items():
            if str(timepoint) not in SUPPORTED_TIMEPOINTS:
                continue
            day_dir = Path(value)
            if not day_dir.is_absolute():
                day_dir = data_root / day_dir
            timepoint_directories.append((str(timepoint), day_dir))
    else:
        timepoint_directories = [
            (_timepoint_for_folder(day_dir.name, aliases), day_dir)
            for day_dir in sorted(
                path for path in data_root.iterdir() if path.is_dir()
            )
        ]

    for timepoint, day_dir in timepoint_directories:
        if (
            timepoint is None
            or str(timepoint) not in SUPPORTED_TIMEPOINTS
            or not day_dir.exists()
        ):
            continue
        session = _find_session(day_dir)
        if session is None:
            continue
        for well in all_wells(exp["plate_rows"], int(exp["plate_columns"])):
            raw = session / f"{well}.tif"
            cf = session / f"{well}-cf.tif"
            cells = session / f"{well}-cells.csv"
            metadata = inspect_tiff(raw) if raw.exists() else {
                "width_px": "",
                "height_px": "",
                "bit_depth": "",
                "channels": "",
                "pyramid_levels": "",
                "decode_status": "missing",
                "decode_error": "raw image missing",
            }
            cf_meta = inspect_tiff(cf) if cf.exists() else {"decode_status": "missing", "decode_error": "cf image missing"}
            decode_status = metadata["decode_status"]
            decode_error = metadata["decode_error"]
            if cf_meta["decode_status"] != "ok":
                decode_status = "error" if raw.exists() else "missing"
                decode_error = f"{decode_error}; cf={cf_meta['decode_error']}".strip("; ")
            rows.append(
                {
                    "experiment_id": exp["experiment_id"],
                    "plate_id": exp["plate_id"],
                    "well": well,
                    "timepoint": timepoint,
                    "acquisition_datetime": _acquisition_datetime(session),
                    "raw_image_path": str(raw.resolve()) if raw.exists() else "",
                    "cf_image_path": str(cf.resolve()) if cf.exists() else "",
                    "cells_csv_path": str(cells.resolve()) if cells.exists() else "",
                    "metrics_csv_path": str((session / "metricsummary.csv").resolve()),
                    **metadata,
                    "cf_decode_status": cf_meta["decode_status"],
                    "cf_decode_error": cf_meta["decode_error"],
                    "resolution_um_per_pixel": config["calibration"]["resolution_um_per_pixel"],
                    "decode_status": decode_status,
                    "decode_error": decode_error,
                    "source_hash": quick_hash(raw) if raw.exists() else "",
                }
            )

    images = pd.DataFrame(rows).sort_values(["well", "timepoint"]).reset_index(drop=True)
    images_path = artifact_path(config, "manifests", "images.csv")
    images.to_csv(images_path, index=False, encoding="utf-8")

    sequence_rows: list[dict[str, Any]] = []
    for well, group in images.groupby("well"):
        available = {
            row.timepoint: row.decode_status == "ok"
            for row in group.itertuples(index=False)
        }
        t0_t2 = all(available.get(tp, False) for tp in ("T0", "T1", "T2"))
        sequence_rows.append(
            {
                "sequence_id": stable_sequence_id(exp["experiment_id"], exp["plate_id"], well),
                "experiment_id": exp["experiment_id"],
                "plate_id": exp["plate_id"],
                "well": well,
                "t0_available": available.get("T0", False),
                "t1_available": available.get("T1", False),
                "t2_available": available.get("T2", False),
                "t3_available": available.get("T3", False),
                "t4_available": available.get("T4", False),
                "sequence_status": "complete_t0_t2" if t0_t2 else "incomplete",
            }
        )
    sequences = pd.DataFrame(sequence_rows).sort_values("well").reset_index(drop=True)
    sequences.to_csv(artifact_path(config, "manifests", "sequences.csv"), index=False, encoding="utf-8")
    return images, sequences


def split_sequences(config: dict[str, Any]) -> dict[str, list[str]]:
    source = artifact_path(config, "manifests", "sequences.csv")
    if not source.exists():
        build_manifest(config)
    sequences = pd.read_csv(source)
    wells = [normalize_well(value) for value in sequences["well"].tolist()]
    rng = random.Random(int(config["split"]["seed"]))
    rng.shuffle(wells)
    n_total = len(wells)
    n_train = round(n_total * float(config["split"]["train_fraction"]))
    n_val = round(n_total * float(config["split"]["validation_fraction"]))
    split = {
        "train": wells[:n_train],
        "validation": wells[n_train:n_train + n_val],
        "test": wells[n_train + n_val:],
    }
    sets = [set(values) for values in split.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Well-level leakage detected")
    if set().union(*sets) != set(wells):
        raise RuntimeError("Split does not cover every sequence")

    split_dir = artifact_path(config, "splits", "split_manifest.json").parent
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "unit": "well",
        "seed": int(config["split"]["seed"]),
        "counts": {name: len(values) for name, values in split.items()},
        "overlap_count": 0,
        "splits": split,
        "warning": "Single-plate internal split; an independent plate is required for external validation.",
    }
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, selected in split.items():
        sequences[sequences["well"].isin(selected)].to_csv(
            split_dir / f"{name}_sequences.csv", index=False, encoding="utf-8"
        )
    return split

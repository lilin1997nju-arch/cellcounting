"""Materialize the image subset owned by a production Project.

The acquisition folder is an ingest source, not a runtime dependency.  After
the endpoint gate has selected reviewable wells, this module copies the exact
files needed by inference and normal review into the Project and rewrites the
plate config to those Project-owned directories.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


EARLY_TIMEPOINTS = ("T0", "T1", "T2")
LATE_TIMEPOINTS = ("T3", "T4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(source: Path, destination: Path) -> tuple[int, str]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_size = int(source.stat().st_size)
    source_hash = _sha256(source)
    if destination.is_file() and int(destination.stat().st_size) == source_size:
        if _sha256(destination) == source_hash:
            return source_size, source_hash
    temporary = destination.with_suffix(destination.suffix + ".copying")
    if temporary.exists():
        temporary.unlink()
    try:
        shutil.copy2(source, temporary)
        if int(temporary.stat().st_size) != source_size or _sha256(temporary) != source_hash:
            raise OSError(f"Project image verification failed: {source}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return source_size, source_hash


def _record_copy(
    records: list[dict[str, Any]],
    *,
    project_root: Path,
    source: Path,
    destination: Path,
    board_slug: str,
    timepoint: str,
    well: str,
    role: str,
    required: bool,
) -> None:
    if not source.is_file():
        if required:
            raise FileNotFoundError(source)
        return
    size, digest = _copy_verified(source, destination)
    records.append({
        "board_slug": board_slug,
        "timepoint": timepoint,
        "well": well,
        "role": role,
        "project_path": destination.relative_to(project_root).as_posix(),
        "source_path": str(source.resolve()),
        "bytes": size,
        "sha256": digest,
    })


def materialize_project_images(
    config_path: str | Path,
    config: dict[str, Any],
    *,
    positive_wells: set[str],
    gate_rows: list[dict[str, Any]],
    endpoint_csv: str | Path,
    group_id: str,
) -> dict[str, Any]:
    """Copy retained-well assets and make the plate config Project-local.

    T0--T2 retain raw/CF/cells inputs because they are computation inputs.
    T3/T4 retain raw TIFFs because the normal review UI presents all five
    frames.  Endpoint-positive controls retain T4 as gate evidence but remain
    excluded from early inference.
    """

    settings = config.get("project_images", {})
    if not bool(settings.get("enabled")):
        return {"enabled": False, "config": config, "endpoint_csv": str(endpoint_csv)}

    config_file = Path(config_path).expanduser().resolve()
    project_root = Path(str(settings["project_root"])).expanduser().resolve()
    board_slug = str(settings.get("board_slug") or config["experiment"]["plate_id"])
    image_root = Path(str(settings.get("image_root") or project_root / "data" / "images" / board_slug))
    if not image_root.is_absolute():
        image_root = project_root / image_root
    image_root = image_root.resolve()
    try:
        image_root.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Project image root must stay inside Project: {image_root}") from exc

    normalized_positive = {str(well).upper() for well in positive_wells}
    control_wells = {
        str(row.get("well") or "").upper()
        for row in gate_rows
        if bool(row.get("is_positive_control"))
    }
    retained_endpoint_wells = normalized_positive | control_wells
    early_sources = {
        str(timepoint).upper(): Path(str(directory)).expanduser().resolve()
        for timepoint, directory in config["experiment"].get("timepoint_directories", {}).items()
        if str(timepoint).upper() in EARLY_TIMEPOINTS
    }
    late_sources = {
        str(timepoint).upper(): Path(str(directory)).expanduser().resolve()
        for timepoint, directory in config.get("review", {}).get("late_timepoint_directories", {}).items()
        if str(timepoint).upper() in LATE_TIMEPOINTS
    }
    if set(early_sources) != set(EARLY_TIMEPOINTS):
        raise ValueError(f"Project image ingest requires T0/T1/T2, found {sorted(early_sources)}")

    records: list[dict[str, Any]] = []
    for timepoint in EARLY_TIMEPOINTS:
        source_dir = early_sources[timepoint]
        destination_dir = image_root / timepoint
        for well in sorted(normalized_positive):
            for suffix, role, required in (
                (".tif", "raw", True),
                ("-cf.tif", "cf", True),
                ("-cells.csv", "cells", False),
            ):
                _record_copy(
                    records,
                    project_root=project_root,
                    source=source_dir / f"{well}{suffix}",
                    destination=destination_dir / f"{well}{suffix}",
                    board_slug=board_slug,
                    timepoint=timepoint,
                    well=well,
                    role=role,
                    required=required,
                )
        _record_copy(
            records,
            project_root=project_root,
            source=source_dir / "metricsummary.csv",
            destination=destination_dir / "metricsummary.csv",
            board_slug=board_slug,
            timepoint=timepoint,
            well="",
            role="metrics",
            required=False,
        )

    for timepoint in LATE_TIMEPOINTS:
        source_dir = late_sources.get(timepoint)
        if source_dir is None:
            continue
        wells = retained_endpoint_wells if timepoint == "T4" else normalized_positive
        for well in sorted(wells):
            _record_copy(
                records,
                project_root=project_root,
                source=source_dir / f"{well}.tif",
                destination=image_root / timepoint / f"{well}.tif",
                board_slug=board_slug,
                timepoint=timepoint,
                well=well,
                role="raw",
                required=timepoint == "T4",
            )

    local_endpoint = image_root / "endpoint_screening.csv"
    endpoint_frame = pd.read_csv(endpoint_csv, low_memory=False)
    group_mask = endpoint_frame["group_id"].astype(str) == str(group_id)
    endpoint_frame = endpoint_frame[group_mask].copy()
    if endpoint_frame.empty:
        raise ValueError(f"Endpoint screening has no rows for group: {group_id}")
    endpoint_frame["well"] = endpoint_frame["well"].astype(str).str.upper()
    endpoint_frame["raw_image_path"] = endpoint_frame["well"].map(
        lambda well: str((image_root / "T4" / f"{well}.tif").resolve())
        if well in retained_endpoint_wells and (image_root / "T4" / f"{well}.tif").is_file()
        else ""
    )
    if "cf_mask_path" in endpoint_frame.columns:
        endpoint_frame["cf_mask_path"] = ""
    local_endpoint.parent.mkdir(parents=True, exist_ok=True)
    endpoint_frame.to_csv(local_endpoint, index=False, encoding="utf-8")

    local_directories = {tp: str((image_root / tp).resolve()) for tp in EARLY_TIMEPOINTS}
    local_late = {tp: str((image_root / tp).resolve()) for tp in LATE_TIMEPOINTS if tp in late_sources}
    config["paths"]["data_root"] = str((project_root / "data").resolve())
    config["experiment"]["timepoint_directories"] = local_directories
    config.setdefault("review", {})["late_timepoint_directories"] = local_late
    config.setdefault("gated_report", {})["day14_csv"] = str(local_endpoint)
    config["gated_report"]["endpoint_csv"] = str(local_endpoint)
    config["project_images"] = {
        **settings,
        "enabled": True,
        "state": "materialized",
        "project_root": str(project_root),
        "image_root": str(image_root),
        "board_slug": board_slug,
        "active_endpoint_csv": str(local_endpoint),
    }

    raw_config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    raw_config.setdefault("paths", {})["data_root"] = config["paths"]["data_root"]
    raw_config.setdefault("experiment", {})["timepoint_directories"] = local_directories
    raw_config.setdefault("review", {})["late_timepoint_directories"] = local_late
    raw_config.setdefault("gated_report", {})["day14_csv"] = str(local_endpoint)
    raw_config["gated_report"]["endpoint_csv"] = str(local_endpoint)
    raw_config["project_images"] = config["project_images"]
    config_file.write_text(
        yaml.safe_dump(raw_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    inventory_dir = project_root / "data" / "manifests"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    inventory_csv = inventory_dir / f"{board_slug}-images.csv"
    pd.DataFrame(records).to_csv(inventory_csv, index=False, encoding="utf-8")
    no_growth_wells = sorted(
        str(row.get("well") or "").upper()
        for row in gate_rows
        if not bool(row.get("is_positive_control"))
        and str(row.get("well") or "").upper() not in normalized_positive
    )
    summary = {
        "format": "cellvision-project-images",
        "version": 1,
        "board_slug": board_slug,
        "group_id": str(group_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "path_mode": "project_owned",
        "reviewable_wells": sorted(normalized_positive),
        "positive_control_wells": sorted(control_wells),
        "no_growth_wells_without_images": no_growth_wells,
        "file_count": len(records),
        "copied_bytes": int(sum(int(row["bytes"]) for row in records)),
        "inventory_csv": inventory_csv.relative_to(project_root).as_posix(),
        "endpoint_screening_csv": local_endpoint.relative_to(project_root).as_posix(),
    }
    summary_path = inventory_dir / f"{board_slug}-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**summary, "enabled": True, "config": config, "endpoint_csv": str(local_endpoint)}

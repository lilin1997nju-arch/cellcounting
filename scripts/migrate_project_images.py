"""Migrate a completed legacy Project to Project-owned review images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from cellvision.config import artifact_path, load_config
from cellvision.gated_screening import build_gated_plate_report
from cellvision.project_images import materialize_project_images


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return False
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _rewrite_plate_paths(root: Path, replacements: dict[str, str]) -> int:
    variants = dict(replacements)
    variants.update({
        source.replace("\\", "/"): destination.replace("\\", "/")
        for source, destination in replacements.items()
    })
    changed = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in {".csv", ".json", ".yaml", ".yml"}:
            continue
        if "cache" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for source, destination in variants.items():
            updated = updated.replace(source, destination)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def migrate(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    project_root = manifest_file.parent
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    previous_root = str(manifest.get("root") or "")
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    if not source:
        source = {
            "root": previous_root,
            "index": str(manifest.get("source_index") or ""),
            "access_policy": "ingest_only",
        }

    migrated_plates: list[dict[str, Any]] = []
    for plate in manifest.get("plates", []):
        if not isinstance(plate, dict):
            continue
        config_path = Path(str(plate["config"])).expanduser().resolve()
        config = load_config(config_path)
        group_id = str(plate.get("group_id") or config.get("gated_report", {}).get("group_id") or "")
        board_slug = str(plate.get("slug") or config["experiment"]["plate_id"])
        source_endpoint = Path(
            str(
                config.get("gated_report", {}).get("endpoint_csv")
                or config.get("gated_report", {}).get("day14_csv")
                or manifest.get("source_endpoint_csv")
                or manifest.get("source_day14_csv")
            )
        ).expanduser().resolve()
        endpoint_frame = pd.read_csv(source_endpoint, low_memory=False)
        selected = endpoint_frame[endpoint_frame["group_id"].astype(str) == group_id].copy()
        if selected.empty:
            raise ValueError(f"Endpoint screening has no rows for group: {group_id}")
        growth_column = (
            "endpoint_obvious_sheet_growth"
            if "endpoint_obvious_sheet_growth" in selected.columns
            else "day14_obvious_sheet_growth"
        )
        positive_wells = {
            str(row["well"]).upper()
            for row in selected.to_dict(orient="records")
            if _truth(row.get(growth_column)) and not _truth(row.get("is_positive_control"))
        }
        source_directories = {
            str(timepoint).upper(): str(directory)
            for timepoint, directory in config["experiment"].get("timepoint_directories", {}).items()
        }
        source_directories.update({
            str(timepoint).upper(): str(directory)
            for timepoint, directory in config.get("review", {}).get("late_timepoint_directories", {}).items()
        })
        config["project_images"] = {
            "enabled": True,
            "state": "pending_migration",
            "project_root": str(project_root),
            "image_root": str(project_root / "data" / "images" / board_slug),
            "board_slug": board_slug,
        }
        result = materialize_project_images(
            config_path,
            config,
            positive_wells=positive_wells,
            gate_rows=selected.to_dict(orient="records"),
            endpoint_csv=source_endpoint,
            group_id=group_id,
        )
        local_directories = {
            **result["config"]["experiment"]["timepoint_directories"],
            **result["config"].get("review", {}).get("late_timepoint_directories", {}),
        }
        replacements = {
            str(Path(source_path).expanduser().resolve()): str(Path(local_directories[timepoint]).resolve())
            for timepoint, source_path in source_directories.items()
            if timepoint in local_directories
        }
        artifact_root = Path(str(plate["artifact_root"])).expanduser().resolve()
        rewritten = _rewrite_plate_paths(artifact_root, replacements)

        early_screening = artifact_path(result["config"], "predictions", "latest_well_screening.csv")
        report = build_gated_plate_report(
            result["endpoint_csv"],
            group_id,
            Path(str(plate.get("gated_output_dir") or artifact_root / "gated")),
            early_screening_csv=early_screening if early_screening.is_file() else None,
            sessions_csv=manifest.get("source_sessions_csv"),
            locate_day7=False,
            endpoint_day_label=str(manifest.get("endpoint_day_label") or "Day14"),
        )
        migrated_plates.append({
            "slug": board_slug,
            "reviewable_wells": len(positive_wells),
            "file_count": int(result["file_count"]),
            "copied_bytes": int(result["copied_bytes"]),
            "rewritten_artifacts": rewritten,
            "report_json": report["report_json"],
        })

    manifest["root"] = str(project_root)
    manifest["source"] = {**source, "access_policy": "ingest_only"}
    manifest["image_storage"] = {
        "mode": "project_owned_after_endpoint_gate",
        "root": str(project_root / "data" / "images"),
        "no_growth_images_retained": False,
    }
    manifest["image_migration"] = {
        "version": 1,
        "plates": migrated_plates,
    }
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "project": str(manifest_file),
        "plate_count": len(migrated_plates),
        "copied_bytes": sum(item["copied_bytes"] for item in migrated_plates),
        "plates": migrated_plates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

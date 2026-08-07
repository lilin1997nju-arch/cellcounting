"""Create one project manifest and one T0/T2-only config per QL2603 board."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-csv", default="artifacts/ingest/ql2603_sessions.csv")
    parser.add_argument("--day14-csv", default="artifacts/day14_screening/ql2603/ql2603_day14_well_screening.csv")
    parser.add_argument("--endpoint-csv", default="", help="Endpoint screening CSV; defaults to --day14-csv")
    parser.add_argument("--endpoint-day-label", default="Day14")
    parser.add_argument("--output", default="artifacts/projects/ql2603/project.json")
    parser.add_argument("--data-root", default="", help="Raw export root; defaults to the historical QL2603 location")
    parser.add_argument("--artifact-root", default="", help="Artifact root; defaults to artifacts/projects/ql2603")
    parser.add_argument("--config-dir", default="", help="Generated config directory; defaults to configs/generated")
    args = parser.parse_args()

    sessions = pd.read_csv(resolve(args.sessions_csv))
    sessions["group_id"] = sessions["group_id"].astype(str)
    sessions = sessions[sessions["group_id"].str.startswith("QL2603 ")].copy()
    if sessions.empty:
        raise SystemExit("No QL2603 groups found in sessions CSV")
    endpoint_csv_value = args.endpoint_csv or args.day14_csv
    endpoint = pd.read_csv(resolve(endpoint_csv_value), usecols=["group_id"])
    available_groups = set(endpoint["group_id"].astype(str))
    config_dir = resolve(args.config_dir) if args.config_dir else ROOT / "configs" / "generated"
    config_dir.mkdir(parents=True, exist_ok=True)
    project_root = resolve(args.artifact_root) if args.artifact_root else ROOT / "artifacts" / "projects" / "ql2603"
    project_root.mkdir(parents=True, exist_ok=True)
    data_root = resolve(args.data_root) if args.data_root else ROOT.parent / "20260623 QL2603"
    plates: list[dict[str, object]] = []

    for group_id, group in sessions.groupby("group_id", sort=True):
        group = group.sort_values("session_ordinal")
        board_id = str(group.iloc[0].get("board_id") or group_id.split()[-1])
        board_slug = slug(f"ql2603-{board_id}")
        by_tp = {str(row.timepoint_label): str(row.session_path) for row in group.itertuples()}
        missing = [label for label in ("T0", "T1", "T2", "T3", "T4") if not by_tp.get(label)]
        if missing:
            raise SystemExit(f"{group_id} missing timepoints: {missing}")
        artifact_root = project_root / "plates" / board_slug
        gated_dir = artifact_root / "gated"
        config_path = config_dir / f"{board_slug}.yaml"
        config = {
            "base_config": "configs/default.yaml",
            "paths": {"data_root": str(data_root), "artifact_root": str(artifact_root)},
            "experiment": {
                "experiment_id": f"{group_id} full project",
                "plate_id": f"QL2603_{board_id}",
                "timepoint_directories": {label: by_tp[label] for label in ("T0", "T1", "T2")},
            },
            "morphology_classifier": {"excluded_wells": ["A1"], "reuse_candidate_manifest": False},
            "dense_detection": {"include_manual_anchors": False, "reuse_unchanged_stage": False},
            "v2_inference": {"reuse_unchanged_stage": False},
            "review_queue": {"excluded_wells": ["A1"], "apply_manual_point_overrides": False},
            "joint_training": {"sources": []},
            "review": {
                "late_timepoint_directories": {"T3": by_tp["T3"], "T4": by_tp["T4"]},
                "timepoint_display_names": {"T0": "T0", "T1": "T1", "T2": "T2", "T3": "Day7", "T4": "Day14"},
                "day14_overlay": {"downsample": 4, "minimum_component_coverage_pct": 1.0, "minimum_radius_px": 40.0, "minimum_mean_distance_px": 10.0},
            },
            "gated_report": {
                "group_id": group_id,
                "day14_csv": str(resolve(args.day14_csv)),
                "endpoint_csv": str(resolve(endpoint_csv_value)),
                "endpoint_day_label": args.endpoint_day_label,
                "sessions_csv": str(resolve(args.sessions_csv)),
                "output_dir": str(gated_dir),
            },
        }
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        plates.append({
            "slug": board_slug,
            "group_id": group_id,
            "board_id": board_id,
            "config": str(config_path),
            "artifact_root": str(artifact_root),
            "gated_output_dir": str(gated_dir),
            "images_manifest": str(artifact_root / "manifests" / "images.csv"),
            "pipeline_summary": str(gated_dir / "pipeline_summary.json"),
            "report_json": str(gated_dir / "plate_overview.json"),
            "status": "ready" if group_id in available_groups else "missing_day14_screening",
        })
    manifest = {
        "project_id": "ql2603",
        "project_name": "QL2603",
        "root": str(data_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sessions_csv": str(resolve(args.sessions_csv)),
        "source_day14_csv": str(resolve(args.day14_csv)),
        "source_endpoint_csv": str(resolve(endpoint_csv_value)),
        "endpoint_day_label": args.endpoint_day_label,
        "plates": plates,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"project": str(output.resolve()), "plates": len(plates), "configs": len(plates)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

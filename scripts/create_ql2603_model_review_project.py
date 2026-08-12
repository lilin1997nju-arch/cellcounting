"""Create a standalone project manifest for the six-board QL2603 model run.

The deployment runner intentionally writes to a timestamped shadow directory
so production QL2603 artifacts remain unchanged.  This script turns that
shadow directory into a normal project-server project: it creates per-plate
configs, rebuilds the gated summary from the new screening output, and writes
one manifest that the existing 8777 review hub can mount.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cellvision.config import load_config
from cellvision.gated_screening import build_gated_plate_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT
    / "artifacts"
    / "v2"
    / "runs"
    / "ql2603-model-deployment-20260812-142424"
    / "report.json"
)
DEFAULT_SOURCE_MANIFEST = ROOT / "artifacts" / "projects" / "ql2603" / "project.json"
DEFAULT_PROJECT_ID = "ql2603-model-review-20260812"
DEFAULT_PROJECT_NAME = "QL2603 六板新模型审核"
DEFAULT_PLATES = (
    "ql2603-t1-1",
    "ql2603-t1-2",
    "ql2603-t4-2",
    "ql2603-t2-4",
    "ql2603-t5-1",
    "ql2603-t5-2",
)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def latest_late_growth_overrides(database: Path) -> dict[str, str]:
    """Carry the source board's existing late-growth decisions into the copy."""

    if not database.exists():
        return {}
    try:
        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                """
                SELECT well, decision, updated_at
                FROM late_growth_reviews
                ORDER BY updated_at, late_growth_review_id
                """
            ).fetchall()
    except sqlite3.Error:
        return {}
    latest: dict[str, str] = {}
    for well, decision, _ in rows:
        latest[str(well).upper()] = str(decision)
    return latest


def write_plate_config(
    source: Path,
    target: Path,
    artifact_root: Path,
    gated_output_dir: Path,
    group_id: str,
) -> Path:
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {source}")
    paths = raw.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"paths must be a mapping: {source}")
    paths["artifact_root"] = str(artifact_root.resolve())
    gated = raw.setdefault("gated_report", {})
    if not isinstance(gated, dict):
        raise ValueError(f"gated_report must be a mapping: {source}")
    gated["group_id"] = group_id
    gated["output_dir"] = str(gated_output_dir.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    # Resolve base_config references and validate the result before mounting.
    load_config(target)
    return target.resolve()


def make_pipeline_summary(
    deployment: dict[str, Any],
    deployment_summary: Path,
    report_json: Path,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "plate": deployment.get("plate"),
        "review_group": deployment.get("review_group"),
        "total_elapsed_seconds": deployment.get("elapsed_seconds"),
        "source_deployment_summary": str(deployment_summary.resolve()),
        "report_json": str(report_json.resolve()),
        "model_checkpoints": {
            "instance": deployment.get("instance_checkpoint"),
            "temporal": deployment.get("temporal_checkpoint"),
            "v3_config": deployment.get("v3_config"),
        },
        "final_integrated_label_counts": deployment.get(
            "final_integrated_label_counts", {}
        ),
        "screening_summary": deployment.get("screening_summary", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--project-name", default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--plates", nargs="+", default=list(DEFAULT_PLATES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    report_path = resolve(args.report)
    source_manifest_path = resolve(args.source_manifest)
    deployment_root = report_path.parent
    report = read_json(report_path)
    source_manifest = read_json(source_manifest_path)
    project_dir = ROOT / "artifacts" / "projects" / str(args.project_id)
    manifest_path = project_dir / "project.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"project already exists; use --force to refresh generated files: {manifest_path}"
        )

    source_plates = {
        str(item.get("slug")): item
        for item in source_manifest.get("plates", [])
        if isinstance(item, dict) and item.get("slug")
    }
    deployment_summaries = {
        str(item.get("plate")): item
        for item in report.get("plates", [])
        if isinstance(item, dict) and item.get("plate")
    }

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    plates: list[dict[str, Any]] = []
    for slug in args.plates:
        shadow_root = deployment_root / "deployment" / slug
        deployment_summary_path = shadow_root / "deployment_summary.json"
        deployment = read_json(deployment_summary_path)
        source_plate = source_plates.get(slug, {})
        source_config = ROOT / "configs" / "generated" / f"{slug}.yaml"
        if not source_config.exists():
            raise FileNotFoundError(source_config)
        if not (shadow_root / "manifests" / "images.csv").exists():
            raise FileNotFoundError(shadow_root / "manifests" / "images.csv")
        if not (shadow_root / "predictions" / "latest_v3_predictions.csv").exists():
            raise FileNotFoundError(
                shadow_root / "predictions" / "latest_v3_predictions.csv"
            )

        group_id = str(
            source_plate.get("group_id")
            or deployment.get("plate")
            or slug
        )
        config_path = project_dir / "configs" / f"{slug}.yaml"
        gated_output_dir = shadow_root / "gated"
        write_plate_config(
            source_config,
            config_path,
            shadow_root,
            gated_output_dir,
            group_id,
        )
        config = load_config(config_path)
        gated = config.get("gated_report", {})
        endpoint_csv = gated.get("endpoint_csv") or gated.get("day14_csv")
        if not endpoint_csv:
            raise ValueError(f"gated endpoint CSV is missing in {config_path}")
        early_screening_csv = shadow_root / "predictions" / "latest_well_screening.csv"
        if not early_screening_csv.exists():
            raise FileNotFoundError(early_screening_csv)
        report_payload = build_gated_plate_report(
            endpoint_csv,
            group_id,
            gated_output_dir,
            early_screening_csv=early_screening_csv,
            sessions_csv=gated.get("sessions_csv"),
            locate_day7=False,
            day14_growth_overrides=latest_late_growth_overrides(
                shadow_root / "annotations" / "annotations.db"
            ),
            endpoint_day_label=str(gated.get("endpoint_day_label", "Day14")),
        )
        report_json_path = gated_output_dir / "plate_overview.json"
        pipeline_summary_path = gated_output_dir / "pipeline_summary.json"
        write_json(
            pipeline_summary_path,
            make_pipeline_summary(
                deployment,
                deployment_summary_path,
                report_json_path,
            ),
        )

        category_counts = {
            str(key): int(value or 0)
            for key, value in report_payload.get("category_counts", {}).items()
        }
        plates.append(
            {
                "slug": slug,
                "group_id": group_id,
                "board_id": str(
                    source_plate.get("board_id") or slug.removeprefix("ql2603-")
                ),
                "config": str(config_path.resolve()),
                "artifact_root": str(shadow_root.resolve()),
                "gated_output_dir": str(gated_output_dir.resolve()),
                "images_manifest": str(
                    (shadow_root / "manifests" / "images.csv").resolve()
                ),
                "pipeline_summary": str(pipeline_summary_path.resolve()),
                "report_json": str(report_json_path.resolve()),
                "status": "completed",
                "started_at": generated_at,
                "finished_at": generated_at,
                "elapsed_seconds": deployment.get("elapsed_seconds"),
                "category_counts": category_counts,
                "review_group": deployment.get("review_group"),
                "source_artifact_root": deployment.get("input", {}).get(
                    "source_root"
                ),
                "deployment_summary": str(deployment_summary_path.resolve()),
            }
        )

    project = {
        "project_id": str(args.project_id),
        "project_name": str(args.project_name),
        "created_by": "Codex",
        "root": source_manifest.get("root"),
        "generated_at": generated_at,
        "status": "completed",
        "source_project_id": source_manifest.get("project_id"),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_sessions_csv": source_manifest.get("source_sessions_csv"),
        "model_deployment_report": str(report_path.resolve()),
        "model_run_root": str(deployment_root.resolve()),
        "model_selection": {
            "reviewed_plates": [
                slug for slug in args.plates if slug in {"ql2603-t1-1", "ql2603-t1-2", "ql2603-t4-2"}
            ],
            "unreviewed_plates": [
                slug for slug in args.plates if slug not in {"ql2603-t1-1", "ql2603-t1-2", "ql2603-t4-2"}
            ],
            "deployment_report": str(report_path.resolve()),
        },
        "plates": plates,
        "last_updated_at": generated_at,
    }
    write_json(manifest_path, project)
    print(json.dumps({
        "project_id": project["project_id"],
        "project_name": project["project_name"],
        "manifest": str(manifest_path.resolve()),
        "plate_count": len(plates),
        "plates": [item["slug"] for item in plates],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

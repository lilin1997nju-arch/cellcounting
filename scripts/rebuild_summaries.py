"""Rebuild all per-plate quick-review summaries across every project.

Use this after a code change that alters review matching (for example a
SUMMARY_VERSION bump) so the persisted summaries no longer undercount reviewed
progress.  It walks every artifacts/projects/*/project.json and regenerates
each plate's quick_review_summary.json with the current matching logic.
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import pandas as pd

from cellvision.config import load_config
from cellvision.review_quick_review import build_quick_review_service
from cellvision.review_server import _review_images_manifest
from cellvision.review_summary import summary_path


def rebuild_plate(plate: dict, images_manifest: pd.DataFrame) -> str:
    config = load_config(plate["config"])
    database = Path(config["paths"]["artifact_root"]) / "annotations" / "annotations.db"
    service = build_quick_review_service(
        config=config,
        database=database,
        images_manifest=images_manifest,
        prediction_cache={"mtime": None, "source": None, "frame": pd.DataFrame()},
        proposal_cache={"mtime": None, "frame": pd.DataFrame()},
        completion_review_cache={"mtime": None, "frame": pd.DataFrame()},
        quick_frame_cache={"key": None, "frame": pd.DataFrame()},
        quick_summary_cache={"signature": None, "payload": None},
        quick_summary_lock=threading.RLock(),
        quick_summary_file=str(summary_path(config["paths"]["artifact_root"])),
        gated_lookup=lambda: {},
        gated_report_path=lambda: None,
        ui_screening_status=lambda g, f: f,
        ui_screening_status_label=lambda s: s,
        ui_status_aliases={},
    )
    summary = service.quick_review_summary(force=True)
    return f"{summary.get('completed_well_count')}/{summary.get('well_count')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--projects-root",
        default="artifacts/projects",
        help="Directory containing one subdirectory per project (project.json inside)",
    )
    parser.add_argument("--only", nargs="*", default=None, help="Only rebuild these project slugs")
    args = parser.parse_args()

    root = Path(args.projects_root)
    wanted = {str(value) for value in (args.only or [])}
    rebuilt = 0
    for manifest_file in sorted(root.glob("*/project.json")):
        project_id = manifest_file.parent.name
        if wanted and project_id not in wanted:
            continue
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for plate in manifest.get("plates", []):
            if not isinstance(plate, dict) or not plate.get("config"):
                continue
            try:
                config = load_config(plate["config"])
                database = Path(config["paths"]["artifact_root"]) / "annotations" / "annotations.db"
                if not database.exists():
                    continue
                manifest_path = Path(config["paths"]["artifact_root"]) / "manifests" / "images.csv"
                images_manifest = _review_images_manifest(
                    config, pd.read_csv(manifest_path)
                )
                progress = rebuild_plate(plate, images_manifest)
                rebuilt += 1
                print(f"{project_id}/{plate.get('slug')}: {progress}", flush=True)
            except Exception as exc:  # keep going past one bad plate
                print(f"{project_id}/{plate.get('slug')}: SKIP {type(exc).__name__}", flush=True)
    print(f"rebuilt plates: {rebuilt}")


if __name__ == "__main__":
    main()

"""Run all QL2603 boards sequentially with the Day14 compute gate."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from run_day14_gated_plate import run as run_plate
from cellvision.config import load_config
from cellvision.review_server import rebuild_quick_review_summary


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/projects/ql2603/project.json")
    parser.add_argument("--source-artifacts", default="artifacts")
    parser.add_argument("--only", default="", help="Comma-separated board ids/slugs; default is all")
    args = parser.parse_args()
    manifest_path = resolve(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = {item.strip().casefold() for item in args.only.split(",") if item.strip()}
    project_started = time.perf_counter()
    run_rows: list[dict] = []
    for index, plate in enumerate(manifest.get("plates", []), start=1):
        board_id = str(plate.get("board_id", ""))
        slug = str(plate.get("slug", ""))
        if selected and board_id.casefold() not in selected and slug.casefold() not in selected:
            continue
        config = resolve(plate["config"])
        output_dir = resolve(plate["gated_output_dir"])
        endpoint_csv = resolve(manifest.get("source_endpoint_csv", manifest["source_day14_csv"]))
        endpoint_day_label = str(manifest.get("endpoint_day_label", "Day14"))
        sessions_csv = resolve(manifest["source_sessions_csv"])
        plate["status"] = "running"
        plate["started_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)
        print(json.dumps({"event": "plate_started", "index": index, "board_id": board_id}, ensure_ascii=False), flush=True)
        started = time.perf_counter()
        try:
            summary = run_plate(
                argparse.Namespace(
                    config=str(config),
                    day14_csv=str(endpoint_csv),
                    endpoint_day_label=endpoint_day_label,
                    sessions_csv=str(sessions_csv),
                    group_id=str(plate["group_id"]),
                    output_dir=str(output_dir),
                    source_artifacts=str(resolve(args.source_artifacts)),
                )
            )
            try:
                rebuild_quick_review_summary(load_config(config))
            except Exception as summary_error:
                print(
                    json.dumps(
                        {
                            "event": "review_summary_warning",
                            "board_id": board_id,
                            "error": f"{type(summary_error).__name__}: {summary_error}",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            elapsed = round(time.perf_counter() - started, 3)
            plate["status"] = "completed"
            plate["finished_at"] = datetime.now(timezone.utc).isoformat()
            plate["elapsed_seconds"] = elapsed
            plate["category_counts"] = summary.get("category_counts", {})
            row = {"board_id": board_id, "group_id": plate["group_id"], "status": "completed", "elapsed_seconds": elapsed, "category_counts": summary.get("category_counts", {})}
            print(json.dumps({"event": "plate_finished", **row}, ensure_ascii=False), flush=True)
        except Exception as exc:  # keep the project queue moving after a bad board
            elapsed = round(time.perf_counter() - started, 3)
            plate["status"] = "error"
            plate["finished_at"] = datetime.now(timezone.utc).isoformat()
            plate["elapsed_seconds"] = elapsed
            plate["error"] = f"{type(exc).__name__}: {exc}"
            row = {"board_id": board_id, "group_id": plate["group_id"], "status": "error", "elapsed_seconds": elapsed, "error": plate["error"]}
            print(json.dumps({"event": "plate_failed", **row}, ensure_ascii=False), flush=True)
        run_rows.append(row)
        manifest["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)
    summary = {
        "project_id": manifest.get("project_id"),
        "board_count": len(run_rows),
        "completed_count": sum(row["status"] == "completed" for row in run_rows),
        "error_count": sum(row["status"] == "error" for row in run_rows),
        "elapsed_seconds": round(time.perf_counter() - project_started, 3),
        "boards": run_rows,
    }
    summary_path = manifest_path.parent / "project_run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "project_finished", "summary": str(summary_path), **{key: value for key, value in summary.items() if key != "boards"}}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

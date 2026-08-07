"""Rebuild the project-level QL2603 summary from per-board reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/projects/ql2603/project.json")
    args = parser.parse_args()
    manifest_path = resolve(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate: dict[str, int] = {}
    rows: list[dict] = []
    for plate in manifest.get("plates", []):
        report_path = resolve(plate.get("report_json", ""))
        pipeline_path = resolve(plate.get("pipeline_summary", ""))
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8")) if pipeline_path.exists() else {}
        counts = {str(key): int(value or 0) for key, value in (report.get("category_counts") or {}).items()}
        for key, value in counts.items(): aggregate[key] = aggregate.get(key, 0) + value
        rows.append({
            "board_id": plate.get("board_id"),
            "group_id": plate.get("group_id"),
            "status": "completed" if report else plate.get("status", "ready"),
            "elapsed_seconds": pipeline.get("total_elapsed_seconds", plate.get("elapsed_seconds")),
            "day14_positive_sample_wells": report.get("day14_positive_sample_wells", 0),
            "day14_skipped_sample_wells": report.get("day14_skipped_sample_wells", 0),
            "category_counts": counts,
        })
    summary = {
        "project_id": manifest.get("project_id"),
        "project_name": manifest.get("project_name"),
        "board_count": len(rows),
        "completed_count": sum(row["status"] == "completed" for row in rows),
        "error_count": sum(row["status"] == "error" for row in rows),
        "sum_board_elapsed_seconds": round(sum(float(row["elapsed_seconds"] or 0) for row in rows), 3),
        "day14_positive_sample_wells": int(sum(int(row["day14_positive_sample_wells"] or 0) for row in rows)),
        "day14_skipped_sample_wells": int(sum(int(row["day14_skipped_sample_wells"] or 0) for row in rows)),
        "category_counts": aggregate,
        "boards": rows,
    }
    output = manifest_path.parent / "project_run_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "boards"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

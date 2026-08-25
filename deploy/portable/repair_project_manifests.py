"""Back up and safely recover Cell Vision project metadata.

The tool never deletes project data.  It restores an absent project.json from
an exact metadata backup first, and only falls back to the project catalog when
every catalogued plate still has the files required by the review UI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKUP_FOLDER = ".metadata-backups"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def _project_directories(projects_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in projects_root.iterdir()
            if path.is_dir() and path.name != BACKUP_FOLDER
        ),
        key=lambda path: path.name.casefold(),
    )


def backup_project_metadata(projects_root: str | Path) -> dict[str, Any]:
    root = Path(projects_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    backup_root = root / BACKUP_FOLDER
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:8]}"
    destination = backup_root / stamp
    destination.mkdir(parents=True, exist_ok=False)
    copied: list[str] = []

    for project_dir in _project_directories(root):
        relative_files = [Path("project.json"), Path("task_queue.json")]
        plan_dir = project_dir / "task_plans"
        if plan_dir.is_dir():
            relative_files.extend(
                path.relative_to(project_dir)
                for path in sorted(plan_dir.glob("*.json"))
                if path.is_file()
            )
        for relative in relative_files:
            source = project_dir / relative
            if not source.is_file():
                continue
            target = destination / project_dir.name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(Path(project_dir.name) / relative))

    catalog_path = root / "project_catalog.sqlite"
    catalog_backup = destination / "project_catalog.sqlite"
    if catalog_path.is_file():
        source_connection = sqlite3.connect(str(catalog_path), timeout=30)
        try:
            destination_connection = sqlite3.connect(str(catalog_backup))
            try:
                source_connection.backup(destination_connection)
            finally:
                destination_connection.close()
        finally:
            source_connection.close()

    metadata = {
        "format": "cellvision-project-metadata-backup",
        "version": 1,
        "created_at": _now(),
        "projects_root": str(root),
        "files": copied,
        "catalog_included": catalog_backup.is_file(),
    }
    _atomic_write_json(destination / "BACKUP.json", metadata)
    return {
        "backup_path": str(destination),
        "file_count": len(copied),
        "catalog_included": catalog_backup.is_file(),
    }


def _backup_candidates(projects_root: Path, project_name: str) -> list[Path]:
    backup_root = projects_root / BACKUP_FOLDER
    if not backup_root.is_dir():
        return []
    return [
        backup / project_name / "project.json"
        for backup in sorted(backup_root.iterdir(), reverse=True)
        if backup.is_dir() and (backup / project_name / "project.json").is_file()
    ]


def _restore_companion_metadata(
    projects_root: Path,
    project_dir: Path,
    manifest_backup: Path,
) -> None:
    backup_project_dir = manifest_backup.parent
    for name in ("task_queue.json",):
        source = backup_project_dir / name
        target = project_dir / name
        if source.is_file() and not target.exists():
            _atomic_copy(source, target)
    source_plans = backup_project_dir / "task_plans"
    if source_plans.is_dir():
        for source in source_plans.glob("*.json"):
            target = project_dir / "task_plans" / source.name
            if not target.exists():
                _atomic_copy(source, target)


def _catalog_connection(projects_root: Path) -> sqlite3.Connection | None:
    catalog = projects_root / "project_catalog.sqlite"
    if not catalog.is_file():
        return None
    connection = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def _catalog_project(
    connection: sqlite3.Connection,
    project_dir: Path,
) -> sqlite3.Row | None:
    rows = connection.execute(
        "SELECT * FROM projects WHERE project_id=? OR manifest_path=? "
        "ORDER BY CASE WHEN manifest_path=? THEN 0 ELSE 1 END LIMIT 1",
        (project_dir.name, str(project_dir / "project.json"), str(project_dir / "project.json")),
    ).fetchone()
    return rows


def _reconstruct_from_catalog(
    connection: sqlite3.Connection,
    project_dir: Path,
) -> tuple[dict[str, Any] | None, str]:
    project = _catalog_project(connection, project_dir)
    if project is None:
        return None, "no project catalog record"
    project_id = str(project["project_id"])
    plates = connection.execute(
        "SELECT * FROM plates WHERE project_id=? ORDER BY plate_slug",
        (project_id,),
    ).fetchall()
    if not plates:
        return None, "catalog has no plates"

    recovered_plates: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for row in plates:
        artifact_root = Path(str(row["artifact_root"] or "")).expanduser()
        config = Path(str(row["config_path"] or "")).expanduser()
        images = Path(str(row["images_manifest_path"] or "")).expanduser()
        report = Path(str(row["report_path"] or "")).expanduser()
        required = {
            "artifact": artifact_root.is_dir(),
            "config": config.is_file(),
            "images": images.is_file(),
            "report": report.is_file(),
        }
        if not all(required.values()):
            missing = ",".join(name for name, exists in required.items() if not exists)
            incomplete.append(f"{row['plate_slug']}({missing})")
            continue
        gated_root = report.parent
        recovered_plates.append(
            {
                "slug": str(row["plate_slug"]),
                "group_id": str(row["group_id"] or ""),
                "board_id": str(row["board_id"] or ""),
                "config": str(config.resolve()),
                "artifact_root": str(artifact_root.resolve()),
                "gated_output_dir": str(gated_root.resolve()),
                "images_manifest": str(images.resolve()),
                "pipeline_summary": str((gated_root / "pipeline_summary.json").resolve()),
                "report_json": str(report.resolve()),
                "status": str(row["status"] or "completed"),
                "elapsed_seconds": float(row["elapsed_seconds"] or 0),
            }
        )
    if incomplete:
        return None, "incomplete catalogued plate files: " + "; ".join(incomplete)

    task = connection.execute(
        "SELECT task_id, options_json FROM tasks WHERE project_id=? "
        "ORDER BY COALESCE(finished_at, started_at, created_at) DESC LIMIT 1",
        (project_id,),
    ).fetchone()
    options: dict[str, Any] = {}
    if task is not None:
        try:
            parsed = json.loads(str(task["options_json"] or "{}"))
            if isinstance(parsed, dict):
                options = parsed
        except json.JSONDecodeError:
            pass
    all_completed = all(plate["status"] == "completed" for plate in recovered_plates)
    manifest: dict[str, Any] = {
        "project_id": project_id,
        "project_name": str(project["project_name"] or project_id),
        "created_by": str(project["created_by"] or ""),
        "root": str(project_dir),
        "generated_at": str(project["created_at"] or _now()),
        "status": "completed" if all_completed else "ready",
        "plates": recovered_plates,
        "recovered_at": _now(),
        "recovery_source": "project_catalog.sqlite",
    }
    if task is not None:
        manifest["task_id"] = str(task["task_id"] or "")
    selected = options.get("selected_timepoint_labels")
    if isinstance(selected, list):
        manifest["selected_timepoint_labels"] = selected
    return manifest, ""


def recover_missing_project_manifests(projects_root: str | Path) -> dict[str, Any]:
    root = Path(projects_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    restored: list[str] = []
    reconstructed: list[str] = []
    skipped: list[dict[str, str]] = []
    connection = _catalog_connection(root)
    try:
        for project_dir in _project_directories(root):
            manifest = project_dir / "project.json"
            if manifest.is_file():
                if _read_json(manifest) is None:
                    skipped.append({"project": project_dir.name, "reason": "invalid project.json"})
                continue
            candidates = _backup_candidates(root, project_dir.name)
            valid_backup = next((path for path in candidates if _read_json(path) is not None), None)
            if valid_backup is not None:
                _atomic_copy(valid_backup, manifest)
                _restore_companion_metadata(root, project_dir, valid_backup)
                restored.append(project_dir.name)
                continue
            if connection is None:
                skipped.append({"project": project_dir.name, "reason": "no metadata backup or catalog"})
                continue
            recovered, reason = _reconstruct_from_catalog(connection, project_dir)
            if recovered is None:
                skipped.append({"project": project_dir.name, "reason": reason})
                continue
            _atomic_write_json(manifest, recovered)
            reconstructed.append(project_dir.name)
    finally:
        if connection is not None:
            connection.close()
    return {
        "restored_from_backup": restored,
        "reconstructed_from_catalog": reconstructed,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects-root", required=True)
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    if not args.backup and not args.recover:
        parser.error("select --backup, --recover, or both")
    result: dict[str, Any] = {}
    if args.backup:
        result["backup"] = backup_project_metadata(args.projects_root)
    if args.recover:
        result["recovery"] = recover_missing_project_manifests(args.projects_root)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

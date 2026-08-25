"""Safely import a historical Cell Vision Workspace into the current installation.

Existing destination files and projects are never overwritten. JSON metadata in
newly copied projects is rebased from the old Workspace path to the new one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .repair_project_manifests import backup_project_metadata, recover_missing_project_manifests
except ImportError:  # Direct execution from a portable installation root.
    from repair_project_manifests import backup_project_metadata, recover_missing_project_manifests


IGNORED_PROJECT_DIRECTORIES = {"active", ".metadata-backups"}
MERGED_WORKSPACE_DIRECTORIES = ("Inbox", "Database")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rebase_string(value: str, source: Path, target: Path) -> str:
    source_values = (str(source), source.as_posix())
    for prefix in source_values:
        prefix = prefix.rstrip("\\/")
        if value.casefold() == prefix.casefold():
            return str(target)
        if (
            len(value) > len(prefix)
            and value[: len(prefix)].casefold() == prefix.casefold()
            and value[len(prefix)] in "\\/"
        ):
            suffix = value[len(prefix) + 1 :].replace("/", os.sep).replace("\\", os.sep)
            return str(target / suffix)
    return value


def _rebase_value(value: Any, source: Path, target: Path) -> Any:
    if isinstance(value, str):
        return _rebase_string(value, source, target)
    if isinstance(value, list):
        return [_rebase_value(item, source, target) for item in value]
    if isinstance(value, dict):
        return {key: _rebase_value(item, source, target) for key, item in value.items()}
    return value


def rebase_project_metadata(project_dir: Path, source_workspace: Path, target_workspace: Path) -> int:
    changed = 0
    for path in project_dir.rglob("*.json"):
        if not path.is_file():
            continue
        try:
            original = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        updated = _rebase_value(original, source_workspace, target_workspace)
        if updated != original:
            _atomic_write_json(path, updated)
            changed += 1
    return changed


def _copy_missing_tree(source: Path, target: Path) -> tuple[int, int]:
    copied = 0
    skipped = 0
    if not source.is_dir():
        return copied, skipped
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        target_path = target / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        if not source_path.is_file():
            continue
        if target_path.exists():
            skipped += 1
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied += 1
    return copied, skipped


def import_workspace(source_workspace: str | Path, target_workspace: str | Path) -> dict[str, Any]:
    source = Path(source_workspace).expanduser().resolve()
    target = Path(target_workspace).expanduser().resolve()
    source_projects = source / "Projects"
    target_projects = target / "Projects"
    if source == target:
        raise ValueError("source and target Workspace are the same folder")
    if not source_projects.is_dir():
        raise ValueError(f"source Workspace has no Projects directory: {source_projects}")
    target_projects.mkdir(parents=True, exist_ok=True)

    backup = backup_project_metadata(target_projects)
    imported_projects: list[str] = []
    skipped_projects: list[dict[str, str]] = []
    rebased_json_files = 0
    for source_project in sorted(source_projects.iterdir(), key=lambda path: path.name.casefold()):
        if not source_project.is_dir() or source_project.name in IGNORED_PROJECT_DIRECTORIES:
            continue
        target_project = target_projects / source_project.name
        if target_project.exists():
            skipped_projects.append(
                {"project": source_project.name, "reason": "destination project folder already exists"}
            )
            continue
        shutil.copytree(source_project, target_project, copy_function=shutil.copy2)
        rebased_json_files += rebase_project_metadata(target_project, source, target)
        imported_projects.append(source_project.name)

    merged: dict[str, dict[str, int]] = {}
    for directory_name in MERGED_WORKSPACE_DIRECTORIES:
        copied, skipped = _copy_missing_tree(source / directory_name, target / directory_name)
        merged[directory_name] = {"copied_files": copied, "existing_files_skipped": skipped}

    recovery = recover_missing_project_manifests(target_projects)
    result = {
        "format": "cellvision-workspace-import",
        "version": 1,
        "created_at": _now(),
        "source_workspace": str(source),
        "target_workspace": str(target),
        "metadata_backup": backup,
        "imported_projects": imported_projects,
        "skipped_projects": skipped_projects,
        "rebased_json_files": rebased_json_files,
        "merged_directories": merged,
        "recovery": recovery,
    }
    logs = target / "Logs"
    logs.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(logs / "last-workspace-import.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", required=True)
    parser.add_argument("--target-workspace", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            import_workspace(args.source_workspace, args.target_workspace),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

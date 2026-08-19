"""Platform-independent review-data packages.

A ``.cvreview`` package is a directory containing only review data.  It has no
Python runtime, application code, model checkpoint, training input or cache.
All active paths inside the project snapshot are relative, so the same package
can be opened by the installed Windows or macOS review platform.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml


DATA_PACKAGE_FORMAT = "cellvision-review-data"
DATA_PACKAGE_VERSION = 1
DATA_PACKAGE_MANIFEST = "CVREVIEW.json"
ProgressCallback = Callable[[int, int, str], None]

_EXCLUDED_DIRECTORIES = {
    "exports",
    "cache",
    "models",
    "checkpoints",
    "task_plans",
    "endpoint_screening",
    ".venv-production",
    ".venv-review",
    "__pycache__",
}
_EXCLUDED_NAME_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".onnx",
    "-cf.tif",
    "-cells.csv",
    "metricsummary.csv",
}
_EXCLUDED_NAMES = {"sessions.csv"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if set(relative.parts[:-1]) & _EXCLUDED_DIRECTORIES:
            continue
        lowered = path.name.casefold()
        if lowered in _EXCLUDED_NAMES:
            continue
        if any(lowered.endswith(suffix) for suffix in _EXCLUDED_NAME_SUFFIXES):
            continue
        yield path


def project_export_signature(project_root: str | Path, git_commit: str) -> str:
    root = Path(project_root).expanduser().resolve()
    file_count = 0
    total_bytes = 0
    latest_mtime_ns = 0
    mtime_total = 0
    for path in _files(root):
        if path.relative_to(root).as_posix() == "task_queue.json":
            continue
        stat = path.stat()
        file_count += 1
        total_bytes += int(stat.st_size)
        latest_mtime_ns = max(latest_mtime_ns, int(stat.st_mtime_ns))
        mtime_total += int(stat.st_mtime_ns)
    # This is a fast source-state marker, not a content hash.  It deliberately
    # avoids opening TIFF files during export freshness checks.
    return f"v2-stat:{git_commit}:{file_count}:{total_bytes}:{latest_mtime_ns}:{mtime_total}"


def _promote(staging: Path, target: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(20):
        try:
            staging.replace(target)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(0.25 + attempt * 0.1, 1.0))
    assert last_error is not None
    raise last_error


def _portable_path(value: str, *, source_file: Path, project_root: Path) -> str:
    candidate = Path(os.path.expandvars(value)).expanduser()
    if not candidate.is_absolute():
        return value.replace("\\", "/")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        # Acquisition, model and host-runtime locations are provenance only;
        # exposing them would make the review package host-dependent.
        return ""
    return Path(os.path.relpath(resolved, source_file.parent)).as_posix()


def _portable_value(value: Any, *, source_file: Path, project_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _portable_value(item, source_file=source_file, project_root=project_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _portable_value(item, source_file=source_file, project_root=project_root)
            for item in value
        ]
    if isinstance(value, str):
        return _portable_path(value, source_file=source_file, project_root=project_root)
    return value


def _normalize_json(source: Path, destination: Path, project_root: Path) -> None:
    try:
        value = json.loads(destination.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    portable = _portable_value(value, source_file=source, project_root=project_root)
    if source.name == "project.json" and isinstance(portable, dict):
        portable["root"] = "."
        portable["portable_review"] = True
        portable["review_data_package"] = True
        portable["source"] = {"access_policy": "not_included"}
        for key in (
            "source_sessions_csv",
            "source_day14_csv",
            "source_endpoint_csv",
            "source_index",
        ):
            portable.pop(key, None)
    if source.name == "task_queue.json":
        tasks = portable.get("tasks") if isinstance(portable, dict) else portable
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                task["project_manifest"] = "project.json"
                task["path"] = "."
                task["source_path"] = ""
                task["index"] = ""
                task.pop("offline_export", None)
    destination.write_text(json.dumps(portable, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_yaml(source: Path, destination: Path, project_root: Path) -> None:
    try:
        value = yaml.safe_load(destination.read_text(encoding="utf-8")) or {}
    except (UnicodeDecodeError, yaml.YAMLError):
        return
    if not isinstance(value, dict):
        return
    portable = _portable_value(value, source_file=source, project_root=project_root)
    portable.setdefault("runtime", {})["resolve_paths_relative_to_config"] = True
    portable["runtime"]["ignore_path_env_overrides"] = True
    destination.write_text(
        yaml.safe_dump(portable, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _normalize_csv(source: Path, destination: Path, project_root: Path) -> None:
    try:
        with destination.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            path_columns = {
                column
                for column in fieldnames
                if (
                    "path" in column.casefold()
                    or column.casefold() in {"root", "directory", "dir"}
                    or column.casefold().endswith(("_root", "_directory", "_dir"))
                )
            }
            if not path_columns:
                return
            rows = list(reader)
    except (UnicodeDecodeError, csv.Error):
        return
    if not fieldnames:
        return
    changed = False
    portable_cache: dict[str, str] = {}
    for row in rows:
        for column in path_columns:
            value = row.get(column)
            if not value:
                continue
            portable = portable_cache.get(value)
            if portable is None:
                portable = _portable_path(
                    value, source_file=source, project_root=project_root
                )
                portable_cache[value] = portable
            if portable != value:
                row[column] = portable
                changed = True
    if changed:
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _normalize_snapshot(snapshot: Path, source_root: Path) -> None:
    for destination in snapshot.rglob("*"):
        if not destination.is_file():
            continue
        source = source_root / destination.relative_to(snapshot)
        suffix = destination.suffix.casefold()
        if suffix == ".json":
            _normalize_json(source, destination, source_root)
        elif suffix in {".yaml", ".yml"}:
            _normalize_yaml(source, destination, source_root)
        elif suffix == ".csv":
            _normalize_csv(source, destination, source_root)


def _inventory(package_root: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total = 0
    for path in sorted(package_root.rglob("*"), key=lambda value: value.as_posix().casefold()):
        if not path.is_file() or path.name == DATA_PACKAGE_MANIFEST:
            continue
        size = int(path.stat().st_size)
        total += size
        relative = path.relative_to(package_root).as_posix()
        mutable = (
            relative == "project/task_queue.json"
            or "/annotations/" in relative
            or "/gated/" in relative
            or "/predictions/" in relative
            or relative.endswith("/review_summary.json")
        )
        rows.append({
            "path": relative,
            "bytes": size,
            "modified_ns": int(path.stat().st_mtime_ns),
            "mutable": mutable,
        })
    return rows, total


def validate_review_data_package(
    package_root: str | Path,
    *,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    root = Path(package_root).expanduser().resolve()
    manifest_path = root / DATA_PACKAGE_MANIFEST
    if root.suffix.casefold() != ".cvreview" or not manifest_path.is_file():
        raise ValueError("不是 Cell Vision .cvreview 数据包")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != DATA_PACKAGE_FORMAT
        or int(manifest.get("version", 0)) != DATA_PACKAGE_VERSION
    ):
        raise ValueError("不支持的 .cvreview 数据包版本")
    entrypoint = root / str(manifest.get("entrypoint") or "")
    if not entrypoint.is_file():
        raise ValueError(".cvreview 缺少项目入口")
    for item in manifest.get("files", []):
        relative = Path(str(item.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(".cvreview 文件清单包含越界路径")
        path = root / relative
        if not path.is_file():
            raise ValueError(f".cvreview 文件缺失：{relative.as_posix()}")
        mutable = bool(item.get("mutable"))
        if not mutable and int(path.stat().st_size) != int(item.get("bytes", -1)):
            raise ValueError(f".cvreview 文件大小不符：{relative.as_posix()}")
        expected_hash = str(item.get("sha256") or "")
        if verify_hashes and expected_hash and not mutable and _sha256(path) != expected_hash:
            raise ValueError(f".cvreview 文件校验失败：{relative.as_posix()}")
    return manifest


def prepare_review_data_package(
    manifest_path: str | Path,
    task: dict[str, Any],
    *,
    git_commit: str,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    project_root = manifest_file.parent
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    project_id = str(manifest.get("project_id") or project_root.name)
    export_signature = project_export_signature(project_root, git_commit)
    exports = project_root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    for existing_target in exports.glob(f"{project_id}-*.cvreview"):
        existing_metadata = existing_target / DATA_PACKAGE_MANIFEST
        if not existing_metadata.is_file():
            continue
        try:
            existing = json.loads(existing_metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            existing.get("format") == DATA_PACKAGE_FORMAT
            and existing.get("export_signature") == export_signature
            and existing.get("production_git_commit") == git_commit
        ):
            if progress_callback:
                progress_callback(1, 1, "审核数据包已经是最新版本")
            return existing_target, {
                "format": DATA_PACKAGE_FORMAT,
                "package_id": str(existing.get("package_id") or ""),
                "package_path": str(existing_target),
                "git_commit": git_commit,
                "export_signature": export_signature,
                "copied_bytes": int(existing.get("total_bytes", 0)),
                "file_count": int(existing.get("file_count", 0)),
                "reused": True,
            }

    package_id = uuid.uuid4().hex
    target = exports / f"{project_id}-{package_id[:12]}.cvreview"

    staging = exports / f".{target.name}.building"
    if staging.exists():
        raise FileExistsError(f"已有未完成的数据包导出：{staging}")
    staging.mkdir(parents=True)
    try:
        entries = []
        for source in _files(project_root):
            relative = source.relative_to(project_root)
            entries.append((source, staging / "project" / relative, int(source.stat().st_size)))
        total = sum(size for _, _, size in entries)
        copied = 0
        for source, destination, size in entries:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += size
            if progress_callback:
                progress_callback(copied, total, f"正在导出审核数据：{source.name}")

        snapshot = staging / "project"
        if progress_callback:
            progress_callback(total, total, "正在整理审核包内的相对路径")
        _normalize_snapshot(snapshot, project_root)
        files, normalized_total = _inventory(staging)
        metadata = {
            "format": DATA_PACKAGE_FORMAT,
            "version": DATA_PACKAGE_VERSION,
            "package_id": package_id,
            "export_signature": export_signature,
            "integrity_mode": "size-and-presence",
            "project_id": project_id,
            "project_name": str(manifest.get("project_name") or project_id),
            "task_id": str(task.get("task_id") or manifest.get("task_id") or ""),
            "task_name": str(task.get("name") or manifest.get("task_name") or ""),
            "production_git_commit": git_commit,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "entrypoint": "project/project.json",
            "path_mode": "relative",
            "platform_independent": True,
            "environment_included": False,
            "models_included": False,
            "compute_inputs_included": False,
            "total_bytes": normalized_total,
            "file_count": len(files),
            "files": files,
        }
        metadata_path = staging / DATA_PACKAGE_MANIFEST
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        _promote(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if progress_callback:
        progress_callback(total, total, "无环境依赖的审核数据包已生成")
    return target, {
        "format": DATA_PACKAGE_FORMAT,
        "package_id": package_id,
        "package_path": str(target),
        "git_commit": git_commit,
        "export_signature": export_signature,
        "copied_bytes": normalized_total,
        "file_count": len(files),
        "reused": False,
    }

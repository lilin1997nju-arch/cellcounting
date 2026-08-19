"""Prepare a copyable review workspace that runs the production review UI."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ProgressCallback = Callable[[int, int, str], None]
_APPLICATION_ENTRIES = (
    "src",
    "review-ui",
    "configs",
    "scripts",
    "deploy/offline_review",
    "pyproject.toml",
    "requirements-production.txt",
    "requirements-portable-review.txt",
    "RELEASE_GIT_COMMIT.txt",
)


def _promote_staging_directory(staging: Path, target: Path) -> None:
    """Rename a completed workspace, tolerating short Windows scanner locks."""

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


_PROJECT_EXCLUDED_DIRECTORIES = {
    "exports",
    "cache",
    "models",
    "checkpoints",
    ".venv-production",
    ".venv-review",
}
_PROJECT_EXCLUDED_SUFFIXES = {".pt", ".pth", ".onnx"}
_PORTABLE_WHEEL_EXCLUDED_PREFIXES = (
    "torch-",
    "torchvision-",
    "sympy-",
    "mpmath-",
    "fsspec-",
    "filelock-",
)


def _files(
    root: Path,
    *,
    excluded_directories: set[str] | None = None,
    excluded_suffixes: set[str] | None = None,
    portable_wheels_only: bool = False,
) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    excluded_directories = excluded_directories or set()
    excluded_suffixes = excluded_suffixes or set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = set(path.relative_to(root).parts[:-1])
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if relative_parts & excluded_directories:
            continue
        if path.suffix.casefold() in excluded_suffixes:
            continue
        if portable_wheels_only and path.name.casefold().startswith(
            _PORTABLE_WHEEL_EXCLUDED_PREFIXES
        ):
            continue
        yield path


def _copy_sources(
    sources: list[tuple[Path, Path, set[str], set[str], bool]],
    *,
    progress_callback: ProgressCallback | None,
) -> tuple[int, int]:
    entries: list[tuple[Path, Path, int]] = []
    for source, destination, excluded_directories, excluded_suffixes, wheels_only in sources:
        if source.is_file():
            entries.append((source, destination, int(source.stat().st_size)))
            continue
        for file in _files(
            source,
            excluded_directories=excluded_directories,
            excluded_suffixes=excluded_suffixes,
            portable_wheels_only=wheels_only,
        ):
            entries.append((file, destination / file.relative_to(source), int(file.stat().st_size)))
    total = sum(size for _, _, size in entries)
    copied = 0
    for source, destination, size in entries:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += size
        if progress_callback is not None:
            progress_callback(copied, total, f"正在复制 {source.name}")
    return copied, total


def prepare_portable_review_workspace(
    manifest_path: str | Path,
    task: dict[str, Any],
    *,
    application_root: str | Path,
    release_root: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create a self-contained normal-UI review export from one Project."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    application = Path(application_root).expanduser().resolve()
    release = Path(release_root).expanduser().resolve() if release_root else application.parent
    project_root = manifest_file.parent
    image_storage = manifest.get("image_storage")
    if isinstance(image_storage, dict) and image_storage.get("mode") == "project_owned_after_endpoint_gate":
        image_root = Path(str(image_storage.get("root") or project_root / "data" / "images"))
        if not image_root.is_dir():
            raise FileNotFoundError(f"Project 自有图像目录不存在：{image_root}")
    wheelhouse = release / "wheelhouse"
    runtime = release / "runtime"
    if not wheelhouse.is_dir() or not runtime.is_dir():
        raise FileNotFoundError("当前生产发布目录缺少 wheelhouse 或 runtime")

    commit_file = application / "RELEASE_GIT_COMMIT.txt"
    commit = commit_file.read_text(encoding="utf-8").strip() if commit_file.is_file() else "development"
    suffix = commit[:10] if commit else "development"
    platform_id = "windows-x64"
    exports_root = project_root / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    target = exports_root / f"CellVisionReview-{suffix}-{platform_id}"
    metadata_path = target / "PACKAGE.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(existing.get("task_id") or "") == str(task.get("task_id") or ""):
            summary = {
                "package_path": str(target),
                "git_commit": commit,
                "reused": True,
            }
            if progress_callback is not None:
                progress_callback(1, 1, "完整审核目录已经准备好")
            return target, summary

    staging = exports_root / f".{target.name}.building"
    if staging.exists():
        raise FileExistsError(f"已有未完成的审核目录准备任务：{staging}")
    staging.mkdir(parents=True)
    try:
        sources: list[tuple[Path, Path, set[str], set[str], bool]] = [
            (
                project_root,
                staging / "project",
                _PROJECT_EXCLUDED_DIRECTORIES,
                _PROJECT_EXCLUDED_SUFFIXES,
                False,
            ),
            (
                wheelhouse,
                staging / "platform" / platform_id / "wheelhouse",
                set(),
                set(),
                True,
            ),
            (
                runtime,
                staging / "platform" / platform_id / "runtime",
                set(),
                set(),
                False,
            ),
        ]
        for relative in _APPLICATION_ENTRIES:
            source = application / relative
            if source.exists():
                sources.append((source, staging / "application" / relative, set(), set(), False))
        copied, total = _copy_sources(sources, progress_callback=progress_callback)

        launcher_root = staging / "application" / "deploy" / "offline_review"
        shutil.copy2(launcher_root / "Start-Offline-Review.cmd", staging / "Start-Offline-Review.cmd")
        shutil.copy2(launcher_root / "start_offline_review.ps1", staging / "start_offline_review.ps1")
        metadata = {
            "format": "cellvision-portable-normal-review",
            "version": 1,
            "layout_version": 2,
            "task_id": str(task.get("task_id") or ""),
            "task_name": str(task.get("name") or manifest.get("project_name") or ""),
            "project_id": str(manifest.get("project_id") or manifest_file.parent.name),
            "git_commit": commit,
            "source_data_root": str(project_root),
            "source_project_root": str(project_root),
            "platform_layers": [platform_id],
            "default_platform": platform_id,
            "model_runtime_included": False,
            "project_cache_included": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "copied_bytes": copied,
            "total_bytes": total,
            "instructions": "复制整个 CellVisionReview-* 目录；在 Windows x64 审核电脑双击 Start-Offline-Review.cmd。",
        }
        (staging / "PACKAGE.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _promote_staging_directory(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if progress_callback is not None:
        progress_callback(total, total, "完整审核目录已经准备好")
    return target, {
        "package_path": str(target),
        "git_commit": commit,
        "copied_bytes": copied,
        "reused": False,
    }

"""Prepare a copyable review workspace that runs the production review UI."""

from __future__ import annotations

import json
import shutil
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
    "RELEASE_GIT_COMMIT.txt",
)


def _files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if ".venv-production" in path.parts or ".venv-review" in path.parts:
            continue
        yield path


def _copy_sources(
    sources: list[tuple[Path, Path]],
    *,
    progress_callback: ProgressCallback | None,
) -> tuple[int, int]:
    entries: list[tuple[Path, Path, int]] = []
    for source, destination in sources:
        if source.is_file():
            entries.append((source, destination, int(source.stat().st_size)))
            continue
        for file in _files(source):
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
    """Create a full normal-UI review workspace inside the task data folder."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    application = Path(application_root).expanduser().resolve()
    release = Path(release_root).expanduser().resolve() if release_root else application.parent
    data_root_value = task.get("path") or manifest.get("root")
    if not data_root_value:
        raise ValueError("任务没有可用的数据目录")
    data_root = Path(str(data_root_value)).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    wheelhouse = release / "wheelhouse"
    runtime = release / "runtime"
    if not wheelhouse.is_dir() or not runtime.is_dir():
        raise FileNotFoundError("当前生产发布目录缺少 wheelhouse 或 runtime")

    commit_file = application / "RELEASE_GIT_COMMIT.txt"
    commit = commit_file.read_text(encoding="utf-8").strip() if commit_file.is_file() else "development"
    suffix = commit[:10] if commit else "development"
    target = data_root / f"CellVisionReview-{suffix}"
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

    staging = data_root / f".{target.name}.building"
    if staging.exists():
        raise FileExistsError(f"已有未完成的审核目录准备任务：{staging}")
    staging.mkdir(parents=True)
    try:
        sources: list[tuple[Path, Path]] = [
            (manifest_file.parent, staging / "project"),
            (wheelhouse, staging / "wheelhouse"),
            (runtime, staging / "runtime"),
        ]
        for relative in _APPLICATION_ENTRIES:
            source = application / relative
            if source.exists():
                sources.append((source, staging / "application" / relative))
        copied, total = _copy_sources(sources, progress_callback=progress_callback)

        launcher_root = staging / "application" / "deploy" / "offline_review"
        shutil.copy2(launcher_root / "Start-Offline-Review.cmd", staging / "Start-Offline-Review.cmd")
        shutil.copy2(launcher_root / "start_offline_review.ps1", staging / "start_offline_review.ps1")
        metadata = {
            "format": "cellvision-portable-normal-review",
            "version": 1,
            "task_id": str(task.get("task_id") or ""),
            "task_name": str(task.get("name") or manifest.get("project_name") or ""),
            "project_id": str(manifest.get("project_id") or manifest_file.parent.name),
            "git_commit": commit,
            "source_data_root": str(Path(str(manifest.get("root") or data_root)).expanduser().resolve()),
            "source_project_root": str(manifest_file.parent),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "copied_bytes": copied,
            "total_bytes": total,
            "instructions": "复制整个任务数据文件夹；在审核电脑双击 CellVisionReview-*\\Start-Offline-Review.cmd。",
        }
        (staging / "PACKAGE.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        staging.replace(target)
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

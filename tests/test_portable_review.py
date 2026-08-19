import json
import shutil
from pathlib import Path

from cellvision.portable_review import (
    _promote_staging_directory,
    prepare_portable_review_workspace,
)
from scripts.rebase_portable_review import rebase


def test_portable_review_workspace_is_created_inside_task_data_and_rebased(tmp_path: Path):
    data_root = tmp_path / "task-data"
    data_root.mkdir()
    (data_root / "raw.tif").write_bytes(b"raw")
    project_root = tmp_path / "project"
    (project_root / "configs").mkdir(parents=True)
    manifest = project_root / "project.json"
    manifest.write_text(json.dumps({
        "project_id": "demo-project",
        "project_name": "Demo",
        "root": str(data_root),
        "plates": [{"config": str(project_root / "configs" / "plate.yaml")}],
    }), encoding="utf-8")
    (project_root / "configs" / "plate.yaml").write_text(
        f"data_root: {data_root}\nartifact_root: {project_root}\n",
        encoding="utf-8",
    )

    application = tmp_path / "release" / "Application"
    for directory in ["src", "review-ui", "configs", "scripts", "deploy/offline_review"]:
        (application / directory).mkdir(parents=True, exist_ok=True)
    (application / "src" / "app.py").write_text("# app", encoding="utf-8")
    (application / "review-ui" / "auto-review.js").write_text("// review", encoding="utf-8")
    (application / "pyproject.toml").write_text("[project]\nname='demo'", encoding="utf-8")
    (application / "requirements-production.txt").write_text("", encoding="utf-8")
    (application / "RELEASE_GIT_COMMIT.txt").write_text("abcdef1234567890\n", encoding="utf-8")
    (application / "deploy" / "offline_review" / "Start-Offline-Review.cmd").write_text("start", encoding="utf-8")
    (application / "deploy" / "offline_review" / "start_offline_review.ps1").write_text("start", encoding="utf-8")
    release = application.parent
    (release / "wheelhouse").mkdir()
    (release / "wheelhouse" / "dependency.whl").write_bytes(b"wheel")
    (release / "runtime").mkdir()
    (release / "runtime" / "python-installer.exe").write_bytes(b"python")
    progress: list[tuple[int, int, str]] = []

    workspace, summary = prepare_portable_review_workspace(
        manifest,
        {"task_id": "task-1", "name": "Demo", "path": str(data_root)},
        application_root=application,
        release_root=release,
        progress_callback=lambda current, total, message: progress.append((current, total, message)),
    )

    assert workspace == data_root / "CellVisionReview-abcdef1234"
    assert (workspace / "project" / "project.json").is_file()
    assert (workspace / "application" / "review-ui" / "auto-review.js").is_file()
    assert (workspace / "wheelhouse" / "dependency.whl").is_file()
    assert (workspace / "Start-Offline-Review.cmd").is_file()
    assert summary["reused"] is False
    assert progress[-1][0] == progress[-1][1]

    rebased_manifest = rebase(workspace)
    rebased = json.loads(rebased_manifest.read_text(encoding="utf-8"))
    assert rebased["portable_review"] is True
    assert Path(rebased["root"]) == data_root
    config_text = (workspace / "project" / "configs" / "plate.yaml").read_text(encoding="utf-8")
    assert str(workspace / "project") in config_text
    assert str(data_root) in config_text

    moved_data_root = tmp_path / "copied task data"
    moved_workspace = moved_data_root / workspace.name
    shutil.copytree(workspace, moved_workspace)
    rebase(moved_workspace)
    moved_config = (
        moved_workspace / "project" / "configs" / "plate.yaml"
    ).read_text(encoding="utf-8")
    assert str(moved_workspace / "project") in moved_config
    assert str(moved_data_root) in moved_config
    assert str(workspace) not in moved_config


def test_promote_staging_directory_retries_short_windows_lock(tmp_path: Path, monkeypatch):
    staging = tmp_path / "staging"
    target = tmp_path / "target"
    staging.mkdir()
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(path: Path, destination: Path):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("scanner still holds a file")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr("cellvision.portable_review.time.sleep", lambda _: None)

    _promote_staging_directory(staging, target)

    assert attempts == 3
    assert target.is_dir()

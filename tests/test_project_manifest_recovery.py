from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from deploy.portable.repair_project_manifests import (
    backup_project_metadata,
    recover_missing_project_manifests,
)


def _catalog(path: Path, project_dir: Path, *, complete: bool = True) -> None:
    artifact = project_dir / "plates" / "plate-1"
    config = project_dir / "configs" / "plate-1.yaml"
    images = artifact / "manifests" / "images.csv"
    report = artifact / "gated" / "plate_overview.json"
    artifact.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    config.write_text("name: plate-1\n", encoding="utf-8")
    images.parent.mkdir(parents=True)
    images.write_text("well,path\n", encoding="utf-8")
    report.parent.mkdir(parents=True)
    if complete:
        report.write_text("{}", encoding="utf-8")

    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE projects (
              project_id TEXT PRIMARY KEY, project_name TEXT, source_root TEXT,
              manifest_path TEXT, created_by TEXT, status TEXT, created_at TEXT,
              updated_at TEXT, revision INTEGER, deleted_at TEXT
            );
            CREATE TABLE plates (
              plate_id TEXT PRIMARY KEY, project_id TEXT, plate_slug TEXT,
              board_id TEXT, group_id TEXT, status TEXT, current_stage TEXT,
              progress_percent REAL, elapsed_seconds REAL, artifact_root TEXT,
              config_path TEXT, images_manifest_path TEXT, report_path TEXT,
              review_database_path TEXT, updated_at TEXT
            );
            CREATE TABLE tasks (
              task_id TEXT PRIMARY KEY, project_id TEXT, options_json TEXT,
              finished_at TEXT, started_at TEXT, created_at TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO projects VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                project_dir.name,
                "Recovered project",
                str(project_dir),
                str(project_dir / "project.json"),
                "tester",
                "deleted",
                "2026-01-01T00:00:00+00:00",
                "2026-01-02T00:00:00+00:00",
                1,
                "2026-01-02T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO plates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"{project_dir.name}:plate-1",
                project_dir.name,
                "plate-1",
                "P1",
                "Group P1",
                "completed",
                "completed",
                100,
                12.5,
                str(artifact),
                str(config),
                str(images),
                str(report),
                str(artifact / "annotations" / "annotations.db"),
                "2026-01-02T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO tasks VALUES (?,?,?,?,?,?)",
            (
                "task-1",
                project_dir.name,
                json.dumps({"selected_timepoint_labels": ["Day0", "Day14"]}),
                "2026-01-02T00:00:00+00:00",
                None,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def test_exact_metadata_backup_restores_missing_manifest_and_queue(tmp_path: Path):
    projects = tmp_path / "Projects"
    project = projects / "alpha"
    project.mkdir(parents=True)
    manifest = project / "project.json"
    original = {"project_id": "alpha", "project_name": "Alpha", "plates": []}
    manifest.write_text(json.dumps(original), encoding="utf-8")
    (project / "task_queue.json").write_text("[]", encoding="utf-8")

    backup = backup_project_metadata(projects)
    manifest.unlink()
    (project / "task_queue.json").unlink()
    recovered = recover_missing_project_manifests(projects)

    assert Path(backup["backup_path"]).is_dir()
    assert recovered["restored_from_backup"] == ["alpha"]
    assert json.loads(manifest.read_text(encoding="utf-8")) == original
    assert (project / "task_queue.json").read_text(encoding="utf-8") == "[]"


def test_complete_catalog_reconstructs_reviewable_project_manifest(tmp_path: Path):
    projects = tmp_path / "Projects"
    project = projects / "beta"
    project.mkdir(parents=True)
    _catalog(projects / "project_catalog.sqlite", project)

    recovered = recover_missing_project_manifests(projects)
    manifest = json.loads((project / "project.json").read_text(encoding="utf-8"))

    assert recovered["reconstructed_from_catalog"] == ["beta"]
    assert manifest["project_id"] == "beta"
    assert manifest["task_id"] == "task-1"
    assert manifest["selected_timepoint_labels"] == ["Day0", "Day14"]
    assert manifest["plates"][0]["images_manifest"].endswith("images.csv")


def test_incomplete_catalog_does_not_create_false_project(tmp_path: Path):
    projects = tmp_path / "Projects"
    project = projects / "gamma"
    project.mkdir(parents=True)
    _catalog(projects / "project_catalog.sqlite", project, complete=False)

    recovered = recover_missing_project_manifests(projects)

    assert not (project / "project.json").exists()
    assert recovered["reconstructed_from_catalog"] == []
    assert "incomplete catalogued plate files" in recovered["skipped"][0]["reason"]

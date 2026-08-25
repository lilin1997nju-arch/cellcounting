from __future__ import annotations

import json
from pathlib import Path

import pytest

from deploy.portable.import_cellvision_workspace import import_workspace


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_workspace_import_preserves_destination_and_rebases_new_projects(tmp_path: Path):
    source = tmp_path / "old-install" / "Workspace"
    target = tmp_path / "new-install" / "Workspace"
    source_project = source / "Projects" / "historical-project"
    _write_json(
        source_project / "project.json",
        {
            "project_id": "historical-project",
            "root": str(source_project),
            "plates": [
                {
                    "slug": "plate-1",
                    "artifact_root": str(source_project / "plates" / "plate-1"),
                }
            ],
        },
    )
    _write_json(
        source_project / "task_plans" / "task.json",
        {"output": (source / "Projects" / "historical-project" / "report.json").as_posix()},
    )
    (source / "Inbox").mkdir(parents=True)
    (source / "Inbox" / "sessions.idx").write_text("source", encoding="utf-8")
    (source / "Database").mkdir(parents=True)
    (source / "Database" / "existing.db").write_text("source", encoding="utf-8")

    _write_json(
        target / "Projects" / "active" / "project.json",
        {"project_id": "active", "system_placeholder": True, "plates": []},
    )
    (target / "Database").mkdir(parents=True)
    (target / "Database" / "existing.db").write_text("target", encoding="utf-8")

    result = import_workspace(source, target)

    assert result["imported_projects"] == ["historical-project"]
    assert result["rebased_json_files"] == 2
    imported = json.loads(
        (target / "Projects" / "historical-project" / "project.json").read_text(encoding="utf-8")
    )
    assert imported["root"] == str(target / "Projects" / "historical-project")
    assert imported["plates"][0]["artifact_root"].startswith(str(target))
    assert (target / "Inbox" / "sessions.idx").read_text(encoding="utf-8") == "source"
    assert (target / "Database" / "existing.db").read_text(encoding="utf-8") == "target"
    assert Path(result["metadata_backup"]["backup_path"]).is_dir()
    assert (target / "Logs" / "last-workspace-import.json").is_file()


def test_workspace_import_never_overwrites_an_existing_project(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_json(source / "Projects" / "same" / "project.json", {"project_id": "source"})
    _write_json(target / "Projects" / "same" / "project.json", {"project_id": "target"})

    result = import_workspace(source, target)

    assert result["imported_projects"] == []
    assert result["skipped_projects"] == [
        {"project": "same", "reason": "destination project folder already exists"}
    ]
    current = json.loads((target / "Projects" / "same" / "project.json").read_text())
    assert current["project_id"] == "target"


def test_workspace_import_rejects_the_active_workspace(tmp_path: Path):
    workspace = tmp_path / "Workspace"
    (workspace / "Projects").mkdir(parents=True)
    with pytest.raises(ValueError, match="same folder"):
        import_workspace(workspace, workspace)

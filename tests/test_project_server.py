import json
from pathlib import Path

from cellvision.project_server import (
    ProjectRenamePayload,
    TaskPayload,
    _folder_name,
    create_project_app,
)


def test_folder_name_uses_parent_for_sessions_index():
    assert _folder_name(r"E:\\CM\\20260623 QL2603") == "20260623 QL2603"
    assert _folder_name(r"E:\\CM\\20260623 QL2603\\sessions.idx") == "20260623 QL2603"


def test_empty_project_can_be_renamed_and_deleted(tmp_path: Path):
    projects_root = tmp_path / "projects"
    main_dir = projects_root / "main"
    empty_dir = projects_root / "empty"
    main_dir.mkdir(parents=True)
    empty_dir.mkdir(parents=True)
    (main_dir / "project.json").write_text(
        json.dumps({"project_id": "main", "project_name": "Main", "plates": []}),
        encoding="utf-8",
    )
    (empty_dir / "project.json").write_text(
        json.dumps({"project_id": "empty", "project_name": "Empty", "plates": []}),
        encoding="utf-8",
    )

    app = create_project_app(main_dir / "project.json")
    rename_endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", "") == "/api/project/{project_id}"
        and "PATCH" in getattr(route, "methods", set())
    )
    delete_endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", "") == "/api/project/{project_id}"
        and "DELETE" in getattr(route, "methods", set())
    )
    renamed = rename_endpoint("empty", ProjectRenamePayload(project_name="Renamed"))
    assert renamed["project_name"] == "Renamed"

    deleted = delete_endpoint("empty")
    assert deleted["status"] == "deleted"
    assert not (empty_dir / "project.json").exists()


def test_new_task_records_creator_and_uses_folder_name_for_project(
    tmp_path: Path,
    monkeypatch,
):
    import cellvision.project_server as project_server

    projects_root = tmp_path / "projects"
    main_dir = projects_root / "main"
    main_dir.mkdir(parents=True)
    main_manifest = main_dir / "project.json"
    main_manifest.write_text(
        json.dumps({"project_id": "main", "project_name": "Main", "plates": []}),
        encoding="utf-8",
    )
    source = tmp_path / "FolderNamedProject"
    source.mkdir()
    options = [
        {
            "day_label": "Day0",
            "day_number": 0,
            "timepoint_labels": ["T0"],
            "required_early": True,
            "eligible_endpoint": False,
        },
        {
            "day_label": "Day1",
            "day_number": 1,
            "timepoint_labels": ["T1"],
            "required_early": True,
            "eligible_endpoint": False,
        },
        {
            "day_label": "Day2",
            "day_number": 2,
            "timepoint_labels": ["T2"],
            "required_early": True,
            "eligible_endpoint": False,
        },
        {
            "day_label": "Day7",
            "day_number": 7,
            "timepoint_labels": ["T3"],
            "required_early": False,
            "eligible_endpoint": True,
        },
    ]
    records = [
        {
            "group_id": "DEMO T1-1",
            "day_label": item["day_label"],
            "timepoint_label": item["timepoint_labels"][0],
            "group_complete": True,
        }
        for item in options
    ]
    monkeypatch.setattr(
        project_server,
        "_parse_folder",
        lambda _path: {
            "root": str(source),
            "folder_name": source.name,
            "index": str(source / "sessions.idx"),
            "group_count": 1,
            "session_count": 4,
            "timepoint_labels": ["T0", "T1", "T2", "T3"],
            "day_labels": ["Day0", "Day1", "Day2", "Day7"],
            "timepoint_options": options,
            "session_records": records,
        },
    )
    app = create_project_app(main_manifest)
    add_endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", "") == "/api/project/tasks"
        and "POST" in getattr(route, "methods", set())
    )
    task = add_endpoint(
        TaskPayload(
            path=str(source),
            name="计算任务 1",
            created_by="张三",
        )
    )
    assert task["created_by"] == "张三"
    project = json.loads(Path(task["project_manifest"]).read_text(encoding="utf-8"))
    assert project["project_name"] == source.name
    assert project["created_by"] == "张三"

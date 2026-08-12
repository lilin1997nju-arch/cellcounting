import asyncio
import json
from pathlib import Path

from cellvision.project_server import (
    _PlateReviewManager,
    ProjectRenamePayload,
    TaskPayload,
    _folder_name,
    create_project_app,
)


def test_folder_name_uses_parent_for_sessions_index():
    assert _folder_name(r"E:\\CM\\20260623 QL2603") == "20260623 QL2603"
    assert _folder_name(r"E:\\CM\\20260623 QL2603\\sessions.idx") == "20260623 QL2603"


def test_plate_review_manager_resolves_sibling_projects_and_uses_lru_limit(tmp_path: Path):
    projects_root = tmp_path / "projects"
    main_dir = projects_root / "main"
    sibling_dir = projects_root / "sibling"
    main_dir.mkdir(parents=True)
    sibling_dir.mkdir(parents=True)
    for directory, project_id, plate_slug in [
        (main_dir, "main", "main-board"),
        (sibling_dir, "sibling", "sibling-board"),
    ]:
        config = tmp_path / f"{project_id}.yaml"
        images = tmp_path / f"{project_id}.csv"
        config.write_text("paths: {}\n", encoding="utf-8")
        images.write_text("well,timepoint,decode_status\nA1,T0,ok\n", encoding="utf-8")
        (directory / "project.json").write_text(
            json.dumps({
                "project_id": project_id,
                "project_name": project_id,
                "plates": [{
                    "slug": plate_slug,
                    "config": str(config),
                    "images_manifest": str(images),
                }],
            }),
            encoding="utf-8",
        )
    manager = _PlateReviewManager(main_dir / "project.json", max_loaded=1)
    resolved = manager.resolve_plate("sibling", "sibling-board")
    assert resolved is not None
    assert resolved[0] == "sibling"
    assert manager.resolve_plate("main", "sibling-board") is None


def test_dynamic_plate_route_forwards_the_requested_project_and_board(
    tmp_path: Path,
    monkeypatch,
):
    import cellvision.project_server as project_server

    projects_root = tmp_path / "projects"
    main_dir = projects_root / "main"
    sibling_dir = projects_root / "sibling"
    main_dir.mkdir(parents=True)
    sibling_dir.mkdir(parents=True)
    manifests = []
    for directory, project_id, plate_slug in [
        (main_dir, "main", "main-board"),
        (sibling_dir, "sibling", "sibling-board"),
    ]:
        config = tmp_path / f"{project_id}.yaml"
        images = tmp_path / f"{project_id}.csv"
        config.write_text("paths: {}\n", encoding="utf-8")
        images.write_text("well,timepoint,decode_status\nA1,T0,ok\n", encoding="utf-8")
        manifest = directory / "project.json"
        manifest.write_text(
            json.dumps({
                "project_id": project_id,
                "project_name": project_id,
                "plates": [{
                    "slug": plate_slug,
                    "config": str(config),
                    "images_manifest": str(images),
                }],
            }),
            encoding="utf-8",
        )
        manifests.append(manifest)

    monkeypatch.setattr(project_server, "load_config", lambda _path: {})

    def fake_create_app(_config, *, project_back_url=None, review_base_url=None):
        async def fake_review_app(scope, receive, send):
            body = json.dumps({
                "path": scope.get("path"),
                "app_root_path": scope.get("app_root_path"),
                "review_base_url": review_base_url,
                "project_back_url": project_back_url,
            }).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            })
            await send({
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            })

        return fake_review_app

    monkeypatch.setattr(project_server, "create_app", fake_create_app)
    app = create_project_app(manifests[0])

    async def request(path: str):
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": [],
            "client": ("test", 1),
            "server": ("test", 80),
            "root_path": "",
        }
        await app(scope, receive, send)
        status = next(item["status"] for item in sent if item["type"] == "http.response.start")
        body = b"".join(item.get("body", b"") for item in sent if item["type"] == "http.response.body")
        return status, json.loads(body)

    status, payload = asyncio.run(
        request("/projects/sibling/plates/sibling-board/auto-review")
    )
    assert status == 200
    assert payload["path"] == "/auto-review"
    assert payload["review_base_url"] == "/projects/sibling/plates/sibling-board"
    assert payload["project_back_url"] == "/projects/sibling/"

    status, _ = asyncio.run(request("/projects/main/plates/sibling-board/auto-review"))
    assert status == 404


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
    project_queue = Path(task["project_manifest"]).parent / "task_queue.json"
    main_queue = main_dir / "task_queue.json"
    assert json.loads(project_queue.read_text(encoding="utf-8"))[0]["task_id"] == task["task_id"]
    assert not main_queue.exists()
    assert project["created_by"] == "张三"

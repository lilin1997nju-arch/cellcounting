import asyncio
import json
import sqlite3
from pathlib import Path

from cellvision.project_server import (
    _PlateReviewManager,
    ProjectRenamePayload,
    TaskPayload,
    _folder_name,
    _manual_verdict_counts_for_plate,
    create_project_app,
)


def test_folder_name_uses_parent_for_sessions_index():
    assert _folder_name(r"E:\\CM\\20260623 QL2603") == "20260623 QL2603"
    assert _folder_name(r"E:\\CM\\20260623 QL2603\\sessions.idx") == "20260623 QL2603"


def test_production_project_hub_hides_specialist_training_and_mask_routes(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("CELLVISION_PRODUCTION", "1")
    manifest_path = tmp_path / "projects" / "main" / "project.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"project_id": "main", "project_name": "Main", "plates": []}),
        encoding="utf-8",
    )

    app = create_project_app(manifest_path)
    paths = {getattr(route, "path", "") for route in app.routes}

    assert "/single-doublet-review" not in paths
    assert "/mask-review" not in paths
    assert "/api/multiplicity-training-candidates" not in paths
    assert "/api/mask-review-rounds" not in paths
    assert "/api/project/export-results" in paths
    assert "/api/project/review-filter-counts" in paths
    assert "/api/project/tasks/{task_id_value}/offline-review-export" in paths
    assert "/api/project/tasks/{task_id_value}/export-offline-review-results" in paths
    assert "/api/project/tasks/{task_id_value}/import-offline-review" in paths

    server_source = (
        Path(__file__).parents[1] / "src" / "cellvision" / "project_server.py"
    ).read_text(encoding="utf-8")
    assert "previous_commit == current_release_commit" in server_source


def test_production_dashboard_has_no_training_or_mask_review_entry():
    html = (
        Path(__file__).parents[1] / "review-ui" / "project-dashboard.html"
    ).read_text(encoding="utf-8")
    quick_review = (
        Path(__file__).parents[1] / "review-ui" / "auto-review.html"
    ).read_text(encoding="utf-8")
    dashboard_js = (
        Path(__file__).parents[1] / "review-ui" / "project-dashboard.js"
    ).read_text(encoding="utf-8")

    assert "单/粘连训练审核" not in html
    assert "Mask 轮廓审核" not in html
    assert "用本轮结果训练并生成下一轮" not in quick_review
    assert "导出 .cvreview 审核数据包" in dashboard_js
    assert "轮廓和快捷键与生产审核一致" in html
    assert "统一审核筛选条件" in html
    assert "筛选命中孔数" in html
    assert "合格孔" in dashboard_js
    assert "待定孔" in dashboard_js
    assert "排除孔" in dashboard_js
    assert "openReviewFilterDialog" not in dashboard_js
    assert "refreshReviewFilterCounts" in dashboard_js


def test_plate_manual_verdict_counts_keep_unreviewed_wells_unclassified(tmp_path: Path):
    artifact_root = tmp_path / "plate"
    predictions = artifact_root / "predictions"
    predictions.mkdir(parents=True)
    (predictions / "latest_well_screening.csv").write_text(
        "well,review_decision\nA1,approved\nA2,pending\nA3,rejected\nA4,\n",
        encoding="utf-8",
    )
    database = artifact_root / "annotations" / "annotations.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE well_screening_reviews(well TEXT PRIMARY KEY, decision TEXT)"
        )
        connection.executemany(
            "INSERT INTO well_screening_reviews(well, decision) VALUES (?, ?)",
            [("A1", "approved"), ("A2", "pending"), ("A3", "rejected")],
        )

    counts = _manual_verdict_counts_for_plate({"artifact_root": str(artifact_root)})

    assert counts == {"approved": 1, "pending": 1, "rejected": 1, "unclassified": 1}


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


def test_legacy_task_plans_are_restored_to_the_task_list(tmp_path: Path):
    projects_root = tmp_path / "projects"
    main_dir = projects_root / "main"
    main_dir.mkdir(parents=True)
    manifest_path = main_dir / "project.json"
    manifest_path.write_text(
        json.dumps({
            "project_id": "main",
            "project_name": "Main",
            "task_id": "task-legacy-1",
            "status": "queued",
            "plates": [],
        }),
        encoding="utf-8",
    )
    plan_dir = main_dir / "task_plans"
    plan_dir.mkdir()
    (plan_dir / "task-legacy-1.json").write_text(
        json.dumps({
            "task_id": "task-legacy-1",
            "name": "历史任务",
            "created_by": "LL",
            "root": str(tmp_path / "source"),
            "index": str(tmp_path / "source" / "sessions.idx"),
            "selected_timepoint_labels": ["Day0", "Day1"],
            "endpoint_day_label": "Day1",
            "endpoint_day_number": 1,
            "sessions": [{
                "group_id": "G1",
                "board_id": "T1-1",
                "timepoint_label": "T0",
                "day_label": "Day0",
            }],
            "created_at": "2026-08-10T00:00:00+00:00",
        }),
        encoding="utf-8",
    )

    app = create_project_app(manifest_path)
    tasks_endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", "") == "/api/project/tasks"
        and "GET" in getattr(route, "methods", set())
    )

    restored = tasks_endpoint()
    assert [task["task_id"] for task in restored] == ["task-legacy-1"]
    assert restored[0]["status"] == "queued"
    assert restored[0]["created_by"] == "LL"
    assert restored[0]["project_id"] == "main"
    assert restored[0]["plan_path"].endswith("task_plans\\task-legacy-1.json")

    queue_path = main_dir / "task_queue.json"
    queued = json.loads(queue_path.read_text(encoding="utf-8"))
    assert [task["task_id"] for task in queued] == ["task-legacy-1"]
    assert [task["task_id"] for task in tasks_endpoint("main")] == ["task-legacy-1"]

    delete_endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", "") == "/api/project/tasks/{task_id}"
        and "DELETE" in getattr(route, "methods", set())
    )
    deleted = delete_endpoint("task-legacy-1")
    assert deleted["status"] == "deleted"
    assert (plan_dir / "task-legacy-1.json.deleted").exists()
    assert tasks_endpoint() == []
    assert tasks_endpoint("main") == []

    # A new app instance must not resurrect the deleted plan either.
    fresh_app = create_project_app(manifest_path)
    fresh_tasks_endpoint = next(
        route.endpoint for route in fresh_app.routes
        if getattr(route, "path", "") == "/api/project/tasks"
        and "GET" in getattr(route, "methods", set())
    )
    assert fresh_tasks_endpoint() == []


def test_deleting_a_migrated_task_removes_duplicate_queue_copies(tmp_path: Path):
    projects_root = tmp_path / "projects"
    first_dir = projects_root / "first"
    second_dir = projects_root / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first_manifest = first_dir / "project.json"
    second_manifest = second_dir / "project.json"
    first_manifest.write_text(
        json.dumps({"project_id": "first", "project_name": "First", "plates": []}),
        encoding="utf-8",
    )
    second_manifest.write_text(
        json.dumps({"project_id": "second", "project_name": "Second", "plates": []}),
        encoding="utf-8",
    )
    task = {
        "task_id": "task-duplicate",
        "name": "Duplicate legacy task",
        "project_id": "first",
        "project_manifest": str(first_manifest),
        "plan_path": str(first_dir / "task_plans" / "task-duplicate.json"),
        "status": "cancelled",
    }
    (first_dir / "task_plans").mkdir()
    (first_dir / "task_plans" / "task-duplicate.json").write_text(
        json.dumps(task), encoding="utf-8"
    )
    (first_dir / "task_queue.json").write_text(json.dumps([task]), encoding="utf-8")
    (second_dir / "task_queue.json").write_text(json.dumps([task]), encoding="utf-8")

    app = create_project_app(first_manifest)
    tasks_endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", "") == "/api/project/tasks"
        and "GET" in getattr(route, "methods", set())
    )
    delete_endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", "") == "/api/project/tasks/{task_id}"
        and "DELETE" in getattr(route, "methods", set())
    )
    assert [item["task_id"] for item in tasks_endpoint()] == ["task-duplicate"]
    assert delete_endpoint("task-duplicate")["status"] == "deleted"
    assert tasks_endpoint() == []
    assert json.loads((first_dir / "task_queue.json").read_text(encoding="utf-8")) == []
    assert json.loads((second_dir / "task_queue.json").read_text(encoding="utf-8")) == []


def test_deleting_a_task_from_a_legacy_queue_removes_its_project_queue_copy(
    tmp_path: Path,
):
    projects_root = tmp_path / "projects"
    main_dir = projects_root / "main"
    child_dir = projects_root / "child"
    main_dir.mkdir(parents=True)
    child_dir.mkdir(parents=True)
    main_manifest = main_dir / "project.json"
    child_manifest = child_dir / "project.json"
    main_manifest.write_text(
        json.dumps({"project_id": "main", "project_name": "Main", "plates": []}),
        encoding="utf-8",
    )
    child_manifest.write_text(
        json.dumps({"project_id": "child", "project_name": "Child", "plates": []}),
        encoding="utf-8",
    )
    task = {
        "task_id": "task-legacy-copy",
        "name": "Legacy copy",
        "project_id": "child",
        "project_manifest": str(child_manifest),
        "status": "error",
    }
    (main_dir / "task_queue.json").write_text(json.dumps([task]), encoding="utf-8")
    (child_dir / "task_queue.json").write_text(json.dumps([task]), encoding="utf-8")

    app = create_project_app(main_manifest)
    tasks_endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", "") == "/api/project/tasks"
        and "GET" in getattr(route, "methods", set())
    )
    delete_endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", "") == "/api/project/tasks/{task_id}"
        and "DELETE" in getattr(route, "methods", set())
    )
    assert [item["task_id"] for item in tasks_endpoint()] == ["task-legacy-copy"]
    deleted = delete_endpoint("task-legacy-copy")
    assert deleted["status"] == "deleted"
    assert deleted["cleaned_project"] is True
    assert tasks_endpoint() == []
    assert json.loads((main_dir / "task_queue.json").read_text(encoding="utf-8")) == []
    assert not child_dir.exists()

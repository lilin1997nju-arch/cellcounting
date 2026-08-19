from pathlib import Path

from cellvision.project_worker import ProjectTaskWorker
from cellvision.runtime import detect_compute_runtime
from cellvision.task_queue import TaskQueueStore


def test_project_worker_waits_for_start_and_completes_one_task(tmp_path: Path):
    store = TaskQueueStore(tmp_path / "tasks.json")
    store.add({"task_id": "task-1", "name": "demo"})
    worker = ProjectTaskWorker(
        store,
        runtime=detect_compute_runtime("cpu"),
        worker_id="cpu-test",
        runtime_path=tmp_path / "worker_runtime.json",
        executor=lambda task: {"task_id": task["task_id"], "device": "cpu"},
    )

    assert worker.run_once() is None
    store.start("task-1")
    result = worker.run_once()

    assert result is not None
    assert result["status"] == "completed"
    assert result["result"]["device"] == "cpu"
    assert store.get("task-1")["worker_id"] == "cpu-test"


def _project_store(root: Path, project_id: str) -> TaskQueueStore:
    project_dir = root / project_id
    project_dir.mkdir(parents=True)
    (project_dir / "project.json").write_text(
        '{"project_id": "' + project_id + '", "plates": []}',
        encoding="utf-8",
    )
    return TaskQueueStore(project_dir / "task_queue.json")


def test_project_worker_claims_oldest_started_task_across_projects(tmp_path: Path):
    project_root = tmp_path / "projects"
    main_store = _project_store(project_root, "main")
    other_store = _project_store(project_root, "other")
    main_store.add({"task_id": "task-main", "name": "main"})
    other_store.add({"task_id": "task-other", "name": "other"})
    main_store.start("task-main")
    other_store.start("task-other")
    main_store.update("task-main", started_at="2026-08-19T02:00:00+00:00")
    other_store.update("task-other", started_at="2026-08-19T01:00:00+00:00")
    executed: list[str] = []
    worker = ProjectTaskWorker(
        main_store,
        queue_root=project_root,
        runtime=detect_compute_runtime("cpu"),
        worker_id="cpu-test",
        runtime_path=project_root / "worker_runtime.json",
        executor=lambda task: executed.append(task["task_id"]) or {"ok": True},
    )

    first = worker.run_once()
    second = worker.run_once()

    assert first is not None and first["task_id"] == "task-other"
    assert second is not None and second["task_id"] == "task-main"
    assert executed == ["task-other", "task-main"]
    assert other_store.get("task-other")["status"] == "completed"
    assert main_store.get("task-main")["status"] == "completed"


def test_project_worker_from_manifest_uses_collection_runtime_and_recovers_sibling_task(
    tmp_path: Path,
):
    project_root = tmp_path / "projects"
    main_store = _project_store(project_root, "main")
    sibling_store = _project_store(project_root, "sibling")
    sibling_store.add({"task_id": "task-stuck", "name": "stuck"})
    sibling_store.start("task-stuck")

    worker = ProjectTaskWorker.from_manifest(
        main_store.path.parent / "project.json",
        requested_device="cpu",
        worker_id="cpu-restarted",
    )
    worker.executor = lambda task: {"recovered": task["task_id"]}
    result = worker.run_once()

    assert result is not None
    assert result["task_id"] == "task-stuck"
    assert result["status"] == "completed"
    assert sibling_store.get("task-stuck")["worker_id"] == "cpu-restarted"
    assert worker.runtime_path == project_root / "worker_runtime.json"

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


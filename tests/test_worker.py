from __future__ import annotations

from pathlib import Path

from cellvision.task_queue import TaskQueueStore
from cellvision.worker import TaskWorker


def test_worker_claims_and_completes_task(tmp_path: Path):
    store = TaskQueueStore(tmp_path / "tasks.json")
    store.add({"task_id": "task-1"})
    worker = TaskWorker(store, lambda task: {"task_id": task["task_id"], "ok": True}, worker_id="test")
    result = worker.run_once()
    assert result is not None
    assert result["status"] == "completed"
    assert store.get("task-1")["result"]["ok"] is True


def test_worker_records_executor_failure(tmp_path: Path):
    store = TaskQueueStore(tmp_path / "tasks.json")
    store.add({"task_id": "task-1"})

    def fail(_task):
        raise RuntimeError("broken")

    result = TaskWorker(store, fail, worker_id="test").run_once()
    assert result is not None
    assert result["status"] == "error"
    assert "broken" in result["error"]


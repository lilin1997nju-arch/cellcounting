from __future__ import annotations

from pathlib import Path

from cellvision.task_queue import TaskQueueStore


def test_task_queue_lifecycle_is_atomic_and_explicit(tmp_path: Path):
    store = TaskQueueStore(tmp_path / "tasks.json")
    task = store.add({"task_id": "task-1", "name": "demo"})
    assert task["status"] == "queued"
    claimed = store.claim_next("worker-1")
    assert claimed is not None
    assert claimed["status"] == "running"
    assert claimed["attempts"] == 1
    assert store.complete("task-1", {"ok": True})["status"] == "completed"
    assert store.get("task-1")["result"] == {"ok": True}


def test_task_queue_can_cancel_queued_task(tmp_path: Path):
    store = TaskQueueStore(tmp_path / "tasks.json")
    store.add({"task_id": "task-1"})
    cancelled = store.cancel("task-1")
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    assert store.claim_next("worker-1") is None


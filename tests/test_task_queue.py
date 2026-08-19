from __future__ import annotations

from pathlib import Path

from cellvision.task_queue import TaskQueueStore


def test_task_queue_lifecycle_is_atomic_and_explicit(tmp_path: Path):
    store = TaskQueueStore(tmp_path / "tasks.json")
    task = store.add({"task_id": "task-1", "name": "demo", "group_count": 4})
    assert task["status"] == "queued"
    assert task["progress_percent"] == 0
    assert task["progress_current"] == 0
    assert task["progress_total"] == 4
    assert task["progress_message"] == "排队中，尚未开始"
    claimed = store.claim_next("worker-1")
    assert claimed is not None
    assert claimed["status"] == "running"
    assert claimed["attempts"] == 1
    assert claimed["progress_message"] == "任务执行中"
    updated = store.update_progress("task-1", 2, stage="inference", message="正在处理第 2/4 组")
    assert updated is not None
    assert updated["progress_percent"] == 50
    assert updated["progress_stage"] == "inference"
    assert store.complete("task-1", {"ok": True})["status"] == "completed"
    assert store.get("task-1")["result"] == {"ok": True}
    assert store.get("task-1")["progress_percent"] == 100


def test_task_queue_can_cancel_queued_task(tmp_path: Path):
    store = TaskQueueStore(tmp_path / "tasks.json")
    store.add({"task_id": "task-1"})
    cancelled = store.cancel("task-1")
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    assert store.claim_next("worker-1") is None


def test_task_queue_can_start_and_delete_unfinished_task(tmp_path: Path):
    store = TaskQueueStore(tmp_path / "tasks.json")
    store.add({"task_id": "task-1"})

    started = store.start("task-1")
    assert started is not None
    assert started["status"] == "running"
    assert started["progress_stage"] == "starting"
    assert store.start("task-1")["status"] == "running"
    assert store.delete("task-1")["task_id"] == "task-1"
    assert store.get("task-1") is None


def test_task_queue_worker_claims_only_explicitly_started_task(tmp_path: Path):
    store = TaskQueueStore(tmp_path / "tasks.json")
    store.add({"task_id": "task-1"})

    # A newly queued task is intentionally invisible to the project worker.
    assert store.claim_started("worker-1") is None
    started = store.start("task-1")
    assert started is not None
    claimed = store.claim_started("worker-1")
    assert claimed is not None
    assert claimed["worker_id"] == "worker-1"
    assert claimed["progress_stage"] == "running"
    assert store.claim_started("worker-2") is None


def test_task_queue_can_claim_a_specific_started_task(tmp_path: Path):
    store = TaskQueueStore(tmp_path / "tasks.json")
    store.add({"task_id": "task-1"})
    store.add({"task_id": "task-2"})
    store.start("task-1")
    store.start("task-2")

    claimed = store.claim_started("worker-1", task_id="task-2")

    assert claimed is not None
    assert claimed["task_id"] == "task-2"
    assert store.get("task-1")["progress_stage"] == "starting"


def test_task_queue_preserves_board_progress_and_runtime(tmp_path: Path):
    store = TaskQueueStore(tmp_path / "tasks.json")
    store.add({"task_id": "task-1", "group_count": 1})
    store.start("task-1")
    store.update(
        "task-1",
        progress_boards=[{
            "slug": "demo-board",
            "board_id": "T1-1",
            "status": "running",
            "stage": "morphology_inference",
            "progress_percent": 45,
            "elapsed_seconds": 12.5,
        }],
    )
    running = store.get("task-1")
    assert running["progress_boards"][0]["stage"] == "morphology_inference"
    assert running["elapsed_seconds"] >= 0
    completed = store.complete("task-1")
    assert completed["elapsed_seconds"] >= 0
    assert completed["progress_boards"][0]["progress_percent"] == 45


def test_task_queue_delete_does_not_enforce_completed_guard_at_store_layer(tmp_path: Path):
    store = TaskQueueStore(tmp_path / "tasks.json")
    store.add({"task_id": "task-1"})
    store.complete("task-1")

    # The HTTP layer enforces the product rule that completed tasks cannot be
    # deleted; the store remains a small, reusable persistence primitive.
    assert store.delete("task-1")["status"] == "completed"

"""Reusable worker loop for long-running project tasks.

The worker deliberately receives an executor callback.  Pipeline-specific
execution stays outside the API process, while the queue lifecycle and error
handling are shared by future project import, inference, and training jobs.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from .task_queue import TaskQueueStore


TaskExecutor = Callable[[dict[str, Any]], dict[str, Any] | None]


class TaskWorker:
    def __init__(self, store: TaskQueueStore, executor: TaskExecutor, *, worker_id: str | None = None):
        self.store = store
        self.executor = executor
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"

    def run_once(self) -> dict[str, Any] | None:
        task = self.store.claim_next(self.worker_id)
        if task is None:
            return None
        task_id = str(task["task_id"])
        try:
            result = self.executor(task) or {}
        except Exception as exc:  # keep the worker alive for the next task
            return self.store.fail(task_id, f"{type(exc).__name__}: {exc}")
        return self.store.complete(task_id, result)

    def run_forever(self, *, poll_seconds: float = 2.0) -> None:
        while True:
            if self.run_once() is None:
                time.sleep(max(0.1, float(poll_seconds)))


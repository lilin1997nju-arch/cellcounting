"""Small durable task-state store used by the project hub.

The current project runner is still the safest pipeline executor.  This module
separates task state from the HTTP layer so a future GPU worker can claim,
retry, cancel, and complete jobs without changing the browser API again.
JSON remains the on-disk format for backwards compatibility with existing task
files; writes are atomic and all lifecycle transitions are explicit.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    path = path.resolve()
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path, threading.Lock())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskQueueStore:
    """Atomic JSON-backed task lifecycle storage."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    def write(self, tasks: list[dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def list(self) -> list[dict[str, Any]]:
        with _lock_for(self.path):
            return self.read()

    def get(self, task_id: str) -> dict[str, Any] | None:
        with _lock_for(self.path):
            return next((task for task in self.read() if str(task.get("task_id")) == str(task_id)), None)

    def add(self, task: dict[str, Any]) -> dict[str, Any]:
        value = dict(task)
        value.setdefault("status", "queued")
        value.setdefault("attempts", 0)
        now = _now()
        value.setdefault("created_at", now)
        value["updated_at"] = now
        with _lock_for(self.path):
            tasks = self.read()
            tasks.append(value)
            self.write(tasks)
        return value

    def update(self, task_id: str, **fields: Any) -> dict[str, Any] | None:
        with _lock_for(self.path):
            tasks = self.read()
            found = None
            for task in tasks:
                if str(task.get("task_id")) == str(task_id):
                    task.update(fields)
                    task["updated_at"] = _now()
                    found = dict(task)
                    break
            if found is not None:
                self.write(tasks)
            return found

    def claim_next(self, worker_id: str) -> dict[str, Any] | None:
        """Atomically claim one queued task for a worker."""

        with _lock_for(self.path):
            tasks = self.read()
            for task in tasks:
                if str(task.get("status", "queued")) != "queued":
                    continue
                task["status"] = "running"
                task["worker_id"] = worker_id
                task["started_at"] = _now()
                task["attempts"] = int(task.get("attempts", 0) or 0) + 1
                task["updated_at"] = _now()
                self.write(tasks)
                return dict(task)
        return None

    def complete(self, task_id: str, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return self.update(task_id, status="completed", finished_at=_now(), result=result or {}, error=None)

    def fail(self, task_id: str, error: str, *, retry: bool = False) -> dict[str, Any] | None:
        return self.update(
            task_id,
            status="queued" if retry else "error",
            error=str(error),
            retry_requested=bool(retry),
        )

    def cancel(self, task_id: str) -> dict[str, Any] | None:
        task = self.get(task_id)
        if task is None or str(task.get("status")) not in {"queued", "running"}:
            return task
        return self.update(task_id, status="cancelled", cancelled_at=_now())


def task_id(prefix: str = "task") -> str:
    return f"{prefix}-{int(time.time() * 1000)}"


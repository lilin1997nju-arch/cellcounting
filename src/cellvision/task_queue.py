"""Small durable task-state store used by the project hub.

This module separates task state from the HTTP layer so an independent GPU or
CPU worker can claim, retry, cancel, and complete jobs without changing the
browser API again.
JSON remains the on-disk format for backwards compatibility with existing task
files; writes are atomic and all lifecycle transitions are explicit.
"""

from __future__ import annotations

import json
import os
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


def _elapsed_seconds(started_at: str | None, finished_at: str | None = None) -> float:
    """Return a safe wall-clock duration for an old or current task record."""

    if not started_at:
        return 0.0
    try:
        started = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        finished = datetime.fromisoformat(
            str(finished_at or _now()).replace("Z", "+00:00")
        )
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if finished.tzinfo is None:
            finished = finished.replace(tzinfo=timezone.utc)
        return max(0.0, round((finished - started).total_seconds(), 3))
    except (TypeError, ValueError, OverflowError):
        return 0.0


class TaskQueueStore:
    """Atomic JSON-backed task lifecycle storage."""

    @staticmethod
    def _with_progress_defaults(task: dict[str, Any]) -> dict[str, Any]:
        """Keep old queue records readable by the progress-aware UI."""

        status = str(task.get("status") or "queued")
        task.setdefault("status", status)
        task.setdefault("progress_current", 0)
        task.setdefault(
            "progress_total",
            int(task.get("group_count", 0) or 0),
        )
        task.setdefault("progress_percent", 0)
        task.setdefault("progress_stage", status)
        task.setdefault("progress_boards", [])
        task.setdefault("elapsed_seconds", 0.0)
        task.setdefault(
            "progress_message",
            {
                "queued": "排队中，尚未开始",
                "running": "任务执行中",
                "completed": "任务已完成",
                "error": "任务执行失败",
                "cancelled": "任务已取消",
            }.get(status, status),
        )
        return task

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
        return [
            self._with_progress_defaults(item)
            for item in value
            if isinstance(item, dict)
        ] if isinstance(value, list) else []

    def write(self, tasks: list[dict[str, Any]]) -> None:
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            for attempt in range(20):
                try:
                    os.replace(temporary, self.path)
                    return
                except PermissionError:
                    if attempt == 19:
                        raise
                    time.sleep(min(0.02 * (attempt + 1), 0.2))
        finally:
            temporary.unlink(missing_ok=True)

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
        value.setdefault("progress_current", 0)
        value.setdefault("progress_total", int(value.get("group_count", 0) or 0))
        value.setdefault("progress_percent", 0)
        value.setdefault("progress_stage", "queued")
        value.setdefault("progress_message", "排队中，尚未开始")
        value.setdefault("progress_boards", [])
        value.setdefault("elapsed_seconds", 0.0)
        now = _now()
        value.setdefault("created_at", now)
        value["updated_at"] = now
        self._with_progress_defaults(value)
        with _lock_for(self.path):
            tasks = self.read()
            tasks.append(value)
            self.write(tasks)
        return value

    def ensure(self, task: dict[str, Any]) -> dict[str, Any]:
        """Insert a task only when its id is not already present.

        This is used when importing task records written by older versions of
        the project hub.  The operation is guarded by the same per-file lock
        as the normal lifecycle methods, so repeated page refreshes or a
        concurrent worker cannot create duplicate task rows.
        """

        value = dict(task)
        value.setdefault("status", "queued")
        value.setdefault("attempts", 0)
        value.setdefault("progress_current", 0)
        value.setdefault("progress_total", int(value.get("group_count", 0) or 0))
        value.setdefault("progress_percent", 0)
        value.setdefault("progress_stage", str(value.get("status") or "queued"))
        value.setdefault("progress_boards", [])
        value.setdefault("elapsed_seconds", 0.0)
        now = _now()
        value.setdefault("created_at", now)
        value.setdefault("updated_at", now)
        self._with_progress_defaults(value)
        task_key = str(value.get("task_id") or "")
        with _lock_for(self.path):
            tasks = self.read()
            for existing in tasks:
                if task_key and str(existing.get("task_id") or "") == task_key:
                    return existing
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
                task["elapsed_seconds"] = 0.0
                task["attempts"] = int(task.get("attempts", 0) or 0) + 1
                task["progress_stage"] = "running"
                task["progress_message"] = "任务执行中"
                task["updated_at"] = _now()
                self.write(tasks)
                return dict(task)
        return None

    def claim_started(
        self,
        worker_id: str,
        task_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Claim one task that the user explicitly started.

        The browser's start button intentionally changes ``queued`` to
        ``running`` with ``progress_stage=starting``.  A project worker must
        not silently execute newly-created queued tasks before the user starts
        them, so it uses this narrower claim operation instead of
        :meth:`claim_next`.
        """

        with _lock_for(self.path):
            tasks = self.read()
            for task in tasks:
                if task_id is not None and str(task.get("task_id")) != str(task_id):
                    continue
                if str(task.get("status", "queued")) != "running":
                    continue
                if str(task.get("progress_stage", "")) != "starting":
                    continue
                owner = str(task.get("worker_id", ""))
                if owner not in {"", "manual"}:
                    continue
                task["worker_id"] = worker_id
                task["progress_stage"] = "running"
                task["progress_message"] = "任务执行器已接管"
                task["updated_at"] = _now()
                self.write(tasks)
                return dict(task)
        return None

    def start(self, task_id: str, worker_id: str = "manual") -> dict[str, Any] | None:
        """Manually move one queued task into the running lifecycle state."""

        with _lock_for(self.path):
            tasks = self.read()
            for task in tasks:
                if str(task.get("task_id")) != str(task_id):
                    continue
                if str(task.get("status", "queued")) != "queued":
                    return dict(task)
                task["status"] = "running"
                task["worker_id"] = worker_id
                task["started_at"] = _now()
                task["elapsed_seconds"] = 0.0
                task["attempts"] = int(task.get("attempts", 0) or 0) + 1
                task["progress_stage"] = "starting"
                task["progress_message"] = "已开始计算，等待执行器接管"
                task["updated_at"] = _now()
                self.write(tasks)
                return dict(task)
        return None

    def complete(self, task_id: str, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
        task = self.get(task_id)
        total = int(task.get("progress_total", 0) or 0) if task else 0
        finished_at = _now()
        return self.update(
            task_id,
            status="completed",
            finished_at=finished_at,
            elapsed_seconds=_elapsed_seconds(task.get("started_at") if task else None, finished_at),
            result=result or {},
            error=None,
            progress_current=total if total else (task.get("progress_current", 0) if task else 0),
            progress_percent=100,
            progress_stage="completed",
            progress_message="任务已完成",
        )

    def fail(self, task_id: str, error: str, *, retry: bool = False) -> dict[str, Any] | None:
        task = self.get(task_id)
        finished_at = _now()
        return self.update(
            task_id,
            status="queued" if retry else "error",
            error=str(error),
            retry_requested=bool(retry),
            **({
                "finished_at": finished_at,
                "elapsed_seconds": _elapsed_seconds(
                    task.get("started_at") if task else None,
                    finished_at,
                ),
            } if not retry else {}),
            progress_stage="queued" if retry else "error",
            progress_message="已重新排队，等待执行" if retry else f"任务执行失败：{error}",
        )

    def cancel(self, task_id: str) -> dict[str, Any] | None:
        task = self.get(task_id)
        if task is None or str(task.get("status")) not in {"queued", "running"}:
            return task
        cancelled_at = _now()
        fields: dict[str, Any] = {
            "status": "cancelled",
            "cancelled_at": cancelled_at,
            "progress_stage": "cancelled",
            "progress_message": "任务已取消",
        }
        if str(task.get("status")) == "running":
            fields.update({
                "finished_at": cancelled_at,
                "elapsed_seconds": _elapsed_seconds(task.get("started_at"), cancelled_at),
            })
        return self.update(
            task_id,
            **fields,
        )

    def delete(self, task_id: str) -> dict[str, Any] | None:
        """Remove a task record; callers must guard completed/running states."""

        with _lock_for(self.path):
            tasks = self.read()
            for index, task in enumerate(tasks):
                if str(task.get("task_id")) != str(task_id):
                    continue
                removed = dict(task)
                del tasks[index]
                self.write(tasks)
                return removed
        return None

    def update_progress(
        self,
        task_id: str,
        current: int | float,
        total: int | float | None = None,
        *,
        stage: str | None = None,
        message: str | None = None,
    ) -> dict[str, Any] | None:
        """Persist worker progress without coupling the queue to a pipeline."""

        task = self.get(task_id)
        if task is None:
            return None
        current_value = max(0, float(current or 0))
        total_value = float(
            total if total is not None else task.get("progress_total", 0) or 0
        )
        percent = (
            min(100, max(0, round(current_value / total_value * 100)))
            if total_value > 0
            else int(task.get("progress_percent", 0) or 0)
        )
        fields: dict[str, Any] = {
            "progress_current": int(current_value) if current_value.is_integer() else current_value,
            "progress_total": int(total_value) if total_value.is_integer() else total_value,
            "progress_percent": percent,
        }
        if str(task.get("status")) == "running":
            fields["elapsed_seconds"] = _elapsed_seconds(task.get("started_at"))
        if stage:
            fields["progress_stage"] = stage
        if message:
            fields["progress_message"] = message
        return self.update(task_id, **fields)


def task_id(prefix: str = "task") -> str:
    return f"{prefix}-{int(time.time() * 1000)}"

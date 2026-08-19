"""Project task worker used by the queue UI and standalone deployments.

The worker is deliberately outside the FastAPI request handlers.  It claims
only tasks that a user has explicitly started, prepares a project manifest when
needed, runs boards sequentially, persists progress, and terminates the active
subprocess when a task is cancelled.
"""

from __future__ import annotations

import json
import os
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from .runtime import ComputeRuntime, detect_compute_runtime
from .project_catalog import ProjectCatalog, catalog_path_for_manifest
from .session_index import parse_sessions_index
from .task_queue import TaskQueueStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STREAM_END = object()
_STAGE_PROGRESS = {
    "day14_gate": 5,
    "initialize_database": 10,
    "build_positive_only_t0_t2_manifest": 18,
    "build_cf_candidates": 26,
    "dense_candidate_augmentation": 34,
    "morphology_inference": 45,
    "auto_annotation_round": 52,
    "multiplicity_inference": 60,
    "integrated_round": 68,
    "v2_instance_segmentation": 78,
    "v2_temporal_evidence": 88,
    "early_well_screening": 95,
    "final_report": 99,
}


class TaskCancelledError(RuntimeError):
    """Raised internally when the queue record is cancelled."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _day_number(label: str) -> int | None:
    match = re.match(r"^day\s*(-?\d+)$", str(label).strip(), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _duration_seconds(started_at: str | None, finished_at: str | None = None) -> float:
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


class ProjectTaskWorker:
    """Execute started project tasks with one adaptive compute worker."""

    def __init__(
        self,
        store: TaskQueueStore,
        *,
        queue_root: str | Path | None = None,
        runtime: ComputeRuntime | None = None,
        requested_device: str | None = None,
        worker_id: str | None = None,
        runtime_path: str | Path | None = None,
        catalog_path: str | Path | None = None,
        executor: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    ):
        self.store = store
        self._default_store = store
        self.queue_root = (
            Path(queue_root).expanduser().resolve()
            if queue_root is not None
            else None
        )
        self.runtime = runtime or detect_compute_runtime(requested_device)
        self.worker_id = worker_id or f"{self.runtime.worker_kind}-worker-{os.getpid()}"
        self.runtime_path = (
            Path(runtime_path).expanduser().resolve()
            if runtime_path
            else self.store.path.parent / "worker_runtime.json"
        )
        self.catalog = ProjectCatalog(
            catalog_path
            if catalog_path is not None
            else self.store.path.parent / "project_catalog.sqlite"
        )
        self.executor = executor or self.execute_project_task
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = _now()

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        requested_device: str | None = None,
        worker_id: str | None = None,
    ) -> "ProjectTaskWorker":
        manifest = _resolve(manifest_path)
        queue_path = manifest.parent / "task_queue.json"
        queue_root = manifest.parent.parent
        return cls(
            TaskQueueStore(queue_path),
            queue_root=queue_root,
            requested_device=requested_device,
            worker_id=worker_id,
            runtime_path=queue_root / "worker_runtime.json",
            catalog_path=catalog_path_for_manifest(manifest),
        )

    def _queue_stores(self) -> list[TaskQueueStore]:
        """Return every live project queue visible to this worker."""

        paths = {self._default_store.path}
        if self.queue_root is not None and self.queue_root.is_dir():
            paths.update(
                manifest.parent / "task_queue.json"
                for manifest in self.queue_root.glob("*/project.json")
                if manifest.is_file()
            )
        ordered_paths = sorted(paths, key=lambda value: value.as_posix().casefold())
        return [TaskQueueStore(path) for path in ordered_paths]

    @staticmethod
    def _started_sort_key(
        task: dict[str, Any],
        store: TaskQueueStore,
    ) -> tuple[str, str, str]:
        started_at = str(
            task.get("started_at")
            or task.get("updated_at")
            or task.get("created_at")
            or "9999-12-31T23:59:59+00:00"
        )
        return (
            started_at,
            str(task.get("task_id", "")),
            store.path.as_posix().casefold(),
        )

    def _claim_started_task(self) -> tuple[TaskQueueStore, dict[str, Any]] | None:
        """Claim the oldest explicitly-started task across all project queues."""

        candidates: list[tuple[dict[str, Any], TaskQueueStore]] = []
        for store in self._queue_stores():
            for task in store.list():
                if str(task.get("status", "queued")) != "running":
                    continue
                if str(task.get("progress_stage", "")) != "starting":
                    continue
                if str(task.get("worker_id", "")) not in {"", "manual"}:
                    continue
                candidates.append((task, store))

        candidates.sort(key=lambda item: self._started_sort_key(item[0], item[1]))
        for candidate, store in candidates:
            claimed = store.claim_started(
                self.worker_id,
                task_id=str(candidate.get("task_id", "")),
            )
            if claimed is not None:
                return store, claimed
        return None

    def _write_runtime(
        self,
        status: str,
        *,
        task: dict[str, Any] | None = None,
        last_task_id: str = "",
    ) -> None:
        payload = {
            **self.runtime.as_dict(),
            "status": status,
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "started_at": self._started_at,
            "updated_at": _now(),
        }
        if task is not None:
            payload.update({
                "task_id": str(task.get("task_id", "")),
                "task_name": str(task.get("name", "")),
            })
        elif last_task_id:
            payload["last_task_id"] = last_task_id
        try:
            _atomic_json_write(self.runtime_path, payload)
        except OSError:
            # A read-only artifact mount should not stop inference itself.
            pass

    def start_background(self, *, poll_seconds: float = 2.0) -> threading.Thread:
        """Start the worker loop in a daemon thread for ``review-project``."""

        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._stop_event.clear()
        self._write_runtime("starting")
        self._thread = threading.Thread(
            target=self._background_loop,
            args=(max(0.1, float(poll_seconds)),),
            name=f"cellvision-{self.worker_id}",
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def stop(self, *, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(0.1, float(timeout)))
        self._write_runtime("stopped")

    def _background_loop(self, poll_seconds: float) -> None:
        try:
            while not self._stop_event.is_set():
                result = self.run_once()
                if result is None:
                    self._stop_event.wait(poll_seconds)
        finally:
            self._write_runtime("stopped")

    def run_forever(self, *, poll_seconds: float = 2.0) -> None:
        """Run in the foreground for a dedicated worker process."""

        self._write_runtime("starting")
        try:
            while True:
                result = self.run_once()
                if result is None:
                    time.sleep(max(0.1, float(poll_seconds)))
        except KeyboardInterrupt:
            self._write_runtime("stopped")

    def run_once(self) -> dict[str, Any] | None:
        """Claim and execute one explicitly-started task."""

        claimed = self._claim_started_task()
        if claimed is None:
            self._write_runtime("idle")
            return None

        store, task = claimed
        previous_store = self.store
        self.store = store

        try:
            return self._run_claimed_task(task)
        finally:
            self.store = previous_store

    def _run_claimed_task(self, task: dict[str, Any]) -> dict[str, Any] | None:
        """Execute a task after its owning queue has been selected."""

        task_id = str(task["task_id"])
        self._sync_catalog_task(task)
        self._write_runtime("running", task=task)
        try:
            result = self.executor(task) or {}
        except TaskCancelledError:
            current = self.store.get(task_id)
            self._mark_project_manifest(task, "cancelled")
            self._sync_catalog_task(current or task, source="task_cancelled")
            self._write_runtime("idle", last_task_id=task_id)
            return current
        except Exception as exc:  # keep the worker alive for the next task
            error = f"{type(exc).__name__}: {exc}"
            self._mark_project_manifest(task, "error", error=error)
            failed = self.store.fail(task_id, error)
            self._sync_catalog_task(failed or task, source="task_failed")
            self._write_runtime("idle", last_task_id=task_id)
            return failed

        current = self.store.get(task_id)
        if current is not None and str(current.get("status")) == "cancelled":
            self._mark_project_manifest(task, "cancelled")
            self._sync_catalog_task(current, source="task_cancelled")
            self._write_runtime("idle", last_task_id=task_id)
            return current
        completed = self.store.complete(task_id, result)
        self._sync_catalog_task(completed or task, source="task_completed")
        self._write_runtime("idle", last_task_id=task_id)
        return completed

    def _sync_catalog_task(
        self,
        task: dict[str, Any] | None,
        *,
        source: str = "task_progress",
        force_manifest: bool = False,
    ) -> None:
        if not task:
            return
        try:
            manifest_value = task.get("project_manifest")
            if manifest_value:
                manifest_path = _resolve(manifest_value)
                if manifest_path.exists():
                    self.catalog.sync_manifest(manifest_path, force=force_manifest, source=source)
            self.catalog.sync_task(task)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, sqlite3.Error):
            # Inference remains independent from the derived catalog.
            return

    def _update_progress(
        self,
        task_id: str,
        current: int | float,
        total: int | float | None = None,
        *,
        stage: str | None = None,
        message: str | None = None,
    ) -> dict[str, Any] | None:
        updated = self.store.update_progress(
            task_id, current, total, stage=stage, message=message
        )
        self._sync_catalog_task(updated)
        return updated

    def _ensure_not_cancelled(self, task_id: str) -> dict[str, Any]:
        current = self.store.get(task_id)
        if current is None or str(current.get("status")) == "cancelled":
            raise TaskCancelledError(f"task {task_id} was cancelled")
        if str(current.get("status")) != "running":
            raise RuntimeError(
                f"task {task_id} left the running state: {current.get('status')}"
            )
        return current

    def _set_board_progress(
        self,
        task_id: str,
        plates: list[dict[str, Any]],
        *,
        active_slug: str = "",
        active_stage: str = "",
        active_message: str = "",
    ) -> None:
        """Persist a compact stage/progress row for every board in the task."""

        task = self.store.get(task_id)
        if task is None:
            return
        previous = {
            str(row.get("slug")): row
            for row in task.get("progress_boards", [])
            if isinstance(row, dict) and row.get("slug")
        }
        now = _now()
        rows: list[dict[str, Any]] = []
        for plate in plates:
            slug = _text(plate.get("slug"))
            old = previous.get(slug, {})
            plate_status = str(plate.get("status", "queued"))
            row_status = plate_status if plate_status in {"queued", "running", "completed", "error", "cancelled"} else "queued"
            started_at = str(old.get("started_at") or plate.get("started_at") or "")
            finished_at = str(old.get("finished_at") or plate.get("finished_at") or "")
            stage = str(old.get("stage") or ("completed" if row_status == "completed" else row_status))
            message = str(old.get("message") or ("等待执行" if row_status == "queued" else ""))
            percent = float(old.get("progress_percent", 100 if row_status == "completed" else 0) or 0)
            elapsed = float(old.get("elapsed_seconds", 0) or 0)
            if row_status == "completed":
                stage = "completed"
                message = "板子计算完成"
                percent = 100.0
                elapsed = float(plate.get("elapsed_seconds", elapsed) or elapsed)
                finished_at = finished_at or now
            elif row_status == "error":
                stage = "error"
                message = str(plate.get("error") or old.get("message") or "板子计算失败")
                finished_at = finished_at or now
                elapsed = _duration_seconds(started_at, finished_at)
            elif started_at:
                elapsed = _duration_seconds(started_at)

            if slug and slug == active_slug:
                row_status = "error" if active_stage == "error" else "running"
                started_at = started_at or now
                stage = active_stage or stage or "running"
                message = active_message or message
                if active_stage in _STAGE_PROGRESS:
                    percent = max(percent, float(_STAGE_PROGRESS[active_stage]))
                percent = min(99.0, max(0.0, percent))
                if row_status == "error":
                    finished_at = finished_at or now
                    elapsed = _duration_seconds(started_at, finished_at)
                else:
                    elapsed = _duration_seconds(started_at)

            rows.append({
                "slug": slug,
                "board_id": _text(plate.get("board_id") or plate.get("group_id") or slug),
                "group_id": _text(plate.get("group_id")),
                "status": row_status,
                "stage": stage,
                "message": message,
                "progress_percent": round(min(100.0, max(0.0, percent)), 1),
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": round(max(0.0, elapsed), 3),
                "error": str(plate.get("error") or ""),
            })
        updated = self.store.update(
            task_id,
            progress_boards=rows,
            elapsed_seconds=_duration_seconds(task.get("started_at"))
            if str(task.get("status")) == "running"
            else float(task.get("elapsed_seconds", 0) or 0),
        )
        self._sync_catalog_task(updated)

    def _child_env(self, task: dict[str, Any]) -> dict[str, str]:
        env = os.environ.copy()
        env["CELLVISION_DEVICE"] = self.runtime.selected_device
        env["CELLVISION_WORKER_DEVICE"] = self.runtime.selected_device
        env["CELLVISION_WORKER_ID"] = self.worker_id
        env["PYTHONUNBUFFERED"] = "1"
        # This is set before the child imports torch, so an explicit CPU worker
        # cannot accidentally allocate on an available GPU.
        if self.runtime.selected_device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        source_artifacts = task.get("source_artifacts")
        if source_artifacts:
            env["CELLVISION_SOURCE_ARTIFACTS"] = str(_resolve(source_artifacts))
        return env

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=8)

    def _run_command(
        self,
        command: list[str],
        *,
        task_id: str,
        env: dict[str, str],
        on_line: Callable[[str], None] | None = None,
    ) -> list[str]:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        lines: list[str] = []
        output: queue.Queue[object] = queue.Queue()

        def pump() -> None:
            try:
                if process.stdout is not None:
                    for line in process.stdout:
                        output.put(line.rstrip())
            finally:
                output.put(_STREAM_END)

        reader = threading.Thread(target=pump, name=f"cellvision-output-{task_id}", daemon=True)
        reader.start()
        try:
            while True:
                self._ensure_not_cancelled(task_id)
                try:
                    item = output.get(timeout=0.25)
                except queue.Empty:
                    if process.poll() is not None and not reader.is_alive():
                        break
                    continue
                if item is _STREAM_END:
                    if process.poll() is not None and not reader.is_alive() and output.empty():
                        break
                    continue
                line = str(item)
                lines.append(line)
                if on_line is not None:
                    on_line(line)
            return_code = process.wait()
        except TaskCancelledError:
            self._terminate_process(process)
            reader.join(timeout=1)
            raise
        if return_code != 0:
            tail = "\n".join(lines[-12:])
            raise RuntimeError(
                f"子进程退出码 {return_code}: {' '.join(command[:3])}\n{tail}"
            )
        return lines

    def _mark_project_manifest(
        self,
        task: dict[str, Any],
        status: str,
        *,
        error: str = "",
    ) -> None:
        value = task.get("project_manifest")
        if not value:
            return
        path = _resolve(value)
        if not path.exists():
            return
        try:
            manifest = _read_json(path)
            manifest["status"] = status
            manifest["last_updated_at"] = _now()
            if error:
                manifest["error"] = error
            _atomic_json_write(path, manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    def execute_project_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Prepare and run all boards in one project task."""

        task_id = str(task["task_id"])
        manifest_path = _resolve(task["project_manifest"])
        if not manifest_path.exists():
            raise FileNotFoundError(f"Project manifest not found: {manifest_path}")
        manifest = _read_json(manifest_path)
        if not isinstance(manifest.get("plates"), list) or not manifest.get("plates"):
            self._update_progress(
                task_id,
                0,
                int(task.get("group_count", 0) or 0),
                stage="prepare",
                message=f"{self.runtime.label} 正在解析 sessions.idx 并准备项目",
            )
            manifest = self._prepare_project(task, manifest_path)

        plates = [plate for plate in manifest.get("plates", []) if isinstance(plate, dict)]
        if not plates:
            raise RuntimeError("项目没有可执行的板子")
        total = len(plates)
        self._mark_project_manifest(task, "running")
        self._sync_catalog_task(task, source="task_running", force_manifest=True)
        self._set_board_progress(task_id, plates)
        self._update_progress(
            task_id,
            0,
            total,
            stage="running",
            message=f"{self.runtime.label} 已接管，准备计算 {total} 块板",
        )

        completed_count = 0
        for index, plate in enumerate(plates, start=1):
            self._ensure_not_cancelled(task_id)
            board_id = _text(plate.get("board_id") or plate.get("slug"))
            if str(plate.get("status", "")).casefold() == "completed":
                completed_count += 1
                self._update_progress(
                    task_id,
                    completed_count,
                    total,
                    stage="plate",
                    message=f"{self.runtime.label} 已跳过已完成板 {board_id} ({index}/{total})",
                )
                continue
            self._update_progress(
                task_id,
                completed_count,
                total,
                stage="inference",
                message=f"{self.runtime.label} 正在计算 {board_id} ({index}/{total})",
            )
            self._set_board_progress(
                task_id,
                plates,
                active_slug=str(plate.get("slug", "")),
                active_stage="starting",
                active_message=f"准备计算 {board_id}",
            )
            self._run_plate(task, manifest_path, plate, plates)
            refreshed = _read_json(manifest_path)
            refreshed_plate = next(
                (
                    item
                    for item in refreshed.get("plates", [])
                    if isinstance(item, dict)
                    and str(item.get("slug")) == str(plate.get("slug"))
                ),
                None,
            )
            if not refreshed_plate or str(refreshed_plate.get("status")) != "completed":
                self._set_board_progress(task_id, refreshed.get("plates", plates))
                error = (refreshed_plate or {}).get("error", "board did not complete")
                raise RuntimeError(f"板 {board_id} 计算失败：{error}")
            plates = [item for item in refreshed.get("plates", []) if isinstance(item, dict)]
            self._set_board_progress(task_id, plates)
            completed_count += 1
            self._update_progress(
                task_id,
                completed_count,
                total,
                stage="plate",
                message=f"{self.runtime.label} 已完成 {board_id} ({completed_count}/{total})",
            )

        final_manifest = _read_json(manifest_path)
        final_manifest["status"] = "completed"
        final_manifest["finished_at"] = _now()
        _atomic_json_write(manifest_path, final_manifest)
        self._sync_catalog_task(
            {**task, "project_manifest": str(manifest_path)},
            source="project_completed",
            force_manifest=True,
        )
        return {
            "project_manifest": str(manifest_path),
            "device": self.runtime.as_dict(),
            "plate_count": total,
            "completed_count": completed_count,
        }

    def _run_plate(
        self,
        task: dict[str, Any],
        manifest_path: Path,
        plate: dict[str, Any],
        plates: list[dict[str, Any]],
    ) -> None:
        task_id = str(task["task_id"])
        slug = _text(plate.get("slug"))
        if not slug:
            raise ValueError("Plate slug is missing")
        source_artifacts = _resolve(
            task.get("source_artifacts")
            or os.getenv("CELLVISION_SOURCE_ARTIFACTS", str(PROJECT_ROOT / "artifacts"))
        )
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_ql2603_project.py"),
            "--manifest",
            str(manifest_path),
            "--source-artifacts",
            str(source_artifacts),
            "--only",
            slug,
        ]

        def on_line(line: str) -> None:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return
            if not isinstance(event, dict):
                return
            stage = _text(event.get("stage"))
            if stage:
                board_id = _text(plate.get("board_id") or slug)
                self._set_board_progress(
                    task_id,
                    plates,
                    active_slug=slug,
                    active_stage=stage,
                    active_message=f"{board_id}：{stage}",
                )
                self._update_progress(
                    task_id,
                    float(self.store.get(task_id).get("progress_current", 0) or 0)
                    if self.store.get(task_id)
                    else 0,
                    stage="inference",
                    message=f"{self.runtime.label} {slug}：{stage}",
                )

        try:
            self._run_command(
                command,
                task_id=task_id,
                env=self._child_env(task),
                on_line=on_line,
            )
        except TaskCancelledError:
            raise
        except Exception as exc:
            self._set_board_progress(
                task_id,
                plates,
                active_slug=slug,
                active_stage="error",
                active_message=f"{_text(plate.get('board_id') or slug)}：{exc}",
            )
            raise

    def _prepare_project(
        self,
        task: dict[str, Any],
        manifest_path: Path,
    ) -> dict[str, Any]:
        task_id = str(task["task_id"])
        plan_path = _resolve(task["plan_path"])
        plan = _read_json(plan_path)
        root = _resolve(plan["root"])
        index = _resolve(plan["index"])
        selected_labels = [str(value) for value in plan.get("selected_timepoint_labels", [])]
        endpoint_label = str(plan.get("endpoint_day_label") or "Day14")
        endpoint_day = int(plan.get("endpoint_day_number", _day_number(endpoint_label) or 14))
        sessions = parse_sessions_index(index, root, timepoint_origin=0)
        if sessions.empty:
            raise RuntimeError("sessions.idx 没有可用采集记录")
        endpoint_rows = sessions[
            sessions["day_label"].astype(str).str.casefold() == endpoint_label.casefold()
        ]
        if endpoint_rows.empty:
            raise RuntimeError(f"没有找到末点 {endpoint_label} 的采集记录")

        project_dir = manifest_path.parent
        sessions_csv = project_dir / "sessions.csv"
        sessions.to_csv(sessions_csv, index=False, encoding="utf-8-sig")
        endpoint_dir = project_dir / "endpoint_screening"
        endpoint_dir.mkdir(parents=True, exist_ok=True)
        self._update_progress(
            task_id,
            0,
            int(task.get("group_count", 0) or 0),
            stage="endpoint_screening",
            message=f"{self.runtime.label} 正在计算 {endpoint_label} 快速生长筛选",
        )
        self._run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "analyze_day14_sheet_growth.py"),
                "--root",
                str(root),
                "--index",
                str(index),
                "--output",
                str(endpoint_dir),
                "--endpoint-day",
                str(endpoint_day),
            ],
            task_id=task_id,
            env=self._child_env(task),
        )
        endpoint_candidates = list(endpoint_dir.glob("*_well_screening.csv"))
        if not endpoint_candidates:
            raise FileNotFoundError(f"末点筛选没有生成 well_screening.csv: {endpoint_dir}")
        endpoint_csv = max(endpoint_candidates, key=lambda path: path.stat().st_mtime)

        project = _read_json(manifest_path)
        project_id = _text(project.get("project_id")) or _slug(project_dir.name) or task_id
        config_dir = project_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        plates: list[dict[str, Any]] = []
        used_slugs: set[str] = set()
        for group_id, group in sessions.groupby("group_id", sort=True):
            group = group.sort_values("session_ordinal")
            by_day: dict[str, Any] = {}
            for row in group.itertuples(index=False):
                by_day.setdefault(_text(row.day_label).casefold(), row)
            early_rows = [by_day.get(f"day{day}") for day in (0, 1, 2)]
            if any(row is None for row in early_rows):
                raise RuntimeError(f"{group_id} 缺少 Day0/Day1/Day2，无法运行早期推理")
            endpoint_row = by_day.get(endpoint_label.casefold())
            if endpoint_row is None:
                raise RuntimeError(f"{group_id} 缺少 {endpoint_label}")
            board_id = _text(group.iloc[0].get("board_id")) or str(group_id).split()[-1]
            board_slug = _slug(f"{project_id}-{board_id}") or _slug(str(group_id)) or task_id
            base_slug = board_slug
            suffix = 2
            while board_slug in used_slugs:
                board_slug = f"{base_slug}-{suffix}"
                suffix += 1
            used_slugs.add(board_slug)
            artifact_root = project_dir / "plates" / board_slug
            gated_dir = artifact_root / "gated"
            config_path = config_dir / f"{board_slug}.yaml"

            later_rows = []
            for row in group.itertuples(index=False):
                day = _day_number(_text(row.day_label))
                if day is not None and 7 <= day < endpoint_day:
                    later_rows.append((day, row))
            later_rows.sort(key=lambda item: item[0])
            late_row = next((row for day, row in later_rows if day == 7), None)
            late_row = late_row or (later_rows[-1][1] if later_rows else endpoint_row)
            endpoint_path = _text(endpoint_row.session_path)
            late_path = _text(late_row.session_path)
            endpoint_display = endpoint_label
            late_display = _text(late_row.day_label) or endpoint_label
            config = {
                "base_config": "configs/default.yaml",
                "paths": {
                    "data_root": str(root),
                    "artifact_root": str(artifact_root),
                },
                "runtime": {
                    "device": self.runtime.selected_device,
                    # Preserve the per-board paths written above when the
                    # production parent exports global root defaults.
                    "ignore_path_env_overrides": True,
                },
                "experiment": {
                    "experiment_id": f"{group_id} full project",
                    "plate_id": f"{project_id}_{board_id}",
                    "timepoint_directories": {
                        f"T{index}": _text(row.session_path)
                        for index, row in enumerate(early_rows)
                    },
                },
                "morphology_classifier": {
                    "excluded_wells": ["A1"],
                    "reuse_candidate_manifest": False,
                },
                "dense_detection": {
                    "include_manual_anchors": False,
                    "reuse_unchanged_stage": False,
                },
                "v2_inference": {"reuse_unchanged_stage": False},
                "review_queue": {
                    "excluded_wells": ["A1"],
                    "apply_manual_point_overrides": False,
                },
                "joint_training": {"sources": []},
                "review": {
                    "late_timepoint_directories": {"T3": late_path, "T4": endpoint_path},
                    "timepoint_display_names": {
                        "T0": "T0",
                        "T1": "T1",
                        "T2": "T2",
                        "T3": late_display,
                        "T4": endpoint_display,
                    },
                    "day14_overlay": {
                        "downsample": 4,
                        "minimum_component_coverage_pct": 1.0,
                        "minimum_radius_px": 40.0,
                        "minimum_mean_distance_px": 10.0,
                    },
                },
                "gated_report": {
                    "group_id": str(group_id),
                    "day14_csv": str(endpoint_csv),
                    "endpoint_csv": str(endpoint_csv),
                    "endpoint_day_label": endpoint_label,
                    "sessions_csv": str(sessions_csv),
                    "output_dir": str(gated_dir),
                },
            }
            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            plates.append({
                "slug": board_slug,
                "group_id": str(group_id),
                "board_id": board_id,
                "config": str(config_path),
                "artifact_root": str(artifact_root),
                "gated_output_dir": str(gated_dir),
                "images_manifest": str(artifact_root / "manifests" / "images.csv"),
                "pipeline_summary": str(gated_dir / "pipeline_summary.json"),
                "report_json": str(gated_dir / "plate_overview.json"),
                "status": "queued",
            })

        project.update({
            "project_id": project_id,
            "root": str(root),
            "generated_at": project.get("generated_at") or _now(),
            "source_sessions_csv": str(sessions_csv),
            "source_day14_csv": str(endpoint_csv),
            "source_endpoint_csv": str(endpoint_csv),
            "endpoint_day_label": endpoint_label,
            "endpoint_day_number": endpoint_day,
            "selected_timepoint_labels": selected_labels,
            "plates": plates,
            "status": "queued",
            "last_updated_at": _now(),
        })
        _atomic_json_write(manifest_path, project)
        self._sync_catalog_task(
            {**task, "project_manifest": str(manifest_path)},
            source="project_prepared",
            force_manifest=True,
        )
        self._update_progress(
            task_id,
            0,
            len(plates),
            stage="prepared",
            message=f"{self.runtime.label} 已准备 {len(plates)} 块板，等待逐板计算",
        )
        return project


def detect_worker_runtime(requested_device: str | None = None) -> dict[str, Any]:
    """Small public helper for diagnostics and deployment scripts."""

    return detect_compute_runtime(requested_device).as_dict()

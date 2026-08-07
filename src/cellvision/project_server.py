"""Project-level review hub and local import/task queue.

The existing per-plate review service remains the source of truth for object
labels.  This parent application aggregates gated plate reports, provides a
folder/session parser for new jobs, and lazily mounts each completed plate
under ``/plates/<slug>/`` so a large project does not load every board at
startup.
"""

from __future__ import annotations

import json
import asyncio
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import PROJECT_ROOT, load_config
from .review_server import _visible_v2_review_instances, create_app
from .session_index import parse_sessions_index, summarize_session_groups
from .task_queue import TaskQueueStore, task_id


def _safe(value: Any) -> Any:
    """Convert pandas/numpy values to JSON-safe primitives."""

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except (TypeError, ValueError):
            pass
    if pd.isna(value):
        return None
    return value


def _resolve(path: str | Path) -> Path:
    value = Path(os.path.expandvars(str(path))).expanduser()
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in str(value)).strip("-")


def _report_for_plate(plate: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    candidates = [
        plate.get("report_json"),
        Path(plate.get("artifact_root", "")) / "plate_overview.json"
        if plate.get("artifact_root")
        else None,
        Path(plate.get("gated_output_dir", "")) / "plate_overview.json"
        if plate.get("gated_output_dir")
        else None,
    ]
    for item in candidates:
        if not item:
            continue
        path = _resolve(item)
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8")), path
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


_REVIEWABLE_LABELS = {
    "single",
    "touching_doublet",
    "cluster_3plus",
    "debris",
    "uncertain",
}
_REVIEW_PROGRESS_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}


def _empty_review_progress(*, available: bool) -> dict[str, Any]:
    return {
        "review_data_available": available,
        "reviewable_well_count": 0,
        "reviewed_well_count": 0,
        "reviewable_object_count": 0,
        "reviewed_object_count": 0,
        "review_complete": bool(available),
    }


def _latest_prediction_path(config: dict[str, Any]) -> Path:
    prediction_root = Path(config["paths"]["artifact_root"]) / "predictions"
    base = prediction_root / "latest_integrated_predictions.csv"
    source = base
    for name in (
        "latest_temporally_completed_predictions.csv",
        "latest_v2_predictions.csv",
    ):
        candidate = prediction_root / name
        if candidate.exists() and (
            not source.exists() or candidate.stat().st_mtime_ns >= source.stat().st_mtime_ns
        ):
            source = candidate
    return source


def _review_progress_for_plate(plate: dict[str, Any]) -> dict[str, Any]:
    """Read lightweight well-review progress without starting a plate app.

    The project hub must not construct every review application just to render
    a table.  The prediction table and annotation database already contain the
    information needed to count reviewable wells, so use them directly and
    invalidate the small cache when either file changes.
    """

    config_value = plate.get("config")
    if not config_value:
        return _empty_review_progress(available=False)
    config_path = _resolve(config_value)
    if not config_path.exists():
        return _empty_review_progress(available=False)
    try:
        config = load_config(config_path)
    except (OSError, ValueError, TypeError):
        return _empty_review_progress(available=False)

    prediction_path = _latest_prediction_path(config)
    database_path = Path(config["paths"]["artifact_root"]) / "annotations" / "annotations.db"
    signature = (
        str(config_path),
        prediction_path.stat().st_mtime_ns if prediction_path.exists() else None,
        database_path.stat().st_mtime_ns if database_path.exists() else None,
    )
    cache_key = str(config_path)
    cached = _REVIEW_PROGRESS_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return dict(cached[1])
    if not prediction_path.exists() or not database_path.exists():
        result = _empty_review_progress(available=False)
        _REVIEW_PROGRESS_CACHE[cache_key] = (signature, result)
        return dict(result)

    try:
        predictions = pd.read_csv(prediction_path, low_memory=False)
        if predictions.empty or "integrated_label" not in predictions.columns:
            result = _empty_review_progress(available=True)
            _REVIEW_PROGRESS_CACHE[cache_key] = (signature, result)
            return dict(result)

        reviewable = predictions[
            predictions["integrated_label"].isin(_REVIEWABLE_LABELS)
            & (predictions["well"].astype(str).str.upper() != "A1")
        ].copy()
        if "v2_instance_id" in reviewable.columns:
            try:
                reviewable = _visible_v2_review_instances(reviewable)
            except (KeyError, TypeError, ValueError):
                # Keep the hub usable with older/incomplete prediction tables.
                pass

        round_id = (
            str(reviewable.iloc[0]["integrated_round_id"])
            if not reviewable.empty and "integrated_round_id" in reviewable.columns
            else ""
        )
        with sqlite3.connect(database_path) as connection:
            try:
                reviews = pd.read_sql_query(
                    """
                    SELECT candidate_id, reviewed_label, updated_at
                    FROM integrated_training_reviews
                    WHERE round_id = ?
                    ORDER BY updated_at, integrated_review_id
                    """,
                    connection,
                    params=(round_id,),
                )
            except (sqlite3.OperationalError, pd.errors.DatabaseError):
                reviews = pd.DataFrame()
            try:
                manual = pd.read_sql_query(
                    """
                    SELECT candidate_id, well, reviewed_label
                    FROM quick_missed_objects
                    WHERE round_id = ?
                    """,
                    connection,
                    params=(round_id,),
                )
            except (sqlite3.OperationalError, pd.errors.DatabaseError):
                manual = pd.DataFrame()

        reviewed_ids: set[str] = set()
        if not reviews.empty:
            latest = reviews.drop_duplicates("candidate_id", keep="last")
            reviewed_ids = set(
                latest.loc[latest["reviewed_label"].notna(), "candidate_id"]
                .astype(str)
            )
        reviewable["is_reviewed"] = reviewable["candidate_id"].astype(str).isin(reviewed_ids)

        if not manual.empty:
            existing_ids = set(reviewable["candidate_id"].astype(str))
            manual = manual[~manual["candidate_id"].astype(str).isin(existing_ids)].copy()
            if not manual.empty:
                manual["is_reviewed"] = manual["reviewed_label"].notna()
                manual = manual.rename(columns={"screen_well": "well"})
                reviewable = pd.concat(
                    [reviewable[["well", "is_reviewed"]], manual[["well", "is_reviewed"]]],
                    ignore_index=True,
                )
        else:
            reviewable = reviewable[["well", "is_reviewed"]]

        if reviewable.empty:
            result = _empty_review_progress(available=True)
        else:
            grouped = reviewable.groupby(reviewable["well"].astype(str).str.upper())["is_reviewed"]
            well_progress = grouped.agg(["size", "sum"])
            result = {
                "review_data_available": True,
                "reviewable_well_count": int(len(well_progress)),
                "reviewed_well_count": int((well_progress["size"] == well_progress["sum"]).sum()),
                "reviewable_object_count": int(len(reviewable)),
                "reviewed_object_count": int(reviewable["is_reviewed"].sum()),
                "review_complete": bool((well_progress["size"] == well_progress["sum"]).all()),
            }
    except (OSError, KeyError, ValueError, TypeError, pd.errors.ParserError, sqlite3.Error):
        result = _empty_review_progress(available=False)

    _REVIEW_PROGRESS_CACHE[cache_key] = (signature, result)
    return dict(result)


def _project_detection_dates(value: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return the first/last acquisition dates at day precision."""

    candidates: list[Path] = []
    source = value.get("source_sessions_csv")
    if source:
        candidates.append(_resolve(source))
    for plate in value.get("plates", []):
        if not isinstance(plate, dict) or not plate.get("images_manifest"):
            continue
        candidates.append(_resolve(plate["images_manifest"]))
    for path in candidates:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in {
                "acquisition_date", "acquisition_datetime", "excluded"
            })
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if frame.empty:
            continue
        if "excluded" in frame.columns:
            excluded = frame["excluded"].astype(str).str.strip().str.lower().isin(
                {"1", "true", "yes", "y"}
            )
            frame = frame.loc[~excluded]
        column = "acquisition_date" if "acquisition_date" in frame.columns else "acquisition_datetime"
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
        if not parsed.empty:
            return parsed.min().date().isoformat(), parsed.max().date().isoformat()
    return None, None


def _plate_summary(plate: dict[str, Any]) -> dict[str, Any]:
    report, report_path = _report_for_plate(plate)
    counts: dict[str, int] = {}
    elapsed = None
    pipeline_path = _resolve(plate.get("pipeline_summary", "")) if plate.get("pipeline_summary") else None
    if report:
        counts = {
            str(key): int(value or 0)
            for key, value in (report.get("category_counts") or {}).items()
        }
    if pipeline_path and pipeline_path.exists():
        try:
            pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
            elapsed = pipeline.get("total_elapsed_seconds")
        except (OSError, json.JSONDecodeError):
            pass
    status = str(plate.get("status", "queued"))
    if report:
        status = "completed"
    review_progress = _review_progress_for_plate(plate)
    return {
        "slug": str(plate.get("slug", "")),
        "group_id": str(plate.get("group_id", plate.get("board_id", ""))),
        "board_id": str(plate.get("board_id", "")),
        "status": status,
        "category_counts": counts,
        "well_count": int(report.get("well_count", 0)) if report else 0,
        "positive_well_count": int(report.get("day14_positive_sample_wells", 0)) if report else 0,
        "skipped_well_count": int(report.get("day14_skipped_sample_wells", 0)) if report else 0,
        "endpoint_day_label": report.get("endpoint_day_label", plate.get("endpoint_day_label", "Day14")) if report else plate.get("endpoint_day_label", "Day14"),
        "elapsed_seconds": elapsed,
        "report_path": str(report_path) if report_path else None,
        "config_path": str(_resolve(plate.get("config", ""))) if plate.get("config") else None,
        **review_progress,
    }


class FolderPayload(BaseModel):
    path: str = Field(min_length=1)


class TaskPayload(FolderPayload):
    name: str = Field(default="新建项目任务", min_length=1, max_length=120)
    selected_timepoint_labels: list[str] = Field(default_factory=list)


_DAY_RE = re.compile(r"^day\s*(?P<day>-?\d+)$", re.IGNORECASE)


def _day_number(value: Any) -> int | None:
    """Return an actual culture-day number from a parsed Day label."""

    match = _DAY_RE.match(str(value or "").strip())
    if match:
        try:
            return int(match.group("day"))
        except ValueError:
            return None
    return None


def _timepoint_options(sessions: pd.DataFrame) -> list[dict[str, Any]]:
    """Collapse session rows into actual calendar-day choices.

    ``timepoint_label`` is an acquisition-order label (T0, T1, ...), while
    ``day_label`` is the culture age.  Task selection must use the latter so a
    task remains correct when a plate has a missing or extra acquisition.
    """

    if sessions.empty:
        return []
    frame = sessions.copy()
    frame["day_number"] = frame["culture_day"].map(_day_number)
    # ``culture_day`` is an Int64 column in pandas; use it when available.
    numeric = pd.to_numeric(frame["culture_day"], errors="coerce")
    frame["day_number"] = numeric.where(numeric.notna(), frame["day_number"])
    rows: list[dict[str, Any]] = []
    for day_number, group in frame.dropna(subset=["day_number"]).groupby("day_number", sort=True):
        day = int(day_number)
        labels = sorted({str(value) for value in group["timepoint_label"].dropna()})
        day_label = f"Day{day}"
        rows.append({
            "day_label": day_label,
            "day_number": day,
            "timepoint_labels": labels,
            "session_count": int(len(group)),
            "group_count": int(group["group_id"].nunique()),
            "complete_session_count": int(group["group_complete"].fillna(False).sum()),
            "required_early": day in {0, 1, 2},
            "eligible_endpoint": day >= 7,
        })
    return rows


def _default_timepoint_selection(options: list[dict[str, Any]]) -> list[str]:
    selected = [str(item["day_label"]) for item in options if int(item["day_number"]) in {0, 1, 2}]
    later = [item for item in options if bool(item.get("eligible_endpoint"))]
    if later:
        selected.append(str(max(later, key=lambda item: int(item["day_number"]))["day_label"]))
    return list(dict.fromkeys(selected))


def _validate_timepoint_selection(
    options: list[dict[str, Any]], selected: list[str] | None,
) -> tuple[list[str], str, int]:
    by_label = {str(item["day_label"]).casefold(): item for item in options}
    default = _default_timepoint_selection(options)
    requested = [str(value).strip() for value in (selected or []) if str(value).strip()]
    chosen: list[dict[str, Any]] = []
    for value in requested or default:
        item = by_label.get(value.casefold())
        if item is None:
            # Accept T0/T1/T2 aliases returned by older clients.
            alias = next((candidate for candidate in options if value.casefold() in {label.casefold() for label in candidate["timepoint_labels"]}), None)
            item = alias
        if item is not None and item not in chosen:
            chosen.append(item)
    chosen.sort(key=lambda item: int(item["day_number"]))
    chosen_labels = [str(item["day_label"]) for item in chosen]
    missing = [f"Day{day}" for day in (0, 1, 2) if f"Day{day}".casefold() not in {label.casefold() for label in chosen_labels}]
    later = [item for item in chosen if int(item["day_number"]) >= 7]
    if missing:
        raise HTTPException(status_code=422, detail=f"必须连续选择 Day0、Day1、Day2；缺少：{', '.join(missing)}")
    if not later:
        raise HTTPException(status_code=422, detail="至少选择一个 Day7 或更晚的时间点作为末点")
    endpoint = max(later, key=lambda item: int(item["day_number"]))
    return chosen_labels, str(endpoint["day_label"]), int(endpoint["day_number"])


def _validate_group_timepoint_coverage(
    session_records: list[dict[str, Any]], selected: list[str]
) -> None:
    """Do not enqueue a multi-board task with a silently incomplete board."""

    missing: list[str] = []
    by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for row in session_records:
        group = str(row.get("group_id", ""))
        label = str(row.get("day_label", "")).casefold()
        by_group.setdefault(group, {})[label] = row
    for group, available in sorted(by_group.items()):
        absent = [label for label in selected if str(label).casefold() not in available]
        if absent:
            missing.append(f"{group}: {', '.join(absent)}")
        incomplete = [
            label for label in selected
            if label.casefold() in available
            and not bool(available[label.casefold()].get("group_complete"))
        ]
        if incomplete:
            missing.append(f"{group}: incomplete 96-well session(s) at {', '.join(incomplete)}")
    if missing:
        preview = "; ".join(missing[:4])
        suffix = " …" if len(missing) > 4 else ""
        raise HTTPException(status_code=422, detail=f"所选时间点在部分板子中缺失：{preview}{suffix}")


def _parse_folder(path_value: str) -> dict[str, Any]:
    root = _resolve(path_value)
    index = root if root.is_file() and root.name.casefold() == "sessions.idx" else root / "sessions.idx"
    if not index.exists():
        raise HTTPException(status_code=400, detail=f"未找到 sessions.idx: {index}")
    try:
        sessions = parse_sessions_index(index, root if root.is_dir() else index.parent, timepoint_origin=0)
        groups = summarize_session_groups(sessions)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    options = _timepoint_options(sessions)
    default_selected = _default_timepoint_selection(options)
    try:
        _, default_endpoint, default_endpoint_day = _validate_timepoint_selection(options, default_selected)
        selection_valid = True
        selection_error = ""
    except HTTPException as exc:
        default_endpoint = ""
        default_endpoint_day = None
        selection_valid = False
        selection_error = str(exc.detail)
    day_values = sorted(
        {str(value) for value in sessions["day_label"].dropna()},
        key=lambda label: (
            _day_number(label) is None,
            _day_number(label) if _day_number(label) is not None else 10**9,
            label.casefold(),
        ),
    )
    timepoint_values = sorted(
        {str(value) for value in sessions["timepoint_label"].dropna()},
        key=lambda label: (
            not label.casefold().startswith("t"),
            int(label[1:])
            if label.casefold().startswith("t") and label[1:].isdigit()
            else 10**9,
            label.casefold(),
        ),
    )
    return {
        "root": str(root),
        "index": str(index),
        "session_count": int(len(sessions)),
        "group_count": int(len(groups)),
        "groups": [_safe(row) for row in groups.to_dict(orient="records")],
        "timepoint_labels": timepoint_values,
        "day_labels": day_values,
        "timepoint_options": options,
        "default_selected_timepoint_labels": default_selected,
        "default_endpoint_day_label": default_endpoint,
        "default_endpoint_day_number": default_endpoint_day,
        "selection_valid": selection_valid,
        "selection_error": selection_error,
        "session_records": [
            _safe(row)
            for row in sessions[
                ["group_id", "board_id", "timepoint_label", "day_label", "culture_day", "session_path", "group_complete"]
            ].to_dict(orient="records")
        ],
        "complete_groups": int(groups["complete_96_well_sessions"].eq(groups["session_count"]).sum()) if not groups.empty else 0,
    }


def _queue_path(manifest_path: Path) -> Path:
    configured = manifest_path.parent / "task_queue.json"
    configured.parent.mkdir(parents=True, exist_ok=True)
    return configured


def _project_manifest_paths(current: Path) -> list[Path]:
    """Find sibling project manifests for the hub homepage."""

    root = current.parent.parent
    paths = [path for path in root.glob("*/project.json") if path.is_file()]
    if current.is_file() and current not in paths:
        paths.append(current)
    return sorted(set(paths), key=lambda path: path.as_posix().casefold())


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _project_card(path: Path) -> dict[str, Any]:
    value = _read_manifest(path) or {}
    plates = value.get("plates") if isinstance(value.get("plates"), list) else []
    aggregate: dict[str, int] = {}
    completed = 0
    reviewed = 0
    for plate in plates:
        if not isinstance(plate, dict):
            continue
        report, _ = _report_for_plate(plate)
        if report:
            completed += 1
            for key, count in (report.get("category_counts") or {}).items():
                aggregate[str(key)] = aggregate.get(str(key), 0) + int(count or 0)
        if _review_progress_for_plate(plate).get("review_complete"):
            reviewed += 1
    project_id = str(value.get("project_id") or path.parent.name)
    detection_start, detection_end = _project_detection_dates(value)
    return {
        "project_id": project_id,
        "project_name": str(value.get("project_name") or project_id),
        "root": value.get("root"),
        "manifest_path": str(path),
        "plate_count": len(plates),
        "completed_plate_count": completed,
        "recognized_plate_count": completed,
        "reviewed_plate_count": reviewed,
        "category_counts": aggregate,
        "single_cell_origin_well_count": int(aggregate.get("single_cell_origin", 0)),
        "detection_start_date": detection_start,
        "detection_end_date": detection_end,
        "created_by": str(
            value.get("created_by")
            or value.get("creator")
            or value.get("owner")
            or ""
        ),
        "generated_at": value.get("generated_at"),
        "detail_url": f"/projects/{_slug(project_id)}/",
    }


class _LazyPlateApp:
    """Create a plate review app only when that plate is first opened.

    A project may contain many boards.  Eagerly constructing one full review
    application per board duplicates image/data caches and makes startup scale
    with the whole project.  This ASGI wrapper keeps the same mounted URL while
    deferring the expensive initialization to the first request.
    """

    def __init__(self, config_path: Path, project_back_url: str = ""):
        self.config_path = config_path
        self.project_back_url = project_back_url
        self._app = None
        self._error: str | None = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._app is not None

    async def _ensure_app(self):
        if self._app is not None:
            return self._app
        async with self._lock:
            if self._app is None and self._error is None:
                try:
                    # Review-app creation performs synchronous image/database
                    # discovery.  Keep that work off the event loop.
                    self._app = await asyncio.to_thread(
                        create_app,
                        load_config(self.config_path),
                        project_back_url=self.project_back_url,
                    )
                except Exception as exc:  # pragma: no cover - startup-only path
                    self._error = f"{type(exc).__name__}: {exc}"
        return self._app

    async def __call__(self, scope, receive, send):
        app = await self._ensure_app()
        if app is None:
            if scope.get("type") != "http":
                raise RuntimeError(self._error or "plate review app unavailable")
            body = json.dumps(
                {"detail": "plate review app unavailable", "error": self._error},
                ensure_ascii=False,
            ).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 503,
                "headers": [(b"content-type", b"application/json; charset=utf-8")],
            })
            await send({"type": "http.response.body", "body": body})
            return
        await app(scope, receive, send)


def create_project_app(manifest_path: str | Path) -> FastAPI:
    manifest_file = _resolve(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(manifest_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    project_name = str(manifest.get("project_name", manifest.get("project_id", "Cell Vision Project")))
    project_id = str(manifest.get("project_id") or manifest_file.parent.name)
    project_back_url = f"/projects/{_slug(project_id)}/"
    ui_root = PROJECT_ROOT / "review-ui"
    app = FastAPI(title=f"Cell Vision Project · {project_name}")
    app.mount("/project-assets", StaticFiles(directory=ui_root), name="project-assets")
    queue_file = _queue_path(manifest_file)
    queue_store = TaskQueueStore(queue_file)
    # Register completed boards lazily.  A project can still be opened while a
    # board is queued; that board simply appears as a disabled row until its
    # manifest and inference artifacts are ready.
    mounted: list[str] = []
    lazy_apps: dict[str, _LazyPlateApp] = {}
    for plate in manifest.get("plates", []):
        slug = str(plate.get("slug") or _slug(plate.get("board_id", "")))
        config_value = plate.get("config")
        if not slug or not config_value:
            continue
        config_path = _resolve(config_value)
        images_path = _resolve(plate.get("images_manifest", "")) if plate.get("images_manifest") else None
        if not config_path.exists() or (images_path is not None and not images_path.exists()):
            continue
        try:
            lazy_plate = _LazyPlateApp(config_path, project_back_url=project_back_url)
            app.mount(f"/plates/{slug}", lazy_plate, name=f"plate-{slug}")
            lazy_apps[slug] = lazy_plate
            mounted.append(slug)
        except (OSError, ValueError, KeyError) as exc:
            plate["mount_error"] = str(exc)
    manifest["mounted_plates"] = mounted

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        # The landing page is deliberately project-level.  Opening a board is
        # an explicit second click, so a new task can never unexpectedly
        # replace the currently selected project.
        return (ui_root / "project-list.html").read_text(encoding="utf-8")

    @app.get("/projects/{project_id}/", response_class=HTMLResponse)
    def project_detail_page(project_id: str) -> str:
        target = _slug(project_id)
        # Re-scan here instead of using the startup snapshot: creating a task
        # writes a new project manifest while the hub is already running.
        available = {
            _slug(str((_read_manifest(path) or {}).get("project_id", path.parent.name))): path
            for path in _project_manifest_paths(manifest_file)
        }
        if target not in available:
            raise HTTPException(status_code=404, detail="project not found")
        page = (ui_root / "project-dashboard.html").read_text(encoding="utf-8")
        return page.replace("</head>", f'<meta name="project-id" content="{target}"></head>', 1)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "project": project_name,
            "mounted_plates": mounted,
            "loaded_plates": [slug for slug, item in lazy_apps.items() if item.loaded],
            "project_count": len(_project_manifest_paths(manifest_file)),
        }

    @app.get("/api/ready")
    def ready() -> dict[str, Any]:
        """Readiness probe for a reverse proxy or service manager."""

        missing = []
        if not manifest_file.exists():
            missing.append(str(manifest_file))
        if not ui_root.exists():
            missing.append(str(ui_root))
        if missing:
            raise HTTPException(status_code=503, detail={"missing": missing})
        return {"status": "ready", "project": project_name, "plate_count": len(mounted)}

    @app.get("/api/projects")
    def projects() -> list[dict[str, Any]]:
        return [_project_card(path) for path in _project_manifest_paths(manifest_file)]

    @app.get("/api/project")
    def project(project_id: str | None = None) -> dict[str, Any]:
        selected_manifest = manifest
        selected_path = manifest_file
        if project_id:
            wanted = _slug(project_id)
            for path in _project_manifest_paths(manifest_file):
                candidate = _read_manifest(path) or {}
                candidate_id = _slug(str(candidate.get("project_id") or path.parent.name))
                if candidate_id == wanted:
                    selected_manifest = candidate
                    selected_path = path
                    break
            else:
                raise HTTPException(status_code=404, detail="project not found")
        plates = [_plate_summary(item) for item in selected_manifest.get("plates", [])]
        aggregate: dict[str, int] = {}
        for item in plates:
            for key, value in item["category_counts"].items():
                aggregate[key] = aggregate.get(key, 0) + int(value)
        detection_start, detection_end = _project_detection_dates(selected_manifest)
        return {
            "project_id": selected_manifest.get("project_id", selected_path.parent.name),
            "project_name": selected_manifest.get("project_name", selected_path.parent.name),
            "root": selected_manifest.get("root"),
            "plate_count": len(plates),
            "mounted_plates": mounted if selected_path == manifest_file else [],
            "recognized_plate_count": int(sum(item.get("status") == "completed" for item in plates)),
            "reviewed_plate_count": int(sum(bool(item.get("review_complete")) for item in plates)),
            "category_counts": aggregate,
            "plates": plates,
            "single_cell_origin_well_count": int(aggregate.get("single_cell_origin", 0)),
            "detection_start_date": detection_start,
            "detection_end_date": detection_end,
            "created_by": str(
                selected_manifest.get("created_by")
                or selected_manifest.get("creator")
                or selected_manifest.get("owner")
                or ""
            ),
            "generated_at": selected_manifest.get("generated_at"),
        }

    @app.get("/api/project/tasks")
    def tasks() -> list[dict[str, Any]]:
        return queue_store.list()

    @app.post("/api/project/browse-folder")
    def browse_folder() -> dict[str, str]:
        """Open a Windows folder picker from an explicit user action."""

        if os.name != "nt":
            return {"path": ""}
        focus_helper = r"""
using System;
using System.Text;
using System.Threading;
using System.Runtime.InteropServices;

public static class CellVisionWindowFocus
{
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    [DllImport("user32.dll")]
    private static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetClassName(IntPtr hWnd, StringBuilder className, int maxCount);

    [DllImport("user32.dll")]
    private static extern bool BringWindowToTop(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool SetWindowPos(
        IntPtr hWnd,
        IntPtr insertAfter,
        int x,
        int y,
        int width,
        int height,
        uint flags);

    private static readonly IntPtr HwndTopmost = new IntPtr(-1);
    private const uint SwpNoSize = 0x0001;
    private const uint SwpNoMove = 0x0002;
    private const uint SwpShowWindow = 0x0040;
    private static System.Threading.Timer promotionTimer;

    public static void Start(uint targetProcessId)
    {
        Stop();
        promotionTimer = new System.Threading.Timer(
            unused => Promote(targetProcessId),
            null,
            0,
            50);
    }

    public static void Stop()
    {
        var timer = promotionTimer;
        promotionTimer = null;
        if (timer != null)
        {
            timer.Dispose();
        }
    }

    public static void Promote(uint targetProcessId)
    {
        IntPtr dialogHandle = IntPtr.Zero;
        EnumWindows((hWnd, unused) =>
        {
            uint processId;
            GetWindowThreadProcessId(hWnd, out processId);
            if (processId != targetProcessId || !IsWindowVisible(hWnd))
            {
                return true;
            }

            var className = new StringBuilder(128);
            GetClassName(hWnd, className, className.Capacity);
            if (string.Equals(className.ToString(), "#32770", StringComparison.Ordinal))
            {
                dialogHandle = hWnd;
                return false;
            }

            return true;
        }, IntPtr.Zero);

        if (dialogHandle == IntPtr.Zero)
        {
            return;
        }

        SetWindowPos(
            dialogHandle,
            HwndTopmost,
            0,
            0,
            0,
            0,
            SwpNoMove | SwpNoSize | SwpShowWindow);
        BringWindowToTop(dialogHandle);
        SetForegroundWindow(dialogHandle);
    }
}
"""
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "Add-Type -AssemblyName System.Drawing;"
            "Add-Type -TypeDefinition @'\n"
            + focus_helper
            + "'@;"
            "[System.Windows.Forms.Application]::EnableVisualStyles();"
            "$owner=New-Object System.Windows.Forms.Form;"
            "$owner.FormBorderStyle='None';"
            "$owner.StartPosition='Manual';"
            "$owner.Location=New-Object System.Drawing.Point -ArgumentList -10,-10;"
            "$owner.Size=New-Object System.Drawing.Size -ArgumentList 1,1;"
            "$owner.Opacity=0;"
            "$owner.ShowInTaskbar=$false;"
            "$owner.TopMost=$true;"
            "$owner.Show();"
            "$owner.Activate();"
            "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
            "$d.Description='选择包含 sessions.idx 的数据文件夹';"
            "$d.ShowNewFolderButton=$false;"
            "[CellVisionWindowFocus]::Start([uint32]$PID);"
            "try{if($d.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK){$d.SelectedPath}}"
            "finally{[CellVisionWindowFocus]::Stop();$d.Dispose();$owner.Close();$owner.Dispose()}"
        )
        try:
            # The review service is often started without an attached console
            # (for example from a detached local launcher).  Explicitly ask
            # PowerShell for a normal interactive window without creating a
            # second console that could send a close event to the server.
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = getattr(subprocess, "SW_SHOWNORMAL", 1)
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-STA",
                    "-WindowStyle",
                    "Normal",
                    "-Command",
                    script,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                startupinfo=startupinfo,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(
                f"Folder picker failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return {"path": "", "error": "无法启动 Windows 文件夹选择器，请直接输入文件夹路径。"}
        selected_path = completed.stdout.strip()
        if selected_path:
            return {"path": selected_path}
        error = completed.stderr.strip()
        return {
            "path": "",
            "error": error or "未选择文件夹；也可以直接输入文件夹路径。",
        }

    @app.post("/api/project/analyze-folder")
    def analyze_folder(payload: FolderPayload) -> dict[str, Any]:
        return _parse_folder(payload.path)

    @app.post("/api/project/tasks")
    def add_task(payload: TaskPayload) -> dict[str, Any]:
        analysis = _parse_folder(payload.path)
        selected, endpoint_label, endpoint_day = _validate_timepoint_selection(
            analysis["timepoint_options"], payload.selected_timepoint_labels
        )
        _validate_group_timepoint_coverage(analysis.get("session_records", []), selected)
        selected_set = {label.casefold() for label in selected}
        selected_sessions = [
            row for row in analysis.get("session_records", [])
            if str(row.get("day_label", "")).casefold() in selected_set
        ]
        endpoint_timepoint_labels = sorted({
            str(row.get("timepoint_label"))
            for row in selected_sessions
            if str(row.get("day_label", "")).casefold() == endpoint_label.casefold()
            and row.get("timepoint_label")
        })
        now = datetime.now(timezone.utc).isoformat()
        new_task_id = task_id()
        plan_dir = queue_file.parent / "task_plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / f"{new_task_id}.json"
        plan = {
            "task_id": new_task_id,
            "name": payload.name.strip() or "新建项目任务",
            "root": analysis["root"],
            "index": analysis["index"],
            "selected_timepoint_labels": selected,
            "endpoint_day_label": endpoint_label,
            "endpoint_day_number": endpoint_day,
            "endpoint_timepoint_labels": endpoint_timepoint_labels,
            "early_timepoint_labels": ["Day0", "Day1", "Day2"],
            "endpoint_gate": {
                "enabled": True,
                "rule": "endpoint_obvious_sheet_growth=false 的孔跳过 T0-T2 深度计算",
                "endpoint_day_label": endpoint_label,
            },
            "sessions": selected_sessions,
            "created_at": now,
        }
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        project_id = _slug(payload.name.strip()) or new_task_id
        project_dir = manifest_file.parent.parent / project_id
        project_manifest_path = project_dir / "project.json"
        if project_manifest_path.exists():
            project_id = f"{project_id}-{new_task_id.rsplit('-', 1)[-1]}"
            project_dir = manifest_file.parent.parent / project_id
            project_manifest_path = project_dir / "project.json"
        project_dir.mkdir(parents=True, exist_ok=True)
        project_manifest_path.write_text(
            json.dumps({
                "project_id": project_id,
                "project_name": payload.name.strip() or "新建项目任务",
                "root": analysis["root"],
                "generated_at": now,
                "task_id": new_task_id,
                "status": "queued",
                "selected_timepoint_labels": selected,
                "endpoint_day_label": endpoint_label,
                "endpoint_day_number": endpoint_day,
                "plates": [],
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        task = {
            "task_id": new_task_id,
            "name": payload.name.strip() or "新建项目任务",
            "path": analysis["root"],
            "index": analysis["index"],
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "group_count": analysis["group_count"],
            "session_count": analysis["session_count"],
            "timepoint_labels": analysis["timepoint_labels"],
            "day_labels": analysis["day_labels"],
            "timepoint_options": analysis["timepoint_options"],
            "selected_timepoint_labels": selected,
            "endpoint_day_label": endpoint_label,
            "endpoint_day_number": endpoint_day,
            "endpoint_timepoint_labels": endpoint_timepoint_labels,
            "plan_path": str(plan_path),
            "project_id": project_id,
            "project_manifest": str(project_manifest_path),
        }
        return queue_store.add(task)

    @app.get("/api/project/task-queue")
    def task_queue() -> dict[str, Any]:
        return {"queue_path": str(queue_file), "tasks": queue_store.list()}

    @app.get("/api/project/tasks/{task_id}")
    def task_detail(task_id: str) -> dict[str, Any]:
        task = queue_store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task

    @app.post("/api/project/tasks/{task_id}/cancel")
    def cancel_task(task_id: str) -> dict[str, Any]:
        task = queue_store.cancel(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task

    return app

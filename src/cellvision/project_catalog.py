"""Durable project-level catalog and derived review summaries.

The catalog is deliberately small.  Images, model tables and detailed plate
annotation databases remain external artifacts; this database stores the
queryable project/plate/task state and the latest well conclusions.  Every
write is idempotent and the whole catalog can be rebuilt from project manifests
and their report files.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .review_summary import read_summary, summary_path, summary_signature


SCHEMA_VERSION = 2
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 100


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS catalog_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    source_root TEXT,
    manifest_path TEXT NOT NULL UNIQUE,
    created_by TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT,
    updated_at TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS plates (
    plate_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    plate_slug TEXT NOT NULL,
    board_id TEXT,
    group_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    current_stage TEXT,
    progress_percent REAL NOT NULL DEFAULT 0,
    elapsed_seconds REAL NOT NULL DEFAULT 0,
    artifact_root TEXT,
    config_path TEXT,
    images_manifest_path TEXT,
    report_path TEXT,
    review_database_path TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, plate_slug)
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id) ON DELETE SET NULL,
    task_name TEXT NOT NULL,
    created_by TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    current_stage TEXT,
    progress_current REAL NOT NULL DEFAULT 0,
    progress_total REAL NOT NULL DEFAULT 0,
    progress_percent REAL NOT NULL DEFAULT 0,
    progress_message TEXT,
    worker_id TEXT,
    created_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    elapsed_seconds REAL NOT NULL DEFAULT 0,
    error_message TEXT,
    options_json TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_plates (
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    plate_id TEXT NOT NULL REFERENCES plates(plate_id) ON DELETE CASCADE,
    sequence_no INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(task_id, plate_id)
);

CREATE TABLE IF NOT EXISTS plate_runs (
    run_id TEXT PRIMARY KEY,
    plate_id TEXT NOT NULL REFERENCES plates(plate_id) ON DELETE CASCADE,
    task_id TEXT REFERENCES tasks(task_id) ON DELETE SET NULL,
    worker_id TEXT,
    device TEXT,
    model_version TEXT,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    elapsed_seconds REAL,
    output_root TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS well_current (
    plate_id TEXT NOT NULL REFERENCES plates(plate_id) ON DELETE CASCADE,
    well TEXT NOT NULL,
    category_code TEXT NOT NULL,
    reason_code TEXT,
    source TEXT NOT NULL,
    reviewer TEXT,
    reviewed_at TEXT,
    model_category_code TEXT,
    model_reason_code TEXT,
    model_updated_at TEXT,
    revision INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(plate_id, well)
);

CREATE TABLE IF NOT EXISTS well_decision_history (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_id TEXT NOT NULL REFERENCES plates(plate_id) ON DELETE CASCADE,
    well TEXT NOT NULL,
    category_code TEXT NOT NULL,
    reason_code TEXT,
    source TEXT NOT NULL,
    reviewer TEXT,
    operation TEXT NOT NULL,
    action_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plate_summaries (
    plate_id TEXT PRIMARY KEY REFERENCES plates(plate_id) ON DELETE CASCADE,
    total_wells INTEGER NOT NULL DEFAULT 0,
    reviewed_wells INTEGER NOT NULL DEFAULT 0,
    reviewable_wells INTEGER NOT NULL DEFAULT 0,
    reviewed_objects INTEGER NOT NULL DEFAULT 0,
    reviewable_objects INTEGER NOT NULL DEFAULT 0,
    review_complete INTEGER NOT NULL DEFAULT 0,
    category_counts_json TEXT NOT NULL DEFAULT '{}',
    source_report_revision TEXT,
    calculated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS project_summaries (
    project_id TEXT PRIMARY KEY REFERENCES projects(project_id) ON DELETE CASCADE,
    plate_count INTEGER NOT NULL DEFAULT 0,
    recognized_plate_count INTEGER NOT NULL DEFAULT 0,
    reviewed_plate_count INTEGER NOT NULL DEFAULT 0,
    total_wells INTEGER NOT NULL DEFAULT 0,
    reviewed_wells INTEGER NOT NULL DEFAULT 0,
    category_counts_json TEXT NOT NULL DEFAULT '{}',
    calculated_at TEXT NOT NULL,
    source_revision INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS catalog_sources (
    source_path TEXT PRIMARY KEY,
    signature_json TEXT NOT NULL,
    synced_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS catalog_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    aggregate_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(project_name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_projects_updated ON projects(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_plates_project ON plates(project_id, plate_slug);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_wells_plate ON well_current(plate_id, well);
CREATE INDEX IF NOT EXISTS idx_events_aggregate ON catalog_events(aggregate_type, aggregate_id, event_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def catalog_path_for_manifest(manifest_path: str | Path) -> Path:
    """Return the shared catalog path for a project collection."""

    path = Path(os.path.expandvars(str(manifest_path))).expanduser().resolve()
    return path.parent.parent / "project_catalog.sqlite"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _resolve_reference(value: Any, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _slug(value: Any) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in str(value)).strip("-")


def _mtime_signature(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            result.append({"path": str(path), "mtime_ns": None, "size": None})
            continue
        result.append({"path": str(path), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size})
    return result


def _report_for_plate(plate: dict[str, Any], base: Path) -> tuple[dict[str, Any] | None, Path | None]:
    candidates = [
        _resolve_reference(plate.get("report_json"), base),
        _resolve_reference(plate.get("gated_output_dir"), base),
        _resolve_reference(plate.get("artifact_root"), base),
    ]
    paths: list[Path] = []
    if candidates[0] is not None:
        paths.append(candidates[0])
    for root in candidates[1:]:
        if root is not None:
            paths.append(root / "plate_overview.json")
    for path in paths:
        if not path.exists():
            continue
        value = _read_json(path)
        if value is not None:
            return value, path
    return None, None


def _plate_paths(plate: dict[str, Any], base: Path) -> dict[str, Path | None]:
    artifact_root = _resolve_reference(plate.get("artifact_root"), base)
    gated_root = _resolve_reference(plate.get("gated_output_dir"), base)
    report_path = _resolve_reference(plate.get("report_json"), base)
    if report_path is None and gated_root is not None:
        report_path = gated_root / "plate_overview.json"
    if report_path is None and artifact_root is not None:
        report_path = artifact_root / "gated" / "plate_overview.json"
    database = (artifact_root / "annotations" / "annotations.db") if artifact_root else None
    summary = summary_path(artifact_root) if artifact_root else None
    report_csv = (gated_root / "plate_overview.csv") if gated_root else None
    pipeline = _resolve_reference(plate.get("pipeline_summary"), base)
    return {
        "artifact_root": artifact_root,
        "gated_root": gated_root,
        "report": report_path,
        "report_csv": report_csv,
        "database": database,
        "summary": summary,
        "pipeline": pipeline,
    }


def _fresh_review_summary(paths: dict[str, Path | None]) -> dict[str, Any] | None:
    artifact_root = paths.get("artifact_root")
    if artifact_root is None:
        return None
    summary_file = paths.get("summary")
    if summary_file is None:
        return None
    signature = summary_signature(
        artifact_root,
        database_path=paths.get("database"),
        report_path=paths.get("report_csv"),
    )
    return read_summary(summary_file, signature)


def _category_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(item or 0) for key, item in value.items()}


def _project_id(value: dict[str, Any], path: Path) -> str:
    return str(value.get("project_id") or path.parent.name)


def _project_card_from_row(data: dict[str, Any]) -> dict[str, Any]:
    """Build a project card from a joined projects+summaries row."""

    try:
        categories = json.loads(data.get("category_counts_json") or "{}")
    except json.JSONDecodeError:
        categories = {}
    return {
        "project_id": data["project_id"],
        "project_name": data["project_name"],
        "root": data.get("source_root") or "",
        "manifest_path": data["manifest_path"],
        "plate_count": int(data.get("plate_count") or 0),
        "completed_plate_count": int(data.get("recognized_plate_count") or 0),
        "recognized_plate_count": int(data.get("recognized_plate_count") or 0),
        "reviewed_plate_count": int(data.get("reviewed_plate_count") or 0),
        "category_counts": categories,
        "single_cell_origin_well_count": int(categories.get("single_cell_origin", 0) or 0),
        "detection_start_date": None,
        "detection_end_date": None,
        "created_by": data.get("created_by") or "",
        "generated_at": data.get("created_at"),
        "updated_at": data.get("calculated_at") or data.get("updated_at"),
        "detail_url": f"/projects/{_slug(data['project_id'])}/",
    }


class ProjectCatalog:
    """SQLite-backed project index with idempotent file reconciliation."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            # Keep an already-created catalog forward compatible.  The catalog
            # is a projection and can always be rebuilt, but adding columns in
            # place avoids making a running installation lose its history.
            for statement in (
                "ALTER TABLE well_current ADD COLUMN model_category_code TEXT",
                "ALTER TABLE well_current ADD COLUMN model_reason_code TEXT",
                "ALTER TABLE well_current ADD COLUMN model_updated_at TEXT",
            ):
                try:
                    connection.execute(statement)
                except sqlite3.OperationalError as error:
                    if "duplicate column name" not in str(error).lower():
                        raise
            connection.execute(
                "INSERT INTO catalog_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _manifest_signature(path: Path, value: dict[str, Any]) -> list[dict[str, Any]]:
        sources: list[Path] = [path, path.parent / "task_queue.json"]
        for plate in value.get("plates", []):
            if not isinstance(plate, dict):
                continue
            sources.extend(_plate_paths(plate, path.parent).values())
        return _mtime_signature(path for path in sources if path is not None)

    def _record_event(
        self,
        connection: sqlite3.Connection,
        aggregate_type: str,
        aggregate_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO catalog_events(aggregate_type, aggregate_id, event_type, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (aggregate_type, aggregate_id, event_type, _json(payload), _now()),
        )

    def sync_manifest(
        self,
        manifest_path: str | Path,
        *,
        force: bool = False,
        source: str = "manifest_sync",
    ) -> dict[str, Any] | None:
        path = Path(manifest_path).expanduser().resolve()
        value = _read_json(path)
        if value is None:
            return None
        project_id = _project_id(value, path)
        signature = self._manifest_signature(path, value)
        signature_json = _json(signature)
        now = _now()

        with self._connect() as connection:
            previous = connection.execute(
                "SELECT signature_json FROM catalog_sources WHERE source_path = ?",
                (str(path),),
            ).fetchone()
            if not force and previous is not None and previous["signature_json"] == signature_json:
                return self._project_card_row(connection, project_id)

            connection.execute(
                """
                INSERT INTO projects(
                  project_id, project_name, source_root, manifest_path, created_by,
                  status, created_at, updated_at, revision, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, NULL)
                ON CONFLICT(project_id) DO UPDATE SET
                  project_name=excluded.project_name,
                  source_root=excluded.source_root,
                  manifest_path=excluded.manifest_path,
                  created_by=excluded.created_by,
                  status=excluded.status,
                  updated_at=excluded.updated_at,
                  revision=projects.revision + 1,
                  deleted_at=NULL
                """,
                (
                    project_id,
                    str(value.get("project_name") or project_id),
                    str(value.get("root") or ""),
                    str(path),
                    str(value.get("created_by") or value.get("creator") or value.get("owner") or ""),
                    str(value.get("status") or "queued"),
                    str(value.get("created_at") or value.get("generated_at") or now),
                    now,
                ),
            )

            current_slugs: list[str] = []
            project_counts: Counter[str] = Counter()
            plate_count = 0
            recognized_count = 0
            reviewed_plate_count = 0
            total_wells = 0
            reviewed_wells = 0
            for index, plate in enumerate(value.get("plates", [])):
                if not isinstance(plate, dict):
                    continue
                slug = str(plate.get("slug") or plate.get("board_id") or f"plate-{index + 1}")
                plate_id = f"{project_id}:{slug}"
                current_slugs.append(slug)
                plate_count += 1
                paths = _plate_paths(plate, path.parent)
                report, report_path = _report_for_plate(plate, path.parent)
                counts = _category_counts((report or {}).get("category_counts"))
                review_summary = _fresh_review_summary(paths) or {}
                review_ready = review_summary.get("status") == "ready"
                reviewable = int(review_summary.get("well_count", 0) or 0) if review_ready else 0
                reviewed = int(review_summary.get("completed_well_count", 0) or 0) if review_ready else 0
                reviewable_objects = int(review_summary.get("object_count", 0) or 0) if review_ready else 0
                reviewed_objects = int(review_summary.get("reviewed_object_count", 0) or 0) if review_ready else 0
                review_complete = bool(review_ready and reviewed >= reviewable)
                status = str(plate.get("status") or value.get("status") or "queued")
                if report is not None:
                    status = "completed"
                    recognized_count += 1
                elif status == "completed":
                    recognized_count += 1
                if review_complete:
                    reviewed_plate_count += 1
                total_wells += int((report or {}).get("well_count", 0) or 0)
                reviewed_wells += reviewed
                connection.execute(
                    """
                    INSERT INTO plates(
                      plate_id, project_id, plate_slug, board_id, group_id, status,
                      current_stage, progress_percent, elapsed_seconds, artifact_root,
                      config_path, images_manifest_path, report_path, review_database_path,
                      updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(plate_id) DO UPDATE SET
                      board_id=excluded.board_id,
                      group_id=excluded.group_id,
                      status=excluded.status,
                      artifact_root=excluded.artifact_root,
                      config_path=excluded.config_path,
                      images_manifest_path=excluded.images_manifest_path,
                      report_path=excluded.report_path,
                      review_database_path=excluded.review_database_path,
                      updated_at=excluded.updated_at
                    """,
                    (
                        plate_id,
                        project_id,
                        slug,
                        str(plate.get("board_id") or ""),
                        str(plate.get("group_id") or ""),
                        status,
                        str(plate.get("current_stage") or plate.get("stage") or status),
                        float(plate.get("progress_percent", 100 if status == "completed" else 0) or 0),
                        float(plate.get("elapsed_seconds", 0) or 0),
                        str(paths["artifact_root"] or ""),
                        str(_resolve_reference(plate.get("config"), path.parent) or ""),
                        str(_resolve_reference(plate.get("images_manifest"), path.parent) or ""),
                        str(report_path or ""),
                        str(paths["database"] or ""),
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO plate_summaries(
                      plate_id, total_wells, reviewed_wells, reviewable_wells,
                      reviewed_objects, reviewable_objects, review_complete,
                      category_counts_json, source_report_revision, calculated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(plate_id) DO UPDATE SET
                      total_wells=excluded.total_wells,
                      reviewed_wells=excluded.reviewed_wells,
                      reviewable_wells=excluded.reviewable_wells,
                      reviewed_objects=excluded.reviewed_objects,
                      reviewable_objects=excluded.reviewable_objects,
                      review_complete=excluded.review_complete,
                      category_counts_json=excluded.category_counts_json,
                      source_report_revision=excluded.source_report_revision,
                      calculated_at=excluded.calculated_at
                    """,
                    (
                        plate_id,
                        int((report or {}).get("well_count", 0) or 0),
                        reviewed,
                        reviewable,
                        reviewed_objects,
                        reviewable_objects,
                        int(review_complete),
                        _json(counts),
                        str(paths["report"].stat().st_mtime_ns) if paths["report"] and paths["report"].exists() else None,
                        now,
                    ),
                )
                if report is not None and isinstance(report.get("wells"), list):
                    self._sync_wells(connection, plate_id, report["wells"], source=source)
                    # Human decisions stay in well_current.  The report is
                    # still the model/source projection, so summaries must be
                    # calculated from the preserved current values instead of
                    # blindly replacing them with a later model refresh.
                    current_counts = Counter(
                        {
                            str(row["category_code"]): int(row["count"])
                            for row in connection.execute(
                                "SELECT category_code, COUNT(*) AS count FROM well_current WHERE plate_id = ? GROUP BY category_code",
                                (plate_id,),
                            ).fetchall()
                        }
                    )
                    if current_counts:
                        counts = dict(current_counts)
                        connection.execute(
                            "UPDATE plate_summaries SET category_counts_json = ?, calculated_at = ? WHERE plate_id = ?",
                            (_json(counts), now, plate_id),
                        )
                project_counts.update(counts)

            if current_slugs:
                placeholders = ",".join("?" for _ in current_slugs)
                connection.execute(
                    f"DELETE FROM plates WHERE project_id = ? AND plate_slug NOT IN ({placeholders})",
                    (project_id, *current_slugs),
                )
            else:
                connection.execute("DELETE FROM plates WHERE project_id = ?", (project_id,))
            project_row = connection.execute(
                "SELECT revision FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            revision = int(project_row["revision"] if project_row else 1)
            connection.execute(
                """
                INSERT INTO project_summaries(
                  project_id, plate_count, recognized_plate_count, reviewed_plate_count,
                  total_wells, reviewed_wells, category_counts_json, calculated_at, source_revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                  plate_count=excluded.plate_count,
                  recognized_plate_count=excluded.recognized_plate_count,
                  reviewed_plate_count=excluded.reviewed_plate_count,
                  total_wells=excluded.total_wells,
                  reviewed_wells=excluded.reviewed_wells,
                  category_counts_json=excluded.category_counts_json,
                  calculated_at=excluded.calculated_at,
                  source_revision=excluded.source_revision
                """,
                (
                    project_id,
                    plate_count,
                    recognized_count,
                    reviewed_plate_count,
                    total_wells,
                    reviewed_wells,
                    _json(dict(project_counts)),
                    now,
                    revision,
                ),
            )
            connection.execute(
                "INSERT INTO catalog_sources(source_path, signature_json, synced_at) VALUES (?, ?, ?) "
                "ON CONFLICT(source_path) DO UPDATE SET signature_json=excluded.signature_json, synced_at=excluded.synced_at",
                (str(path), signature_json, now),
            )
            self._record_event(
                connection,
                "project",
                project_id,
                source,
                {"manifest_path": str(path), "plate_count": plate_count},
            )
            return self._project_card_row(connection, project_id)

    def _sync_wells(
        self,
        connection: sqlite3.Connection,
        plate_id: str,
        wells: list[Any],
        *,
        source: str,
        reviewer: str = "",
        action_id: str = "",
        operation: str = "sync",
        force_current: bool = False,
    ) -> None:
        now = _now()
        for item in wells:
            if not isinstance(item, dict) or not item.get("well"):
                continue
            well = str(item["well"]).upper()
            category = str(item.get("final_category") or item.get("category_code") or "undetermined")
            reason = str(item.get("undetermined_reason") or item.get("reason_code") or "")
            old = connection.execute(
                "SELECT category_code, reason_code, source, reviewer, reviewed_at, revision, "
                "model_category_code, model_reason_code FROM well_current WHERE plate_id = ? AND well = ?",
                (plate_id, well),
            ).fetchone()
            old_source = str(old["source"] or "").lower() if old else ""
            human_current = bool(old and (old_source.endswith("_review") or old_source in {
                "human_review",
                "quick_review",
                "screening_review",
                "late_growth_review",
                "integrated_review",
                "auto_review",
                "multiplicity_review",
                "mask_review",
                "lineage_review",
                "manual",
            }))
            preserve_current = human_current and not force_current
            current_category = str(old["category_code"]) if preserve_current else category
            current_reason = str(old["reason_code"] or "") if preserve_current else reason
            current_source = str(old["source"]) if preserve_current else source
            current_reviewer = str(old["reviewer"] or "") if preserve_current else reviewer
            current_reviewed_at = old["reviewed_at"] if preserve_current else (now if reviewer else None)
            revision = int(old["revision"] if old else 0)
            changed = old is None or old["category_code"] != current_category or (old["reason_code"] or "") != current_reason
            if changed:
                revision += 1
            if changed or force_current:
                connection.execute(
                    """
                    INSERT INTO well_decision_history(
                      plate_id, well, category_code, reason_code, source,
                      reviewer, operation, action_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (plate_id, well, current_category, current_reason, current_source, current_reviewer, operation, action_id, now),
                )
            connection.execute(
                """
                INSERT INTO well_current(
                  plate_id, well, category_code, reason_code, source,
                  reviewer, reviewed_at, model_category_code, model_reason_code,
                  model_updated_at, revision, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plate_id, well) DO UPDATE SET
                  category_code=excluded.category_code,
                  reason_code=excluded.reason_code,
                  source=excluded.source,
                  reviewer=excluded.reviewer,
                  reviewed_at=excluded.reviewed_at,
                  model_category_code=excluded.model_category_code,
                  model_reason_code=excluded.model_reason_code,
                  model_updated_at=excluded.model_updated_at,
                  revision=excluded.revision,
                  updated_at=excluded.updated_at
                """,
                (
                    plate_id,
                    well,
                    current_category,
                    current_reason,
                    current_source,
                    current_reviewer,
                    current_reviewed_at,
                    category,
                    reason,
                    now,
                    revision,
                    now,
                ),
            )

    def sync_task(self, task: dict[str, Any]) -> dict[str, Any] | None:
        task_id = str(task.get("task_id") or "")
        if not task_id:
            return None
        manifest_path = Path(str(task.get("project_manifest") or "")).expanduser()
        project_id = str(task.get("project_id") or "")
        if not project_id and manifest_path.exists():
            value = _read_json(manifest_path)
            if value is not None:
                project_id = _project_id(value, manifest_path)
        now = _now()
        if project_id and manifest_path.exists():
            with self._connect() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM projects WHERE project_id = ?", (project_id,)
                ).fetchone()
            if exists is None:
                self.sync_manifest(manifest_path, force=True, source="task_sync")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks(
                  task_id, project_id, task_name, created_by, status, current_stage,
                  progress_current, progress_total, progress_percent, progress_message,
                  worker_id, created_at, started_at, finished_at, elapsed_seconds,
                  error_message, options_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                  project_id=excluded.project_id,
                  task_name=excluded.task_name,
                  created_by=excluded.created_by,
                  status=excluded.status,
                  current_stage=excluded.current_stage,
                  progress_current=excluded.progress_current,
                  progress_total=excluded.progress_total,
                  progress_percent=excluded.progress_percent,
                  progress_message=excluded.progress_message,
                  worker_id=excluded.worker_id,
                  created_at=COALESCE(tasks.created_at, excluded.created_at),
                  started_at=excluded.started_at,
                  finished_at=excluded.finished_at,
                  elapsed_seconds=excluded.elapsed_seconds,
                  error_message=excluded.error_message,
                  options_json=excluded.options_json,
                  updated_at=excluded.updated_at
                """,
                (
                    task_id,
                    project_id or None,
                    str(task.get("name") or task.get("task_name") or task_id),
                    str(task.get("created_by") or ""),
                    str(task.get("status") or "queued"),
                    str(task.get("progress_stage") or task.get("stage") or "queued"),
                    float(task.get("progress_current", 0) or 0),
                    float(task.get("progress_total", 0) or 0),
                    float(task.get("progress_percent", 0) or 0),
                    str(task.get("progress_message") or ""),
                    str(task.get("worker_id") or ""),
                    str(task.get("created_at") or now),
                    task.get("started_at"),
                    task.get("finished_at"),
                    float(task.get("elapsed_seconds", 0) or 0),
                    str(task.get("error") or task.get("error_message") or ""),
                    _json(task.get("options") or {"selected_timepoint_labels": task.get("selected_timepoint_labels", [])}),
                    now,
                ),
            )
            boards = task.get("progress_boards") if isinstance(task.get("progress_boards"), list) else []
            connection.execute("DELETE FROM task_plates WHERE task_id = ?", (task_id,))
            for index, board in enumerate(boards):
                if not isinstance(board, dict) or not board.get("slug") or not project_id:
                    continue
                plate = connection.execute(
                    "SELECT plate_id FROM plates WHERE project_id = ? AND plate_slug = ?",
                    (project_id, str(board["slug"])),
                ).fetchone()
                if plate is not None:
                    connection.execute(
                        "INSERT INTO task_plates(task_id, plate_id, sequence_no) VALUES (?, ?, ?) "
                        "ON CONFLICT(task_id, plate_id) DO UPDATE SET sequence_no=excluded.sequence_no",
                        (task_id, plate["plate_id"], index),
                    )
                    connection.execute(
                        """
                        UPDATE plates
                        SET status = ?, current_stage = ?, progress_percent = ?,
                            elapsed_seconds = ?, updated_at = ?
                        WHERE plate_id = ?
                        """,
                        (
                            str(board.get("status") or "queued"),
                            str(board.get("stage") or board.get("status") or "queued"),
                            float(board.get("progress_percent", 0) or 0),
                            float(board.get("elapsed_seconds", 0) or 0),
                            now,
                            plate["plate_id"],
                        ),
                    )
            self._record_event(connection, "task", task_id, "task_sync", {"project_id": project_id})
            row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            return dict(row) if row is not None else None

    def sync_review_update(
        self,
        manifest_path: str | Path,
        plate_id: str,
        *,
        wells: Iterable[str] | None = None,
        source: str = "human_review",
        reviewer: str = "",
        action_id: str = "",
        operation: str = "review_save",
    ) -> dict[str, Any] | None:
        """Synchronize a just-saved review without replacing human decisions.

        The report and review summary are first refreshed into the catalog.  If
        ``wells`` is supplied, those wells are promoted to the current human
        decision from the newest report and an audit record is written.  Model
        values are retained separately on ``well_current`` so a later report
        refresh cannot erase the reviewer decision.
        """

        # The review-save path already rewrites the quick-review summary (whose
        # mtime participates in the manifest signature), so a normal signature
        # check detects the change without forcing a full manifest re-read.
        # Selected wells below are the only rows promoted to a human source.
        card = self.sync_manifest(manifest_path, force=False, source="review_report_sync")
        if card is None:
            return None
        selected = {str(value).upper() for value in (wells or []) if str(value).strip()}
        if not selected:
            return card
        manifest = _read_json(Path(manifest_path).expanduser().resolve()) or {}
        project_id = _project_id(manifest, Path(manifest_path).expanduser().resolve())
        slug = str(plate_id).split(":", 1)[1] if ":" in str(plate_id) else str(plate_id)
        plate = next(
            (
                item
                for item in manifest.get("plates", [])
                if isinstance(item, dict)
                and str(item.get("slug") or item.get("board_id") or "") == slug
            ),
            None,
        )
        if plate is None:
            return card
        report, _ = _report_for_plate(plate, Path(manifest_path).expanduser().resolve().parent)
        report_wells = [
            item for item in (report or {}).get("wells", [])
            if isinstance(item, dict) and str(item.get("well") or "").upper() in selected
        ]
        if not report_wells:
            return card
        with self._connect() as connection:
            self._sync_wells(
                connection,
                f"{project_id}:{slug}",
                report_wells,
                source=source,
                reviewer=reviewer,
                action_id=action_id,
                operation=operation,
                force_current=True,
            )
            self._refresh_summaries_for_project(connection, project_id)
            self._record_event(
                connection,
                "plate",
                f"{project_id}:{slug}",
                operation,
                {"wells": sorted(selected), "reviewer": reviewer, "action_id": action_id},
            )
        return self._card_after_sync(project_id)

    def _refresh_summaries_for_project(
        self,
        connection: sqlite3.Connection,
        project_id: str,
    ) -> None:
        """Recalculate category totals after an in-place well decision."""

        rows = connection.execute(
            """
            SELECT ps.plate_id, ps.total_wells, ps.reviewed_wells, ps.reviewable_wells,
                   ps.reviewed_objects, ps.reviewable_objects, ps.review_complete,
                   ps.source_report_revision, ps.calculated_at
            FROM plate_summaries ps
            JOIN plates p ON p.plate_id = ps.plate_id
            WHERE p.project_id = ?
            """,
            (project_id,),
        ).fetchall()
        counts: Counter[str] = Counter()
        total_wells = reviewed_wells = 0
        reviewed_plates = 0
        for row in rows:
            well_rows = connection.execute(
                "SELECT category_code, COUNT(*) AS count FROM well_current WHERE plate_id = ? GROUP BY category_code",
                (row["plate_id"],),
            ).fetchall()
            plate_counts = {str(item["category_code"]): int(item["count"]) for item in well_rows}
            if plate_counts:
                connection.execute(
                    "UPDATE plate_summaries SET category_counts_json = ?, calculated_at = ? WHERE plate_id = ?",
                    (_json(plate_counts), _now(), row["plate_id"]),
                )
                counts.update(plate_counts)
            total_wells += int(row["total_wells"] or 0)
            reviewed_wells += int(row["reviewed_wells"] or 0)
            reviewed_plates += int(row["review_complete"] or 0)
        project_row = connection.execute(
            "SELECT revision FROM projects WHERE project_id = ?", (project_id,)
        ).fetchone()
        plate_count = int(
            connection.execute("SELECT COUNT(*) FROM plates WHERE project_id = ?", (project_id,)).fetchone()[0]
            or 0
        )
        recognized_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM plates WHERE project_id = ? AND status = 'completed'", (project_id,)
            ).fetchone()[0]
            or 0
        )
        now = _now()
        connection.execute(
            """
            INSERT INTO project_summaries(
              project_id, plate_count, recognized_plate_count, reviewed_plate_count,
              total_wells, reviewed_wells, category_counts_json, calculated_at, source_revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
              plate_count=excluded.plate_count,
              recognized_plate_count=excluded.recognized_plate_count,
              reviewed_plate_count=excluded.reviewed_plate_count,
              total_wells=excluded.total_wells,
              reviewed_wells=excluded.reviewed_wells,
              category_counts_json=excluded.category_counts_json,
              calculated_at=excluded.calculated_at,
              source_revision=excluded.source_revision
            """,
            (
                project_id,
                plate_count,
                recognized_count,
                reviewed_plates,
                total_wells,
                reviewed_wells,
                _json(dict(counts)),
                now,
                int(project_row["revision"] if project_row else 1),
            ),
        )

    def _card_after_sync(self, project_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            return self._project_card_row(connection, project_id)

    def project_detail(self, project_id: str) -> dict[str, Any] | None:
        """Return one project card with plate-level catalog summaries."""

        with self._connect() as connection:
            card = self._project_card_row(connection, project_id)
            if card is None:
                return None
            rows = connection.execute(
                """
                SELECT p.*, ps.total_wells, ps.reviewed_wells, ps.reviewable_wells,
                       ps.reviewed_objects, ps.reviewable_objects, ps.review_complete,
                       ps.category_counts_json, ps.source_report_revision, ps.calculated_at
                FROM plates p
                LEFT JOIN plate_summaries ps ON ps.plate_id = p.plate_id
                WHERE p.project_id = ?
                ORDER BY p.plate_slug COLLATE NOCASE
                """,
                (str(project_id),),
            ).fetchall()
            plates: list[dict[str, Any]] = []
            for row in rows:
                try:
                    categories = json.loads(row["category_counts_json"] or "{}")
                except json.JSONDecodeError:
                    categories = {}
                plates.append({
                    "plate_id": row["plate_id"],
                    "plate_slug": row["plate_slug"],
                    "slug": row["plate_slug"],
                    "board_id": row["board_id"] or "",
                    "group_id": row["group_id"] or "",
                    "status": row["status"],
                    "current_stage": row["current_stage"] or "",
                    "progress_percent": float(row["progress_percent"] or 0),
                    "elapsed_seconds": float(row["elapsed_seconds"] or 0),
                    "total_wells": int(row["total_wells"] or 0),
                    "well_count": int(row["total_wells"] or 0),
                    "reviewed_wells": int(row["reviewed_wells"] or 0),
                    "reviewable_wells": int(row["reviewable_wells"] or 0),
                    "reviewed_well_count": int(row["reviewed_wells"] or 0),
                    "reviewable_well_count": int(row["reviewable_wells"] or 0),
                    "reviewed_objects": int(row["reviewed_objects"] or 0),
                    "reviewable_objects": int(row["reviewable_objects"] or 0),
                    "reviewed_object_count": int(row["reviewed_objects"] or 0),
                    "reviewable_object_count": int(row["reviewable_objects"] or 0),
                    "review_data_available": bool(row["reviewable_wells"] or row["reviewable_objects"]),
                    "review_complete": bool(row["review_complete"]),
                    "category_counts": categories,
                    "source_report_revision": row["source_report_revision"],
                    "updated_at": row["calculated_at"] or row["updated_at"],
                })
            card["plates"] = plates
            return card

    def list_wells(
        self,
        plate_id: str,
        *,
        category: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Read current and model conclusions for a plate without images."""

        safe_limit = max(1, min(int(limit), 10000))
        clauses = ["plate_id = ?"]
        params: list[Any] = [str(plate_id)]
        if category:
            clauses.append("category_code = ?")
            params.append(str(category))
        params.append(safe_limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM well_current WHERE {' AND '.join(clauses)} ORDER BY well LIMIT ?",
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def status(self) -> dict[str, Any]:
        with self._connect() as connection:
            schema = connection.execute(
                "SELECT value FROM catalog_meta WHERE key = 'schema_version'"
            ).fetchone()
            return {
                "path": str(self.path),
                "schema_version": int(schema[0]) if schema is not None else None,
                "projects": int(connection.execute("SELECT COUNT(*) FROM projects WHERE deleted_at IS NULL").fetchone()[0] or 0),
                "plates": int(connection.execute("SELECT COUNT(*) FROM plates").fetchone()[0] or 0),
                "tasks": int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] or 0),
                "wells": int(connection.execute("SELECT COUNT(*) FROM well_current").fetchone()[0] or 0),
                "last_event_at": connection.execute(
                    "SELECT created_at FROM catalog_events ORDER BY event_id DESC LIMIT 1"
                ).fetchone()[0]
                if connection.execute("SELECT COUNT(*) FROM catalog_events").fetchone()[0]
                else None,
            }

    def delete_task(self, task_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM tasks WHERE task_id = ?", (str(task_id),))
            self._record_event(connection, "task", str(task_id), "task_deleted", {})

    def mark_project_deleted(self, project_id: str) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE projects SET status='deleted', deleted_at=?, updated_at=? WHERE project_id = ?",
                (now, now, str(project_id)),
            )
            self._record_event(connection, "project", str(project_id), "project_deleted", {})

    def _project_card_row(self, connection: sqlite3.Connection, project_id: str) -> dict[str, Any] | None:
        row = connection.execute(
            """
            SELECT p.*, s.plate_count, s.recognized_plate_count, s.reviewed_plate_count,
                   s.total_wells, s.reviewed_wells, s.category_counts_json, s.calculated_at
            FROM projects p
            LEFT JOIN project_summaries s ON s.project_id = p.project_id
            WHERE p.project_id = ? AND p.deleted_at IS NULL
            """,
            (str(project_id),),
        ).fetchone()
        if row is None:
            return None
        return _project_card_from_row(dict(row))


    def list_projects(
        self,
        *,
        query: str = "",
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = max(1, min(MAX_PAGE_SIZE, int(page_size)))
        needle = str(query or "").strip()
        where = "p.deleted_at IS NULL"
        params: list[Any] = []
        if needle:
            where += " AND (p.project_id LIKE ? COLLATE NOCASE OR p.project_name LIKE ? COLLATE NOCASE OR p.source_root LIKE ? COLLATE NOCASE OR p.created_by LIKE ? COLLATE NOCASE)"
            params.extend([f"%{needle}%"] * 4)
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM projects p WHERE {where}", params).fetchone()[0])
            pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, pages)
            rows = connection.execute(
                f"""
                SELECT p.*, s.plate_count, s.recognized_plate_count, s.reviewed_plate_count,
                       s.total_wells, s.reviewed_wells, s.category_counts_json, s.calculated_at
                FROM projects p
                LEFT JOIN project_summaries s ON s.project_id = p.project_id
                WHERE {where}
                ORDER BY COALESCE(s.calculated_at, p.updated_at) DESC, p.project_name COLLATE NOCASE
                LIMIT ? OFFSET ?
                """,
                (*params, page_size, (page - 1) * page_size),
            ).fetchall()
            items: list[dict[str, Any]] = []
            for row in rows:
                items.append(_project_card_from_row(dict(row)))
            aggregate_rows = connection.execute(
                f"SELECT s.category_counts_json FROM projects p LEFT JOIN project_summaries s ON s.project_id=p.project_id WHERE {where}",
                params,
            ).fetchall()
            categories: Counter[str] = Counter()
            for row in aggregate_rows:
                try:
                    categories.update(_category_counts(json.loads(row["category_counts_json"] or "{}")))
                except json.JSONDecodeError:
                    pass
            aggregate = {
                "project_count": total,
                "plate_count": int(connection.execute(f"SELECT COALESCE(SUM(COALESCE(s.plate_count, 0)), 0) FROM projects p LEFT JOIN project_summaries s ON s.project_id=p.project_id WHERE {where}", params).fetchone()[0] or 0),
                "recognized_plate_count": int(connection.execute(f"SELECT COALESCE(SUM(COALESCE(s.recognized_plate_count, 0)), 0) FROM projects p LEFT JOIN project_summaries s ON s.project_id=p.project_id WHERE {where}", params).fetchone()[0] or 0),
                "reviewed_plate_count": int(connection.execute(f"SELECT COALESCE(SUM(COALESCE(s.reviewed_plate_count, 0)), 0) FROM projects p LEFT JOIN project_summaries s ON s.project_id=p.project_id WHERE {where}", params).fetchone()[0] or 0),
                "category_counts": dict(categories),
            }
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": pages,
            "query": needle,
            "aggregate": aggregate,
            "updated_at": _now(),
        }

    def reconcile_all(self, current_manifest: str | Path) -> int:
        current = Path(current_manifest).expanduser().resolve()
        try:
            current_value = json.loads(current.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current_value = {}
        current_dict = current_value if isinstance(current_value, dict) else {}
        configured = current_dict.get("project_manifest_paths")
        paths: list[Path] = []
        if isinstance(configured, list):
            for value in configured:
                candidate = Path(os.path.expandvars(str(value))).expanduser()
                if not candidate.is_absolute():
                    candidate = current.parent / candidate
                candidate = candidate.resolve()
                if candidate.is_file():
                    paths.append(candidate)
        if not bool(current_dict.get("review_hub")):
            root = current.parent.parent
            paths.extend(path for path in root.glob("*/project.json") if path.is_file())
            if current.is_file() and current not in paths:
                paths.append(current)
        live_paths = {str(path.resolve()) for path in paths}
        now = _now()
        with self._connect() as connection:
            stale = connection.execute(
                "SELECT project_id, manifest_path FROM projects WHERE deleted_at IS NULL"
            ).fetchall()
            for row in stale:
                if str(row["manifest_path"]) not in live_paths:
                    connection.execute(
                        "UPDATE projects SET status='deleted', deleted_at=?, updated_at=? WHERE project_id = ?",
                        (now, now, row["project_id"]),
                    )
        synced = 0
        for path in sorted(set(paths), key=lambda item: item.as_posix().casefold()):
            if self.sync_manifest(path) is not None:
                synced += 1
        return synced


def sync_catalog_manifest(
    catalog_path: str | Path,
    manifest_path: str | Path,
    *,
    force: bool = False,
    source: str = "manifest_sync",
) -> dict[str, Any] | None:
    return ProjectCatalog(catalog_path).sync_manifest(manifest_path, force=force, source=source)

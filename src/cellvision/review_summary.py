"""Persistent, timestamped summaries used by the review list views.

The full prediction table remains the source of truth for detail review and
training.  This module only owns the small derived cache that lets list views
avoid loading that table when the underlying files have not changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUMMARY_VERSION = 3  # bump: well rows now include pre-review filter metrics and manual verdicts
SUMMARY_FILENAME = "quick_review_summary.json"
PREDICTION_FILENAMES = (
    "latest_integrated_predictions.csv",
    "latest_temporally_completed_predictions.csv",
    "latest_v2_predictions.csv",
    "latest_v3_predictions.csv",
)


def latest_prediction_path(artifact_root: str | Path) -> Path | None:
    """Return the newest supported integrated prediction table."""

    prediction_root = Path(artifact_root) / "predictions"
    candidates = [
        prediction_root / name
        for name in PREDICTION_FILENAMES
        if (prediction_root / name).exists()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def _mtime_ns(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def summary_path(artifact_root: str | Path) -> Path:
    return Path(artifact_root) / "cache" / SUMMARY_FILENAME


def summary_signature(
    artifact_root: str | Path,
    *,
    database_path: str | Path | None = None,
    screening_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a cheap signature without reading any large data file."""

    root = Path(artifact_root)
    prediction = latest_prediction_path(root)
    database = Path(database_path) if database_path else root / "annotations" / "annotations.db"
    screening = (
        Path(screening_path)
        if screening_path
        else root / "predictions" / "latest_well_screening.csv"
    )
    report = (
        Path(report_path)
        if report_path
        else root / "gated" / "plate_overview.csv"
    )
    return {
        "prediction_source": str(prediction) if prediction else None,
        "prediction_mtime_ns": _mtime_ns(prediction),
        "database_mtime_ns": _mtime_ns(database),
        "screening_mtime_ns": _mtime_ns(screening),
        "report_mtime_ns": _mtime_ns(report),
    }


def read_cached_summary(path: str | Path) -> dict[str, Any] | None:
    """Read the last structurally valid summary, even when its inputs changed."""

    summary_file = Path(path)
    try:
        payload = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != SUMMARY_VERSION:
        return None
    return payload


def read_summary(path: str | Path, signature: dict[str, Any]) -> dict[str, Any] | None:
    """Read a summary only when it was generated for the current inputs."""

    payload = read_cached_summary(path)
    if payload is None:
        return None
    if payload.get("signature") != signature:
        return None
    return payload


def write_summary(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically replace a derived summary cache."""

    summary_file = Path(path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_file.with_name(f".{summary_file.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(summary_file)

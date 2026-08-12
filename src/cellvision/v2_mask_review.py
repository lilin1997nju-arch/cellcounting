from __future__ import annotations

"""Versioned V2 mask review rounds.

The review round is deliberately separate from the active ``latest_*``
artifacts.  A reviewer can therefore correct a mask, inspect the derived
contour/area, and only later choose whether the reviewed CSV should become an
input to a downstream stage.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, artifact_path, load_config
from .v2_instance_inference import (
    _contour,
    _rle,
    decode_rle,
    finalize_v2_instances,
    infer_v2_instances,
)


MASK_SIZE = 96
ROUND_PREFIX = "p0-mask-review-"
_ROUND_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REVIEWABLE_LABELS = {
    "single",
    "touching_doublet",
    "cluster_3plus",
    "uncertain",
}
_GROUP_LABELS = {"touching_doublet", "cluster_3plus"}


def _now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _round_root(config: dict[str, Any]) -> Path:
    root = Path(config["paths"]["artifact_root"]) / "v2" / "mask_review"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def round_directory(config: dict[str, Any], round_id: str) -> Path:
    round_id = str(round_id)
    if not _ROUND_ID_PATTERN.fullmatch(round_id):
        raise ValueError("invalid mask review round id")
    root = _round_root(config)
    target = (root / round_id).resolve()
    if target.parent != root:
        raise ValueError("mask review round is outside the configured artifact root")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _manifest_path(config: dict[str, Any], round_id: str) -> Path:
    return round_directory(config, round_id) / "manifest.json"


def load_mask_review_manifest(config: dict[str, Any], round_id: str) -> dict[str, Any]:
    path = _manifest_path(config, round_id)
    if not path.exists():
        raise FileNotFoundError(f"mask review round does not exist: {round_id}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid mask review manifest: {round_id}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid mask review manifest: {round_id}")
    return manifest


def _manifest_file(config: dict[str, Any], round_id: str, key: str) -> Path:
    manifest = load_mask_review_manifest(config, round_id)
    value = manifest.get(key)
    if not value:
        raise ValueError(f"mask review manifest is missing {key}")
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = round_directory(config, round_id) / candidate
    return candidate.resolve()


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value


def _read_round_frame(
    config: dict[str, Any], round_id: str, *, reviewed: bool = False
) -> pd.DataFrame:
    key = "reviewed_predictions" if reviewed else "pre_temporal_predictions"
    manifest = load_mask_review_manifest(config, round_id)
    value = manifest.get(key)
    # Comparison rounds are assembled from two isolated inference outputs and
    # use the combined reviewed file as their single review input.  Keep this
    # fallback so older comparison manifests remain readable as well.
    if not value and manifest.get("kind") == "comparison" and not reviewed:
        value = manifest.get("reviewed_predictions")
    if not value:
        raise ValueError(f"mask review manifest is missing {key}")
    path = Path(str(value))
    if not path.is_absolute():
        path = round_directory(config, round_id) / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"mask review predictions are missing: {path}")
    return pd.read_csv(path, low_memory=False)


def _reviewable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "candidate_id" not in frame.columns:
        return frame.iloc[0:0].copy()
    labels = frame.get("integrated_label", pd.Series("uncertain", index=frame.index)).astype(str)
    original_labels = frame.get(
        "v2_original_integrated_label", pd.Series("", index=frame.index)
    ).astype(str)
    valid = frame.get("v2_mask_valid", pd.Series(True, index=frame.index)).map(_bool_value)
    suppressed = frame.get("v2_is_suppressed", pd.Series(False, index=frame.index)).map(_bool_value)
    is_candidate = labels.isin(_REVIEWABLE_LABELS) | original_labels.isin(_REVIEWABLE_LABELS)
    # A group row is the authoritative object for a touching group.  Hide
    # nested single proposals so the reviewer does not correct the same cell
    # twice, but retain a cell-labelled row with an empty mask for recovery.
    visible = is_candidate & (~suppressed | labels.isin(_GROUP_LABELS) | ~valid)
    return frame[visible].copy()


def _review_priority(row: pd.Series) -> tuple[float, ...]:
    status = str(row.get("v2_refinement_status", ""))
    label = str(row.get("integrated_label", ""))
    valid = _bool_value(row.get("v2_mask_valid", False))
    ratio = _number(row.get("v2_refinement_area_ratio", 1.0), 1.0)
    iou = _number(row.get("v2_refinement_iou", 1.0), 1.0)
    fallback = 1.0 if status.startswith("fallback") or status in {"empty", "fallback_empty"} else 0.0
    missing = 1.0 if not valid else 0.0
    # Touching groups are the main P0 failure mode, so put them before
    # ordinary singles in the review queue.  Within each bucket, fallback and
    # missing masks still rise to the top.
    group = 0.0 if label in _GROUP_LABELS else 1.0
    drift = abs(1.0 - ratio) + max(0.0, 0.75 - iou)
    return (missing, fallback, group, drift, -_number(row.get("v2_instance_confidence", 0.0)))


def _load_review_records(database: str | Path, round_id: str) -> dict[str, dict[str, Any]]:
    try:
        import sqlite3

        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM v2_mask_reviews WHERE round_id = ?",
                (str(round_id),),
            ).fetchall()
    except Exception:
        return {}
    return {str(row["candidate_id"]): dict(row) for row in rows}


def _row_payload(
    row: pd.Series,
    record: dict[str, Any] | None,
    *,
    include_masks: bool,
    mask_size: int,
) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id", ""))
    is_comparison = "comparison_old_mask_rle" in row.index
    model_rle = str(
        row.get("comparison_new_mask_rle", row.get("v2_mask_rle", "[]"))
    )
    decision = str(record.get("decision", "pending")) if record else "pending"
    reviewed_rle = str(record.get("reviewed_mask_rle", model_rle)) if record else model_rle
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "well": str(row.get("well", "")),
        "timepoint": str(row.get("timepoint", "")),
        "x_px": _number(row.get("x_px")),
        "y_px": _number(row.get("y_px")),
        "integrated_label": str(row.get("integrated_label", "")),
        "integrated_confidence": _number(row.get("integrated_confidence")),
        "cell_probability": _number(row.get("cell_probability")),
        "invalid_probability": _number(row.get("invalid_probability")),
        "model_mask_valid": _bool_value(
            row.get("comparison_new_mask_valid", row.get("v2_mask_valid", False))
        ),
        "model_area_px": int(
            round(
                _number(
                    row.get("comparison_new_area_px", row.get("v2_instance_area_px"))
                )
            )
        ),
        "model_diameter_px": _number(
            row.get("comparison_new_diameter_px", row.get("v2_instance_diameter_px"))
        ),
        "model_confidence": _number(
            row.get("comparison_new_confidence", row.get("v2_instance_confidence"))
        ),
        "refinement_status": str(
            row.get("comparison_new_refinement_status", row.get("v2_refinement_status", ""))
        ),
        "refinement_area_ratio": _number(
            row.get("comparison_new_refinement_area_ratio", row.get("v2_refinement_area_ratio")),
            1.0,
        ),
        "refinement_iou": _number(
            row.get("comparison_new_refinement_iou", row.get("v2_refinement_iou")),
            1.0,
        ),
        "mask_origin_x": int(round(_number(row.get("v2_mask_origin_x")))),
        "mask_origin_y": int(round(_number(row.get("v2_mask_origin_y")))),
        "mask_size": mask_size,
        "decision": decision,
        "reviewer": str(record.get("reviewer", "")) if record else "",
        "notes": str(record.get("notes", "")) if record else "",
        "reviewed_area_px": int(round(_number(record.get("reviewed_area_px"))))
        if record
        else int(round(_number(row.get("v2_instance_area_px")))),
        "reviewed_diameter_px": _number(record.get("reviewed_diameter_px"))
        if record
        else _number(row.get("v2_instance_diameter_px")),
        "comparison": is_comparison,
        "source_config": str(row.get("comparison_source_config", ""))
        if is_comparison
        else "",
        "old_model_area_px": int(round(_number(row.get("comparison_old_area_px"))))
        if is_comparison
        else 0,
        "new_model_area_px": int(round(_number(row.get("comparison_new_area_px"))))
        if is_comparison
        else 0,
        "old_model_confidence": _number(row.get("comparison_old_confidence"))
        if is_comparison
        else 0.0,
        "new_model_confidence": _number(row.get("comparison_new_confidence"))
        if is_comparison
        else 0.0,
    }
    if include_masks:
        payload["model_mask_rle"] = model_rle
        payload["reviewed_mask_rle"] = reviewed_rle
        payload["model_contour_json"] = str(row.get("v2_contour_json", "[]"))
        payload["reviewed_contour_json"] = (
            str(record.get("contour_json", row.get("v2_contour_json", "[]")))
            if record
            else str(row.get("v2_contour_json", "[]"))
        )
        if is_comparison:
            payload.update(
                {
                    "old_model_mask_rle": str(row.get("comparison_old_mask_rle", "[]")),
                    "new_model_mask_rle": model_rle,
                    "old_model_area_px": int(round(_number(row.get("comparison_old_area_px")))),
                    "new_model_area_px": int(round(_number(row.get("comparison_new_area_px")))),
                    "old_model_confidence": _number(row.get("comparison_old_confidence")),
                    "new_model_confidence": _number(row.get("comparison_new_confidence")),
                    "old_model_refinement_status": str(
                        row.get("comparison_old_refinement_status", "")
                    ),
                    "new_model_refinement_status": str(
                        row.get("comparison_new_refinement_status", "")
                    ),
                    "old_checkpoint": str(row.get("comparison_old_checkpoint", "")),
                    "new_checkpoint": str(row.get("comparison_new_checkpoint", "")),
                }
            )
    return {key: _json_value(value) for key, value in payload.items()}


def mask_review_candidates(
    config: dict[str, Any],
    database: str | Path,
    round_id: str,
    *,
    status: str = "pending",
    limit: int = 500,
) -> list[dict[str, Any]]:
    manifest = load_mask_review_manifest(config, round_id)
    mask_size = int(manifest.get("mask_size", MASK_SIZE))
    frame = _reviewable_frame(_read_round_frame(config, round_id))
    records = _load_review_records(database, round_id)
    if frame.empty:
        return []
    frame = frame.assign(
        _review_priority=frame.apply(_review_priority, axis=1),
        _candidate_status=frame["candidate_id"].astype(str).map(
            lambda value: str(records.get(value, {}).get("decision", "pending"))
        ),
    )
    normalized_status = str(status).lower()
    if normalized_status == "pending":
        frame = frame[frame["_candidate_status"] == "pending"]
    elif normalized_status == "reviewed":
        frame = frame[frame["_candidate_status"] != "pending"]
    elif normalized_status != "all":
        raise ValueError("status must be pending, reviewed, or all")
    frame = frame.sort_values(["_review_priority", "candidate_id"], kind="stable")
    output: list[dict[str, Any]] = []
    for _, row in frame.head(max(1, min(int(limit), 5000))).iterrows():
        record = records.get(str(row.candidate_id))
        output.append(_row_payload(row, record, include_masks=False, mask_size=mask_size))
    return output


def mask_review_candidate(
    config: dict[str, Any],
    database: str | Path,
    round_id: str,
    candidate_id: str,
) -> dict[str, Any] | None:
    manifest = load_mask_review_manifest(config, round_id)
    mask_size = int(manifest.get("mask_size", MASK_SIZE))
    frame = _reviewable_frame(_read_round_frame(config, round_id))
    selected = frame[frame["candidate_id"].astype(str) == str(candidate_id)]
    if selected.empty:
        return None
    records = _load_review_records(database, round_id)
    return _row_payload(
        selected.iloc[0],
        records.get(str(candidate_id)),
        include_masks=True,
        mask_size=mask_size,
    )


def mask_review_summary(
    config: dict[str, Any], database: str | Path, round_id: str
) -> dict[str, Any]:
    manifest = load_mask_review_manifest(config, round_id)
    candidates = mask_review_candidates(
        config, database, round_id, status="all", limit=5000
    )
    counts: dict[str, int] = {}
    for item in candidates:
        decision = str(item.get("decision", "pending"))
        counts[decision] = counts.get(decision, 0) + 1
    return {
        "round_id": str(round_id),
        "kind": str(manifest.get("kind", "mask_review")),
        "created_at": manifest.get("created_at"),
        "checkpoint": manifest.get("checkpoint"),
        "algorithm_version": manifest.get("algorithm_version"),
        "candidate_count": len(candidates),
        "pending_count": counts.get("pending", 0),
        "accepted_count": counts.get("accepted", 0),
        "edited_count": counts.get("edited", 0),
        "rejected_count": counts.get("rejected", 0),
        "decision_counts": counts,
        "reviewed_predictions": manifest.get("reviewed_predictions"),
    }


def list_mask_review_rounds(
    config: dict[str, Any], database: str | Path
) -> list[dict[str, Any]]:
    root = _round_root(config)
    output: list[dict[str, Any]] = []
    for manifest_path in root.glob("*/manifest.json"):
        round_id = manifest_path.parent.name
        try:
            output.append(mask_review_summary(config, database, round_id))
        except (FileNotFoundError, ValueError, OSError):
            continue
    return sorted(output, key=lambda item: str(item.get("created_at", "")), reverse=True)


def _resolve_repo_path(value: str | Path) -> Path:
    candidate = Path(str(value)).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _comparison_holdout_sources() -> list[dict[str, Any]]:
    training_config_path = PROJECT_ROOT / "configs" / "v2_training.yaml"
    training_config = load_config(training_config_path)
    sources: list[dict[str, Any]] = []
    for source_value in training_config.get("validation_holdout_sources", []) or []:
        source_path = _resolve_repo_path(source_value)
        if not source_path.exists():
            continue
        source_config = load_config(source_path)
        artifact_root = Path(source_config["paths"]["artifact_root"])
        predictions = artifact_root / "predictions" / "latest_integrated_predictions.csv"
        images = artifact_root / "manifests" / "images.csv"
        sources.append(
            {
                "source_config": str(source_path),
                "label": str(source_config.get("experiment", {}).get("experiment_id", source_path.stem)),
                "plate_id": str(source_config.get("experiment", {}).get("plate_id", source_path.stem)),
                "predictions_available": predictions.exists(),
                "images_available": images.exists(),
                "candidate_count": int(len(pd.read_csv(predictions, low_memory=False)))
                if predictions.exists()
                else 0,
            }
        )
    return sources


def _comparison_checkpoints() -> dict[str, Any]:
    old = PROJECT_ROOT / "artifacts" / "v2" / "models" / "latest_instance_segmenter.pt"
    candidates = sorted(
        (
            path
            for path in (PROJECT_ROOT / "artifacts" / "v2" / "runs").glob(
                "v2-instance-*/model.pt"
            )
            if path.exists() and path.resolve() != old.resolve()
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    return {
        "old_checkpoint": str(old.resolve()),
        "old_available": old.exists(),
        "new_candidates": [
            {"path": str(path.resolve()), "label": path.parent.name}
            for path in candidates[:10]
        ],
        "new_checkpoint": str(candidates[0].resolve()) if candidates else "",
    }


def mask_comparison_options(
    config: dict[str, Any], database: str | Path
) -> dict[str, Any]:
    rounds = [
        item
        for item in list_mask_review_rounds(config, database)
        if item.get("kind") == "comparison"
    ]
    checkpoints = _comparison_checkpoints()
    return {
        "holdout_sources": _comparison_holdout_sources(),
        "old_checkpoint": checkpoints["old_checkpoint"],
        "old_available": checkpoints["old_available"],
        "new_candidates": checkpoints["new_candidates"],
        "new_checkpoint": checkpoints["new_checkpoint"],
        "rounds": rounds,
    }


def create_model_comparison_round(
    config: dict[str, Any],
    *,
    old_checkpoint: str | Path | None = None,
    new_checkpoint: str | Path | None = None,
    source_configs: list[str] | None = None,
    round_id: str | None = None,
) -> dict[str, Any]:
    """Create an isolated old/new checkpoint comparison review round."""

    sources = _comparison_holdout_sources()
    allowed = {str(item["source_config"]): item for item in sources}
    selected_paths = source_configs or list(allowed)
    selected = [
        allowed[str(_resolve_repo_path(value))]
        for value in selected_paths
        if str(_resolve_repo_path(value)) in allowed
    ]
    if not selected:
        raise ValueError("no configured validation holdout sources are available")

    checkpoints = _comparison_checkpoints()
    old_path = _resolve_repo_path(old_checkpoint or checkpoints["old_checkpoint"])
    new_path = _resolve_repo_path(new_checkpoint or checkpoints["new_checkpoint"])
    if not old_path.exists():
        raise FileNotFoundError(f"old comparison checkpoint is missing: {old_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"new comparison checkpoint is missing: {new_path}")
    if old_path.resolve() == new_path.resolve():
        raise ValueError("old and new comparison checkpoints must be different")

    if round_id is None:
        round_id = "model-comparison-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    round_dir = round_directory(config, round_id)
    combined: list[pd.DataFrame] = []
    source_manifests: list[dict[str, Any]] = []
    for source in selected:
        source_config_path = Path(source["source_config"])
        source_config = load_config(source_config_path)
        artifact_root = Path(source_config["paths"]["artifact_root"])
        predictions_path = artifact_root / "predictions" / "latest_integrated_predictions.csv"
        if not predictions_path.exists():
            continue
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", source_config_path.stem)
        old_output = round_dir / f"{slug}-old-v2.csv"
        old_pre_temporal = round_dir / f"{slug}-old-v2-pre-temporal.csv"
        old_summary = round_dir / f"{slug}-old-v2.json"
        new_output = round_dir / f"{slug}-new-v2.csv"
        new_pre_temporal = round_dir / f"{slug}-new-v2-pre-temporal.csv"
        new_summary = round_dir / f"{slug}-new-v2.json"
        infer_v2_instances(
            source_config,
            old_path,
            predictions_path=predictions_path,
            output_path=old_output,
            pre_temporal_output_path=old_pre_temporal,
            summary_path=old_summary,
        )
        infer_v2_instances(
            source_config,
            new_path,
            predictions_path=predictions_path,
            output_path=new_output,
            pre_temporal_output_path=new_pre_temporal,
            summary_path=new_summary,
        )
        old_frame = pd.read_csv(old_output, low_memory=False)
        old_frame["candidate_id"] = old_frame["candidate_id"].astype(str)
        old_frame = old_frame.set_index("candidate_id")
        new_frame = pd.read_csv(new_output, low_memory=False)
        new_frame["candidate_id"] = new_frame["candidate_id"].astype(str)
        visible = _reviewable_frame(new_frame).copy()
        if visible.empty:
            continue
        old_frame = old_frame.reindex(visible["candidate_id"].astype(str))
        visible["comparison_old_mask_rle"] = old_frame["v2_mask_rle"].fillna("[]").to_numpy()
        visible["comparison_old_mask_valid"] = old_frame["v2_mask_valid"].fillna(False).to_numpy()
        visible["comparison_old_area_px"] = old_frame["v2_instance_area_px"].fillna(0).to_numpy()
        visible["comparison_old_diameter_px"] = old_frame["v2_instance_diameter_px"].fillna(0).to_numpy()
        visible["comparison_old_confidence"] = old_frame["v2_instance_confidence"].fillna(0).to_numpy()
        visible["comparison_old_refinement_status"] = old_frame["v2_refinement_status"].fillna("").to_numpy()
        visible["comparison_old_refinement_area_ratio"] = old_frame["v2_refinement_area_ratio"].fillna(0).to_numpy()
        visible["comparison_old_refinement_iou"] = old_frame["v2_refinement_iou"].fillna(0).to_numpy()
        visible["comparison_new_mask_rle"] = visible["v2_mask_rle"]
        visible["comparison_new_mask_valid"] = visible["v2_mask_valid"]
        visible["comparison_new_area_px"] = visible["v2_instance_area_px"]
        visible["comparison_new_diameter_px"] = visible["v2_instance_diameter_px"]
        visible["comparison_new_confidence"] = visible["v2_instance_confidence"]
        visible["comparison_new_refinement_status"] = visible["v2_refinement_status"]
        visible["comparison_source_config"] = str(source_config_path.resolve())
        visible["comparison_source_id"] = slug
        visible["comparison_old_checkpoint"] = str(old_path.resolve())
        visible["comparison_new_checkpoint"] = str(new_path.resolve())
        combined.append(visible)
        source_manifests.append({**source, "candidate_count": int(len(visible))})

    if not combined:
        raise RuntimeError("comparison inference produced no reviewable candidates")
    frame = _ensure_review_columns(pd.concat(combined, ignore_index=True, sort=False))
    reviewed = round_dir / "reviewed_v2_predictions.csv"
    _atomic_write_csv(frame, reviewed)
    manifest = {
        "round_id": str(round_id),
        "kind": "comparison",
        "created_at": _now_text(),
        "algorithm_version": "v2-model-comparison-review-20260812",
        "mask_size": MASK_SIZE,
        "old_checkpoint": str(old_path.resolve()),
        "new_checkpoint": str(new_path.resolve()),
        "source_configs": [item["source_config"] for item in source_manifests],
        "sources": source_manifests,
        # The comparison frame already contains the new-model output plus the
        # old-model columns, so it is the authoritative review input for both
        # candidate listing and persistence.
        "pre_temporal_predictions": str(reviewed.resolve()),
        "reviewed_predictions": str(reviewed.resolve()),
        "candidate_count": int(len(frame)),
        "valid_instance_count": int(frame["v2_mask_valid"].map(_bool_value).sum()),
    }
    _atomic_write_json(manifest, round_dir / "manifest.json")
    return manifest


def _decode_valid_rle(value: str | None, size: int) -> tuple[str, np.ndarray]:
    try:
        runs = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError("reviewed_mask_rle must be valid JSON") from exc
    if not isinstance(runs, list):
        raise ValueError("reviewed_mask_rle must be a list of runs")
    total = size * size
    for run in runs:
        if not isinstance(run, (list, tuple)) or len(run) != 2:
            raise ValueError("each mask run must contain start and length")
        start, length = int(run[0]), int(run[1])
        if start < 0 or length < 0 or start + length > total:
            raise ValueError("reviewed mask run is outside the mask patch")
    mask = decode_rle(json.dumps(runs), size)
    return _rle(mask), mask


def _ensure_review_columns(frame: pd.DataFrame) -> pd.DataFrame:
    text_defaults: dict[str, str] = {
        "v2_mask_review_status": "pending",
        "v2_mask_reviewed_by": "",
        "v2_mask_reviewed_at": "",
        "v2_mask_review_notes": "",
    }
    numeric_defaults: dict[str, pd.Series] = {
        "v2_reviewed_area_px": pd.to_numeric(
            frame.get("v2_instance_area_px", pd.Series(0, index=frame.index)), errors="coerce"
        ).fillna(0),
        "v2_reviewed_diameter_px": pd.to_numeric(
            frame.get("v2_instance_diameter_px", pd.Series(0.0, index=frame.index)), errors="coerce"
        ).fillna(0.0),
    }
    # A freshly created review CSV contains empty strings in these fields.
    # ``read_csv`` infers an all-empty column as float64, so a later reviewer
    # name/timestamp/note assignment would fail with Pandas' strict upcast
    # check.  Normalize existing columns as well as newly created ones.
    for name, default in text_defaults.items():
        if name not in frame.columns:
            frame[name] = default
        else:
            frame[name] = frame[name].fillna("").astype(object)
    for name, default in numeric_defaults.items():
        if name not in frame.columns:
            frame[name] = default
    return frame


def save_mask_review(
    config: dict[str, Any],
    database: str | Path,
    *,
    round_id: str,
    candidate_id: str,
    decision: str,
    reviewed_mask_rle: str | None,
    reviewer: str = "local_user",
    notes: str = "",
) -> dict[str, Any]:
    import sqlite3

    manifest = load_mask_review_manifest(config, round_id)
    mask_size = int(manifest.get("mask_size", MASK_SIZE))
    if mask_size != MASK_SIZE:
        raise ValueError(f"unsupported mask size: {mask_size}")
    frame = _read_round_frame(config, round_id)
    selected = frame[frame["candidate_id"].astype(str) == str(candidate_id)]
    if selected.empty:
        raise ValueError(f"candidate is not in mask review round: {candidate_id}")
    row = selected.iloc[0]
    model_rle, model_mask = _decode_valid_rle(str(row.get("v2_mask_rle", "[]")), mask_size)
    normalized_decision = str(decision).strip().lower()
    if normalized_decision in {"accepted", "accept", "keep_original"}:
        normalized_decision = "accepted"
        reviewed_rle, reviewed_mask = model_rle, model_mask
    elif normalized_decision == "rejected":
        reviewed_rle, reviewed_mask = _rle(np.zeros((mask_size, mask_size), dtype=bool)), np.zeros(
            (mask_size, mask_size), dtype=bool
        )
    elif normalized_decision == "edited":
        reviewed_rle, reviewed_mask = _decode_valid_rle(reviewed_mask_rle, mask_size)
        if not reviewed_mask.any():
            raise ValueError("an edited review must contain at least one mask pixel")
    else:
        raise ValueError("decision must be accepted, edited, or rejected")

    origin_x = int(round(_number(row.get("v2_mask_origin_x"))))
    origin_y = int(round(_number(row.get("v2_mask_origin_y"))))
    reviewed_area = int(reviewed_mask.sum())
    reviewed_diameter = float(2.0 * np.sqrt(reviewed_area / np.pi)) if reviewed_area else 0.0
    contour = _contour(reviewed_mask, origin_x, origin_y) if reviewed_area else "[]"
    timestamp = _now_text()
    database_path = Path(database)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    reviewed_path = _manifest_file(config, round_id, "reviewed_predictions")
    reviewed_frame = _ensure_review_columns(pd.read_csv(reviewed_path, low_memory=False))
    row_indices = reviewed_frame.index[
        reviewed_frame["candidate_id"].astype(str) == str(candidate_id)
    ]
    if row_indices.empty:
        raise ValueError(f"candidate is missing from reviewed predictions: {candidate_id}")
    reviewed_frame.loc[row_indices, "v2_mask_rle"] = reviewed_rle
    reviewed_frame.loc[row_indices, "v2_mask_valid"] = bool(reviewed_area > 0)
    reviewed_frame.loc[row_indices, "v2_contour_json"] = contour
    reviewed_frame.loc[row_indices, "v2_instance_area_px"] = reviewed_area
    reviewed_frame.loc[row_indices, "v2_instance_diameter_px"] = reviewed_diameter
    reviewed_frame.loc[row_indices, "v2_mask_review_status"] = normalized_decision
    reviewed_frame.loc[row_indices, "v2_mask_reviewed_by"] = str(reviewer or "local_user")
    reviewed_frame.loc[row_indices, "v2_mask_reviewed_at"] = timestamp
    reviewed_frame.loc[row_indices, "v2_mask_review_notes"] = str(notes or "")
    reviewed_frame.loc[row_indices, "v2_reviewed_area_px"] = reviewed_area
    reviewed_frame.loc[row_indices, "v2_reviewed_diameter_px"] = reviewed_diameter
    # Recompute overlap ownership and all downstream instance eligibility flags
    # before either persistence target is committed.
    reviewed_frame = finalize_v2_instances(reviewed_frame, mask_size, set())
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO v2_mask_reviews (
                round_id, candidate_id, well, timepoint,
                model_mask_rle, reviewed_mask_rle, decision,
                model_area_px, reviewed_area_px,
                model_diameter_px, reviewed_diameter_px,
                contour_json, reviewer, notes, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(round_id, candidate_id) DO UPDATE SET
                well=excluded.well,
                timepoint=excluded.timepoint,
                model_mask_rle=excluded.model_mask_rle,
                reviewed_mask_rle=excluded.reviewed_mask_rle,
                decision=excluded.decision,
                model_area_px=excluded.model_area_px,
                reviewed_area_px=excluded.reviewed_area_px,
                model_diameter_px=excluded.model_diameter_px,
                reviewed_diameter_px=excluded.reviewed_diameter_px,
                contour_json=excluded.contour_json,
                reviewer=excluded.reviewer,
                notes=excluded.notes,
                updated_at=excluded.updated_at
            """,
            (
                str(round_id),
                str(candidate_id),
                str(row.get("well", "")),
                str(row.get("timepoint", "")),
                model_rle,
                reviewed_rle,
                normalized_decision,
                int(model_mask.sum()),
                reviewed_area,
                float(2.0 * np.sqrt(model_mask.sum() / np.pi)) if model_mask.any() else 0.0,
                reviewed_diameter,
                contour,
                str(reviewer or "local_user"),
                str(notes or ""),
                timestamp,
            ),
        )
        connection.commit()

    _atomic_write_csv(reviewed_frame, reviewed_path)
    return {
        "status": "saved",
        "round_id": str(round_id),
        "candidate_id": str(candidate_id),
        "decision": normalized_decision,
        "model_area_px": int(model_mask.sum()),
        "reviewed_area_px": reviewed_area,
        "reviewed_diameter_px": reviewed_diameter,
        "contour_json": contour,
        "reviewed_predictions": str(reviewed_path),
        "updated_at": timestamp,
    }


def create_mask_review_round(
    config: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    round_id: str | None = None,
) -> dict[str, Any]:
    """Run P0 inference into a new review round and initialize its CSV."""

    if round_id is None:
        round_id = ROUND_PREFIX + datetime.now().strftime("%Y%m%d-%H%M%S")
    round_dir = round_directory(config, round_id)
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
    else:
        checkpoint = artifact_path(config, "v2", "models", "latest_instance_segmenter.pt")
        if not checkpoint.exists():
            shared_root = config.get("late_growth", {}).get("shared_model_root")
            if shared_root:
                checkpoint = Path(str(shared_root)) / "v2" / "models" / "latest_instance_segmenter.pt"
    source = Path(predictions_path) if predictions_path is not None else artifact_path(
        config, "predictions", "latest_integrated_predictions.csv"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"V2 checkpoint is missing: {checkpoint}")
    if not source.exists():
        raise FileNotFoundError(f"source predictions are missing: {source}")
    output = round_dir / "latest_v2_predictions.csv"
    pre_temporal = round_dir / "latest_v2_pre_temporal_predictions.csv"
    summary_path = round_dir / "inference_summary.json"
    infer_v2_instances(
        config,
        checkpoint,
        predictions_path=source,
        output_path=output,
        pre_temporal_output_path=pre_temporal,
        summary_path=summary_path,
    )
    frame = _ensure_review_columns(pd.read_csv(output, low_memory=False))
    reviewed = round_dir / "reviewed_v2_predictions.csv"
    _atomic_write_csv(frame, reviewed)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = {
        "round_id": str(round_id),
        "created_at": _now_text(),
        "algorithm_version": "p0-mask-review-20260812",
        "mask_size": MASK_SIZE,
        "checkpoint": str(checkpoint.resolve()),
        "source_predictions": str(source.resolve()),
        "p0_predictions": str(output.resolve()),
        "pre_temporal_predictions": str(pre_temporal.resolve()),
        "reviewed_predictions": str(reviewed.resolve()),
        "inference_summary": str(summary_path.resolve()),
        "candidate_count": int(len(frame)),
        "valid_instance_count": int(
            frame.get("v2_mask_valid", pd.Series(False, index=frame.index)).map(_bool_value).sum()
        ),
        "contour_refinement_status_counts": summary.get("contour_refinement_status_counts", {}),
    }
    _atomic_write_json(manifest, round_dir / "manifest.json")
    return {**manifest, "inference_summary_payload": summary}

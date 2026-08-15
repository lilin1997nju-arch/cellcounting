"""Activate the pooled QL2603 classifier heads with a recoverable backup.

The classifier checkpoints are trained on the pooled QL2603/legacy review set,
while the V2 instance contours and temporal evidence already published for the
project are retained.  Each plate is rebuilt in this order:

1. write new morphology and multiplicity head predictions;
2. regenerate the auto/integrated base round and carry human decisions;
3. merge that base round into the existing V2 pre-temporal table;
4. rerun the existing V2/V3 temporal policy and publish it as both the V2
   source used by the categorized review page and the current V3 source used by
   the project review page;
5. rebuild well screening and the gated plate report.

All active files and annotation databases touched by the operation are copied
to a timestamped backup directory before the first plate is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cellvision.config import artifact_path, load_config
from cellvision.gated_screening import build_gated_plate_report
from cellvision.model_inference import (
    predict_multiplicity_checkpoint,
    predict_teaching_checkpoint,
)
from cellvision.multiplicity import generate_integrated_training_round
from cellvision.teaching import generate_auto_annotation_round
from cellvision.v2_temporal_inference import infer_v2_temporal_evidence
from cellvision.well_screening import build_well_screening


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "projects" / "ql2603" / "project.json"
MODEL_ROOT = (
    ROOT
    / "artifacts"
    / "projects"
    / "ql2603"
    / "plates"
    / "ql2603-t1-1"
    / "models"
)
BACKUP_ROOT = ROOT / "artifacts" / "v2" / "runs" / "ql2603_classifier_activation"

# These are the active files changed by the refresh.  Round directories are
# append-only audit artifacts and do not need to be copied for rollback.
TOUCHED_FILES = (
    "annotations/annotations.db",
    "annotations/auto_review_queue.csv",
    "predictions/teaching_classifier_predictions.csv",
    "predictions/multiplicity_predictions.csv",
    "predictions/latest_auto_annotations.csv",
    "predictions/latest_auto_annotation.json",
    "predictions/latest_temporal_classifications.csv",
    "predictions/latest_integrated_predictions.csv",
    "predictions/latest_integrated_summary.json",
    "predictions/latest_v2_pre_temporal_predictions.csv",
    "predictions/latest_v2_predictions.csv",
    "predictions/latest_v2_temporal_summary.json",
    "predictions/latest_v3_predictions.csv",
    "predictions/latest_v3_temporal_summary.json",
    "predictions/latest_well_screening.csv",
    "predictions/latest_screening_objects.csv",
    "predictions/latest_well_screening_summary.json",
    "predictions/v3_activation.json",
    "predictions/classifier_activation.json",
    "cache/quick_review_summary.json",
)

# These fields describe candidate geometry/ownership created by the existing
# V2 stage.  The new classifier round should not replace them with the simpler
# legacy integrated table, otherwise the pale contour UI would lose masks or
# duplicate suppression state.
PRESERVE_STRUCTURAL_COLUMNS = frozenset(
    {
        "is_duplicate_suppressed",
        "duplicate_of_candidate_id",
        "duplicate_suppression_reason",
        "is_hierarchy_suppressed",
        "suppressed_by_candidate_id",
        "hierarchy_suppression_reason",
        "parent_candidate_id",
        "is_counting_instance",
        "instance_component_id",
        "instance_component_distance_px",
        "instance_footprint_diameter_px",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.classifier-refresh.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.classifier-refresh.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.classifier-refresh.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _port_project_snapshot() -> dict[str, Any] | None:
    """Read the live 8777 plate progress when the current server is available."""

    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8777/api/project", timeout=5
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None


def _plate_progress(snapshot: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not snapshot:
        return {}
    return {
        str(item.get("slug")): item
        for item in snapshot.get("plates", [])
        if isinstance(item, dict) and item.get("slug")
    }


def _backup_active_files(plates: list[dict[str, Any]], backup_root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for plate in plates:
        slug = str(plate["slug"])
        root = Path(str(plate["artifact_root"]))
        plate_backup = backup_root / slug
        plate_backup.mkdir(parents=True, exist_ok=True)
        files: dict[str, Any] = {}
        for relative in TOUCHED_FILES:
            source = root / relative
            target = plate_backup / relative
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                files[relative] = {
                    "existed": True,
                    "backup": str(target.resolve()),
                    "sha256": _sha256(source),
                    "size": int(source.stat().st_size),
                }
            else:
                files[relative] = {"existed": False}
        records[slug] = files
    return records


def _restore_active_files(
    plates: list[dict[str, Any]], backup_root: Path, records: dict[str, Any]
) -> None:
    for plate in plates:
        slug = str(plate["slug"])
        root = Path(str(plate["artifact_root"]))
        for relative, record in records.get(slug, {}).items():
            target = root / relative
            backup = backup_root / slug / relative
            if record.get("existed") and backup.exists():
                _atomic_copy(backup, target)
            elif not record.get("existed"):
                target.unlink(missing_ok=True)


def _merge_classifier_round(
    template_path: Path,
    new_base: pd.DataFrame,
    *,
    set_pre_temporal_label: bool,
) -> pd.DataFrame:
    """Overlay new classifier/base labels while retaining V2/V3 evidence."""

    base = new_base.copy()
    if "candidate_id" not in base.columns:
        raise ValueError(f"New integrated table has no candidate_id: {template_path}")
    base["candidate_id"] = base["candidate_id"].astype(str)
    base = base.drop_duplicates("candidate_id", keep="last").set_index(
        "candidate_id", drop=False
    )
    if not template_path.exists():
        return base.reset_index(drop=True)

    template = pd.read_csv(template_path, low_memory=False)
    if "candidate_id" not in template.columns:
        raise ValueError(f"Existing V2 table has no candidate_id: {template_path}")
    template["candidate_id"] = template["candidate_id"].astype(str)
    template = template.drop_duplicates("candidate_id", keep="last").set_index(
        "candidate_id", drop=False
    )
    ordered_ids = list(template.index) + [
        candidate_id for candidate_id in base.index if candidate_id not in template.index
    ]
    merged = template.reindex(ordered_ids)

    override_columns = [
        column
        for column in base.columns
        if column != "candidate_id"
        and not column.startswith("v2_")
        and not column.startswith("v3_")
        and column not in PRESERVE_STRUCTURAL_COLUMNS
    ]
    common_ids = base.index.intersection(merged.index)
    for column in override_columns:
        if column not in merged.columns:
            merged[column] = pd.NA
        merged.loc[common_ids, column] = base.loc[common_ids, column]

    if set_pre_temporal_label and "integrated_label" in base.columns:
        for column in ("v2_pre_temporal_integrated_label", "v2_original_integrated_label"):
            if column in merged.columns:
                merged.loc[common_ids, column] = base.loc[common_ids, "integrated_label"]

    return merged.reset_index(drop=True)


def _refresh_gated_report(config: dict[str, Any], database: Path) -> dict[str, Any] | None:
    settings = config.get("gated_report", {})
    endpoint_csv = settings.get("endpoint_csv") or settings.get("day14_csv")
    if not endpoint_csv or not settings.get("group_id") or not settings.get("output_dir"):
        return None
    endpoint_timepoint = str(settings.get("endpoint_timepoint", "T4")).upper()
    endpoint_day_label = str(settings.get("endpoint_day_label", "Day14"))
    with sqlite3.connect(database) as connection:
        try:
            late_rows = connection.execute(
                "SELECT well, decision FROM late_growth_reviews WHERE timepoint = ?",
                (endpoint_timepoint,),
            ).fetchall()
        except sqlite3.OperationalError:
            late_rows = []
    return build_gated_plate_report(
        endpoint_csv,
        settings["group_id"],
        settings["output_dir"],
        early_screening_csv=artifact_path(
            config, "predictions", "latest_well_screening.csv"
        ),
        sessions_csv=settings.get("sessions_csv"),
        locate_day7=False,
        day14_growth_overrides={
            str(well).upper(): str(decision) for well, decision in late_rows
        },
        endpoint_day_label=endpoint_day_label,
    )


def _load_plates() -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    plates = [item for item in manifest.get("plates", []) if isinstance(item, dict)]
    if not plates:
        raise RuntimeError(f"No plates found in {MANIFEST}")
    return plates


def main() -> None:
    global MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--backup-root", default="")
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument(
        "--temporal-only-incomplete",
        action="store_true",
        help="Refresh V2/V3 temporal evidence only for plates still pending review.",
    )
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    MANIFEST = Path(args.manifest).expanduser().resolve()
    plates = _load_plates()
    if args.only:
        wanted = {str(value) for value in args.only}
        plates = [plate for plate in plates if str(plate["slug"]) in wanted]
        missing = wanted - {str(plate["slug"]) for plate in plates}
        if missing:
            raise RuntimeError(f"Unknown plate slug(s): {sorted(missing)}")
    if not plates:
        raise RuntimeError("No plates selected")

    teaching_checkpoint = MODEL_ROOT / "teaching_classifier.pt"
    multiplicity_checkpoint = MODEL_ROOT / "multiplicity_classifier.pt"
    if not teaching_checkpoint.exists() or not multiplicity_checkpoint.exists():
        raise FileNotFoundError("The new pooled classifier checkpoints are missing")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_root = (
        Path(args.backup_root).expanduser().resolve()
        if args.backup_root
        else BACKUP_ROOT / timestamp
    )
    backup_root.mkdir(parents=True, exist_ok=False)
    before_port = _port_project_snapshot()
    backup_records = _backup_active_files(plates, backup_root)
    _atomic_json(
        backup_root / "backup_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(MANIFEST),
            "plates": [str(plate["slug"]) for plate in plates],
            "files": backup_records,
            "port_snapshot_before": before_port,
        },
    )

    progress_before = _plate_progress(before_port)
    incomplete_before = sorted(
        slug
        for slug, plate in progress_before.items()
        if not bool(plate.get("review_complete"))
    )
    temporal_slugs = {
        str(plate["slug"])
        for plate in plates
        if not args.temporal_only_incomplete or str(plate["slug"]) in set(incomplete_before)
    }
    print(
        json.dumps(
            {
                "selected_plates": [str(plate["slug"]) for plate in plates],
                "incomplete_before": incomplete_before,
                "backup_root": str(backup_root),
                "temporal_checkpoint": None,
                "temporal_plates": sorted(temporal_slugs),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    results: list[dict[str, Any]] = []
    try:
        for index, plate in enumerate(plates, start=1):
            slug = str(plate["slug"])
            config = load_config(Path(str(plate["config"])))
            database = artifact_path(config, "annotations", "annotations.db")
            prediction_root = artifact_path(config, "predictions")
            print(f"[{index}/{len(plates)}] {slug}: classifier heads", flush=True)
            teaching = predict_teaching_checkpoint(config, teaching_checkpoint)
            multiplicity = predict_multiplicity_checkpoint(config, multiplicity_checkpoint)
            auto_round = generate_auto_annotation_round(config, database)
            integrated_round = generate_integrated_training_round(config, database)
            integrated_path = artifact_path(
                config, "predictions", "latest_integrated_predictions.csv"
            )
            new_base = pd.read_csv(integrated_path, low_memory=False)

            pre_temporal_path = prediction_root / "latest_v2_pre_temporal_predictions.csv"
            merged_pre_temporal = _merge_classifier_round(
                pre_temporal_path,
                new_base,
                set_pre_temporal_label=True,
            )
            _atomic_csv(merged_pre_temporal, pre_temporal_path)

            temporal_summary: dict[str, Any] | None = None
            if not args.no_temporal and slug in temporal_slugs:
                print(f"[{index}/{len(plates)}] {slug}: V2/V3 temporal refresh", flush=True)
                infer_v2_temporal_evidence(config)
                summary_path = prediction_root / "latest_v2_temporal_summary.json"
                if summary_path.exists():
                    temporal_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                latest_v2 = prediction_root / "latest_v2_predictions.csv"
                latest_v3 = prediction_root / "latest_v3_predictions.csv"
                if latest_v2.exists():
                    _atomic_copy(latest_v2, latest_v3)
                    v2_summary = prediction_root / "latest_v2_temporal_summary.json"
                    v3_summary = prediction_root / "latest_v3_temporal_summary.json"
                    if v2_summary.exists():
                        _atomic_copy(v2_summary, v3_summary)
            else:
                latest_v2 = prediction_root / "latest_v2_predictions.csv"
                merged_v2 = _merge_classifier_round(
                    latest_v2, new_base, set_pre_temporal_label=False
                )
                _atomic_csv(merged_v2, latest_v2)
                if latest_v2.exists():
                    _atomic_copy(latest_v2, prediction_root / "latest_v3_predictions.csv")

            screening = build_well_screening(config, database)
            gated = _refresh_gated_report(config, database)
            activation = {
                "activated_at": datetime.now(timezone.utc).isoformat(),
                "plate": slug,
                "new_teaching_checkpoint": str(teaching_checkpoint.resolve()),
                "new_multiplicity_checkpoint": str(multiplicity_checkpoint.resolve()),
                "temporal_checkpoint": None,
                "training_head_prediction_counts": {
                    "teaching": int(teaching.get("prediction_count", 0)),
                    "multiplicity": int(multiplicity.get("prediction_count", 0)),
                },
                "auto_round": auto_round,
                "integrated_round": integrated_round,
                "temporal_summary": temporal_summary,
                "screening": screening,
                "gated_report_refreshed": gated is not None,
                "review_complete_before": progress_before.get(slug, {}).get("review_complete"),
            }
            _atomic_json(prediction_root / "classifier_activation.json", activation)
            results.append(activation)
            print(
                json.dumps(
                    {
                        "plate": slug,
                        "auto_candidates": auto_round.get("candidate_count"),
                        "integrated_candidates": integrated_round.get("candidate_count"),
                        "carried_reviews": integrated_round.get("carried_review_count"),
                        "screening_wells": screening.get("well_count"),
                        "gated_report_refreshed": gated is not None,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    except Exception:
        _restore_active_files(plates, backup_root, backup_records)
        raise

    after_port = _port_project_snapshot()
    report = {
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "project": "ql2603",
        "manifest": str(MANIFEST),
        "backup_root": str(backup_root),
        "new_teaching_checkpoint": str(teaching_checkpoint.resolve()),
        "new_multiplicity_checkpoint": str(multiplicity_checkpoint.resolve()),
        "temporal_checkpoint": None,
        "temporal_refresh": not args.no_temporal,
        "temporal_plates": sorted(temporal_slugs),
        "plates_processed": [str(plate["slug"]) for plate in plates],
        "incomplete_before": incomplete_before,
        "port_snapshot_before": before_port,
        "port_snapshot_after": after_port,
        "plates": results,
    }
    _atomic_json(backup_root / "activation_report.json", report)
    print(str((backup_root / "activation_report.json").resolve()), flush=True)


if __name__ == "__main__":
    main()

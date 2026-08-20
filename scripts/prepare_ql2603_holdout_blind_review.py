"""Prepare unreviewed old-vs-new single/doublet disagreements on QL2603 plates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cellvision.config import artifact_path, load_config
from cellvision.model_inference import predict_multiplicity_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLATES = ("ql2603-t1-2", "ql2603-t4-2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predict_isolated(config: dict, checkpoint: Path, output: Path) -> None:
    active = artifact_path(config, "predictions", "multiplicity_predictions.csv")
    backup = active.with_name(f".{active.name}.blind-review-backup")
    existed = active.exists()
    if existed:
        shutil.copy2(active, backup)
    try:
        predict_multiplicity_checkpoint(config, checkpoint)
        shutil.copy2(active, output)
    finally:
        if existed:
            shutil.move(backup, active)
        else:
            active.unlink(missing_ok=True)


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default)
    return values.fillna(default).astype(str).str.lower().isin({"1", "true", "yes"})


def _reviewed_ids(database: Path) -> set[str]:
    reviewed: set[str] = set()
    with sqlite3.connect(database) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for table in ("integrated_training_reviews", "multiplicity_labels"):
            if table in tables:
                reviewed.update(
                    str(row[0])
                    for row in connection.execute(
                        f"SELECT DISTINCT candidate_id FROM {table}"
                    )
                )
    return reviewed


def _select_disagreements(
    slug: str, config: dict, old_path: Path, new_path: Path
) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "single_probability",
        "touching_doublet_probability",
        "cluster_3plus_probability",
        "predicted_multiplicity",
        "multiplicity_confidence",
    ]
    old = pd.read_csv(old_path, usecols=columns, low_memory=False).rename(
        columns={column: f"old_{column}" for column in columns if column != "candidate_id"}
    )
    new = pd.read_csv(new_path, usecols=columns, low_memory=False).rename(
        columns={column: f"new_{column}" for column in columns if column != "candidate_id"}
    )
    v2_path = artifact_path(config, "predictions", "latest_v2_predictions.csv")
    source_path = (
        v2_path
        if v2_path.exists()
        else artifact_path(config, "predictions", "latest_integrated_predictions.csv")
    )
    source = pd.read_csv(source_path, low_memory=False).drop_duplicates(
        "candidate_id", keep="last"
    )
    merged = source.merge(old, on="candidate_id").merge(new, on="candidate_id")
    pair = {"single", "touching_doublet"}
    reviewed = _reviewed_ids(
        artifact_path(config, "annotations", "annotations.db")
    )
    keep = (
        merged.old_predicted_multiplicity.astype(str).isin(pair)
        & merged.new_predicted_multiplicity.astype(str).isin(pair)
        & merged.old_predicted_multiplicity.astype(str).ne(
            merged.new_predicted_multiplicity.astype(str)
        )
        & pd.to_numeric(
            merged.get("cell_probability", 0.0), errors="coerce"
        ).fillna(0.0).ge(0.50)
        & pd.to_numeric(
            merged.get("invalid_probability", 1.0), errors="coerce"
        ).fillna(1.0).le(0.50)
        & pd.to_numeric(
            merged.get("debris_probability", 1.0), errors="coerce"
        ).fillna(1.0).le(0.50)
        & merged.timepoint.astype(str).isin({"T0", "T1", "T2"})
        & ~merged.candidate_id.astype(str).isin(reviewed)
        & ~_bool_series(merged, "is_hierarchy_suppressed")
        & ~_bool_series(merged, "is_duplicate_suppressed")
        & ~_bool_series(merged, "v2_wall_rejected")
    )
    if "v2_mask_valid" in merged:
        keep &= _bool_series(merged, "v2_mask_valid")
    selected = merged[keep].copy()
    selected.insert(0, "plate", slug)
    selected["disagreement_strength"] = (
        pd.to_numeric(
            selected.old_multiplicity_confidence, errors="coerce"
        ).fillna(0.0)
        + pd.to_numeric(
            selected.new_multiplicity_confidence, errors="coerce"
        ).fillna(0.0)
    ) / 2.0
    selected = selected.sort_values(
        ["disagreement_strength", "well", "timepoint"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    selected["blind_case_id"] = [
        f"{slug}-{index:04d}" for index in range(1, len(selected) + 1)
    ]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plates", nargs="+", default=list(DEFAULT_PLATES))
    args = parser.parse_args()
    old_checkpoint = Path(args.old_checkpoint).expanduser().resolve()
    new_checkpoint = Path(args.new_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases = []
    summaries = {}
    plates = tuple(str(value) for value in args.plates)
    for slug in plates:
        config_path = ROOT / "configs" / "generated" / f"{slug}.yaml"
        config = load_config(config_path)
        plate_dir = output_dir / slug
        plate_dir.mkdir(parents=True, exist_ok=True)
        old_path = plate_dir / "old_multiplicity_predictions.csv"
        new_path = plate_dir / "new_multiplicity_predictions.csv"
        _predict_isolated(config, old_checkpoint, old_path)
        print(f"{slug}: old checkpoint complete", flush=True)
        _predict_isolated(config, new_checkpoint, new_path)
        print(f"{slug}: new checkpoint complete", flush=True)
        cases = _select_disagreements(slug, config, old_path, new_path)
        cases.to_csv(
            plate_dir / "unreviewed_single_doublet_disagreements.csv",
            index=False,
            encoding="utf-8",
        )
        all_cases.append(cases)
        summaries[slug] = {
            "unreviewed_single_doublet_disagreement_count": int(len(cases)),
            "old_single_new_doublet": int(
                (
                    cases.old_predicted_multiplicity.eq("single")
                    & cases.new_predicted_multiplicity.eq("touching_doublet")
                ).sum()
            ),
            "old_doublet_new_single": int(
                (
                    cases.old_predicted_multiplicity.eq("touching_doublet")
                    & cases.new_predicted_multiplicity.eq("single")
                ).sum()
            ),
        }
    combined = pd.concat(all_cases, ignore_index=True)
    combined.to_csv(
        output_dir / "blind_review_cases.csv", index=False, encoding="utf-8"
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "blind review of unreviewed old/new single-doublet disagreements",
        "plates": list(plates),
        "training_overlap": (
            "all reviewed/training candidate IDs are excluded from this blind batch; "
            "plate-level exposure is reported separately"
        ),
        "already_reviewed_candidates_excluded": True,
        "old_checkpoint": {
            "path": str(old_checkpoint),
            "sha256": _sha256(old_checkpoint),
        },
        "new_checkpoint": {
            "path": str(new_checkpoint),
            "sha256": _sha256(new_checkpoint),
        },
        "plate_summary": summaries,
        "total_cases": int(len(combined)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

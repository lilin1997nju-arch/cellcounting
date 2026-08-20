"""Compare two multiplicity checkpoints on frozen, human-reviewed QL2603 boards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from cellvision.config import artifact_path, load_config
from cellvision.model_inference import predict_multiplicity_checkpoint
from cellvision.multiplicity import MULTIPLICITY_CLASSES, read_multiplicity_labels


ROOT = Path(__file__).resolve().parents[1]
HOLDOUTS = ("ql2603-t1-2", "ql2603-t4-2")
LABELS = tuple(MULTIPLICITY_CLASSES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_truth(database: Path) -> pd.DataFrame:
    labels: dict[str, str] = {}
    with sqlite3.connect(database) as connection:
        integrated = pd.read_sql_query(
            """
            SELECT candidate_id, reviewed_label, updated_at,
                   integrated_review_id
            FROM integrated_training_reviews
            WHERE reviewed_label IN (
              'single', 'touching_doublet', 'cluster_3plus'
            )
            ORDER BY updated_at, integrated_review_id
            """,
            connection,
        )
    for row in integrated.itertuples(index=False):
        labels[str(row.candidate_id)] = str(row.reviewed_label)
    focused = read_multiplicity_labels(database)
    for row in focused.itertuples(index=False):
        if str(row.label) in LABELS:
            labels[str(row.candidate_id)] = str(row.label)
    return pd.DataFrame(
        [
            {"candidate_id": candidate_id, "truth": label}
            for candidate_id, label in labels.items()
        ]
    )


def _metrics(frame: pd.DataFrame, prediction: str) -> dict[str, Any]:
    confusion = {
        truth: {
            predicted: int(
                ((frame["truth"] == truth) & (frame[prediction] == predicted)).sum()
            )
            for predicted in LABELS
        }
        for truth in LABELS
    }
    per_class: dict[str, Any] = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[truth][label] for truth in LABELS if truth != label)
        false_negative = sum(confusion[label][other] for other in LABELS if other != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": int((frame["truth"] == label).sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return {
        "evaluated_count": int(len(frame)),
        "exact_accuracy": float((frame["truth"] == frame[prediction]).mean()),
        "macro_f1": float(sum(value["f1"] for value in per_class.values()) / len(LABELS)),
        "per_class": per_class,
        "confusion": confusion,
    }


def _predict(config: dict[str, Any], checkpoint: Path) -> pd.DataFrame:
    path = artifact_path(config, "predictions", "multiplicity_predictions.csv")
    backup = path.with_name(f".{path.name}.holdout-comparison-backup")
    existed = path.exists()
    if existed:
        shutil.copy2(path, backup)
    try:
        predict_multiplicity_checkpoint(config, checkpoint)
        return pd.read_csv(
            path,
            usecols=["candidate_id", "predicted_multiplicity"],
            low_memory=False,
        ).drop_duplicates("candidate_id", keep="last")
    finally:
        if existed:
            shutil.move(backup, path)
        else:
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    old_checkpoint = Path(args.old_checkpoint).expanduser().resolve()
    new_checkpoint = Path(args.new_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plate_frames: list[pd.DataFrame] = []
    plate_results: dict[str, Any] = {}
    for slug in HOLDOUTS:
        config = load_config(ROOT / "configs" / "generated" / f"{slug}.yaml")
        database = artifact_path(config, "annotations", "annotations.db")
        truth = _latest_truth(database)
        old = _predict(config, old_checkpoint).rename(
            columns={"predicted_multiplicity": "old"}
        )
        new = _predict(config, new_checkpoint).rename(
            columns={"predicted_multiplicity": "new"}
        )
        paired = truth.merge(old, on="candidate_id").merge(new, on="candidate_id")
        paired.insert(0, "plate", slug)
        paired.to_csv(output_dir / f"{slug}_comparison.csv", index=False, encoding="utf-8")
        plate_frames.append(paired)
        plate_results[slug] = {
            "old": _metrics(paired, "old"),
            "new": _metrics(paired, "new"),
        }

    combined = pd.concat(plate_frames, ignore_index=True)
    combined.to_csv(output_dir / "combined_comparison.csv", index=False, encoding="utf-8")
    report = {
        "evaluation": "latest human-reviewed cell candidates on frozen QL2603 boards",
        "holdouts": list(HOLDOUTS),
        "old_checkpoint": {"path": str(old_checkpoint), "sha256": _sha256(old_checkpoint)},
        "new_checkpoint": {"path": str(new_checkpoint), "sha256": _sha256(new_checkpoint)},
        "plates": plate_results,
        "combined": {
            "old": _metrics(combined, "old"),
            "new": _metrics(combined, "new"),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["combined"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

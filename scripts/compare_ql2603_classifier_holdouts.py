"""Compare the newly trained classifier heads on the frozen QL2603 boards."""

from __future__ import annotations

import json
import sqlite3
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from cellvision.config import artifact_path, load_config
from cellvision.model_inference import (
    predict_multiplicity_checkpoint,
    predict_teaching_checkpoint,
)
from cellvision.multiplicity import read_multiplicity_labels
from cellvision.teaching import read_teaching_labels


ROOT = Path(__file__).resolve().parents[1]
HOLDOUTS = ("ql2603-t1-2", "ql2603-t4-2")
MODEL_ROOT = (
    ROOT
    / "artifacts"
    / "projects"
    / "ql2603"
    / "plates"
    / "ql2603-t1-1"
    / "models"
)
OUTPUT_ROOT = ROOT / "artifacts" / "v2" / "runs" / "ql2603_classifier_holdouts"
CELL_LABELS = ("single", "touching_doublet", "cluster_3plus")
MORPHOLOGY_LABELS = ("invalid", "debris", "cell")


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _metrics(frame: pd.DataFrame, truth: str, prediction: str, labels: tuple[str, ...]) -> dict[str, Any]:
    frame = frame[[truth, prediction]].copy()
    frame[truth] = frame[truth].astype(str)
    frame[prediction] = frame[prediction].astype(str)
    confusion = {
        expected: {
            actual: int(((frame[truth] == expected) & (frame[prediction] == actual)).sum())
            for actual in labels
        }
        for expected in labels
    }
    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        true_positive = int(((frame[truth] == label) & (frame[prediction] == label)).sum())
        false_positive = int(((frame[truth] != label) & (frame[prediction] == label)).sum())
        false_negative = int(((frame[truth] == label) & (frame[prediction] != label)).sum())
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        per_class[label] = {
            "support": int((frame[truth] == label).sum()),
            "precision": precision,
            "recall": recall,
            "f1": _safe_divide(2 * precision * recall, precision + recall),
        }
    return {
        "evaluated_count": int(len(frame)),
        "exact_accuracy": float((frame[truth] == frame[prediction]).mean()) if len(frame) else 0.0,
        "macro_f1": float(sum(item["f1"] for item in per_class.values()) / len(labels)),
        "per_class": per_class,
        "confusion": confusion,
    }


def _load_prediction(path: Path, label_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    return frame[["candidate_id", label_column]].drop_duplicates(
        "candidate_id", keep="last"
    )


def _latest_integrated_reviews(database: str | Path) -> pd.DataFrame:
    with sqlite3.connect(database) as connection:
        reviews = pd.read_sql_query(
            """
            SELECT candidate_id, reviewed_label, decision, updated_at,
                   integrated_review_id
            FROM integrated_training_reviews
            ORDER BY updated_at, integrated_review_id
            """,
            connection,
        )
    return reviews.drop_duplicates("candidate_id", keep="last")


def _human_truth(database: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the latest integrated human review plus focused label overrides."""

    integrated = _latest_integrated_reviews(database)
    labels: dict[str, str] = {}
    if not integrated.empty:
        latest = integrated.drop_duplicates("candidate_id", keep="last")
        labels.update(
            {
                str(row.candidate_id): str(row.reviewed_label)
                for row in latest.itertuples(index=False)
            }
        )

    # Focused categorized reviews are more recent, explicit overrides of the
    # integrated round and should win when the same candidate appears in both.
    focused = read_multiplicity_labels(database)
    if not focused.empty:
        for row in focused.itertuples(index=False):
            label = str(row.label)
            if label in set(CELL_LABELS) | {"debris", "invalid"}:
                labels[str(row.candidate_id)] = label
    truth = pd.DataFrame(
        [{"candidate_id": candidate_id, "label": label} for candidate_id, label in labels.items()]
    )
    if truth.empty:
        empty = pd.DataFrame(columns=["candidate_id", "label"])
        return empty, empty
    multiplicity = truth[truth["label"].isin(CELL_LABELS)].copy()
    morphology_label = truth["label"].map(
        {
            "single": "cell",
            "touching_doublet": "cell",
            "cluster_3plus": "cell",
            "cell": "cell",
            "debris": "debris",
            "invalid": "invalid",
        }
    )
    morphology = truth.assign(label=morphology_label).dropna(subset=["label"])
    focused_morphology = read_teaching_labels(database)
    if not focused_morphology.empty:
        focused_morphology = focused_morphology[
            focused_morphology["label"].isin(MORPHOLOGY_LABELS)
        ][["candidate_id", "label"]].drop_duplicates(
            "candidate_id", keep="first"
        )
        morphology = pd.concat(
            [
                morphology[~morphology["candidate_id"].isin(focused_morphology["candidate_id"])],
                focused_morphology,
            ],
            ignore_index=True,
        )
    return multiplicity, morphology


def _run_one(slug: str) -> dict[str, Any]:
    config = load_config(ROOT / "configs" / "generated" / f"{slug}.yaml")
    database = artifact_path(config, "annotations", "annotations.db")
    prediction_root = artifact_path(config, "predictions")
    old_multiplicity_path = prediction_root / "multiplicity_predictions.csv"
    old_morphology_path = prediction_root / "teaching_classifier_predictions.csv"
    if not old_multiplicity_path.exists() or not old_morphology_path.exists():
        raise FileNotFoundError(f"Missing old predictions for {slug}")

    output_dir = OUTPUT_ROOT / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_multiplicity = output_dir / "old_multiplicity_predictions.csv"
    backup_morphology = output_dir / "old_teaching_classifier_predictions.csv"
    shutil.copy2(old_multiplicity_path, backup_multiplicity)
    shutil.copy2(old_morphology_path, backup_morphology)

    try:
        predict_multiplicity_checkpoint(
            config, MODEL_ROOT / "multiplicity_classifier.pt"
        )
        predict_teaching_checkpoint(
            config, MODEL_ROOT / "teaching_classifier.pt"
        )
        shutil.copy2(
            old_multiplicity_path,
            output_dir / "new_multiplicity_predictions.csv",
        )
        shutil.copy2(
            old_morphology_path,
            output_dir / "new_teaching_classifier_predictions.csv",
        )
        new_multiplicity = _load_prediction(
            old_multiplicity_path, "predicted_multiplicity"
        )
        new_morphology = _load_prediction(
            old_morphology_path, "predicted_label"
        )
    finally:
        # Holdout artifacts remain the pre-comparison state; only the copied
        # predictions under OUTPUT_ROOT are retained for auditability.
        shutil.copy2(backup_multiplicity, old_multiplicity_path)
        shutil.copy2(backup_morphology, old_morphology_path)

    old_multiplicity = _load_prediction(
        backup_multiplicity, "predicted_multiplicity"
    )
    old_morphology = _load_prediction(backup_morphology, "predicted_label")

    multiplicity_truth, morphology_truth = _human_truth(database)
    corrected_ids = set(
        _latest_integrated_reviews(database)
        .loc[lambda frame: frame["decision"].astype(str).eq("corrected"), "candidate_id"]
        .astype(str)
    )
    multiplicity_rows = (
        multiplicity_truth.rename(columns={"label": "truth"})
        .merge(old_multiplicity.rename(columns={"predicted_multiplicity": "old"}), on="candidate_id")
        .merge(new_multiplicity.rename(columns={"predicted_multiplicity": "new"}), on="candidate_id")
    )
    multiplicity_rows.to_csv(
        output_dir / "multiplicity_comparison.csv", index=False, encoding="utf-8"
    )

    morphology_rows = (
        morphology_truth.rename(columns={"label": "truth"})
        .merge(old_morphology.rename(columns={"predicted_label": "old"}), on="candidate_id")
        .merge(new_morphology.rename(columns={"predicted_label": "new"}), on="candidate_id")
    )
    morphology_rows.to_csv(
        output_dir / "morphology_comparison.csv", index=False, encoding="utf-8"
    )

    corrected_multiplicity = multiplicity_rows[
        multiplicity_rows["candidate_id"].astype(str).isin(corrected_ids)
    ]
    corrected_morphology = morphology_rows[
        morphology_rows["candidate_id"].astype(str).isin(corrected_ids)
    ]

    return {
        "plate": slug,
        "truth_counts": {
            "multiplicity": multiplicity_truth["label"].value_counts().to_dict(),
            "morphology": morphology_truth["label"].value_counts().to_dict(),
        },
        "multiplicity": {
            "old": _metrics(multiplicity_rows, "truth", "old", CELL_LABELS),
            "new": _metrics(multiplicity_rows, "truth", "new", CELL_LABELS),
            "corrected_only": {
                "old": _metrics(corrected_multiplicity, "truth", "old", CELL_LABELS),
                "new": _metrics(corrected_multiplicity, "truth", "new", CELL_LABELS),
                "paired_count": int(len(corrected_multiplicity)),
            },
            "paired_count": int(len(multiplicity_rows)),
        },
        "morphology": {
            "old": _metrics(morphology_rows, "truth", "old", MORPHOLOGY_LABELS),
            "new": _metrics(morphology_rows, "truth", "new", MORPHOLOGY_LABELS),
            "corrected_only": {
                "old": _metrics(corrected_morphology, "truth", "old", MORPHOLOGY_LABELS),
                "new": _metrics(corrected_morphology, "truth", "new", MORPHOLOGY_LABELS),
                "paired_count": int(len(corrected_morphology)),
            },
            "paired_count": int(len(morphology_rows)),
        },
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {slug: _run_one(slug) for slug in HOLDOUTS}
    combined: dict[str, Any] = {}
    for head, filename, labels in (
        ("multiplicity", "multiplicity_comparison.csv", CELL_LABELS),
        ("morphology", "morphology_comparison.csv", MORPHOLOGY_LABELS),
    ):
        frames = [
            pd.read_csv(OUTPUT_ROOT / slug / filename)
            for slug in HOLDOUTS
        ]
        paired = pd.concat(frames, ignore_index=True)
        combined[head] = {
            "old": _metrics(paired, "truth", "old", labels),
            "new": _metrics(paired, "truth", "new", labels),
            "paired_count": int(len(paired)),
        }
        corrected_frames = []
        for slug, frame in zip(HOLDOUTS, frames):
            config = load_config(ROOT / "configs" / "generated" / f"{slug}.yaml")
            database = artifact_path(config, "annotations", "annotations.db")
            corrected_ids = set(
                _latest_integrated_reviews(database)
                .loc[
                    lambda reviews: reviews["decision"].astype(str).eq("corrected"),
                    "candidate_id",
                ]
                .astype(str)
            )
            corrected_frames.append(
                frame[frame["candidate_id"].astype(str).isin(corrected_ids)]
            )
        corrected_paired = pd.concat(corrected_frames, ignore_index=True)
        combined[head]["corrected_only"] = {
            "old": _metrics(corrected_paired, "truth", "old", labels),
            "new": _metrics(corrected_paired, "truth", "new", labels),
            "paired_count": int(len(corrected_paired)),
        }
    report = {
        "evaluation": "human-labeled candidates on frozen QL2603 boards",
        "holdouts": list(HOLDOUTS),
        "new_model": {
            "teaching_checkpoint": str(MODEL_ROOT / "teaching_classifier.pt"),
            "multiplicity_checkpoint": str(MODEL_ROOT / "multiplicity_classifier.pt"),
        },
        "results": results,
        "combined": combined,
    }
    output_path = OUTPUT_ROOT / "metrics.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for slug, result in results.items():
        print(slug)
        for head in ("multiplicity", "morphology"):
            old = result[head]["old"]
            new = result[head]["new"]
            print(
                head,
                f"paired={result[head]['paired_count']}",
                f"old_acc={old['exact_accuracy']:.4f}",
                f"new_acc={new['exact_accuracy']:.4f}",
                f"delta={new['exact_accuracy'] - old['exact_accuracy']:+.4f}",
                f"old_macro_f1={old['macro_f1']:.4f}",
                f"new_macro_f1={new['macro_f1']:.4f}",
            )
    for head, result in combined.items():
        print(
            "combined",
            head,
            f"paired={result['paired_count']}",
            f"old_acc={result['old']['exact_accuracy']:.4f}",
            f"new_acc={result['new']['exact_accuracy']:.4f}",
            f"delta={result['new']['exact_accuracy'] - result['old']['exact_accuracy']:+.4f}",
            f"old_macro_f1={result['old']['macro_f1']:.4f}",
            f"new_macro_f1={result['new']['macro_f1']:.4f}",
        )
        corrected = result["corrected_only"]
        print(
            "corrected combined",
            head,
            f"paired={corrected['paired_count']}",
            f"old_acc={corrected['old']['exact_accuracy']:.4f}",
            f"new_acc={corrected['new']['exact_accuracy']:.4f}",
            f"delta={corrected['new']['exact_accuracy'] - corrected['old']['exact_accuracy']:+.4f}",
            f"old_macro_f1={corrected['old']['macro_f1']:.4f}",
            f"new_macro_f1={corrected['new']['macro_f1']:.4f}",
        )
    print(output_path)


if __name__ == "__main__":
    main()

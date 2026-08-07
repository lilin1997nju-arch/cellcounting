from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}
REVIEWABLE_LABELS = CELL_LABELS | {"debris", "uncertain"}


def _division(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_metrics(
    ground_truth: pd.Series,
    prediction: pd.Series,
    positive: set[str],
    missed_positive: int,
    recovered_positive: int = 0,
) -> dict[str, float | int]:
    truth = ground_truth.astype(str).isin(positive)
    predicted = prediction.astype(str).isin(positive)
    tp = int((truth & predicted).sum()) + int(recovered_positive)
    fp = int((~truth & predicted).sum())
    fn = (
        int((truth & ~predicted).sum())
        + int(missed_positive)
        - int(recovered_positive)
    )
    precision = _division(tp, tp + fp)
    recall = _division(tp, tp + fn)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative_including_manual_misses": fn,
        "precision": precision,
        "recall": recall,
        "f1": _division(2 * precision * recall, precision + recall),
    }


def _match_manual_misses(
    predictions: pd.DataFrame,
    audited: pd.DataFrame,
    missed: pd.DataFrame,
    *,
    radius_px: float = 16.0,
) -> pd.DataFrame:
    """Credit new automatic candidates that recover former manual misses."""
    available = predictions[
        ~predictions["candidate_id"].astype(str).isin(
            set(audited["candidate_id"].astype(str))
        )
    ].copy()
    used: set[str] = set()
    rows = []
    for target in missed.itertuples(index=False):
        local = available[
            (available["well"].astype(str) == str(target.well))
            & (available["timepoint"].astype(str) == str(target.timepoint))
            & ~available["candidate_id"].astype(str).isin(used)
        ].copy()
        recovered = False
        matched_id = ""
        matched_label = "missing"
        distance = np.inf
        if not local.empty:
            distances = np.hypot(
                local["x_px"].to_numpy(float) - float(target.x_px),
                local["y_px"].to_numpy(float) - float(target.y_px),
            )
            position = int(np.argmin(distances))
            candidate = local.iloc[position]
            distance = float(distances[position])
            expected = str(target.reviewed_label)
            predicted = str(candidate.integrated_label)
            same_object_class = (
                expected in CELL_LABELS and predicted in CELL_LABELS
            ) or (expected == "debris" and predicted == "debris")
            recovered = distance <= radius_px and same_object_class
            if recovered:
                matched_id = str(candidate.candidate_id)
                matched_label = predicted
                used.add(matched_id)
        rows.append(
            {
                "candidate_id": str(target.candidate_id),
                "well": str(target.well),
                "timepoint": str(target.timepoint),
                "reviewed_label": str(target.reviewed_label),
                "recovered": bool(recovered),
                "matched_candidate_id": matched_id,
                "matched_label": matched_label,
                "distance_px": distance if np.isfinite(distance) else None,
            }
        )
    return pd.DataFrame(rows)


def _macro_f1(ground_truth: pd.Series, prediction: pd.Series) -> float:
    labels = sorted(set(ground_truth.astype(str)))
    scores = []
    for label in labels:
        metrics = _binary_metrics(
            ground_truth, prediction, {label}, missed_positive=0
        )
        scores.append(float(metrics["f1"]))
    return float(np.mean(scores)) if scores else 0.0


def _evaluate(
    name: str,
    predictions: pd.DataFrame,
    audited: pd.DataFrame,
    missed: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    selected = audited[
        ["candidate_id", "well", "timepoint", "reviewed_label"]
    ].merge(
        predictions[["candidate_id", "integrated_label"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    selected["integrated_label"] = selected["integrated_label"].fillna("missing")
    missed_counts = missed["reviewed_label"].astype(str).value_counts()
    recovered = _match_manual_misses(predictions, audited, missed)
    missed_cells = int(sum(missed_counts.get(label, 0) for label in CELL_LABELS))
    missed_debris = int(missed_counts.get("debris", 0))
    recovered_cells = int(
        (
            recovered["recovered"]
            & recovered["reviewed_label"].isin(CELL_LABELS)
        ).sum()
    )
    recovered_debris = int(
        (
            recovered["recovered"]
            & recovered["reviewed_label"].eq("debris")
        ).sum()
    )
    cell = _binary_metrics(
        selected["reviewed_label"], selected["integrated_label"],
        CELL_LABELS, missed_cells, recovered_cells,
    )
    debris = _binary_metrics(
        selected["reviewed_label"], selected["integrated_label"],
        {"debris"}, missed_debris, recovered_debris,
    )
    true_cells = selected["reviewed_label"].isin(CELL_LABELS)
    multiplicity_accuracy = float(
        (
            selected.loc[true_cells, "reviewed_label"].astype(str)
            == selected.loc[true_cells, "integrated_label"].astype(str)
        ).mean()
    ) if int(true_cells.sum()) else 0.0
    invalid = selected["reviewed_label"].astype(str) == "invalid"
    invalid_rejected = selected["integrated_label"].astype(str).isin(
        {"invalid", "unmarked", "suppressed"}
    )
    well_exact = []
    missed_wells = set(missed["well"].astype(str)) if not missed.empty else set()
    for well, local in selected.groupby("well"):
        well_exact.append(
            bool(
                (local["reviewed_label"].astype(str)
                 == local["integrated_label"].astype(str)).all()
                and str(well) not in missed_wells
            )
        )
    audited_ids = set(audited["candidate_id"].astype(str))
    new_reviewable = predictions[
        predictions["integrated_label"].astype(str).isin(REVIEWABLE_LABELS)
        & ~predictions["candidate_id"].astype(str).isin(audited_ids)
    ]
    per_timepoint = {}
    for timepoint, local in selected.groupby("timepoint"):
        local_missed = missed[missed["timepoint"].astype(str) == str(timepoint)]
        local_missed_cells = int(
            local_missed["reviewed_label"].astype(str).isin(CELL_LABELS).sum()
        )
        local_recovered_cells = int(
            (
                recovered["recovered"]
                & recovered["reviewed_label"].isin(CELL_LABELS)
                & recovered["timepoint"].astype(str).eq(str(timepoint))
            ).sum()
        )
        per_timepoint[str(timepoint)] = _binary_metrics(
            local["reviewed_label"], local["integrated_label"],
            CELL_LABELS, local_missed_cells, local_recovered_cells,
        )
    report = {
        "name": name,
        "audited_candidate_count": int(len(selected)),
        "manual_missed_count": int(len(missed)),
        "manual_missed_recovered_count": int(recovered["recovered"].sum()),
        "manual_missed_recovered_cells": recovered_cells,
        "manual_missed_recovered_debris": recovered_debris,
        "exact_class_accuracy": float(
            (selected["reviewed_label"].astype(str)
             == selected["integrated_label"].astype(str)).mean()
        ),
        "macro_f1": _macro_f1(
            selected["reviewed_label"], selected["integrated_label"]
        ),
        "cell": cell,
        "debris": debris,
        "multiplicity_accuracy_end_to_end": multiplicity_accuracy,
        "invalid_rejection_rate": _division(
            int((invalid & invalid_rejected).sum()), int(invalid.sum())
        ),
        "exact_well_rate": float(np.mean(well_exact)) if well_exact else 0.0,
        "new_reviewable_candidates_without_holdout_label": int(len(new_reviewable)),
        "cell_metrics_by_timepoint": per_timepoint,
    }
    selected = selected.rename(columns={"integrated_label": f"{name}_label"})
    return report, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--optimized", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ground_truth = Path(args.ground_truth)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    audited = pd.read_csv(ground_truth / "audited_candidates.csv")
    missed = pd.read_csv(ground_truth / "manual_missed_objects.csv")
    baseline = pd.read_csv(args.baseline, low_memory=False)
    optimized = pd.read_csv(args.optimized, low_memory=False)
    before, before_rows = _evaluate("before", baseline, audited, missed)
    after, after_rows = _evaluate("after", optimized, audited, missed)
    comparison = before_rows.merge(
        after_rows[["candidate_id", "after_label"]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    comparison["before_correct"] = (
        comparison["reviewed_label"].astype(str)
        == comparison["before_label"].astype(str)
    )
    comparison["after_correct"] = (
        comparison["reviewed_label"].astype(str)
        == comparison["after_label"].astype(str)
    )
    comparison.to_csv(
        output / "candidate_comparison.csv", index=False, encoding="utf-8"
    )
    result = {
        "evaluation_scope": (
            "Frozen A12-22 model-assisted audit; manual misses are included "
            "as false negatives unless a new automatic candidate recovers "
            "the same object class within 16 pixels."
        ),
        "before": before,
        "after": after,
        "delta": {
            "exact_class_accuracy": after["exact_class_accuracy"] - before["exact_class_accuracy"],
            "macro_f1": after["macro_f1"] - before["macro_f1"],
            "cell_precision": after["cell"]["precision"] - before["cell"]["precision"],
            "cell_recall": after["cell"]["recall"] - before["cell"]["recall"],
            "cell_f1": after["cell"]["f1"] - before["cell"]["f1"],
            "debris_f1": after["debris"]["f1"] - before["debris"]["f1"],
            "multiplicity_accuracy_end_to_end": (
                after["multiplicity_accuracy_end_to_end"]
                - before["multiplicity_accuracy_end_to_end"]
            ),
            "exact_well_rate": after["exact_well_rate"] - before["exact_well_rate"],
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

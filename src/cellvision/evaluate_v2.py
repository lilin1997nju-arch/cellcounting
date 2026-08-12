from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import artifact_path


POSITIVE_LABELS = {"single", "touching_doublet", "cluster_3plus", "debris"}
CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}


def _latest_ground_truth(database: Path, round_id: str) -> pd.DataFrame:
    with sqlite3.connect(database) as connection:
        reviewed = pd.read_sql_query(
            "SELECT candidate_id, reviewed_label, updated_at FROM integrated_training_reviews ORDER BY updated_at, integrated_review_id",
            connection,
        ).drop_duplicates("candidate_id", keep="last")
        reviewed["well"] = None
        reviewed["timepoint"] = None
        missed = pd.read_sql_query(
            "SELECT candidate_id, well, timepoint, x_px, y_px, reviewed_label, updated_at FROM quick_missed_objects ORDER BY updated_at, quick_missed_id",
            connection,
        ).drop_duplicates("candidate_id", keep="last")
    reviewed["manual_missed"] = False
    missed["manual_missed"] = True
    return pd.concat([reviewed, missed], ignore_index=True, sort=False)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate_v2_plate(config: dict[str, Any]) -> dict[str, Any]:
    predictions_path = artifact_path(config, "predictions", "latest_v2_predictions.csv")
    frame = pd.read_csv(predictions_path, low_memory=False)
    round_id = str(frame.iloc[0]["integrated_round_id"])
    truth = _latest_ground_truth(artifact_path(config, "annotations", "annotations.db"), round_id)
    indexed = frame.set_index("candidate_id", drop=False)
    truth = truth[
        truth["manual_missed"]
        | truth["candidate_id"].astype(str).isin(set(frame["candidate_id"].astype(str)))
    ].copy()
    rows = []
    for item in truth.itertuples(index=False):
        matched = indexed.loc[item.candidate_id] if item.candidate_id in indexed.index else None
        if isinstance(matched, pd.DataFrame):
            matched = matched.iloc[0]
        if matched is None and bool(item.manual_missed) and pd.notna(getattr(item, "x_px", np.nan)):
            local = frame[(frame["well"] == item.well) & (frame["timepoint"] == item.timepoint)]
            if not local.empty:
                distances = np.hypot(local["x_px"].astype(float) - float(item.x_px), local["y_px"].astype(float) - float(item.y_px))
                nearest = distances.idxmin()
                if float(distances.loc[nearest]) <= 16.0:
                    matched = frame.loc[nearest]
        v2_match = matched
        v1_match = matched
        if matched is not None and item.reviewed_label in POSITIVE_LABELS:
            owner_id = str(matched.get("v2_suppressed_by", ""))
            if owner_id and owner_id != "nan" and owner_id in indexed.index:
                v2_match = indexed.loc[owner_id]
                if isinstance(v2_match, pd.DataFrame):
                    v2_match = v2_match.iloc[0]
            v1_owner_value = matched.get("suppressed_by_candidate_id", "")
            if pd.isna(v1_owner_value) or not str(v1_owner_value):
                v1_owner_value = matched.get("duplicate_of_candidate_id", "")
            v1_owner_id = str(v1_owner_value)
            if v1_owner_id and v1_owner_id != "nan" and v1_owner_id in indexed.index:
                v1_match = indexed.loc[v1_owner_id]
                if isinstance(v1_match, pd.DataFrame):
                    v1_match = v1_match.iloc[0]
        rows.append({
            "candidate_id": item.candidate_id,
            "well": str(matched.well) if matched is not None and pd.isna(item.well) else item.well,
            "timepoint": str(matched.timepoint) if matched is not None and pd.isna(item.timepoint) else item.timepoint,
            "truth": item.reviewed_label, "manual_missed": bool(item.manual_missed),
            "detected": bool(v2_match is not None and v2_match.v2_is_counting_instance),
            "v1_detected": bool(
                v1_match is not None
                and str(v1_match.v2_original_integrated_label if "v2_original_integrated_label" in v1_match else v1_match.integrated_label)
                in {"single", "touching_doublet", "cluster_3plus", "debris", "uncertain"}
            ),
            "prediction": str(v2_match.integrated_label) if v2_match is not None else "missed",
            "radial_fraction": float(matched.radial_fraction) if matched is not None else np.nan,
        })
    aligned = pd.DataFrame(rows)
    positive = aligned[aligned["truth"].isin(POSITIVE_LABELS)]
    cell_truth = aligned[aligned["truth"].isin(CELL_LABELS)]
    invalid_truth = aligned[aligned["truth"] == "invalid"]
    near_wall = cell_truth[cell_truth["radial_fraction"] >= 0.40]
    multiplicity = cell_truth[cell_truth["detected"]]
    multiplicity_exact = int((multiplicity["truth"] == multiplicity["prediction"]).sum())
    multiplicity_labels = sorted(CELL_LABELS)
    multiplicity_matrix = pd.crosstab(
        multiplicity["truth"], multiplicity["prediction"]
    ).reindex(index=multiplicity_labels, columns=multiplicity_labels, fill_value=0)
    multiplicity_per_class: dict[str, dict[str, float | int | None]] = {}
    for label in multiplicity_labels:
        true_positive = int(multiplicity_matrix.loc[label, label])
        truth_count = int(multiplicity_matrix.loc[label].sum())
        predicted_count = int(multiplicity_matrix[label].sum())
        precision = _safe_ratio(true_positive, predicted_count)
        recall = _safe_ratio(true_positive, truth_count)
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision is not None and recall is not None and precision + recall
            else None
        )
        multiplicity_per_class[label] = {
            "support": truth_count,
            "predicted": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    # End-to-end class recall counts a missed/invalid prediction as a miss.
    # The conditional confusion above is useful for measuring the head's
    # discrimination after segmentation, but it must not hide segmentation
    # misses when reporting the biological cell recall.
    cell_predictions = np.where(
        cell_truth["detected"].astype(bool), cell_truth["prediction"], "missed"
    )
    cell_truth_with_prediction = cell_truth.assign(
        prediction_or_missed=cell_predictions
    )
    end_to_end_per_class: dict[str, dict[str, float | int | None]] = {}
    for label in multiplicity_labels:
        support = int((cell_truth_with_prediction["truth"] == label).sum())
        true_positive = int(
            (
                (cell_truth_with_prediction["truth"] == label)
                & (cell_truth_with_prediction["prediction_or_missed"] == label)
            ).sum()
        )
        predicted_count = int(
            (cell_truth_with_prediction["prediction_or_missed"] == label).sum()
        )
        precision = _safe_ratio(true_positive, predicted_count)
        recall = _safe_ratio(true_positive, support)
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision is not None and recall is not None and precision + recall
            else None
        )
        end_to_end_per_class[label] = {
            "support": support,
            "predicted": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    per_timepoint = {}
    for timepoint in ("T0", "T1", "T2"):
        local = positive[positive["timepoint"] == timepoint]
        per_timepoint[timepoint] = {"reviewed_objects": int(len(local)), "recall": _safe_ratio(int(local["detected"].sum()), len(local))}
    valid_count = int(frame["v2_mask_valid"].sum())
    reviewed_wells = max(int(truth["well"].nunique()), 1)
    with sqlite3.connect(artifact_path(config, "annotations", "annotations.db")) as connection:
        try:
            sessions = pd.read_sql_query("SELECT * FROM quick_review_sessions WHERE round_id = ?", connection, params=(round_id,))
        except Exception:
            sessions = pd.DataFrame()
    result = {
        "algorithm_version": "v2",
        "plate_id": str(config["experiment"]["plate_id"]),
        "round_id": round_id,
        "ground_truth_scope": "latest human decision for candidates present in the current frozen proposal set plus all saved manual misses; labels are evaluation-only",
        "target_level_miss_rate": _safe_ratio(int((~positive["detected"]).sum()), len(positive)),
        "target_level_recall": _safe_ratio(int(positive["detected"].sum()), len(positive)),
        "v1_equivalent_baseline": {
            "target_level_miss_rate": _safe_ratio(int((~positive["v1_detected"]).sum()), len(positive)),
            "target_level_recall": _safe_ratio(int(positive["v1_detected"].sum()), len(positive)),
            "wall_false_positive_rate": _safe_ratio(int(invalid_truth["v1_detected"].sum()), len(invalid_truth)),
        },
        "v2_minus_v1_target_recall": (
            _safe_ratio(int(positive["detected"].sum()), len(positive))
            - _safe_ratio(int(positive["v1_detected"].sum()), len(positive))
            if len(positive) else None
        ),
        "cell_level_recall": _safe_ratio(int(cell_truth["detected"].sum()), len(cell_truth)),
        "cell_level_reviewed_objects": int(len(cell_truth)),
        "duplicate_candidate_rate": _safe_ratio(int(frame["v2_is_suppressed"].sum()), valid_count),
        "wall_false_positive_rate": _safe_ratio(int(invalid_truth["detected"].sum()), len(invalid_truth)),
        "single_doublet_cluster_confusion": {
            "evaluated": int(len(multiplicity)),
            "exact_rate": _safe_ratio(multiplicity_exact, len(multiplicity)),
            "matrix": pd.crosstab(multiplicity["truth"], multiplicity["prediction"]).to_dict(),
            "per_class": multiplicity_per_class,
            "touching_doublet_recall": multiplicity_per_class["touching_doublet"]["recall"],
            "touching_doublet_f1": multiplicity_per_class["touching_doublet"]["f1"],
            "end_to_end_per_class": end_to_end_per_class,
            "touching_doublet_end_to_end_recall": end_to_end_per_class["touching_doublet"]["recall"],
            "touching_doublet_end_to_end_f1": end_to_end_per_class["touching_doublet"]["f1"],
        },
        "near_wall_cell_recall": _safe_ratio(int(near_wall["detected"].sum()), len(near_wall)),
        "near_wall_reviewed_cells": int(len(near_wall)),
        "recall_by_timepoint": per_timepoint,
        "human_review": {
            "reviewed_wells": int(truth["well"].nunique()),
            "mean_corrections_per_reviewed_well": float(len(truth) / reviewed_wells),
            "mean_review_time_seconds": float(sessions["duration_ms"].mean() / 1000) if not sessions.empty else None,
            "timing_status": "measured by V2 review sessions" if not sessions.empty else "not recorded by historical V1 rounds; V2 now records it prospectively",
        },
        "well_level_active_single_cell": {
            "sensitivity": None, "specificity": None, "positive_predictive_value": None,
            "status": "not evaluated: no frozen human well-level activity truth; V1 growth decision intentionally unchanged",
        },
    }
    return result


def write_v2_evaluation(config: dict[str, Any]) -> Path:
    result = evaluate_v2_plate(config)
    output = artifact_path(config, "evaluation", "v2_metrics.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output

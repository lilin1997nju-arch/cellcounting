"""Run an isolated QL2603 model deployment and evaluation batch.

The batch deliberately never writes to the project plate artifact roots.  It
copies only the inputs needed by the inference stages into a timestamped run
directory, then runs:

* the pooled morphology and multiplicity checkpoints;
* the selected instance segmenter;
* the active V3 heuristic/state-fusion layer over the temporal checkpoint;
* a paired old/new classifier evaluation on reviewed boards; and
* a paired old/new mask evaluation against the saved reviewer masks.

The default board split is the current QL2603 set requested for deployment:
T1-1/T1-2/T4-2 are reviewed and T2-4/T5-1/T5-2 are not reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cellvision.config import artifact_path, load_config
from cellvision.model_inference import (
    predict_multiplicity_checkpoint,
    predict_teaching_checkpoint,
)
from cellvision.multiplicity import read_multiplicity_labels
from cellvision.teaching import read_teaching_labels
from cellvision.teaching import generate_auto_annotation_round
from cellvision.multiplicity import generate_integrated_training_round
from cellvision.v2_instance_inference import decode_rle, infer_v2_instances
from cellvision.v2_temporal_inference import infer_v2_temporal_evidence
from cellvision.well_screening import build_well_screening


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "artifacts" / "projects" / "ql2603"
PLATE_ROOT = PROJECT_ROOT / "plates"
CONFIG_ROOT = ROOT / "configs" / "generated"

REVIEWED_PLATES = ("ql2603-t1-1", "ql2603-t1-2", "ql2603-t4-2")
UNREVIEWED_PLATES = ("ql2603-t2-4", "ql2603-t5-1", "ql2603-t5-2")

NEW_INSTANCE = ROOT / "artifacts" / "v2" / "runs" / "v2-instance-20260812-120154" / "model.pt"
OLD_INSTANCE = ROOT / "artifacts" / "v2" / "models" / "latest_instance_segmenter.pt"

NEW_MODEL_ROOT = PLATE_ROOT / "ql2603-t1-1" / "models"
NEW_TEACHING = NEW_MODEL_ROOT / "teaching_classifier.pt"
NEW_MULTIPLICITY = NEW_MODEL_ROOT / "multiplicity_classifier.pt"
OLD_MODEL_ROOT = ROOT / "artifacts" / "backups" / "ql2603_training_round_20260811_pre" / "models"
OLD_TEACHING = OLD_MODEL_ROOT / "teaching_classifier.pt"
OLD_MULTIPLICITY = OLD_MODEL_ROOT / "multiplicity_classifier.pt"

MASK_SIZE = 96
CELL_LABELS = ("single", "touching_doublet", "cluster_3plus")
MORPHOLOGY_LABELS = ("invalid", "debris", "cell")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "sha256": _sha256(path) if path.exists() else "",
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        if path.exists()
        else None,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _source_config(slug: str) -> Path:
    path = CONFIG_ROOT / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _prepare_shadow_inputs(source_root: Path, shadow_root: Path) -> dict[str, Any]:
    """Copy the read inputs required by classifier and V2 stages."""

    required = (
        "annotations/annotations.db",
        "manifests/images.csv",
        "manifests/sequences.csv",
        "pseudo_labels/morphology_candidates.csv",
    )
    copied: list[str] = []
    for relative in required:
        source = source_root / relative
        target = shadow_root / relative
        _copy_file(source, target)
        copied.append(relative)

    optional = (
        "pseudo_labels/morphology_candidates.source.json",
        "annotations/auto_review_queue.csv",
    )
    for relative in optional:
        source = source_root / relative
        if source.exists():
            _copy_file(source, shadow_root / relative)
            copied.append(relative)

    source_cache = source_root / "cache"
    if source_cache.exists():
        for source in source_cache.iterdir():
            if source.is_file():
                _copy_file(source, shadow_root / "cache" / source.name)
                copied.append(str(Path("cache") / source.name))

    return {"source_root": str(source_root.resolve()), "copied_inputs": copied}


def _shadow_config(slug: str, shadow_root: Path) -> dict[str, Any]:
    config = load_config(_source_config(slug))
    config["paths"]["artifact_root"] = str(shadow_root.resolve())
    # The deployment batch is a prediction run, not a review action.  The
    # copied DB still carries existing reviewer context for the actual
    # workflow, while the classifier accuracy evaluation uses raw head output.
    return config


def _run_classifier_stage(
    config: dict[str, Any],
    database: Path,
    teaching_checkpoint: Path,
    multiplicity_checkpoint: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    teaching = predict_teaching_checkpoint(config, teaching_checkpoint)
    multiplicity = predict_multiplicity_checkpoint(config, multiplicity_checkpoint)
    auto = generate_auto_annotation_round(config, database)
    integrated = generate_integrated_training_round(config, database)
    return {
        "teaching": teaching,
        "multiplicity": multiplicity,
        "auto_annotation": auto,
        "integrated": integrated,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _label_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].fillna("<NA>").astype(str).value_counts().items()}


def _run_deployment_plate(slug: str, run_root: Path) -> dict[str, Any]:
    source_root = PLATE_ROOT / slug
    shadow_root = run_root / "deployment" / slug
    input_record = _prepare_shadow_inputs(source_root, shadow_root)
    config = _shadow_config(slug, shadow_root)
    database = shadow_root / "annotations" / "annotations.db"

    print(f"{slug}: latest classifier heads", flush=True)
    classifier = _run_classifier_stage(
        config, database, NEW_TEACHING, NEW_MULTIPLICITY
    )
    print(f"{slug}: new instance segmenter", flush=True)
    started = time.perf_counter()
    v2_path = infer_v2_instances(config, NEW_INSTANCE)
    v2_summary_path = v2_path.with_suffix(".json")
    print(f"{slug}: V3 temporal/state layer", flush=True)
    temporal_path = infer_v2_temporal_evidence(config)
    temporal_summary_path = temporal_path.with_name("latest_v2_temporal_summary.json")
    v3_path = shadow_root / "predictions" / "latest_v3_predictions.csv"
    _copy_file(temporal_path, v3_path)

    screening = build_well_screening(config, database)
    frame = pd.read_csv(v3_path, low_memory=False)
    temporal_summary = (
        json.loads(temporal_summary_path.read_text(encoding="utf-8"))
        if temporal_summary_path.exists()
        else {}
    )
    v2_summary = (
        json.loads(v2_summary_path.read_text(encoding="utf-8"))
        if v2_summary_path.exists()
        else {}
    )
    v3_rows = (
        frame[frame["v3_track_behavior"].astype(str).ne("disabled")]
        if "v3_track_behavior" in frame
        else frame.iloc[0:0]
    )
    deployment = {
        "plate": slug,
        "review_group": "reviewed" if slug in REVIEWED_PLATES else "unreviewed",
        "input": input_record,
        "classifier": classifier,
        "instance_checkpoint": _model_record(NEW_INSTANCE),
        "v3_config": {
            "enabled": bool(config.get("v3_temporal_behavior", {}).get("enabled")),
            "backend": config.get("v3_temporal_behavior", {}).get("backend"),
            "state_fusion": config.get("v3_temporal_behavior", {}).get("state_fusion"),
            "pairwise_checkpoint": temporal_summary.get("v3_pairwise_checkpoint"),
        },
        "candidate_count": int(len(frame)),
        "v2_valid_instance_count": int(frame.get("v2_mask_valid", pd.Series(False, index=frame.index)).fillna(False).astype(bool).sum()),
        "v2_unique_instance_count": int(frame.get("v2_is_unique_instance", pd.Series(False, index=frame.index)).fillna(False).astype(bool).sum()),
        "v2_counting_instance_count": int(frame.get("v2_is_counting_instance", pd.Series(False, index=frame.index)).fillna(False).astype(bool).sum()),
        "v2_temporal_candidate_count": int(frame.get("v2_is_temporal_candidate", pd.Series(False, index=frame.index)).fillna(False).astype(bool).sum()),
        "v3_rows": int(len(v3_rows)),
        "v3_would_change_rows": int(v3_rows.get("v3_would_change", pd.Series(False, index=v3_rows.index)).fillna(False).astype(bool).sum()),
        "v3_track_behavior_counts": _label_counts(v3_rows, "v3_track_behavior"),
        "v3_conclusion_counts": _label_counts(v3_rows, "v3_track_conclusion"),
        "v2_pre_temporal_label_counts": _label_counts(frame, "v2_pre_temporal_integrated_label"),
        "final_integrated_label_counts": _label_counts(frame, "integrated_label"),
        "v2_summary": v2_summary,
        "temporal_summary": temporal_summary,
        "screening_summary": screening,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "prediction_path": str(v3_path.resolve()),
    }
    _write_json(shadow_root / "deployment_summary.json", deployment)
    return deployment


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _metrics(frame: pd.DataFrame, truth: str, prediction: str, labels: tuple[str, ...]) -> dict[str, Any]:
    if frame.empty:
        return {
            "evaluated_count": 0,
            "exact_accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion": {},
        }
    view = frame[[truth, prediction]].copy()
    view[truth] = view[truth].astype(str)
    view[prediction] = view[prediction].astype(str)
    confusion = {
        expected: {
            actual: int(((view[truth] == expected) & (view[prediction] == actual)).sum())
            for actual in labels
        }
        for expected in labels
    }
    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        true_positive = int(((view[truth] == label) & (view[prediction] == label)).sum())
        false_positive = int(((view[truth] != label) & (view[prediction] == label)).sum())
        false_negative = int(((view[truth] == label) & (view[prediction] != label)).sum())
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        per_class[label] = {
            "support": int((view[truth] == label).sum()),
            "precision": precision,
            "recall": recall,
            "f1": _safe_divide(2 * precision * recall, precision + recall),
        }
    return {
        "evaluated_count": int(len(view)),
        "exact_accuracy": float((view[truth] == view[prediction]).mean()),
        "macro_f1": float(sum(item["f1"] for item in per_class.values()) / len(labels)),
        "per_class": per_class,
        "confusion": confusion,
    }


def _human_truth(database: Path) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    with sqlite3.connect(database) as connection:
        integrated = pd.read_sql_query(
            """
            SELECT candidate_id, reviewed_label, decision, updated_at,
                   integrated_review_id
            FROM integrated_training_reviews
            ORDER BY updated_at, integrated_review_id
            """,
            connection,
        )
    integrated = integrated.drop_duplicates("candidate_id", keep="last")
    labels = {
        str(row.candidate_id): str(row.reviewed_label)
        for row in integrated.itertuples(index=False)
    }
    focused_multiplicity = read_multiplicity_labels(database)
    if not focused_multiplicity.empty:
        for row in focused_multiplicity.itertuples(index=False):
            label = str(row.label)
            if label in set(CELL_LABELS) | {"debris", "invalid"}:
                labels[str(row.candidate_id)] = label
    truth = pd.DataFrame(
        [{"candidate_id": candidate_id, "label": label} for candidate_id, label in labels.items()]
    )
    if truth.empty:
        empty = pd.DataFrame(columns=["candidate_id", "label"])
        return empty, empty, set()
    multiplicity = truth[truth["label"].isin(CELL_LABELS)].copy()
    morphology = truth.assign(
        label=truth["label"].map(
            {
                "single": "cell",
                "touching_doublet": "cell",
                "cluster_3plus": "cell",
                "cell": "cell",
                "debris": "debris",
                "invalid": "invalid",
            }
        )
    ).dropna(subset=["label"])
    focused_morphology = read_teaching_labels(database)
    if not focused_morphology.empty:
        focused_morphology = focused_morphology[
            focused_morphology["label"].isin(MORPHOLOGY_LABELS)
        ][["candidate_id", "label"]].drop_duplicates("candidate_id", keep="first")
        morphology = pd.concat(
            [morphology[~morphology["candidate_id"].isin(focused_morphology["candidate_id"])], focused_morphology],
            ignore_index=True,
        )
    corrected_ids = set(
        integrated.loc[integrated["decision"].astype(str).eq("corrected"), "candidate_id"].astype(str)
    )
    return multiplicity, morphology, corrected_ids


def _run_recognition_eval(slug: str, run_root: Path) -> dict[str, Any]:
    source_root = PLATE_ROOT / slug
    database = source_root / "annotations" / "annotations.db"
    multiplicity_truth, morphology_truth, corrected_ids = _human_truth(database)
    output: dict[str, Any] = {"plate": slug, "truth_counts": {"multiplicity": _label_counts(multiplicity_truth, "label"), "morphology": _label_counts(morphology_truth, "label")}}
    heads = (
        ("old", OLD_TEACHING, OLD_MULTIPLICITY),
        ("new", NEW_TEACHING, NEW_MULTIPLICITY),
    )
    predictions: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, teaching_checkpoint, multiplicity_checkpoint in heads:
        shadow_root = run_root / "recognition" / slug / name
        _prepare_shadow_inputs(source_root, shadow_root)
        config = _shadow_config(slug, shadow_root)
        print(f"{slug}: {name} recognition heads", flush=True)
        predict_teaching_checkpoint(config, teaching_checkpoint)
        predict_multiplicity_checkpoint(config, multiplicity_checkpoint)
        teaching_path = shadow_root / "predictions" / "teaching_classifier_predictions.csv"
        multiplicity_path = shadow_root / "predictions" / "multiplicity_predictions.csv"
        teaching = pd.read_csv(teaching_path, low_memory=False)[["candidate_id", "predicted_label"]].drop_duplicates("candidate_id", keep="last")
        multiplicity = pd.read_csv(multiplicity_path, low_memory=False)[["candidate_id", "predicted_multiplicity"]].drop_duplicates("candidate_id", keep="last")
        predictions[name] = (teaching, multiplicity)

    for head, truth_frame, column, labels in (
        ("morphology", morphology_truth, "predicted_label", MORPHOLOGY_LABELS),
        ("multiplicity", multiplicity_truth, "predicted_multiplicity", CELL_LABELS),
    ):
        truth_frame = truth_frame.rename(columns={"label": "truth"})
        rows_by_name: dict[str, pd.DataFrame] = {}
        for name, (teaching, multiplicity) in predictions.items():
            predicted = teaching if head == "morphology" else multiplicity
            rows = truth_frame.merge(predicted.rename(columns={column: name}), on="candidate_id", how="inner")
            rows_by_name[name] = rows
            rows.to_csv(run_root / "recognition" / slug / f"{head}_comparison_{name}.csv", index=False, encoding="utf-8")
        paired = rows_by_name["old"].rename(columns={"old": "old"}).merge(
            rows_by_name["new"][["candidate_id", "new"]], on="candidate_id", how="inner"
        )
        corrected = paired[paired["candidate_id"].astype(str).isin(corrected_ids)]
        output[head] = {
            "old": _metrics(rows_by_name["old"], "truth", "old", labels),
            "new": _metrics(rows_by_name["new"], "truth", "new", labels),
            "paired": {
                "old": _metrics(paired, "truth", "old", labels),
                "new": _metrics(paired, "truth", "new", labels),
                "count": int(len(paired)),
            },
            "corrected_only": {
                "old": _metrics(corrected, "truth", "old", labels),
                "new": _metrics(corrected, "truth", "new", labels),
                "count": int(len(corrected)),
            },
        }
    _write_json(run_root / "recognition" / slug / "metrics.json", output)
    return output


def _latest_review_round_rows() -> list[dict[str, Any]]:
    database = PLATE_ROOT / "ql2603-t1-1" / "annotations" / "annotations.db"
    with sqlite3.connect(database) as connection:
        reviews = pd.read_sql_query(
            """
            SELECT round_id, candidate_id, reviewed_mask_rle, decision,
                   reviewed_area_px, updated_at
            FROM v2_mask_reviews
            ORDER BY updated_at, review_id
            """,
            connection,
        ).drop_duplicates(["round_id", "candidate_id"], keep="last")
    rows: list[dict[str, Any]] = []
    p0_csv = PLATE_ROOT / "ql2603-t1-1" / "v2" / "mask_review" / "p0-mask-review-ql2603-t1-1-20260812" / "reviewed_v2_predictions.csv"
    comparison_csv = PLATE_ROOT / "ql2603-t1-1" / "v2" / "mask_review" / "model-comparison-20260812-125449" / "reviewed_v2_predictions.csv"
    csv_by_round = {
        "p0-mask-review-ql2603-t1-1-20260812": p0_csv,
        "model-comparison-20260812-125449": comparison_csv,
    }
    frame_by_round: dict[str, pd.DataFrame] = {}
    for round_id, path in csv_by_round.items():
        frame_by_round[round_id] = pd.read_csv(path, low_memory=False).set_index("candidate_id")
    for review in reviews.itertuples(index=False):
        round_id = str(review.round_id)
        csv_frame = frame_by_round.get(round_id)
        if csv_frame is None or str(review.candidate_id) not in csv_frame.index:
            continue
        item = csv_frame.loc[str(review.candidate_id)]
        if isinstance(item, pd.DataFrame):
            item = item.iloc[-1]
        if round_id.startswith("p0-"):
            source_slug = "ql2603-t1-1"
        else:
            source_slug = str(item.get("comparison_source_id", ""))
        if source_slug not in REVIEWED_PLATES:
            continue
        rows.append(
            {
                "round_id": round_id,
                "candidate_id": str(review.candidate_id),
                "source_slug": source_slug,
                "decision": str(review.decision),
                "reviewed_mask_rle": str(review.reviewed_mask_rle),
                "reviewed_area_px": int(review.reviewed_area_px or 0),
                "truth_origin_x": int(round(float(item.get("v2_mask_origin_x", 0)))),
                "truth_origin_y": int(round(float(item.get("v2_mask_origin_y", 0)))),
            }
        )
    return rows


def _global_iou(pred_rle: str, pred_x: int, pred_y: int, truth_rle: str, truth_x: int, truth_y: int) -> tuple[float, int, int, int]:
    pred = decode_rle(pred_rle or "[]", MASK_SIZE)
    truth = decode_rle(truth_rle or "[]", MASK_SIZE)
    left = max(pred_x, truth_x)
    top = max(pred_y, truth_y)
    right = min(pred_x + MASK_SIZE, truth_x + MASK_SIZE)
    bottom = min(pred_y + MASK_SIZE, truth_y + MASK_SIZE)
    intersection = 0
    if left < right and top < bottom:
        intersection = int(
            np.logical_and(
                pred[top - pred_y : bottom - pred_y, left - pred_x : right - pred_x],
                truth[top - truth_y : bottom - truth_y, left - truth_x : right - truth_x],
            ).sum()
        )
    pred_area = int(pred.sum())
    truth_area = int(truth.sum())
    union = pred_area + truth_area - intersection
    iou = float(intersection / union) if union else 0.0
    return iou, pred_area, truth_area, intersection


def _mask_metric(rows: pd.DataFrame, model_column: str) -> dict[str, Any]:
    if rows.empty:
        return {"count": 0, "mean_iou": 0.0, "positive_mean_iou": 0.0, "mean_area_abs_error_ratio": 0.0, "positive_recall": 0.0, "negative_specificity": 0.0}
    positive = rows[rows["decision"].isin(["accepted", "edited"])]
    negative = rows[rows["decision"].eq("rejected")]
    area_denominator = rows["truth_area"].clip(lower=1)
    return {
        "count": int(len(rows)),
        "mean_iou": float(rows[model_column].mean()),
        "positive_mean_iou": float(positive[model_column].mean()) if not positive.empty else None,
        "mean_area_abs_error_ratio": float((rows["pred_area"].sub(rows["truth_area"]).abs() / area_denominator).mean()),
        "positive_recall": float((positive["pred_area"] > 0).mean()) if not positive.empty else None,
        "negative_specificity": float((negative["pred_area"] == 0).mean()) if not negative.empty else None,
        "by_decision": {
            decision: {
                "count": int(len(group)),
                "mean_iou": float(group[model_column].mean()),
            }
            for decision, group in rows.groupby("decision", sort=True)
        },
    }


def _run_mask_eval(run_root: Path, deployments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    truth_rows = _latest_review_round_rows()
    by_plate: dict[str, list[dict[str, Any]]] = {slug: [] for slug in REVIEWED_PLATES}
    for item in truth_rows:
        by_plate[item["source_slug"]].append(item)
    report: dict[str, Any] = {"truth_count": len(truth_rows), "plates": {}}
    for slug in REVIEWED_PLATES:
        source_root = PLATE_ROOT / slug
        new_root = Path(deployments[slug]["prediction_path"]).parent.parent
        new_predictions = pd.read_csv(new_root / "predictions" / "latest_v2_predictions.csv", low_memory=False).set_index("candidate_id")
        old_root = run_root / "mask_comparison" / slug / "old"
        _prepare_shadow_inputs(source_root, old_root)
        old_predictions_input = new_root / "predictions" / "latest_integrated_predictions.csv"
        _copy_file(old_predictions_input, old_root / "predictions" / "latest_integrated_predictions.csv")
        old_config = _shadow_config(slug, old_root)
        print(f"{slug}: old segmenter paired mask inference", flush=True)
        infer_v2_instances(old_config, OLD_INSTANCE)
        old_predictions = pd.read_csv(old_root / "predictions" / "latest_v2_predictions.csv", low_memory=False).set_index("candidate_id")

        rows: list[dict[str, Any]] = []
        for truth in by_plate[slug]:
            candidate_id = truth["candidate_id"]
            truth_rle = truth["reviewed_mask_rle"]
            if candidate_id in new_predictions.index:
                new_row = new_predictions.loc[candidate_id]
                if isinstance(new_row, pd.DataFrame):
                    new_row = new_row.iloc[-1]
                new_iou, new_area, truth_area, _ = _global_iou(
                    str(new_row.get("v2_mask_rle", "[]")),
                    int(round(float(new_row.get("v2_mask_origin_x", 0)))),
                    int(round(float(new_row.get("v2_mask_origin_y", 0)))),
                    truth_rle,
                    truth["truth_origin_x"],
                    truth["truth_origin_y"],
                )
            else:
                new_iou, new_area, truth_area = 0.0, 0, int(truth["reviewed_area_px"])
            if candidate_id in old_predictions.index:
                old_row = old_predictions.loc[candidate_id]
                if isinstance(old_row, pd.DataFrame):
                    old_row = old_row.iloc[-1]
                old_iou, old_area, _, _ = _global_iou(
                    str(old_row.get("v2_mask_rle", "[]")),
                    int(round(float(old_row.get("v2_mask_origin_x", 0)))),
                    int(round(float(old_row.get("v2_mask_origin_y", 0)))),
                    truth_rle,
                    truth["truth_origin_x"],
                    truth["truth_origin_y"],
                )
            else:
                old_iou, old_area = 0.0, 0
            rows.append(
                {
                    **truth,
                    "old_iou": old_iou,
                    "new_iou": new_iou,
                    "old_pred_area": old_area,
                    "new_pred_area": new_area,
                    "truth_area": truth_area,
                }
            )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["iou_delta_new_minus_old"] = frame["new_iou"] - frame["old_iou"]
        plate_report = {
            "truth_count": int(len(frame)),
            "old": _mask_metric(frame.rename(columns={"old_iou": "iou", "old_pred_area": "pred_area"}), "iou"),
            "new": _mask_metric(frame.rename(columns={"new_iou": "iou", "new_pred_area": "pred_area"}), "iou"),
            "edited_only": {
                "old": _mask_metric(frame[frame["decision"].eq("edited")].rename(columns={"old_iou": "iou", "old_pred_area": "pred_area"}), "iou"),
                "new": _mask_metric(frame[frame["decision"].eq("edited")].rename(columns={"new_iou": "iou", "new_pred_area": "pred_area"}), "iou"),
            },
            "rejected_only": {
                "old": _mask_metric(frame[frame["decision"].eq("rejected")].rename(columns={"old_iou": "iou", "old_pred_area": "pred_area"}), "iou"),
                "new": _mask_metric(frame[frame["decision"].eq("rejected")].rename(columns={"new_iou": "iou", "new_pred_area": "pred_area"}), "iou"),
            },
            "decision_counts": _label_counts(frame, "decision"),
            "mean_iou_delta_new_minus_old": float(frame["iou_delta_new_minus_old"].mean()) if not frame.empty else 0.0,
            "matched_new": int((frame["new_pred_area"] > 0).sum()) if not frame.empty else 0,
            "matched_old": int((frame["old_pred_area"] > 0).sum()) if not frame.empty else 0,
        }
        report["plates"][slug] = plate_report
        frame.to_csv(run_root / "mask_comparison" / f"{slug}_mask_comparison.csv", index=False, encoding="utf-8")

    all_frames = []
    for slug in REVIEWED_PLATES:
        path = run_root / "mask_comparison" / f"{slug}_mask_comparison.csv"
        if path.exists():
            all_frames.append(pd.read_csv(path, low_memory=False))
    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    report["combined"] = {
        "truth_count": int(len(combined)),
        "old": _mask_metric(combined.rename(columns={"old_iou": "iou", "old_pred_area": "pred_area"}), "iou") if not combined.empty else {},
        "new": _mask_metric(combined.rename(columns={"new_iou": "iou", "new_pred_area": "pred_area"}), "iou") if not combined.empty else {},
        "edited_only": {
            "old": _mask_metric(combined[combined["decision"].eq("edited")].rename(columns={"old_iou": "iou", "old_pred_area": "pred_area"}), "iou") if not combined.empty else {},
            "new": _mask_metric(combined[combined["decision"].eq("edited")].rename(columns={"new_iou": "iou", "new_pred_area": "pred_area"}), "iou") if not combined.empty else {},
        },
        "rejected_only": {
            "old": _mask_metric(combined[combined["decision"].eq("rejected")].rename(columns={"old_iou": "iou", "old_pred_area": "pred_area"}), "iou") if not combined.empty else {},
            "new": _mask_metric(combined[combined["decision"].eq("rejected")].rename(columns={"new_iou": "iou", "new_pred_area": "pred_area"}), "iou") if not combined.empty else {},
        },
        "mean_iou_delta_new_minus_old": float(combined["iou_delta_new_minus_old"].mean()) if not combined.empty else 0.0,
    }
    _write_json(run_root / "mask_comparison" / "metrics.json", report)
    return report


def _validate_paths() -> None:
    for path in (
        NEW_INSTANCE,
        OLD_INSTANCE,
        NEW_TEACHING,
        NEW_MULTIPLICITY,
        OLD_TEACHING,
        OLD_MULTIPLICITY,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated QL2603 new-model deployment and evaluation")
    parser.add_argument("--output-root", default="", help="Existing empty run directory; defaults to a timestamped artifacts/v2/runs directory")
    parser.add_argument("--reviewed", nargs="*", default=list(REVIEWED_PLATES))
    parser.add_argument("--unreviewed", nargs="*", default=list(UNREVIEWED_PLATES))
    args = parser.parse_args()

    reviewed = tuple(args.reviewed)
    unreviewed = tuple(args.unreviewed)
    selected = reviewed + unreviewed
    if len(reviewed) != 3 or len(unreviewed) != 3 or len(set(selected)) != 6:
        raise ValueError("This batch requires three distinct reviewed and three distinct unreviewed plates.")
    for slug in selected:
        if not (PLATE_ROOT / slug).exists():
            raise FileNotFoundError(PLATE_ROOT / slug)
    _validate_paths()

    if args.output_root:
        run_root = Path(args.output_root).expanduser().resolve()
        if run_root.exists() and any(run_root.iterdir()):
            raise FileExistsError(f"output root is not empty: {run_root}")
    else:
        run_root = ROOT / "artifacts" / "v2" / "runs" / f"ql2603-model-deployment-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=False)

    deployment: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for slug in selected:
        deployment[slug] = _run_deployment_plate(slug, run_root)
    recognition = {slug: _run_recognition_eval(slug, run_root) for slug in reviewed}
    mask = _run_mask_eval(run_root, deployment)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": "ql2603",
        "reviewed_plates": list(reviewed),
        "unreviewed_plates": list(unreviewed),
        "scope": "T0/T1/T2 model inference plus V3 temporal/state layer",
        "models": {
            "new_instance": _model_record(NEW_INSTANCE),
            "old_instance": _model_record(OLD_INSTANCE),
            "new_teaching": _model_record(NEW_TEACHING),
            "new_multiplicity": _model_record(NEW_MULTIPLICITY),
            "old_teaching": _model_record(OLD_TEACHING),
            "old_multiplicity": _model_record(OLD_MULTIPLICITY),
            "temporal": None,
        },
        "v3_actual_backend": "heuristic_behavior_v1 / active state fusion; pairwise checkpoint not available",
        "deployment": deployment,
        "recognition_evaluation": recognition,
        "mask_evaluation": mask,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(run_root / "report.json", report)
    print(str((run_root / "report.json").resolve()), flush=True)


if __name__ == "__main__":
    main()

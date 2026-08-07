from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import artifact_path
from .hierarchy import suppress_nested_single_candidates
from .temporal_appearance import refine_ambiguous_temporal_appearance
from .teaching import ensure_teaching_features, joint_training_sources


MULTIPLICITY_LABELS = {
    "single",
    "touching_doublet",
    "cluster_3plus",
    "not_cell",
    "skip",
}
MULTIPLICITY_CLASSES = ["single", "touching_doublet", "cluster_3plus"]
INTEGRATED_REVIEW_LABELS = {
    "single",
    "touching_doublet",
    "cluster_3plus",
    "debris",
    "invalid",
    "uncertain",
}


def ensure_multiplicity_table(database: str | Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS multiplicity_labels (
                multiplicity_label_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL UNIQUE,
                well TEXT NOT NULL,
                timepoint TEXT NOT NULL,
                x_px REAL NOT NULL,
                y_px REAL NOT NULL,
                label TEXT NOT NULL,
                source TEXT NOT NULL,
                reviewer TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )


def read_multiplicity_labels(database: str | Path) -> pd.DataFrame:
    ensure_multiplicity_table(database)
    with sqlite3.connect(database) as connection:
        return pd.read_sql_query(
            "SELECT * FROM multiplicity_labels ORDER BY updated_at DESC",
            connection,
        )


def multiplicity_stats(database: str | Path) -> dict[str, Any]:
    labels = read_multiplicity_labels(database)
    counts = labels["label"].value_counts().to_dict() if not labels.empty else {}
    return {
        "total": int(len(labels)),
        "counts": {
            label: int(counts.get(label, 0))
            for label in MULTIPLICITY_LABELS
        },
        "recommended_minimums": {
            "single": 40,
            "touching_doublet": 20,
            "cluster_3plus": 10,
            "not_cell": 20,
        },
    }


def save_multiplicity_labels(
    database: str | Path,
    items: list[dict[str, Any]],
    reviewer: str,
) -> int:
    ensure_multiplicity_table(database)
    updated = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in items:
        label = str(item["label"])
        if label not in MULTIPLICITY_LABELS:
            raise ValueError(f"Invalid multiplicity label: {label}")
        rows.append(
            (
                str(item["candidate_id"]),
                str(item["well"]).upper(),
                str(item["timepoint"]).upper(),
                float(item["x_px"]),
                float(item["y_px"]),
                label,
                str(item.get("source", "quick_multiplicity")),
                reviewer,
                updated,
            )
        )
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO multiplicity_labels (
              candidate_id, well, timepoint, x_px, y_px, label,
              source, reviewer, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
              label=excluded.label,
              source=excluded.source,
              reviewer=excluded.reviewer,
              updated_at=excluded.updated_at
            """,
            rows,
        )
    return len(rows)


def _human_cell_marker_counts(
    database: str | Path, candidates: pd.DataFrame
) -> pd.Series:
    try:
        with sqlite3.connect(database) as connection:
            points = pd.read_sql_query(
                """
                SELECT well, timepoint, x_px, y_px
                FROM annotations
                WHERE object_type = 'cell'
                """,
                connection,
            )
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        points = pd.DataFrame()
    counts = pd.Series(0, index=candidates.index, dtype=np.int64)
    if points.empty:
        return counts
    for (well, timepoint), indices in candidates.groupby(
        ["well", "timepoint"]
    ).groups.items():
        local_points = points[
            (points["well"] == well)
            & (points["timepoint"] == timepoint)
        ]
        if local_points.empty:
            continue
        point_xy = local_points[["x_px", "y_px"]].to_numpy(float)
        for index in indices:
            row = candidates.loc[index]
            radius = max(12.0, min(40.0, float(row["diameter_px"]) * 0.75))
            distances = np.hypot(
                point_xy[:, 0] - float(row["x_px"]),
                point_xy[:, 1] - float(row["y_px"]),
            )
            counts.at[index] = int((distances <= radius).sum())
    return counts


def multiplicity_queue(
    config: dict[str, Any],
    database: str | Path,
    mode: str,
    limit: int,
) -> list[dict[str, Any]]:
    source = artifact_path(
        config, "predictions", "latest_auto_annotations.csv"
    )
    if not source.exists():
        return []
    frame = pd.read_csv(source)
    excluded = {
        str(well).upper()
        for well in config.get("review_queue", {}).get("excluded_wells", [])
    }
    frame = frame[
        frame["timepoint"].isin(["T0", "T1", "T2"])
        & ~frame["well"].astype(str).str.upper().isin(excluded)
        & (frame["auto_status"] != "deterministic_wall_invalid")
        & frame["area_px"].astype(float).between(8, 1200)
        & (
            (frame["cell_probability"].astype(float) >= 0.12)
            | (
                (frame["predicted_label"] == "debris")
                & (frame["debris_probability"].astype(float) >= 0.45)
            )
        )
    ].copy()
    if frame.empty:
        return []

    labels = read_multiplicity_labels(database)
    labelled_ids = set(labels["candidate_id"].astype(str))
    frame = frame[
        ~frame["candidate_id"].astype(str).isin(labelled_ids)
    ].copy()
    if frame.empty:
        return []

    marker_counts = _human_cell_marker_counts(database, frame)
    area_score = np.clip(
        np.log1p(frame["area_px"].astype(float) / 28.0) / np.log(8.0),
        0,
        1,
    )
    two_lobe_shape = (
        0.30 * area_score
        + 0.18 * frame["eccentricity"].astype(float).clip(0, 1)
        + 0.16 * (1.0 - frame["circularity"].astype(float).clip(0, 1))
        + 0.12 * (1.0 - frame["extent"].astype(float).clip(0, 1))
        + 0.24 * frame["cell_probability"].astype(float).clip(0, 1)
    )
    frame["human_cell_markers_nearby"] = marker_counts
    frame["doublet_priority"] = (
        two_lobe_shape
        + 1.25 * (marker_counts >= 2).astype(float)
        + 0.25 * (marker_counts == 1).astype(float)
    )
    frame["candidate_rank_reason"] = np.select(
        [marker_counts >= 2, marker_counts == 1],
        ["reviewed_multi_center_region", "reviewed_cell_region"],
        default="model_shape_screen",
    )

    if mode == "uncertain":
        frame["_sort"] = (
            frame["cell_probability"].astype(float) - 0.5
        ).abs()
        frame = frame.sort_values(
            ["_sort", "doublet_priority"], ascending=[True, False]
        )
    elif mode == "diverse":
        frame = (
            frame.sort_values("doublet_priority", ascending=False)
            .groupby(["well", "timepoint"], group_keys=False)
            .head(2)
            .sort_values("doublet_priority", ascending=False)
        )
    else:
        frame = frame.sort_values(
            ["doublet_priority", "cell_probability"],
            ascending=[False, False],
        )

    columns = [
        "candidate_id",
        "well",
        "timepoint",
        "x_px",
        "y_px",
        "area_px",
        "diameter_px",
        "cell_probability",
        "debris_probability",
        "invalid_probability",
        "doublet_priority",
        "candidate_rank_reason",
    ]
    selected = frame.head(max(1, min(int(limit), 100))).copy()
    return selected[columns].replace({np.nan: None}).to_dict(orient="records")


def _multiplicity_targets_for_source(
    config: dict[str, Any],
    database: str | Path,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    labels = read_multiplicity_labels(database)
    targets = labels[labels["label"].isin(MULTIPLICITY_CLASSES)].copy()
    # Corrections made on the integrated audit page are higher-value labels
    # because the reviewer has seen the model's proposed class first.
    ensure_integrated_review_table(database)
    with sqlite3.connect(database) as connection:
        integrated = pd.read_sql_query(
            """
            SELECT candidate_id, reviewed_label AS label, updated_at
            FROM integrated_training_reviews
            WHERE reviewed_label IN (
              'single', 'touching_doublet', 'cluster_3plus'
            )
            ORDER BY updated_at
            """,
            connection,
        )
    if not integrated.empty:
        integrated = integrated.drop_duplicates(
            "candidate_id", keep="last"
        )
        targets = targets[
            ~targets["candidate_id"].astype(str).isin(
                integrated["candidate_id"].astype(str)
            )
        ]
        integrated["source"] = "integrated_review"
        targets = pd.concat(
            [targets, integrated[["candidate_id", "label", "source"]]],
            ignore_index=True,
        )
    if targets.empty:
        return targets
    index_lookup = {
        candidate_id: index
        for index, candidate_id in enumerate(
            metadata["candidate_id"].astype(str)
        )
    }
    targets = targets[
        targets["candidate_id"].astype(str).isin(index_lookup)
    ].copy()
    if targets.empty:
        return targets
    geometry = metadata[
        [
            "candidate_id",
            "well",
            "timepoint",
            "x_px",
            "y_px",
            "diameter_px",
        ]
    ].copy()
    geometry["candidate_id"] = geometry["candidate_id"].astype(str)
    target_geometry = targets.drop(
        columns=["well", "timepoint", "x_px", "y_px", "diameter_px"],
        errors="ignore",
    ).merge(geometry, on="candidate_id", how="left")
    target_geometry["training_label"] = target_geometry["label"]
    target_geometry["training_confidence"] = 1.0
    target_geometry["reviewed_label"] = target_geometry["label"]
    target_geometry = suppress_nested_single_candidates(
        target_geometry,
        config,
        label_column="training_label",
        confidence_column="training_confidence",
    )
    suppressed_ids = set(
        target_geometry.loc[
            target_geometry["is_hierarchy_suppressed"]
            | target_geometry["is_duplicate_suppressed"],
            "candidate_id",
        ].astype(str)
    )
    return targets[
        ~targets["candidate_id"].astype(str).isin(suppressed_ids)
    ].copy()


def train_multiplicity_classifier(
    config: dict[str, Any], database: str | Path
) -> dict[str, Any]:
    metadata, features = ensure_teaching_features(config)
    target_parts: list[pd.DataFrame] = []
    feature_parts: list[np.ndarray] = []
    for (
        dataset_name,
        source_config,
        source_database,
        source_metadata,
        source_features,
        _,
    ) in joint_training_sources(config, database, metadata, features):
        source_targets = _multiplicity_targets_for_source(
            source_config, source_database, source_metadata
        )
        if source_targets.empty:
            continue
        source_lookup = {
            candidate_id: index
            for index, candidate_id in enumerate(
                source_metadata["candidate_id"].astype(str)
            )
        }
        source_targets = source_targets[
            source_targets["candidate_id"].astype(str).isin(source_lookup)
        ].copy()
        if source_targets.empty:
            continue
        source_indices = np.asarray(
            [
                source_lookup[str(value)]
                for value in source_targets["candidate_id"]
            ],
            dtype=np.int64,
        )
        source_targets["training_dataset"] = dataset_name
        source_targets["training_key"] = (
            dataset_name + "::" + source_targets["candidate_id"].astype(str)
        )
        target_parts.append(source_targets)
        feature_parts.append(source_features[source_indices])
    if not target_parts:
        raise ValueError("No reviewed multiplicity training labels are available.")
    targets = pd.concat(target_parts, ignore_index=True)
    training_features = np.concatenate(feature_parts).astype(np.float32)
    if set(targets["label"]) != set(MULTIPLICITY_CLASSES):
        raise ValueError(
            "Single, touching-doublet, and cluster labels are all required."
        )
    targets["class_index"] = targets["label"].map(
        {name: index for index, name in enumerate(MULTIPLICITY_CLASSES)}
    )
    x = torch.from_numpy(training_features).float()
    y = torch.from_numpy(
        targets["class_index"].to_numpy(np.int64, copy=True)
    )
    counts = np.bincount(y.numpy(), minlength=len(MULTIPLICITY_CLASSES))
    class_weights = counts.sum() / np.maximum(counts, 1)
    class_weights = class_weights / class_weights.mean()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config.get("teaching", {}).get("seed", 20260730)) + 17
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.LayerNorm(features.shape[1]),
        nn.Linear(features.shape[1], 48),
        nn.GELU(),
        nn.Dropout(0.12),
        nn.Linear(48, len(MULTIPLICITY_CLASSES)),
    ).to(device)
    x, y = x.to(device), y.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.08)
    weight_tensor = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )
    epochs = int(config.get("multiplicity", {}).get("epochs", 260))
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits, y, weight=weight_tensor, label_smoothing=0.05
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        fit_predictions = model(x).argmax(dim=1)
        fit_accuracy = float((fit_predictions == y).float().mean().item())
        all_features = torch.from_numpy(features).float().to(device)
        batches = []
        for start in range(0, len(all_features), 1024):
            batches.append(
                torch.softmax(model(all_features[start : start + 1024]), dim=1)
                .cpu()
                .numpy()
            )
    linear_probabilities = np.concatenate(batches)

    similarities = features @ training_features.T
    neighbour_count = min(5, len(training_features))
    neighbour_indices = np.argpartition(
        similarities, -neighbour_count, axis=1
    )[:, -neighbour_count:]
    neighbour_similarities = np.take_along_axis(
        similarities, neighbour_indices, axis=1
    )
    neighbour_weights = np.exp(
        np.clip((neighbour_similarities - 0.45) * 9.0, -8, 8)
    )
    target_classes = targets["class_index"].to_numpy(np.int64)
    neighbour_probabilities = np.zeros_like(linear_probabilities)
    for class_index in range(len(MULTIPLICITY_CLASSES)):
        neighbour_probabilities[:, class_index] = (
            neighbour_weights
            * (target_classes[neighbour_indices] == class_index)
        ).sum(axis=1)
    neighbour_probabilities /= np.maximum(
        neighbour_probabilities.sum(axis=1, keepdims=True), 1e-8
    )
    probabilities = 0.55 * linear_probabilities + 0.45 * neighbour_probabilities

    predictions = metadata.copy()
    for index, name in enumerate(MULTIPLICITY_CLASSES):
        predictions[f"{name}_probability"] = probabilities[:, index]
    predictions["predicted_multiplicity"] = [
        MULTIPLICITY_CLASSES[index] for index in probabilities.argmax(axis=1)
    ]
    predictions["multiplicity_confidence"] = probabilities.max(axis=1)
    predictions["multiplicity_uncertainty"] = (
        1.0 - predictions["multiplicity_confidence"]
    )
    prediction_path = artifact_path(
        config, "predictions", "multiplicity_predictions.csv"
    )
    predictions.to_csv(prediction_path, index=False, encoding="utf-8")
    checkpoint_path = artifact_path(
        config, "models", "multiplicity_classifier.pt"
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dimensions": int(features.shape[1]),
            "classes": MULTIPLICITY_CLASSES,
            "extractor": "torchvision_resnet18_imagenet1k_v1",
            "training_features": torch.from_numpy(
                training_features.astype(np.float32)
            ),
            "target_classes": torch.from_numpy(
                target_classes.astype(np.int64)
            ),
            "knn_blend": 0.45,
        },
        checkpoint_path,
    )
    report = {
        "status": "ready",
        "device": str(device),
        "training_samples": int(len(targets)),
        "class_counts": {
            name: int((targets["label"] == name).sum())
            for name in MULTIPLICITY_CLASSES
        },
        "training_dataset_counts": {
            str(name): int(count)
            for name, count in targets["training_dataset"].value_counts().items()
        },
        "training_accuracy": fit_accuracy,
        "prediction_count": int(len(predictions)),
        "checkpoint": str(checkpoint_path),
        "predictions": str(prediction_path),
        "metric_scope": "training fit only; continued review required",
    }
    artifact_path(
        config, "models", "multiplicity_classifier.json"
    ).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def ensure_integrated_review_table(database: str | Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS integrated_training_reviews (
                integrated_review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                reviewed_label TEXT NOT NULL,
                decision TEXT NOT NULL,
                reviewer TEXT,
                updated_at TEXT NOT NULL,
                UNIQUE(round_id, candidate_id)
            )
            """
        )


def carry_forward_integrated_reviews(
    database: str | Path,
    round_id: str,
    predictions: pd.DataFrame,
) -> int:
    """Copy the latest human decision for stable candidates into a new round."""
    ensure_integrated_review_table(database)
    with sqlite3.connect(database) as connection:
        previous = pd.read_sql_query(
            """
            SELECT integrated_review_id, candidate_id, reviewed_label,
                   reviewer, updated_at
            FROM integrated_training_reviews
            WHERE round_id <> ?
            ORDER BY updated_at, integrated_review_id
            """,
            connection,
            params=(round_id,),
        )
        try:
            missed = pd.read_sql_query(
                """
                SELECT quick_missed_id AS integrated_review_id,
                       candidate_id, reviewed_label, reviewer, updated_at
                FROM quick_missed_objects
                WHERE round_id <> ?
                """,
                connection,
                params=(round_id,),
            )
            previous = pd.concat(
                [previous, missed], ignore_index=True, sort=False
            )
        except (sqlite3.OperationalError, pd.errors.DatabaseError):
            pass
        if previous.empty:
            return 0
        previous = (
            previous.sort_values(["updated_at", "integrated_review_id"])
            .drop_duplicates("candidate_id", keep="last")
        )
        labels = dict(
            zip(
                predictions["candidate_id"].astype(str),
                predictions["integrated_label"].astype(str),
            )
        )
        updated = datetime.now(timezone.utc).isoformat()
        rows = []
        for review in previous.itertuples(index=False):
            candidate_id = str(review.candidate_id)
            predicted = labels.get(candidate_id)
            reviewed = str(review.reviewed_label)
            if (
                predicted not in INTEGRATED_REVIEW_LABELS
                or reviewed not in INTEGRATED_REVIEW_LABELS
            ):
                continue
            rows.append(
                (
                    round_id,
                    candidate_id,
                    predicted,
                    reviewed,
                    "approved" if predicted == reviewed else "corrected",
                    str(review.reviewer or "") or "carry-forward",
                    updated,
                )
            )
        connection.executemany(
            """
            INSERT INTO integrated_training_reviews (
              round_id, candidate_id, predicted_label, reviewed_label,
              decision, reviewer, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(round_id, candidate_id) DO UPDATE SET
              predicted_label=excluded.predicted_label,
              reviewed_label=excluded.reviewed_label,
              decision=excluded.decision,
              reviewer=excluded.reviewer,
              updated_at=excluded.updated_at
            """,
            rows,
        )
    return len(rows)


def generate_integrated_training_round(
    config: dict[str, Any], database: str | Path
) -> dict[str, Any]:
    morphology_path = artifact_path(
        config, "predictions", "latest_auto_annotations.csv"
    )
    multiplicity_path = artifact_path(
        config, "predictions", "multiplicity_predictions.csv"
    )
    morphology = pd.read_csv(morphology_path)
    multiplicity = pd.read_csv(multiplicity_path)[
        [
            "candidate_id",
            "single_probability",
            "touching_doublet_probability",
            "cluster_3plus_probability",
            "predicted_multiplicity",
            "multiplicity_confidence",
            "multiplicity_uncertainty",
        ]
    ]
    result = morphology.merge(multiplicity, on="candidate_id", how="left")
    threshold = float(
        config.get("multiplicity", {}).get("automatic_threshold", 0.62)
    )
    result["integrated_label"] = "unmarked"
    result.loc[result["auto_label"] == "invalid", "integrated_label"] = "invalid"
    result.loc[result["auto_label"] == "debris", "integrated_label"] = "debris"
    is_cell = result["auto_label"] == "cell"
    confident_count = (
        result["multiplicity_confidence"].astype(float) >= threshold
    )
    result.loc[
        is_cell & confident_count, "integrated_label"
    ] = result.loc[is_cell & confident_count, "predicted_multiplicity"]
    result.loc[
        is_cell & ~confident_count, "integrated_label"
    ] = "uncertain"

    reviewed = read_multiplicity_labels(database)
    reviewed_lookup = dict(
        zip(reviewed["candidate_id"].astype(str), reviewed["label"].astype(str))
    )
    for index, row in result.iterrows():
        label = reviewed_lookup.get(str(row["candidate_id"]))
        if label in MULTIPLICITY_CLASSES:
            result.at[index, "integrated_label"] = label
            result.at[index, "multiplicity_confidence"] = 1.0
        elif label == "not_cell":
            result.at[index, "integrated_label"] = (
                "debris"
                if float(row["debris_probability"])
                >= float(row["invalid_probability"])
                else "invalid"
            )

    # Existing temporal lineage review is the highest-confidence source for
    # the T0 root. It must override a small-data multiplicity prediction.
    try:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            lineage_reviews = connection.execute(
                """
                SELECT canonical_target_id, well, object_type,
                       timepoint_points_json
                FROM lineage_reviews
                """
            ).fetchall()
        result_index = {
            str(candidate_id): index
            for index, candidate_id in result["candidate_id"].items()
        }
        for lineage in lineage_reviews:
            candidate_id = str(lineage["canonical_target_id"])
            points_json = json.loads(
                lineage["timepoint_points_json"] or "{}"
            )
            if candidate_id in result_index:
                index = result_index[candidate_id]
            else:
                t0_point = points_json.get("T0", {})
                if (
                    t0_point.get("x_px") is None
                    or t0_point.get("y_px") is None
                ):
                    continue
                local = result[
                    (result["well"] == lineage["well"])
                    & (result["timepoint"] == "T0")
                ]
                if local.empty:
                    continue
                distances = np.hypot(
                    local["x_px"].to_numpy(float)
                    - float(t0_point["x_px"]),
                    local["y_px"].to_numpy(float)
                    - float(t0_point["y_px"]),
                )
                nearest = int(np.argmin(distances))
                if float(distances[nearest]) > 32:
                    continue
                index = int(local.index[nearest])
            object_type = str(lineage["object_type"])
            if object_type == "cell":
                points = points_json.get("T0", {})
                labels_at_t0 = []
                if points.get("present"):
                    labels_at_t0.append(points.get("object_label"))
                    labels_at_t0.extend(
                        child.get("object_label")
                        for child in points.get("additional_points", [])
                    )
                cell_count = max(
                    1, sum(label == "cell" for label in labels_at_t0)
                )
                result.at[index, "integrated_label"] = (
                    "single"
                    if cell_count == 1
                    else "touching_doublet"
                    if cell_count == 2
                    else "cluster_3plus"
                )
                result.at[index, "multiplicity_confidence"] = 1.0
            elif object_type == "debris":
                result.at[index, "integrated_label"] = "debris"
            elif object_type == "irrelevant":
                result.at[index, "integrated_label"] = "invalid"
    except (sqlite3.OperationalError, json.JSONDecodeError):
        pass

    # Giant connected regions, tiny speckles, and deterministically detected
    # wall chains are not useful review objects. Keep them in full inference
    # output as invalid, but never ask the reviewer to label them.
    filter_settings = config.get("candidate_filter", {})
    wall_start = float(
        filter_settings.get("wall_band_start_fraction", 0.42)
    )
    hard_wall_start = float(
        filter_settings.get("hard_wall_exclusion_fraction", 0.44)
    )
    directional_threshold = float(
        filter_settings.get("directional_wall_anisotropy", 0.60)
    )
    wall_chain_min_neighbors = int(
        filter_settings.get("wall_chain_min_neighbors", 3)
    )
    manual_anchor = (
        result.get(
            "candidate_source",
            pd.Series("", index=result.index),
        ).astype(str)
        == "manual_cell_anchor"
    )
    candidate_source = result.get(
        "candidate_source", pd.Series("", index=result.index)
    ).astype(str)
    legacy_wall_rescue = (
        candidate_source == "wall_cell_rescue_peak"
    ) & (
        result.get(
            "wall_rescue_blobness", pd.Series(0.0, index=result.index)
        ).fillna(0).astype(float)
        >= float(
            config.get("dense_detection", {}).get(
                "wall_rescue_minimum_blobness", 0.34
            )
        )
    )
    residual_wall_rescue = (
        candidate_source == "wall_residual_peak"
    ) & (
        result.get(
            "dense_response", pd.Series(0.0, index=result.index)
        ).fillna(0).astype(float)
        >= float(
            config.get("dense_detection", {}).get(
                "wall_residual_minimum_response", 40.0
            )
        )
    )
    wall_cell_rescue = legacy_wall_rescue | residual_wall_rescue
    dynamic_wall_inner = result.get(
        "detected_wall_inner_fraction",
        pd.Series(hard_wall_start, index=result.index),
    ).fillna(hard_wall_start).astype(float)
    hard_wall = (
        (
            result["radial_fraction"].fillna(0).astype(float)
            >= dynamic_wall_inner
        )
        | (
            (
                result["radial_fraction"].fillna(0).astype(float)
                >= wall_start
            )
            & (
                result["review_anisotropy"].fillna(0).astype(float)
                >= directional_threshold
            )
        )
    ) & ~manual_anchor & ~wall_cell_rescue
    hard_invalid = (
        ~result["area_px"].astype(float).between(8, 1200)
        | (result["auto_status"] == "deterministic_wall_invalid")
        | hard_wall
        | (
            (
                result["radial_fraction"].fillna(0).astype(float)
                >= wall_start
            )
            & (
                result["wall_neighbor_count"].fillna(0).astype(float)
                >= wall_chain_min_neighbors
            )
            & ~manual_anchor
            & ~wall_cell_rescue
        )
    )
    result.loc[hard_invalid, "integrated_label"] = "invalid"
    result["manual_override"] = False

    # Explicit point corrections override every automatic filter. Prefer the
    # exact manual anchor and suppress overlapping automatic proposals so one
    # hand-marked cell cannot appear twice in the review result.
    apply_manual_overrides = bool(
        config.get("review_queue", {}).get(
            "apply_manual_point_overrides", True
        )
    )
    try:
        if apply_manual_overrides:
            with sqlite3.connect(database) as connection:
                manual_points = pd.read_sql_query(
                    """
                    SELECT well, timepoint, x_px, y_px, object_type
                    FROM annotations
                    WHERE timepoint IN ('T0', 'T1', 'T2')
                    ORDER BY updated_at
                    """,
                    connection,
                )
        else:
            manual_points = pd.DataFrame(
                columns=["well", "timepoint", "x_px", "y_px", "object_type"]
            )
        assignments: list[tuple[Any, int]] = []
        for manual in manual_points.itertuples(index=False):
            local = result[
                (result["well"] == manual.well)
                & (result["timepoint"] == manual.timepoint)
            ]
            if local.empty:
                continue
            distances = np.hypot(
                local["x_px"].to_numpy(float) - float(manual.x_px),
                local["y_px"].to_numpy(float) - float(manual.y_px),
            )
            within = np.flatnonzero(distances <= 24)
            if not len(within):
                continue
            local_positions = local.index.to_numpy()[within]
            manual_anchor_positions = [
                index
                for index in local_positions
                if str(result.at[index, "candidate_source"])
                == "manual_cell_anchor"
            ]
            chosen = (
                min(
                    manual_anchor_positions,
                    key=lambda index: np.hypot(
                        float(result.at[index, "x_px"])
                        - float(manual.x_px),
                        float(result.at[index, "y_px"])
                        - float(manual.y_px),
                    ),
                )
                if manual_anchor_positions
                else int(local_positions[int(np.argmin(distances[within]))])
            )
            assignments.append((manual, int(chosen)))

        chosen_positions = {chosen for _, chosen in assignments}
        for manual, chosen in assignments:
            local = result[
                (result["well"] == manual.well)
                & (result["timepoint"] == manual.timepoint)
            ]
            distances = np.hypot(
                local["x_px"].to_numpy(float) - float(manual.x_px),
                local["y_px"].to_numpy(float) - float(manual.y_px),
            )
            for duplicate in local.index[distances <= 8]:
                if int(duplicate) not in chosen_positions:
                    result.at[duplicate, "integrated_label"] = "invalid"

        # Two separately reviewed cells can be so close that candidate
        # generation represents them with one connected component. Preserve
        # that review as a doublet/cluster label instead of collapsing it to
        # one cell. The UI and lineage engine can then render the reviewed
        # child points individually.
        grouped: dict[int, list[Any]] = {}
        for manual, chosen in assignments:
            grouped.setdefault(chosen, []).append(manual)
        for chosen, reviewed_points in grouped.items():
            object_types = [str(point.object_type) for point in reviewed_points]
            cell_count = object_types.count("cell")
            if cell_count:
                mapped = (
                    "single"
                    if cell_count == 1
                    else "touching_doublet"
                    if cell_count == 2
                    else "cluster_3plus"
                )
            elif "debris" in object_types:
                mapped = "debris"
            elif "irrelevant" in object_types:
                mapped = "invalid"
            else:
                continue
            result.at[chosen, "integrated_label"] = mapped
            result.at[chosen, "manual_override"] = True
            if mapped in MULTIPLICITY_CLASSES:
                result.at[chosen, "multiplicity_confidence"] = 1.0
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        pass

    # A manual debris/irrelevant click must not resurrect a structure already
    # proven to be the instrument wall. Only explicit manual cell anchors are
    # allowed to remain in the hard wall band.
    result.loc[
        hard_wall & ~manual_anchor,
        ["integrated_label", "manual_override"],
    ] = ["invalid", False]

    # Background texture is useful as a classifier candidate but should not
    # flood the human audit as "debris". Keep only confident debris plus any
    # object previously confirmed by a reviewer.
    reviewed_debris_ids: set[str] = set()
    try:
        if apply_manual_overrides:
            with sqlite3.connect(database) as connection:
                reviewed_debris_ids = {
                    str(row[0])
                    for row in connection.execute(
                        """
                        SELECT candidate_id
                        FROM integrated_training_reviews
                        WHERE reviewed_label = 'debris'
                        """
                    ).fetchall()
                }
    except sqlite3.OperationalError:
        pass
    formal_debris_threshold = float(
        config.get("auto_annotation", {}).get(
            "formal_debris_threshold", 0.92
        )
    )
    low_confidence_debris = (
        (result["integrated_label"] == "debris")
        & (
            result["debris_probability"].fillna(0).astype(float)
            < formal_debris_threshold
        )
        & ~result["candidate_id"].astype(str).isin(reviewed_debris_ids)
        & ~result["manual_override"]
    )
    result.loc[low_confidence_debris, "integrated_label"] = "unmarked"

    round_id = datetime.now().strftime("integrated-round-%Y%m%d-%H%M%S")
    result["integrated_round_id"] = round_id
    result["integrated_confidence"] = np.where(
        result["integrated_label"] == "debris",
        result["debris_probability"].astype(float),
        np.where(
            result["integrated_label"].isin(MULTIPLICITY_CLASSES),
            result["multiplicity_confidence"].astype(float),
            np.where(
                result["integrated_label"] == "invalid",
                result["invalid_probability"].astype(float),
                1.0 - result["uncertainty"].astype(float),
            ),
        ),
    )
    result.loc[hard_invalid, "integrated_confidence"] = 1.0
    result.loc[
        result["manual_override"], "integrated_confidence"
    ] = 1.0
    # First consolidate proposals so duplicate peaks cannot masquerade as
    # temporal division. Then compare registered local appearance for the
    # deliberately narrow morphology-ambiguous band.
    result = suppress_nested_single_candidates(
        result,
        config,
        label_column="integrated_label",
        confidence_column="integrated_confidence",
    )
    result = refine_ambiguous_temporal_appearance(result, config)
    # Temporal evidence can change a parent label, so materialise the final
    # mutually-exclusive instance graph once more.
    result = suppress_nested_single_candidates(
        result,
        config,
        label_column="integrated_label",
        confidence_column="integrated_confidence",
    )
    result["pre_suppression_label"] = result["integrated_label"]
    result["integrated_review_priority"] = (
        1.0 - result["integrated_confidence"].astype(float)
        + 0.25
        * result["integrated_label"].isin(
            ["touching_doublet", "cluster_3plus", "uncertain"]
        ).astype(float)
    )
    round_dir = (
        Path(config["paths"]["artifact_root"])
        / "predictions"
        / round_id
    )
    round_dir.mkdir(parents=True, exist_ok=False)
    result.to_csv(round_dir / "predictions.csv", index=False, encoding="utf-8")
    result.to_csv(
        artifact_path(
            config, "predictions", "latest_integrated_predictions.csv"
        ),
        index=False,
        encoding="utf-8",
    )
    reviewable = result[
        result["integrated_label"].isin(
            [
                "single",
                "touching_doublet",
                "cluster_3plus",
                "debris",
                "uncertain",
            ]
        )
        & result["is_counting_instance"]
    ].copy()
    reviewable.to_csv(
        round_dir / "review_queue.csv", index=False, encoding="utf-8"
    )
    carry_previous = bool(
        config.get("review_queue", {}).get(
            "carry_forward_previous_reviews", True
        )
    )
    carried_review_count = (
        carry_forward_integrated_reviews(database, round_id, reviewable)
        if carry_previous
        else 0
    )
    summary = {
        "round_id": round_id,
        "candidate_count": int(len(result)),
        "review_queue_count": int(len(reviewable)),
        "review_well_count": int(reviewable["well"].nunique()),
        "carried_review_count": carried_review_count,
        "label_counts": {
            key: int(value)
            for key, value in result["integrated_label"].value_counts().items()
        },
        "threshold": threshold,
        "predictions": str(round_dir / "predictions.csv"),
        "review_queue": str(round_dir / "review_queue.csv"),
    }
    (round_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifact_path(
        config, "predictions", "latest_integrated_summary.json"
    ).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ensure_integrated_review_table(database)
    return summary


def save_integrated_reviews(
    database: str | Path,
    round_id: str,
    items: list[dict[str, Any]],
    reviewer: str,
) -> int:
    ensure_integrated_review_table(database)
    updated = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in items:
        predicted = str(item["predicted_label"])
        reviewed = str(item["reviewed_label"])
        if (
            predicted not in INTEGRATED_REVIEW_LABELS
            or reviewed not in INTEGRATED_REVIEW_LABELS
        ):
            raise ValueError("Invalid integrated review label.")
        rows.append(
            (
                round_id,
                str(item["candidate_id"]),
                predicted,
                reviewed,
                "approved" if predicted == reviewed else "corrected",
                reviewer,
                updated,
            )
        )
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO integrated_training_reviews (
              round_id, candidate_id, predicted_label, reviewed_label,
              decision, reviewer, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(round_id, candidate_id) DO UPDATE SET
              predicted_label=excluded.predicted_label,
              reviewed_label=excluded.reviewed_label,
              decision=excluded.decision,
              reviewer=excluded.reviewer,
              updated_at=excluded.updated_at
            """,
            rows,
        )
    return len(rows)


def integrated_review_stats(
    config: dict[str, Any], database: str | Path
) -> dict[str, Any]:
    summary_path = artifact_path(
        config, "predictions", "latest_integrated_summary.json"
    )
    if not summary_path.exists():
        return {"status": "not_generated"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ensure_integrated_review_table(database)
    with sqlite3.connect(database) as connection:
        reviews = pd.read_sql_query(
            """
            SELECT * FROM integrated_training_reviews
            WHERE round_id = ?
            """,
            connection,
            params=(summary["round_id"],),
        )
    summary["status"] = "ready"
    summary["reviewed_count"] = int(len(reviews))
    summary["approved_count"] = (
        int((reviews["decision"] == "approved").sum())
        if not reviews.empty
        else 0
    )
    summary["corrected_count"] = (
        int((reviews["decision"] == "corrected").sum())
        if not reviews.empty
        else 0
    )
    return summary


def integrated_review_queue(
    config: dict[str, Any],
    database: str | Path,
    mode: str,
    limit: int,
) -> list[dict[str, Any]]:
    source = artifact_path(
        config, "predictions", "latest_integrated_predictions.csv"
    )
    if not source.exists():
        return []
    frame = pd.read_csv(source)
    round_id = str(frame.iloc[0]["integrated_round_id"])
    ensure_integrated_review_table(database)
    with sqlite3.connect(database) as connection:
        reviews = pd.read_sql_query(
            """
            SELECT candidate_id, reviewed_label, decision
            FROM integrated_training_reviews
            WHERE round_id = ?
            """,
            connection,
            params=(round_id,),
        )
    frame = frame.merge(reviews, on="candidate_id", how="left")
    reviewable = frame[
        frame["integrated_label"].isin(
            [
                "single",
                "touching_doublet",
                "cluster_3plus",
                "debris",
                "uncertain",
            ]
        )
    ].copy()
    if mode == "reviewed":
        selected = reviewable[reviewable["reviewed_label"].notna()].copy()
    else:
        selected = reviewable[reviewable["reviewed_label"].isna()].copy()
        if mode == "cell":
            selected = selected[
                selected["integrated_label"].isin(MULTIPLICITY_CLASSES)
            ]
        elif mode == "doublet":
            selected = selected[
                selected["integrated_label"] == "touching_doublet"
            ]
        elif mode == "debris":
            selected = selected[selected["integrated_label"] == "debris"]
        elif mode == "uncertain":
            selected = selected[selected["integrated_label"] == "uncertain"]
    selected = selected.sort_values(
        ["integrated_review_priority", "integrated_confidence"],
        ascending=[False, True],
    ).head(max(1, min(int(limit), 100)))
    columns = [
        "integrated_round_id",
        "candidate_id",
        "well",
        "timepoint",
        "x_px",
        "y_px",
        "area_px",
        "diameter_px",
        "integrated_label",
        "integrated_confidence",
        "cell_probability",
        "debris_probability",
        "invalid_probability",
        "single_probability",
        "touching_doublet_probability",
        "cluster_3plus_probability",
        "reviewed_label",
        "decision",
    ]
    return (
        selected[columns]
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

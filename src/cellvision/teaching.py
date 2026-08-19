from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import artifact_path, is_validation_holdout, load_config
from .dense_candidates import add_wall_neighbor_counts
from .pseudo_labels import _background_anisotropy, _crop_with_padding
from .runtime import ensure_training_allowed


TEACHING_LABELS = {"cell", "debris", "invalid", "skip"}
MORPHOLOGY_CLASSES = ["invalid", "debris", "cell"]


def ensure_teaching_table(database: str | Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS teaching_labels (
                teaching_label_id INTEGER PRIMARY KEY AUTOINCREMENT,
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


def teaching_candidate_pool(config: dict[str, Any]) -> pd.DataFrame:
    source = artifact_path(
        config, "pseudo_labels", "morphology_candidates.csv"
    )
    frame = pd.read_csv(source)
    excluded = {
        str(well).upper()
        for well in config.get("review_queue", {}).get("excluded_wells", [])
    }
    frame = frame[
        frame["timepoint"].isin(["T0", "T1", "T2"])
        & ~frame["well"].str.upper().isin(excluded)
    ].copy()
    return frame.drop_duplicates("candidate_id").reset_index(drop=True)


def save_teaching_labels(
    database: str | Path, items: list[dict[str, Any]], reviewer: str
) -> int:
    ensure_teaching_table(database)
    updated = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in items:
        label = str(item["label"])
        if label not in TEACHING_LABELS:
            raise ValueError(f"Invalid teaching label: {label}")
        rows.append(
            (
                item["candidate_id"],
                item["well"],
                item.get("timepoint", "T0"),
                float(item["x_px"]),
                float(item["y_px"]),
                label,
                item.get("source", "quick_teaching"),
                reviewer,
                updated,
            )
        )
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO teaching_labels (
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


def read_teaching_labels(database: str | Path) -> pd.DataFrame:
    ensure_teaching_table(database)
    with sqlite3.connect(database) as connection:
        return pd.read_sql_query(
            "SELECT * FROM teaching_labels ORDER BY updated_at DESC", connection
        )


def teaching_stats(database: str | Path) -> dict[str, Any]:
    labels = read_teaching_labels(database)
    counts = (
        labels["label"].value_counts().to_dict() if not labels.empty else {}
    )
    return {
        "total": int(len(labels)),
        "counts": {label: int(counts.get(label, 0)) for label in TEACHING_LABELS},
    }


def _feature_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    return (
        artifact_path(config, "cache", "teaching_features.npz"),
        artifact_path(config, "cache", "teaching_feature_candidates.csv"),
        artifact_path(config, "cache", "teaching_feature_environment.json"),
    )


def ensure_teaching_features(
    config: dict[str, Any], *, force: bool = False
) -> tuple[pd.DataFrame, np.ndarray]:
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    cache_path, metadata_path, environment_path = _feature_paths(config)
    candidates = teaching_candidate_pool(config).reset_index(drop=True)
    reusable_features: dict[str, np.ndarray] = {}
    reusable_metadata: dict[str, tuple[str, float, float]] = {}
    if cache_path.exists() and metadata_path.exists() and not force:
        metadata = pd.read_csv(metadata_path)
        cache = np.load(cache_path, allow_pickle=True)
        candidate_ids = cache["candidate_ids"].astype(str)
        if (
            len(metadata) == len(candidates)
            and np.array_equal(
                metadata["candidate_id"].astype(str).to_numpy(), candidate_ids
            )
            and set(candidate_ids) == set(candidates["candidate_id"].astype(str))
        ):
            return metadata, cache["features"].astype(np.float32)

    if cache_path.exists() and metadata_path.exists():
        old_metadata = pd.read_csv(metadata_path)
        old_cache = np.load(cache_path, allow_pickle=True)
        old_features = old_cache["features"].astype(np.float32)
        if len(old_metadata) == len(old_features):
            for index, row in old_metadata.iterrows():
                candidate_id = str(row["candidate_id"])
                reusable_features[candidate_id] = old_features[index]
                reusable_metadata[candidate_id] = (
                    str(row["raw_image_path"]),
                    float(row["x_px"]),
                    float(row["y_px"]),
                )

    reuse_mask = []
    for row in candidates.itertuples(index=False):
        previous = reusable_metadata.get(str(row.candidate_id))
        reuse_mask.append(
            previous is not None
            and previous[0] == str(row.raw_image_path)
            and abs(previous[1] - float(row.x_px)) <= 0.01
            and abs(previous[2] - float(row.y_px)) <= 0.01
        )
    reuse_mask_array = np.asarray(reuse_mask, dtype=bool)
    missing_candidates = candidates.loc[~reuse_mask_array].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    extractor = resnet18(weights=weights)
    extractor.fc = nn.Identity()
    extractor = extractor.eval().to(device)
    means = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    stds = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    batch_size = int(
        config.get("teaching", {}).get("feature_batch_size", 96)
    )
    crop_size = int(config.get("teaching", {}).get("crop_size_px", 128))
    extracted_features: dict[str, np.ndarray] = {}
    pending: list[np.ndarray] = []
    pending_ids: list[str] = []

    def flush() -> None:
        if not pending:
            return
        array = np.stack(pending).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array[:, None]).to(device)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(224, 224), mode="bilinear", align_corners=False
        ).repeat(1, 3, 1, 1)
        tensor = (tensor - means) / stds
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", enabled=device.type == "cuda"
        ):
            embedding = extractor(tensor)
        for candidate_id, feature in zip(
            pending_ids, embedding.float().cpu().numpy()
        ):
            extracted_features[candidate_id] = feature
        pending.clear()
        pending_ids.clear()

    image_groups = list(
        missing_candidates.groupby("raw_image_path", sort=False)
    )
    for group_index, (raw_path, group) in enumerate(image_groups, start=1):
        with Image.open(raw_path) as image:
            raw = np.asarray(image.convert("L"), dtype=np.uint8)
        for row in group.itertuples(index=False):
            pending.append(
                _crop_with_padding(raw, row.x_px, row.y_px, crop_size)
            )
            pending_ids.append(str(row.candidate_id))
            if len(pending) >= batch_size:
                flush()
        if group_index % 20 == 0 or group_index == len(image_groups):
            print(
                "incremental visual feature extraction: "
                f"{group_index}/{len(image_groups)} images",
                flush=True,
            )
    flush()
    metadata = candidates
    feature_rows: list[np.ndarray] = []
    for reusable, row in zip(reuse_mask_array, metadata.itertuples(index=False)):
        candidate_id = str(row.candidate_id)
        feature_rows.append(
            reusable_features[candidate_id]
            if reusable
            else extracted_features[candidate_id]
        )
    features = np.asarray(feature_rows, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8")
    np.savez_compressed(
        cache_path,
        features=features.astype(np.float16),
        candidate_ids=np.asarray(
            metadata["candidate_id"].astype(str).to_list(), dtype=str
        ),
    )
    environment_path.write_text(
        json.dumps(
            {
                "extractor": "torchvision_resnet18_imagenet1k_v1",
                "device": str(device),
                "torch": torch.__version__,
                "candidate_count": int(len(metadata)),
                "reused_feature_count": int(reuse_mask_array.sum()),
                "extracted_feature_count": int((~reuse_mask_array).sum()),
                "crop_size_px": crop_size,
                "embedding_dimensions": int(features.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata, features


def _manual_candidate_labels(
    database: str | Path, candidates: pd.DataFrame
) -> dict[str, str]:
    with sqlite3.connect(database) as connection:
        manual = pd.read_sql_query(
            """
            SELECT well, timepoint, x_px, y_px, object_type
            FROM annotations
            WHERE timepoint IN ('T0', 'T1', 'T2')
            """,
            connection,
        )
    result: dict[str, str] = {}
    mapped = {
        "cell": "cell",
        "debris": "debris",
        "irrelevant": "invalid",
    }
    for row in manual.itertuples(index=False):
        local = candidates[
            (candidates["well"] == row.well)
            & (candidates["timepoint"] == row.timepoint)
        ]
        if local.empty or row.object_type not in mapped:
            continue
        distances = np.hypot(
            local["x_px"].to_numpy(float) - float(row.x_px),
            local["y_px"].to_numpy(float) - float(row.y_px),
        )
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= 32:
            result[str(local.iloc[nearest]["candidate_id"])] = mapped[
                row.object_type
            ]
    return result


def _training_targets(
    config: dict[str, Any],
    database: str | Path,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    labels: dict[str, tuple[str, str, float]] = {}

    # Lower-level teaching comes first. Later sources deliberately override it:
    # a temporal lineage review is more informative than a static quick label.
    teaching = read_teaching_labels(database)
    for row in teaching.itertuples(index=False):
        if row.label != "skip":
            labels[str(row.candidate_id)] = (
                str(row.label),
                "quick_teaching",
                2.0,
            )
    try:
        with sqlite3.connect(database) as connection:
            auto_reviews = pd.read_sql_query(
                """
                SELECT candidate_id, reviewed_label
                FROM auto_annotation_reviews
                ORDER BY updated_at
                """,
                connection,
            )
        for row in auto_reviews.itertuples(index=False):
            labels[str(row.candidate_id)] = (
                str(row.reviewed_label),
                "auto_annotation_review",
                2.25,
            )
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        pass
    try:
        with sqlite3.connect(database) as connection:
            integrated_reviews = pd.read_sql_query(
                """
                SELECT candidate_id, reviewed_label
                FROM integrated_training_reviews
                ORDER BY updated_at
                """,
                connection,
            )
        integrated_mapping = {
            "single": "cell",
            "touching_doublet": "cell",
            "cluster_3plus": "cell",
            "debris": "debris",
            "invalid": "invalid",
        }
        for row in integrated_reviews.itertuples(index=False):
            mapped = integrated_mapping.get(str(row.reviewed_label))
            if mapped:
                labels[str(row.candidate_id)] = (
                    mapped,
                    "integrated_review",
                    2.75,
                )
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        pass
    for candidate_id, label_name in _manual_candidate_labels(
        database, candidates
    ).items():
        labels[candidate_id] = (label_name, "temporal_lineage_review", 3.0)

    # Once a real teaching round contains enough positives, weak pseudo-labels
    # no longer train the classifier. They remain available only for queue
    # ranking and deterministic wall filtering.
    human_cell_count = sum(
        label_name == "cell"
        and source
        in {
            "temporal_lineage_review",
            "auto_annotation_review",
            "quick_teaching",
        }
        for label_name, source, _ in labels.values()
    )
    if human_cell_count < 12:
        for row in candidates[candidates["pseudo_label"] == "cell"].itertuples(
            index=False
        ):
            labels.setdefault(
                str(row.candidate_id),
                ("cell", "strict_pseudo_cell_fallback", 0.45),
            )

    rows = []
    for candidate_id, (label_name, source, weight) in labels.items():
        if label_name not in MORPHOLOGY_CLASSES:
            continue
        rows.append(
            {
                "candidate_id": candidate_id,
                "label": label_name,
                "class_index": MORPHOLOGY_CLASSES.index(label_name),
                "label_source": source,
                "sample_weight": weight,
            }
        )
    targets = pd.DataFrame(rows)
    if targets.empty:
        return targets
    targets = targets.merge(
        candidates[["candidate_id", "timepoint"]].assign(
            candidate_id=lambda frame: frame["candidate_id"].astype(str)
        ),
        on="candidate_id",
        how="left",
    )
    teaching_settings = config.get("teaching", {})
    if bool(teaching_settings.get("timepoint_class_balance", True)):
        group_size = targets.groupby(
            ["timepoint", "label"]
        )["candidate_id"].transform("size").astype(float)
        reference_size = float(
            targets.groupby(["timepoint", "label"]).size().median()
        )
        balance = np.clip(reference_size / group_size, 0.5, 3.0)
        targets["sample_weight"] *= balance
    return targets


def joint_training_sources(
    config: dict[str, Any],
    database: str | Path,
    metadata: pd.DataFrame,
    features: np.ndarray,
) -> list[tuple[str, dict[str, Any], Path, pd.DataFrame, np.ndarray, bool]]:
    """Load independently generated candidate/feature stores for joint training."""
    sources = [
        (
            "primary",
            config,
            Path(database),
            metadata,
            features,
            False,
        )
    ]
    primary_root = Path(config["paths"]["artifact_root"]).resolve()
    for entry in config.get("joint_training", {}).get("sources", []):
        if is_validation_holdout(config, str(entry["config"])):
            continue
        source_config = load_config(str(entry["config"]))
        source_root = Path(source_config["paths"]["artifact_root"]).resolve()
        if source_root == primary_root:
            continue
        candidate_path = artifact_path(
            source_config, "pseudo_labels", "morphology_candidates.csv"
        )
        if not candidate_path.exists():
            continue
        source_database = Path(
            entry.get(
                "database",
                artifact_path(source_config, "annotations", "annotations.db"),
            )
        ).resolve()
        if not source_database.exists():
            continue
        source_metadata, source_features = ensure_teaching_features(source_config)
        sources.append(
            (
                str(entry.get("name", source_root.name)),
                source_config,
                source_database,
                source_metadata,
                source_features,
                bool(entry.get("require_human_labels", True)),
            )
        )
    return sources


def _joint_morphology_training_examples(
    config: dict[str, Any],
    database: str | Path,
    metadata: pd.DataFrame,
    features: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    target_parts: list[pd.DataFrame] = []
    feature_parts: list[np.ndarray] = []
    for (
        dataset_name,
        source_config,
        source_database,
        source_metadata,
        source_features,
        require_human_labels,
    ) in joint_training_sources(config, database, metadata, features):
        targets = _training_targets(
            source_config, source_database, source_metadata
        )
        if require_human_labels and not targets.empty:
            targets = targets[
                ~targets["label_source"].astype(str).str.contains("pseudo")
            ].copy()
        if targets.empty:
            continue
        index_lookup = {
            candidate_id: index
            for index, candidate_id in enumerate(
                source_metadata["candidate_id"].astype(str)
            )
        }
        targets = targets[
            targets["candidate_id"].astype(str).isin(index_lookup)
        ].copy()
        if targets.empty:
            continue
        indices = np.asarray(
            [index_lookup[str(value)] for value in targets["candidate_id"]],
            dtype=np.int64,
        )
        targets["training_dataset"] = dataset_name
        targets["training_key"] = (
            dataset_name + "::" + targets["candidate_id"].astype(str)
        )
        target_parts.append(targets)
        feature_parts.append(source_features[indices])
    if not target_parts:
        return pd.DataFrame(), np.empty((0, features.shape[1]), np.float32)
    return (
        pd.concat(target_parts, ignore_index=True),
        np.concatenate(feature_parts).astype(np.float32),
    )


def train_teaching_classifier(
    config: dict[str, Any], database: str | Path
) -> dict[str, Any]:
    import torch
    from torch import nn

    ensure_training_allowed()
    metadata, features = ensure_teaching_features(config)
    targets, training_features = _joint_morphology_training_examples(
        config, database, metadata, features
    )
    if targets.empty:
        raise ValueError("No reviewed morphology training labels are available.")
    if targets["class_index"].nunique() < 3:
        raise ValueError(
            "Cell, debris, and invalid teaching labels are all required."
        )
    x = torch.from_numpy(training_features).float()
    y = torch.from_numpy(
        targets["class_index"].to_numpy(np.int64, copy=True)
    )
    sample_weights = torch.from_numpy(
        targets["sample_weight"].to_numpy(np.float32, copy=True)
    )
    counts = np.bincount(
        y.numpy(), minlength=len(MORPHOLOGY_CLASSES)
    ).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1)
    class_weights /= class_weights.mean()
    sample_weights *= torch.from_numpy(class_weights[y.numpy()])
    # Keep one large 2603 plate from overwhelming the older reviewed sources
    # while retaining the class/timepoint balancing above.
    source_counts = targets["training_dataset"].value_counts()
    source_target = float(source_counts.median()) if len(source_counts) else 1.0
    source_weights = targets["training_dataset"].map(
        lambda value: np.clip(
            source_target / max(float(source_counts.get(value, 1)), 1.0),
            0.5,
            2.5,
        )
    ).to_numpy(np.float32)
    sample_weights *= torch.from_numpy(source_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, sample_weights = (
        x.to(device),
        y.to(device),
        sample_weights.to(device),
    )
    seed = int(config.get("teaching", {}).get("seed", 20260730))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.LayerNorm(features.shape[1]),
        nn.Linear(features.shape[1], len(MORPHOLOGY_CLASSES)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("teaching", {}).get("learning_rate", 0.003)),
        weight_decay=0.05,
    )
    epochs = int(config.get("teaching", {}).get("epochs", 180))
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        losses = torch.nn.functional.cross_entropy(
            logits, y, reduction="none", label_smoothing=0.04
        )
        loss = (losses * sample_weights).sum() / sample_weights.sum()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        training_predictions = model(x).argmax(dim=1)
        training_accuracy = float(
            (training_predictions == y).float().mean().item()
        )
        all_features = torch.from_numpy(features).float().to(device)
        probability_batches: list[np.ndarray] = []
        for start in range(0, len(all_features), 1024):
            logits = model(all_features[start : start + 1024])
            probability_batches.append(
                torch.softmax(logits, dim=1).cpu().numpy()
            )
    linear_probabilities = np.concatenate(probability_batches)

    # Blend the regularized linear head with a local nearest-neighbour vote.
    # This makes newly confirmed visual examples affect similar candidates
    # immediately without allowing a tiny dataset to drive extreme logits.
    similarities = features @ training_features.T
    neighbour_count = min(7, len(training_features))
    neighbour_indices = np.argpartition(
        similarities, -neighbour_count, axis=1
    )[:, -neighbour_count:]
    neighbour_similarities = np.take_along_axis(
        similarities, neighbour_indices, axis=1
    )
    neighbour_weights = np.exp(
        np.clip((neighbour_similarities - 0.45) * 8.0, -8, 8)
    )
    target_classes = targets["class_index"].to_numpy(np.int64)
    neighbour_probabilities = np.zeros_like(linear_probabilities)
    for class_index in range(len(MORPHOLOGY_CLASSES)):
        neighbour_probabilities[:, class_index] = (
            neighbour_weights
            * (target_classes[neighbour_indices] == class_index)
        ).sum(axis=1)
    neighbour_probabilities /= np.maximum(
        neighbour_probabilities.sum(axis=1, keepdims=True), 1e-8
    )
    probabilities = 0.7 * linear_probabilities + 0.3 * neighbour_probabilities
    timepoint_blend = float(
        config.get("teaching", {}).get("timepoint_knn_blend", 0.40)
    )
    for timepoint in ["T0", "T1", "T2"]:
        training_mask = (
            targets["timepoint"].astype(str).eq(timepoint).to_numpy()
        )
        query_indices = np.flatnonzero(
            metadata["timepoint"].astype(str).eq(timepoint).to_numpy()
        )
        if int(training_mask.sum()) < 3 or not len(query_indices):
            continue
        local_training_features = training_features[training_mask]
        local_target_classes = target_classes[training_mask]
        local_similarities = (
            features[query_indices] @ local_training_features.T
        )
        local_neighbour_count = min(7, len(local_training_features))
        local_neighbour_indices = np.argpartition(
            local_similarities, -local_neighbour_count, axis=1
        )[:, -local_neighbour_count:]
        local_neighbour_similarities = np.take_along_axis(
            local_similarities, local_neighbour_indices, axis=1
        )
        local_neighbour_weights = np.exp(
            np.clip((local_neighbour_similarities - 0.45) * 8.0, -8, 8)
        )
        local_probabilities = np.zeros(
            (len(query_indices), len(MORPHOLOGY_CLASSES)),
            dtype=np.float32,
        )
        for class_index in range(len(MORPHOLOGY_CLASSES)):
            local_probabilities[:, class_index] = (
                local_neighbour_weights
                * (
                    local_target_classes[local_neighbour_indices]
                    == class_index
                )
            ).sum(axis=1)
        local_probabilities /= np.maximum(
            local_probabilities.sum(axis=1, keepdims=True), 1e-8
        )
        probabilities[query_indices] = (
            (1.0 - timepoint_blend) * linear_probabilities[query_indices]
            + timepoint_blend * local_probabilities
        )
    predictions = metadata.copy()
    for class_index, class_name in enumerate(MORPHOLOGY_CLASSES):
        predictions[f"{class_name}_probability"] = probabilities[:, class_index]
    predictions["predicted_label"] = [
        MORPHOLOGY_CLASSES[index] for index in probabilities.argmax(axis=1)
    ]
    predictions["confidence"] = probabilities.max(axis=1)
    predictions["uncertainty"] = 1.0 - predictions["confidence"]
    prediction_path = artifact_path(
        config, "predictions", "teaching_classifier_predictions.csv"
    )
    predictions.to_csv(prediction_path, index=False, encoding="utf-8")
    checkpoint_path = artifact_path(
        config, "models", "teaching_classifier.pt"
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dimensions": int(features.shape[1]),
            "classes": MORPHOLOGY_CLASSES,
            "extractor": "torchvision_resnet18_imagenet1k_v1",
            "training_features": torch.from_numpy(
                training_features.astype(np.float32)
            ),
            "target_classes": torch.from_numpy(
                target_classes.astype(np.int64)
            ),
            "target_timepoints": targets["timepoint"].astype(str).tolist(),
            "global_knn_blend": 0.30,
            "timepoint_knn_blend": timepoint_blend,
        },
        checkpoint_path,
    )
    report = {
        "status": "ready",
        "device": str(device),
        "training_samples": int(len(targets)),
        "class_counts": {
            class_name: int((targets["label"] == class_name).sum())
            for class_name in MORPHOLOGY_CLASSES
        },
        "timepoint_class_counts": {
            f"{timepoint}:{label}": int(count)
            for (timepoint, label), count in targets.groupby(
                ["timepoint", "label"]
            ).size().items()
        },
        "training_dataset_counts": {
            str(name): int(count)
            for name, count in targets["training_dataset"].value_counts().items()
        },
        "source_balance": {
            str(name): float(weight)
            for name, weight in zip(
                source_counts.index,
                source_counts.map(
                    lambda count: np.clip(
                        source_target / max(float(count), 1.0), 0.5, 2.5
                    )
                ),
            )
        },
        "human_teaching_samples": int(
            (targets["label_source"] == "quick_teaching").sum()
        ),
        "temporal_lineage_samples": int(
            (targets["label_source"] == "temporal_lineage_review").sum()
        ),
        "auto_review_samples": int(
            (targets["label_source"] == "auto_annotation_review").sum()
        ),
        "weak_training_samples": int(
            targets["label_source"].str.contains("pseudo").sum()
        ),
        "training_accuracy": training_accuracy,
        "prediction_count": int(len(predictions)),
        "checkpoint": str(checkpoint_path),
        "predictions": str(prediction_path),
        "metric_scope": "training fit only; external validation required",
    }
    artifact_path(
        config, "models", "teaching_classifier.json"
    ).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    targets.to_csv(
        artifact_path(config, "annotations", "teaching_training_targets.csv"),
        index=False,
        encoding="utf-8",
    )
    return report


def ensure_auto_review_table(database: str | Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS auto_annotation_reviews (
                auto_review_id INTEGER PRIMARY KEY AUTOINCREMENT,
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


def ensure_candidate_anisotropy(
    config: dict[str, Any], candidates: pd.DataFrame
) -> pd.DataFrame:
    filter_settings = config.get("candidate_filter", {})
    wall_start = float(
        filter_settings.get("wall_band_start_fraction", 0.42)
    )
    candidates = add_wall_neighbor_counts(
        candidates,
        wall_start=max(0.0, wall_start - 0.02),
        radius=float(
            filter_settings.get("wall_chain_neighbor_radius_px", 110)
        ),
    )
    cache_path = artifact_path(
        config, "cache", "review_candidate_anisotropy_wall_v2.csv"
    )
    measured = (
        pd.read_csv(cache_path)
        if cache_path.exists()
        else pd.DataFrame(columns=["candidate_id", "review_anisotropy"])
    )
    measured_ids = set(measured["candidate_id"].astype(str))
    missing = candidates[
        ~candidates["candidate_id"].astype(str).isin(measured_ids)
    ].copy()
    new_rows: list[dict[str, Any]] = []
    # Measure every timepoint in and immediately inside the wall band.  The old
    # cache used a constant fallback for T1/T2, which allowed rim texture to
    # masquerade as debris.  Interior objects keep their inexpensive feature.
    measure_mask = missing["radial_fraction"].astype(float) >= max(
        0.0, wall_start - 0.02
    )
    fallback = missing[~measure_mask]
    for row in fallback.itertuples(index=False):
        new_rows.append(
            {
                "candidate_id": row.candidate_id,
                "review_anisotropy": float(
                    getattr(row, "background_anisotropy", 1.0)
                ),
            }
        )
    measure_missing = missing[measure_mask]
    groups = list(measure_missing.groupby("raw_image_path", sort=False))
    for group_index, (raw_path, group) in enumerate(groups, start=1):
        with Image.open(raw_path) as image:
            raw = np.asarray(image.convert("L"), dtype=np.uint8)
        for row in group.itertuples(index=False):
            new_rows.append(
                {
                    "candidate_id": row.candidate_id,
                    "review_anisotropy": _background_anisotropy(
                        raw, row.x_px, row.y_px
                    ),
                }
            )
        if group_index % 20 == 0 or group_index == len(groups):
            print(
                f"full wall-feature cache: {group_index}/{len(groups)} images",
                flush=True,
            )
    if new_rows:
        measured = pd.concat(
            [measured, pd.DataFrame(new_rows)], ignore_index=True
        ).drop_duplicates("candidate_id", keep="last")
        measured.to_csv(cache_path, index=False, encoding="utf-8")
    return candidates.merge(
        measured[["candidate_id", "review_anisotropy"]],
        on="candidate_id",
        how="left",
    )


def derive_temporal_classifications(
    config: dict[str, Any],
    database: str | Path,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Classify T0 origins using morphology plus registered T1/T2 behavior."""
    reviewed: dict[str, tuple[str, str]] = {}
    try:
        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                """
                SELECT canonical_target_id, object_type, viability
                FROM lineage_reviews
                """
            ).fetchall()
        reviewed = {
            str(candidate_id): (str(object_type), str(viability))
            for candidate_id, object_type, viability in rows
        }
    except sqlite3.OperationalError:
        pass

    resolution = float(config["calibration"]["resolution_um_per_pixel"])
    significant_motion_um = float(
        config.get("review_queue", {}).get(
            "significant_displacement_um", 20.0
        )
    )
    search_radius_px = float(
        config.get("auto_annotation", {}).get(
            "temporal_search_radius_px", 80
        )
    )
    division_radius_px = float(
        config.get("auto_annotation", {}).get(
            "temporal_division_radius_px", 55
        )
    )
    by_well_timepoint = {
        (str(well), str(timepoint)): group
        for (well, timepoint), group in predictions.groupby(
            ["well", "timepoint"], sort=False
        )
    }
    rows: list[dict[str, Any]] = []
    for t0 in predictions[predictions["timepoint"] == "T0"].itertuples(
        index=False
    ):
        candidate_id = str(t0.candidate_id)
        temporal_label = "uncertain"
        temporal_confidence = 0.5
        evidence = "insufficient_temporal_evidence"
        maximum_motion_um = np.nan
        later_cell_count = 0

        human = reviewed.get(candidate_id)
        if human:
            object_type, viability = human
            if object_type == "cell" and viability in {"live", "dead"}:
                temporal_label = f"{viability}_cell"
            elif object_type == "debris":
                temporal_label = "debris"
            elif object_type == "irrelevant":
                temporal_label = "unmarked"
            temporal_confidence = 1.0
            evidence = "human_temporal_lineage_review"
        elif str(t0.auto_label) == "invalid":
            temporal_label = "unmarked"
            temporal_confidence = float(t0.invalid_probability)
            evidence = "learned_or_geometric_wall_invalid"
        elif str(t0.auto_label) == "debris":
            temporal_label = "debris"
            temporal_confidence = float(t0.debris_probability)
            evidence = "static_debris_morphology"
        else:
            motions: list[float] = []
            area_ratios: list[float] = []
            presence_count = 0
            maximum_nearby_cells = 0
            t0_x = float(t0.aligned_x_px)
            t0_y = float(t0.aligned_y_px)
            t0_area = max(float(t0.area_px), 1.0)
            for timepoint in ("T1", "T2"):
                local = by_well_timepoint.get(
                    (str(t0.well), timepoint),
                    pd.DataFrame(),
                )
                if local.empty:
                    continue
                plausible = local[
                    (local["auto_label"] != "invalid")
                    & (local["cell_probability"].astype(float) >= 0.35)
                ].copy()
                if plausible.empty:
                    continue
                distances = np.hypot(
                    plausible["aligned_x_px"].to_numpy(float) - t0_x,
                    plausible["aligned_y_px"].to_numpy(float) - t0_y,
                )
                nearby = plausible[distances <= search_radius_px].copy()
                if nearby.empty:
                    continue
                nearby_distances = distances[distances <= search_radius_px]
                nearest_offset = int(np.argmin(nearby_distances))
                nearest = nearby.iloc[nearest_offset]
                nearest_distance = float(nearby_distances[nearest_offset])
                presence_count += 1
                motions.append(nearest_distance)
                area_ratios.append(
                    max(float(nearest["area_px"]), 1.0) / t0_area
                )
                maximum_nearby_cells = max(
                    maximum_nearby_cells,
                    int((nearby_distances <= division_radius_px).sum()),
                )
            later_cell_count = maximum_nearby_cells
            maximum_motion_um = (
                max(motions) * resolution if motions else np.nan
            )
            largest_area_ratio = max(area_ratios, default=1.0)
            division_or_growth = (
                maximum_nearby_cells >= 2 or largest_area_ratio >= 1.6
            )
            clear_motion = (
                bool(motions)
                and float(maximum_motion_um) >= significant_motion_um
            )
            stable_persistence = (
                presence_count == 2
                and maximum_nearby_cells <= 1
                and float(maximum_motion_um) <= 10.0
                and all(0.55 <= ratio <= 1.45 for ratio in area_ratios)
            )
            if division_or_growth or clear_motion:
                temporal_label = "live_cell"
                temporal_confidence = min(
                    0.98,
                    0.70
                    + 0.10 * int(division_or_growth)
                    + 0.10 * int(clear_motion),
                )
                evidence = (
                    "registered_division_growth_or_significant_motion"
                )
            elif stable_persistence:
                temporal_label = "dead_cell"
                temporal_confidence = 0.72
                evidence = "persistent_cell_without_division_or_motion"
            else:
                temporal_label = "uncertain"
                temporal_confidence = 0.5
                evidence = "cell_morphology_but_temporal_evidence_incomplete"

        rows.append(
            {
                "candidate_id": candidate_id,
                "well": str(t0.well),
                "timepoint": "T0",
                "temporal_label": temporal_label,
                "temporal_confidence": temporal_confidence,
                "temporal_evidence": evidence,
                "maximum_motion_um": maximum_motion_um,
                "later_cell_count": later_cell_count,
            }
        )
    # Inference-only views can deliberately contain T1/T2 images without a
    # T0 root (for example the late-growth detector reusing the mature-cell
    # pipeline for T3/T4).  Preserve the merge schema even when there are no
    # temporal roots instead of returning a column-less DataFrame.
    result = pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "well",
            "timepoint",
            "temporal_label",
            "temporal_confidence",
            "temporal_evidence",
            "maximum_motion_um",
            "later_cell_count",
        ],
    )
    result.to_csv(
        artifact_path(
            config, "predictions", "latest_temporal_classifications.csv"
        ),
        index=False,
        encoding="utf-8",
    )
    return result


def generate_auto_annotation_round(
    config: dict[str, Any], database: str | Path
) -> dict[str, Any]:
    prediction_path = artifact_path(
        config, "predictions", "teaching_classifier_predictions.csv"
    )
    if not prediction_path.exists():
        raise ValueError("Teaching classifier predictions are unavailable.")
    predictions = pd.read_csv(prediction_path)
    required_probabilities = {
        "cell_probability",
        "debris_probability",
        "invalid_probability",
    }
    if not required_probabilities.issubset(predictions.columns):
        raise ValueError("Retrain the three-class teaching classifier first.")

    round_id = datetime.now().strftime("auto-round-%Y%m%d-%H%M%S")
    settings = config.get("auto_annotation", {})
    cell_threshold = float(settings.get("cell_threshold", 0.90))
    debris_threshold = float(settings.get("debris_threshold", 0.88))
    invalid_threshold = float(settings.get("invalid_threshold", 0.92))
    result = ensure_candidate_anisotropy(config, predictions.copy())
    result["round_id"] = round_id
    result["auto_label"] = result["predicted_label"]
    result["auto_status"] = "needs_review"
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
    radial_fraction = result["radial_fraction"].astype(float)
    dynamic_wall_inner = result.get(
        "detected_wall_inner_fraction",
        pd.Series(hard_wall_start, index=result.index),
    ).fillna(hard_wall_start).astype(float)
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
    in_wall_band = radial_fraction >= wall_start
    hard_outer_wall = (
        (radial_fraction >= dynamic_wall_inner)
        & ~manual_anchor
        & ~wall_cell_rescue
    )
    directional_outer_wall = (
        in_wall_band
        & (
            result["review_anisotropy"].fillna(1.0).astype(float)
            >= directional_threshold
        )
        & ~manual_anchor
        & ~wall_cell_rescue
    )
    directional_wall = (
        result["review_anisotropy"].fillna(1.0).astype(float) >= 0.52
    )
    wall_chain_or_shape = (
        (
            result["wall_neighbor_count"].fillna(0).astype(float)
            >= wall_chain_min_neighbors
        )
        | (result["circularity"].astype(float) < 0.45)
        | (result["solidity"].astype(float) < 0.68)
        | (result["extent"].astype(float) < 0.32)
    )
    learned_wall = (
        (result["invalid_probability"].astype(float) >= 0.70)
        & (
            result["review_anisotropy"].fillna(1.0).astype(float)
            >= 0.45
        )
    )
    strong_cell_override = (
        (result["cell_probability"].astype(float) >= 0.92)
        & (result["circularity"].astype(float) >= 0.55)
        & (result["solidity"].astype(float) >= 0.76)
        & (result["extent"].astype(float) >= 0.42)
    )
    deterministic_wall = (
        hard_outer_wall
        | directional_outer_wall
        | (
            in_wall_band
            & (learned_wall | (directional_wall & wall_chain_or_shape))
            & ~strong_cell_override
            & ~manual_anchor
            & ~wall_cell_rescue
        )
    )
    result.loc[deterministic_wall, "auto_label"] = "invalid"
    result.loc[deterministic_wall, "confidence"] = 1.0
    result.loc[deterministic_wall, "auto_status"] = (
        "deterministic_wall_invalid"
    )

    safe_cell_context = (
        (result["radial_fraction"].astype(float) < 0.42)
        | strong_cell_override
        | wall_cell_rescue
        | (
            result["review_anisotropy"].fillna(1.0).astype(float)
            < 0.45
        )
    )
    high_cell = (
        (result["cell_probability"] >= cell_threshold)
        & safe_cell_context
        & ~deterministic_wall
    )
    high_debris = (
        (result["debris_probability"] >= debris_threshold)
        & ~deterministic_wall
    )
    high_invalid = (
        (result["invalid_probability"] >= invalid_threshold)
        & ~deterministic_wall
    )
    result.loc[high_cell, ["auto_label", "auto_status"]] = [
        "cell",
        "auto_high_confidence",
    ]
    result.loc[high_debris, ["auto_label", "auto_status"]] = [
        "debris",
        "auto_high_confidence",
    ]
    result.loc[high_invalid, ["auto_label", "auto_status"]] = [
        "invalid",
        "auto_high_confidence",
    ]
    result["review_priority"] = (
        result["uncertainty"].astype(float)
        + 0.55 * result["cell_probability"].astype(float)
        + 0.25 * (result["auto_label"] == "cell").astype(float)
    )

    temporal = derive_temporal_classifications(config, database, result)
    temporal_columns = [
        "candidate_id",
        "temporal_label",
        "temporal_confidence",
        "temporal_evidence",
        "maximum_motion_um",
        "later_cell_count",
    ]
    result = result.merge(
        temporal[temporal_columns], on="candidate_id", how="left"
    )

    round_dir = (
        Path(config["paths"]["artifact_root"]) / "predictions" / round_id
    )
    round_dir.mkdir(parents=True, exist_ok=False)
    round_predictions = round_dir / "predictions.csv"
    result.to_csv(round_predictions, index=False, encoding="utf-8")
    latest_path = artifact_path(
        config, "predictions", "latest_auto_annotations.csv"
    )
    result.to_csv(latest_path, index=False, encoding="utf-8")

    review_candidates = result[
        (result["timepoint"] == "T0")
        & (
            result["auto_label"].isin(["cell", "debris"])
            | (
                (result["auto_status"] == "needs_review")
                & (result["uncertainty"].astype(float) >= 0.38)
            )
        )
    ].copy()
    maximum_per_well = int(settings.get("max_review_per_well", 12))
    review_candidates = (
        review_candidates.sort_values(
            ["well", "review_priority"], ascending=[True, False]
        )
        .groupby("well", group_keys=False)
        .head(maximum_per_well)
    )
    review_candidates.to_csv(
        round_dir / "review_queue.csv", index=False, encoding="utf-8"
    )
    review_candidates.to_csv(
        artifact_path(config, "annotations", "auto_review_queue.csv"),
        index=False,
        encoding="utf-8",
    )
    counts = {
        "round_id": round_id,
        "candidate_count": int(len(result)),
        "status_counts": {
            key: int(value)
            for key, value in result["auto_status"].value_counts().items()
        },
        "label_counts": {
            key: int(value)
            for key, value in result["auto_label"].value_counts().items()
        },
        "review_queue_count": int(len(review_candidates)),
        "review_well_count": int(review_candidates["well"].nunique()),
        "temporal_label_counts": {
            key: int(value)
            for key, value in temporal["temporal_label"].value_counts().items()
        },
        "thresholds": {
            "cell": cell_threshold,
            "debris": debris_threshold,
            "invalid": invalid_threshold,
        },
        "predictions": str(round_predictions),
        "review_queue": str(round_dir / "review_queue.csv"),
        "death_classification": "temporal_lineage_evidence_only",
    }
    (round_dir / "summary.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifact_path(
        config, "predictions", "latest_auto_annotation.json"
    ).write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ensure_auto_review_table(database)
    return counts


def save_auto_annotation_reviews(
    database: str | Path,
    round_id: str,
    items: list[dict[str, Any]],
    reviewer: str,
) -> int:
    ensure_auto_review_table(database)
    allowed = set(MORPHOLOGY_CLASSES)
    updated = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in items:
        predicted = str(item["predicted_label"])
        reviewed = str(item["reviewed_label"])
        if predicted not in allowed or reviewed not in allowed:
            raise ValueError("Invalid automatic annotation review label.")
        decision = "approved" if predicted == reviewed else "corrected"
        rows.append(
            (
                round_id,
                item["candidate_id"],
                predicted,
                reviewed,
                decision,
                reviewer,
                updated,
            )
        )
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO auto_annotation_reviews (
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


def auto_annotation_stats(
    config: dict[str, Any], database: str | Path
) -> dict[str, Any]:
    summary_path = artifact_path(
        config, "predictions", "latest_auto_annotation.json"
    )
    if not summary_path.exists():
        return {"status": "not_generated"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ensure_auto_review_table(database)
    with sqlite3.connect(database) as connection:
        reviews = pd.read_sql_query(
            """
            SELECT * FROM auto_annotation_reviews
            WHERE round_id = ?
            """,
            connection,
            params=(summary["round_id"],),
        )
    summary["status"] = "ready"
    summary["reviewed_count"] = int(len(reviews))
    summary["approved_count"] = int(
        (reviews["decision"] == "approved").sum()
    ) if not reviews.empty else 0
    summary["corrected_count"] = int(
        (reviews["decision"] == "corrected").sum()
    ) if not reviews.empty else 0
    return summary


def auto_annotation_queue(
    config: dict[str, Any],
    database: str | Path,
    mode: str,
    limit: int,
) -> list[dict[str, Any]]:
    source = artifact_path(config, "predictions", "latest_auto_annotations.csv")
    if not source.exists():
        return []
    frame = pd.read_csv(source)
    round_id = str(frame.iloc[0]["round_id"])
    ensure_auto_review_table(database)
    with sqlite3.connect(database) as connection:
        reviews = pd.read_sql_query(
            """
            SELECT candidate_id, reviewed_label, decision
            FROM auto_annotation_reviews
            WHERE round_id = ?
            """,
            connection,
            params=(round_id,),
        )
    frame = frame.merge(reviews, on="candidate_id", how="left")
    if mode == "reviewed":
        selected = frame[frame["reviewed_label"].notna()].copy()
    else:
        selected = frame[frame["reviewed_label"].isna()].copy()
        if mode == "cell":
            selected = selected[selected["auto_label"] == "cell"]
        elif mode == "audit":
            selected = selected[
                selected["auto_status"] == "auto_high_confidence"
            ]
        else:
            selected = selected[
                selected["auto_status"] == "needs_review"
            ]
    selected = selected.sort_values(
        ["review_priority", "confidence"], ascending=[False, True]
    ).head(max(1, min(int(limit), 100)))
    columns = [
        "round_id",
        "candidate_id",
        "well",
        "timepoint",
        "x_px",
        "y_px",
        "area_px",
        "diameter_px",
        "auto_label",
        "auto_status",
        "confidence",
        "cell_probability",
        "debris_probability",
        "invalid_probability",
        "reviewed_label",
        "decision",
    ]
    return (
        selected[columns]
        .replace({np.nan: None})
        .to_dict(orient="records")
    )


def teaching_queue(
    config: dict[str, Any],
    database: str | Path,
    mode: str,
    limit: int,
    anchor_id: str | None = None,
) -> list[dict[str, Any]]:
    candidates = teaching_candidate_pool(config)
    candidates = candidates[candidates["timepoint"] == "T0"].copy()
    teaching = read_teaching_labels(database)
    labelled_ids = set(teaching["candidate_id"].astype(str))
    candidates = candidates[
        ~candidates["candidate_id"].astype(str).isin(labelled_ids)
    ].copy()
    prediction_path = artifact_path(
        config, "predictions", "teaching_classifier_predictions.csv"
    )
    if prediction_path.exists():
        scores = pd.read_csv(prediction_path)[
            ["candidate_id", "cell_probability", "uncertainty"]
        ]
        candidates = candidates.merge(scores, on="candidate_id", how="left")
    else:
        candidates["cell_probability"] = np.nan
        candidates["uncertainty"] = np.nan

    if mode == "similar":
        if not anchor_id:
            raise ValueError("anchor_id is required for similar mode")
        metadata, features = ensure_teaching_features(config)
        lookup = {
            value: index
            for index, value in enumerate(metadata["candidate_id"].astype(str))
        }
        if anchor_id not in lookup:
            raise ValueError("Anchor candidate is unavailable")
        similarities = features @ features[lookup[anchor_id]]
        similarity_frame = pd.DataFrame(
            {
                "candidate_id": metadata["candidate_id"].astype(str),
                "similarity": similarities,
            }
        )
        candidates = candidates.merge(
            similarity_frame, on="candidate_id", how="left"
        ).sort_values("similarity", ascending=False)
    elif mode == "hard_negative":
        candidates = candidates[
            candidates["pseudo_label"] != "cell"
        ].sort_values("cell_probability", ascending=False, na_position="last")
    elif mode == "uncertain":
        candidates = candidates.sort_values(
            "uncertainty", ascending=False, na_position="last"
        )
    else:
        candidates["_seed_priority"] = (
            (candidates["pseudo_label"] == "cell").astype(float) * 2
            + (1 - candidates["background_anisotropy"].clip(0, 1))
            + candidates["temporal_support"].clip(0, 2) / 2
        )
        candidates = candidates.sort_values(
            "_seed_priority", ascending=False
        )

    columns = [
        "candidate_id",
        "well",
        "timepoint",
        "x_px",
        "y_px",
        "area_px",
        "diameter_px",
        "pseudo_label",
        "cell_probability",
        "uncertainty",
    ]
    if "similarity" in candidates.columns:
        columns.append("similarity")
    result = candidates.head(max(1, min(int(limit), 100))).copy()
    records = result[columns].replace({np.nan: None}).to_dict(orient="records")
    return records

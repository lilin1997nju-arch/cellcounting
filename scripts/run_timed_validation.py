"""Run a fresh T0--T2 validation pass and record wall-clock time per stage.

This intentionally keeps each validation plate in its own artifact root.  It
is used for external validation only: no annotations are imported and no T3/T4
folders are read because the config contains explicit T0/T1/T2 directories.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from cellvision.config import artifact_path, load_config
from cellvision.dense_candidates import augment_candidates_with_dense_raw_proposals
from cellvision.manifest import build_manifest
from cellvision.model_inference import (
    predict_multiplicity_checkpoint,
    predict_teaching_checkpoint,
)
from cellvision.multiplicity import generate_integrated_training_round
from cellvision.pseudo_labels import build_morphology_pseudo_labels
from cellvision.review_server import initialize_database
from cellvision.teaching import generate_auto_annotation_round
from cellvision.v2_instance_inference import infer_v2_instances
from cellvision.v2_temporal_inference import infer_v2_temporal_evidence
from cellvision.well_screening import build_well_screening


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return {"rows": int(len(value)), "columns": list(value.columns)}
    if isinstance(value, pd.Series):
        return {"length": int(len(value)), "name": str(value.name)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _run_stage(
    stages: list[dict[str, Any]], name: str, fn: Callable[[], Any]
) -> Any:
    _sync_cuda()
    started = time.perf_counter()
    result = fn()
    _sync_cuda()
    elapsed = time.perf_counter() - started
    record = {
        "stage": name,
        "elapsed_seconds": round(elapsed, 3),
        "result": _jsonable(result),
    }
    stages.append(record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return result


def run(config_path: str | Path, source_artifacts: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    stages: list[dict[str, Any]] = []
    started = time.perf_counter()
    source = Path(source_artifacts).resolve()
    instance_checkpoint = source / "v2" / "models" / "latest_instance_segmenter.pt"
    temporal_checkpoint = source / "v2" / "models" / "latest_temporal_evidence.pt"
    if not instance_checkpoint.exists():
        raise FileNotFoundError(instance_checkpoint)
    if not temporal_checkpoint.exists():
        raise FileNotFoundError(temporal_checkpoint)

    database = artifact_path(config, "annotations", "annotations.db")
    _run_stage(stages, "initialize_database", lambda: str(initialize_database(database)))
    def _manifest_stage() -> dict[str, int]:
        images, sequences = build_manifest(config)
        return {"images": int(len(images)), "sequences": int(len(sequences))}

    _run_stage(stages, "build_manifest", _manifest_stage)
    images = pd.read_csv(artifact_path(config, "manifests", "images.csv"))
    sequences = pd.read_csv(artifact_path(config, "manifests", "sequences.csv"))

    # The manifest is a hard scope check: this run must contain exactly T0/T1/T2.
    manifest = pd.read_csv(artifact_path(config, "manifests", "images.csv"))
    timepoints = sorted(set(manifest["timepoint"].astype(str)))
    if timepoints != ["T0", "T1", "T2"]:
        raise RuntimeError(f"Validation scope is not T0/T1/T2 only: {timepoints}")
    if len(manifest) != 96 * 3 or len(sequences) != 96:
        raise RuntimeError(
            f"Expected 288 images and 96 wells, got {len(manifest)} and {len(sequences)}"
        )

    _run_stage(stages, "build_cf_candidate_manifest", lambda: str(build_morphology_pseudo_labels(config)))
    _run_stage(
        stages,
        "dense_raw_candidate_augmentation",
        lambda: augment_candidates_with_dense_raw_proposals(config, database),
    )
    _run_stage(
        stages,
        "morphology_inference",
        lambda: predict_teaching_checkpoint(
            config, source / "models" / "teaching_classifier.pt"
        ),
    )
    _run_stage(
        stages,
        "auto_annotation_round",
        lambda: generate_auto_annotation_round(config, database),
    )
    _run_stage(
        stages,
        "multiplicity_inference",
        lambda: predict_multiplicity_checkpoint(
            config, source / "models" / "multiplicity_classifier.pt"
        ),
    )
    _run_stage(
        stages,
        "integrated_training_round",
        lambda: generate_integrated_training_round(config, database),
    )
    _run_stage(
        stages,
        "v2_instance_segmentation",
        lambda: str(infer_v2_instances(config, instance_checkpoint)),
    )
    _run_stage(
        stages,
        "v2_temporal_evidence",
        lambda: str(infer_v2_temporal_evidence(config, temporal_checkpoint)),
    )
    _run_stage(
        stages,
        "well_screening_summary",
        lambda: build_well_screening(config, database),
    )

    predictions_path = artifact_path(config, "predictions", "latest_v2_predictions.csv")
    predictions = pd.read_csv(predictions_path, low_memory=False)
    timing = {
        "config": str(Path(config_path).resolve()),
        "experiment_id": config["experiment"]["experiment_id"],
        "plate_id": config["experiment"]["plate_id"],
        "scope": {"timepoints": timepoints, "image_count": len(manifest), "well_count": len(sequences)},
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "stages": stages,
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
        "prediction_count": int(len(predictions)),
        "v2_valid_instances": int(predictions.get("v2_mask_valid", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "v2_unique_instances": int(predictions.get("v2_is_unique_instance", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "v2_temporal_candidates": int(predictions.get("v2_is_temporal_candidate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "final_label_counts": {str(key): int(value) for key, value in predictions["integrated_label"].value_counts().items()},
        "note": "Fresh inference-only validation. Human labels and T3/T4 are excluded.",
    }
    output = artifact_path(config, "validation_timing.json")
    output.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(timing, ensure_ascii=False, indent=2), flush=True)
    return timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-artifacts", default="artifacts")
    args = parser.parse_args()
    run(args.config, args.source_artifacts)


if __name__ == "__main__":
    main()

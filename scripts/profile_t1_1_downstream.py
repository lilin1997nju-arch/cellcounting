from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from cellvision.config import load_config
from cellvision.model_inference import (
    predict_multiplicity_checkpoint,
    predict_teaching_checkpoint,
)
from cellvision.multiplicity import generate_integrated_training_round
from cellvision.teaching import generate_auto_annotation_round
from cellvision.v2_instance_inference import infer_v2_instances
from cellvision.v2_temporal_inference import infer_v2_temporal_evidence
from cellvision.well_screening import build_well_screening


def run_stage(name: str, function: Callable[[], Any]) -> tuple[Any, float]:
    print(f"starting {name}", flush=True)
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started
    print(f"finished {name}: {elapsed:.3f}s", flush=True)
    return result, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--teaching-checkpoint", required=True, type=Path)
    parser.add_argument("--multiplicity-checkpoint", required=True, type=Path)
    parser.add_argument("--instance-checkpoint", required=True, type=Path)
    parser.add_argument("--temporal-checkpoint", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    config["paths"]["artifact_root"] = str(args.artifact_root.resolve())
    config.setdefault("v2_inference", {})["reuse_unchanged_stage"] = False
    database = args.artifact_root / "annotations" / "annotations.db"
    timings: dict[str, float] = {}
    outputs: dict[str, str] = {}

    stages: list[tuple[str, Callable[[], Any]]] = [
        (
            "morphology_classifier",
            lambda: predict_teaching_checkpoint(config, args.teaching_checkpoint),
        ),
        (
            "multiplicity_classifier",
            lambda: predict_multiplicity_checkpoint(
                config, args.multiplicity_checkpoint
            ),
        ),
        (
            "auto_annotation_round",
            lambda: generate_auto_annotation_round(config, database),
        ),
        (
            "integrated_round",
            lambda: generate_integrated_training_round(config, database),
        ),
        (
            "v2_instance_segmentation",
            lambda: infer_v2_instances(config, args.instance_checkpoint),
        ),
        (
            "v3_temporal_evidence",
            lambda: infer_v2_temporal_evidence(config, args.temporal_checkpoint),
        ),
        (
            "well_screening",
            lambda: build_well_screening(config, database),
        ),
    ]
    total_started = time.perf_counter()
    for name, function in stages:
        result, elapsed = run_stage(name, function)
        timings[name] = round(elapsed, 3)
        if isinstance(result, Path):
            outputs[name] = str(result.resolve())
    report = {
        "artifact_root": str(args.artifact_root.resolve()),
        "timing_seconds": timings,
        "outputs": outputs,
        "total_elapsed_seconds": round(time.perf_counter() - total_started, 3),
    }
    output = args.artifact_root / "downstream_profile.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

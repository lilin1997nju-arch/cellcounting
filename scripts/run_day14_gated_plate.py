"""Run the complete Day14-gated screening pipeline for one 96-well plate.

The config must describe only the Day0/Day1/Day2 image folders. Day7 and
Day14 are read directly from the session and screening manifests, so they can
never enter instance or temporal inference accidentally.
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
from cellvision.gated_screening import build_gated_plate_report
from cellvision.provenance import create_run_metadata
from cellvision.manifest import build_manifest
from cellvision.model_inference import predict_multiplicity_checkpoint, predict_teaching_checkpoint
from cellvision.multiplicity import generate_integrated_training_round
from cellvision.pseudo_labels import build_morphology_pseudo_labels
from cellvision.review_server import initialize_database
from cellvision.teaching import generate_auto_annotation_round
from cellvision.v2_instance_inference import infer_v2_instances
from cellvision.v2_temporal_inference import infer_v2_temporal_evidence
from cellvision.well_screening import build_well_screening


def _run_stage(stages: list[dict[str, Any]], name: str, fn: Callable[[], Any]) -> Any:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    value = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stages.append({"stage": name, "elapsed_seconds": round(time.perf_counter() - started, 3)})
    print(json.dumps(stages[-1], ensure_ascii=False), flush=True)
    return value


def _restrict_early_manifests(config: dict[str, Any], wells: set[str]) -> dict[str, int]:
    images_path = artifact_path(config, "manifests", "images.csv")
    sequences_path = artifact_path(config, "manifests", "sequences.csv")
    images = pd.read_csv(images_path, low_memory=False)
    sequences = pd.read_csv(sequences_path, low_memory=False)
    present_timepoints = set(images["timepoint"].astype(str))
    if present_timepoints != {"T0", "T1", "T2"}:
        raise RuntimeError(
            "The gated early config must contain only T0/T1/T2; "
            f"found {sorted(present_timepoints)}"
        )
    images = images[images["well"].astype(str).str.upper().isin(wells)].copy()
    sequences = sequences[sequences["well"].astype(str).str.upper().isin(wells)].copy()
    images.to_csv(images_path, index=False, encoding="utf-8")
    sequences.to_csv(sequences_path, index=False, encoding="utf-8")
    expected_images = len(wells) * 3
    if len(images) != expected_images or len(sequences) != len(wells):
        raise RuntimeError(
            f"Gated manifest is incomplete: expected {expected_images} images/"
            f"{len(wells)} wells, got {len(images)}/{len(sequences)}"
        )
    return {"well_count": len(wells), "image_count": len(images)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_artifacts).resolve()
    stages: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    metadata_path = create_run_metadata(
        output_dir,
        config=config,
        input_paths=[args.config, args.day14_csv, args.sessions_csv],
        model_paths=[
            source / "models" / "teaching_classifier.pt",
            source / "models" / "multiplicity_classifier.pt",
            source / "v2" / "models" / "latest_instance_segmenter.pt",
        ],
        extra={"group_id": str(args.group_id), "endpoint_day_label": getattr(args, "endpoint_day_label", "Day14")},
    )

    provisional = _run_stage(
        stages,
        "day14_gate",
        lambda: build_gated_plate_report(
            args.day14_csv,
            args.group_id,
            output_dir,
            sessions_csv=args.sessions_csv,
            locate_day7=False,
            endpoint_day_label=getattr(args, "endpoint_day_label", "Day14"),
        ),
    )
    positive_wells = {
        str(row["well"]).upper()
        for row in provisional["wells"]
        if bool(row["day14_obvious_growth"]) and not bool(row["is_positive_control"])
    }
    if not positive_wells:
        summary = {
            "group_id": args.group_id,
            "positive_well_count": 0,
            "stages": stages,
            "total_elapsed_seconds": round(time.perf_counter() - total_started, 3),
            "report_json": provisional["report_json"],
        }
        create_run_metadata(output_dir, config=config, input_paths=[args.config, args.day14_csv, args.sessions_csv], status="skipped_no_growth", stages=stages, extra={"group_id": str(args.group_id), "metadata_path": str(metadata_path)})
        (output_dir / "pipeline_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    database = artifact_path(config, "annotations", "annotations.db")
    _run_stage(stages, "initialize_database", lambda: initialize_database(database))

    def build_restricted_manifest() -> dict[str, int]:
        build_manifest(config)
        return _restrict_early_manifests(config, positive_wells)

    _run_stage(stages, "build_positive_only_t0_t2_manifest", build_restricted_manifest)
    _run_stage(stages, "build_cf_candidates", lambda: build_morphology_pseudo_labels(config))
    _run_stage(
        stages,
        "dense_candidate_augmentation",
        lambda: augment_candidates_with_dense_raw_proposals(config, database),
    )
    _run_stage(
        stages,
        "morphology_inference",
        lambda: predict_teaching_checkpoint(config, source / "models" / "teaching_classifier.pt"),
    )
    _run_stage(stages, "auto_annotation_round", lambda: generate_auto_annotation_round(config, database))
    _run_stage(
        stages,
        "multiplicity_inference",
        lambda: predict_multiplicity_checkpoint(config, source / "models" / "multiplicity_classifier.pt"),
    )
    _run_stage(stages, "integrated_round", lambda: generate_integrated_training_round(config, database))
    _run_stage(
        stages,
        "v2_instance_segmentation",
        lambda: infer_v2_instances(config, source / "v2" / "models" / "latest_instance_segmenter.pt"),
    )
    _run_stage(
        stages,
        "v2_temporal_evidence",
        lambda: infer_v2_temporal_evidence(config),
    )
    early_path = artifact_path(config, "predictions", "latest_well_screening.csv")
    _run_stage(stages, "early_well_screening", lambda: build_well_screening(config, database))
    final = _run_stage(
        stages,
        "day7_localization_and_final_report",
        lambda: build_gated_plate_report(
            args.day14_csv,
            args.group_id,
            output_dir,
            early_screening_csv=early_path,
            sessions_csv=args.sessions_csv,
            locate_day7=True,
            endpoint_day_label=getattr(args, "endpoint_day_label", "Day14"),
        ),
    )
    summary = {
        "group_id": args.group_id,
        "positive_well_count": len(positive_wells),
        "skipped_sample_well_count": final["day14_skipped_sample_wells"],
        "category_counts": final["category_counts"],
        "undetermined_reason_counts": final["undetermined_reason_counts"],
        "stages": stages,
        "total_elapsed_seconds": round(time.perf_counter() - total_started, 3),
        "report_json": final["report_json"],
        "report_csv": final["report_csv"],
    }
    create_run_metadata(output_dir, config=config, input_paths=[args.config, args.day14_csv, args.sessions_csv], status="completed", stages=stages, extra={"group_id": str(args.group_id), "metadata_path": str(metadata_path)})
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="T0/T1/T2-only plate config")
    parser.add_argument("--day14-csv", required=True)
    parser.add_argument("--sessions-csv", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-artifacts", default="artifacts")
    parser.add_argument("--endpoint-day-label", default="Day14")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

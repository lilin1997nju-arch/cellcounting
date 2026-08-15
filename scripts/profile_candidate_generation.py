from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from cellvision.config import artifact_path, load_config
from cellvision.dense_candidates import augment_candidates_with_dense_raw_proposals
from cellvision.pseudo_labels import build_morphology_pseudo_labels


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _source_counts(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path, low_memory=False)
    if "candidate_source" not in frame:
        return {"cf_component": int(len(frame))}
    return {
        str(name): int(count)
        for name, count in frame["candidate_source"].fillna("unknown").value_counts().items()
    }


def run(
    config_path: Path,
    output_root: Path,
    *,
    registration_cache: Path | None = None,
    cf_only: bool = False,
    base_candidate_root: Path | None = None,
    minimum_interior_response: float | None = None,
    multiscale_gaussian_truncate: float | None = None,
    multiscale_backend: str | None = None,
) -> dict[str, Any]:
    source_config = load_config(config_path)
    source_root = Path(source_config["paths"]["artifact_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "manifests/images.csv",
        "manifests/sequences.csv",
        "annotations/annotations.db",
    ):
        _copy(source_root / relative, output_root / relative)
    if registration_cache is not None:
        destination = output_root / "cache" / "registration"
        destination.mkdir(parents=True, exist_ok=True)
        for source in registration_cache.glob("*.json"):
            _copy(source, destination / source.name)
    if base_candidate_root is not None:
        for relative in (
            "pseudo_labels/morphology_candidates.csv",
            "pseudo_labels/morphology_candidates.source.json",
        ):
            source = base_candidate_root / relative
            if source.exists():
                _copy(source, output_root / relative)

    config = load_config(config_path)
    config["paths"]["artifact_root"] = str(output_root.resolve())
    config.setdefault("morphology_classifier", {})["reuse_candidate_manifest"] = False
    config.setdefault("dense_detection", {})["reuse_unchanged_stage"] = False
    if minimum_interior_response is not None:
        config["dense_detection"]["minimum_interior_peak_response"] = (
            minimum_interior_response
        )
    if multiscale_gaussian_truncate is not None:
        config["dense_detection"]["multiscale_gaussian_truncate"] = (
            multiscale_gaussian_truncate
        )
    if multiscale_backend is not None:
        config["dense_detection"]["multiscale_backend"] = multiscale_backend
    database = artifact_path(config, "annotations", "annotations.db")

    total_started = time.perf_counter()
    if base_candidate_root is None:
        cf_started = time.perf_counter()
        cf_cache = build_morphology_pseudo_labels(config)
        cf_elapsed = time.perf_counter() - cf_started
    else:
        cf_cache = artifact_path(
            config, "pseudo_labels", "morphology_candidates.csv"
        )
        cf_elapsed = 0.0
    dense_elapsed = 0.0
    dense_report: dict[str, Any] = {}
    if not cf_only:
        dense_started = time.perf_counter()
        dense_report = augment_candidates_with_dense_raw_proposals(config, database)
        dense_elapsed = time.perf_counter() - dense_started

    cf_summary = (
        json.loads(cf_cache.with_suffix(".json").read_text(encoding="utf-8"))
        if cf_cache.with_suffix(".json").exists()
        else {}
    )
    candidate_path = artifact_path(config, "pseudo_labels", "morphology_candidates.csv")
    report = {
        "source_config": str(config_path.resolve()),
        "source_artifact_root": str(source_root.resolve()),
        "profile_artifact_root": str(output_root.resolve()),
        "registration_cache_source": (
            str(registration_cache.resolve()) if registration_cache is not None else None
        ),
        "base_candidate_root": (
            str(base_candidate_root.resolve())
            if base_candidate_root is not None
            else None
        ),
        "minimum_interior_peak_response": config["dense_detection"].get(
            "minimum_interior_peak_response", 18.0
        ),
        "multiscale_gaussian_truncate": config["dense_detection"].get(
            "multiscale_gaussian_truncate", 4.0
        ),
        "cf_elapsed_seconds": round(cf_elapsed, 3),
        "cf_timing_seconds": cf_summary.get("timing_seconds", {}),
        "dense_elapsed_seconds": round(dense_elapsed, 3),
        "dense_timing_seconds": dense_report.get("timing_seconds", {}),
        "dense_stage_metadata": {
            key: dense_report.get(key)
            for key in (
                "candidate_count",
                "response_backend",
                "response_backend_detail",
                "cuda_runtime_fallback_count",
                "cuda_runtime_fallback_detail",
                "wall_rescue_peaks_tested",
                "wall_rescue_arc_rejected",
                "wall_residual_peak_count",
            )
            if key in dense_report
        },
        "candidate_source_counts": _source_counts(candidate_path),
        "total_elapsed_seconds": round(time.perf_counter() - total_started, 3),
    }
    (output_root / "candidate_profile.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--registration-cache", type=Path)
    parser.add_argument("--cf-only", action="store_true")
    parser.add_argument("--base-candidate-root", type=Path)
    parser.add_argument("--minimum-interior-response", type=float)
    parser.add_argument("--multiscale-gaussian-truncate", type=float)
    parser.add_argument("--multiscale-backend", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                args.output_root,
                registration_cache=args.registration_cache,
                cf_only=args.cf_only,
                base_candidate_root=args.base_candidate_root,
                minimum_interior_response=args.minimum_interior_response,
                multiscale_gaussian_truncate=args.multiscale_gaussian_truncate,
                multiscale_backend=args.multiscale_backend,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

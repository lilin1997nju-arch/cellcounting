from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from cellvision.well_screening import build_well_screening


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-only external plate validation")
    parser.add_argument("--config", default="configs/ql2202_validation.yaml")
    parser.add_argument("--source-artifacts", default="artifacts")
    parser.add_argument(
        "--reuse-existing-candidates",
        action="store_true",
        help="Reuse the existing CF candidate manifest before rebuilding dense proposals.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    source = Path(args.source_artifacts).resolve()
    database = initialize_database(artifact_path(config, "annotations", "annotations.db"))
    images, sequences = build_manifest(config)
    results = {
        "manifest": {"images": int(len(images)), "wells": int(len(sequences))},
    }
    candidate_manifest = artifact_path(
        config, "pseudo_labels", "morphology_candidates.csv"
    )
    if not args.reuse_existing_candidates or not candidate_manifest.exists():
        build_morphology_pseudo_labels(config)
    results["candidate_manifest"] = str(candidate_manifest)
    results["dense_candidates"] = augment_candidates_with_dense_raw_proposals(config, database)
    results["morphology_inference"] = predict_teaching_checkpoint(
        config, source / "models" / "teaching_classifier.pt"
    )
    results["morphology_round"] = generate_auto_annotation_round(config, database)
    results["multiplicity_inference"] = predict_multiplicity_checkpoint(
        config, source / "models" / "multiplicity_classifier.pt"
    )
    results["integrated_round"] = generate_integrated_training_round(config, database)
    results["well_screening"] = build_well_screening(config, database)
    output = artifact_path(config, "validation_summary.json")
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

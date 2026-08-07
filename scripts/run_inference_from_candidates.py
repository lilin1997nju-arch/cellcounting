from __future__ import annotations

import argparse
import json
from pathlib import Path

from cellvision.config import artifact_path, load_config
from cellvision.model_inference import (
    predict_multiplicity_checkpoint,
    predict_teaching_checkpoint,
)
from cellvision.multiplicity import generate_integrated_training_round
from cellvision.review_server import initialize_database
from cellvision.teaching import generate_auto_annotation_round
from cellvision.well_screening import build_well_screening


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a trained checkpoint on an already prepared plate."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-artifacts", default="artifacts")
    args = parser.parse_args()
    config = load_config(args.config)
    source = Path(args.source_artifacts).resolve()
    database = initialize_database(
        artifact_path(config, "annotations", "annotations.db")
    )
    results = {
        "morphology_inference": predict_teaching_checkpoint(
            config, source / "models" / "teaching_classifier.pt"
        )
    }
    results["morphology_round"] = generate_auto_annotation_round(
        config, database
    )
    results["multiplicity_inference"] = predict_multiplicity_checkpoint(
        config, source / "models" / "multiplicity_classifier.pt"
    )
    results["integrated_round"] = generate_integrated_training_round(
        config, database
    )
    results["well_screening"] = build_well_screening(config, database)
    output = artifact_path(config, "inference_summary.json")
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

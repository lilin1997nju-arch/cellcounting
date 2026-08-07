from __future__ import annotations

import argparse
import json

from cellvision.active_learning import build_review_queue
from cellvision.config import artifact_path, load_config
from cellvision.multiplicity import (
    generate_integrated_training_round,
    train_multiplicity_classifier,
)
from cellvision.teaching import (
    ensure_teaching_features,
    generate_auto_annotation_round,
    train_teaching_classifier,
)
from cellvision.well_screening import build_well_screening


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain from prepared candidates and reviewed labels."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    database = artifact_path(config, "annotations", "annotations.db")
    ensure_teaching_features(config, force=True)
    results = {
        "morphology_training": train_teaching_classifier(config, database),
    }
    results["morphology_round"] = generate_auto_annotation_round(
        config, database
    )
    results["multiplicity_training"] = train_multiplicity_classifier(
        config, database
    )
    results["integrated_round"] = generate_integrated_training_round(
        config, database
    )
    results["well_screening"] = build_well_screening(config, database)
    queue = build_review_queue(config, include_completed=True)
    results["main_review_queue"] = {
        "count": int(len(queue)),
        "cell": int((queue["auto_label"] == "cell").sum()),
        "debris": int((queue["auto_label"] == "debris").sum()),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

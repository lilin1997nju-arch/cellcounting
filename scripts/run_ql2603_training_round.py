"""Train the QL2603 pooled classifier heads without generating new labels."""

from __future__ import annotations

import argparse
import json

from cellvision.config import artifact_path, load_config
from cellvision.multiplicity import train_multiplicity_classifier
from cellvision.teaching import ensure_teaching_features, train_teaching_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ql2603_joint_training.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    database = artifact_path(config, "annotations", "annotations.db")
    # Feature extraction is cached per source; this does not create labels or
    # run an auto-review round.
    ensure_teaching_features(config, force=False)
    result = {
        "config": str(args.config),
        "morphology": train_teaching_classifier(config, database),
        "multiplicity": train_multiplicity_classifier(config, database),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

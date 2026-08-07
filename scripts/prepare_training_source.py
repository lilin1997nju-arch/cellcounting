from __future__ import annotations

import argparse
import json

from cellvision.config import artifact_path, load_config
from cellvision.dense_candidates import augment_candidates_with_dense_raw_proposals
from cellvision.manifest import build_manifest
from cellvision.pseudo_labels import build_morphology_pseudo_labels
from cellvision.review_server import initialize_database
from cellvision.teaching import ensure_teaching_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one plate as an incremental joint-training source."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--reuse-existing-candidates", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    database = initialize_database(
        artifact_path(config, "annotations", "annotations.db")
    )
    images, sequences = build_manifest(config)
    candidate_path = artifact_path(
        config, "pseudo_labels", "morphology_candidates.csv"
    )
    if not args.reuse_existing_candidates or not candidate_path.exists():
        build_morphology_pseudo_labels(config)
    dense = augment_candidates_with_dense_raw_proposals(config, database)
    metadata, _ = ensure_teaching_features(config)
    print(
        json.dumps(
            {
                "images": int(len(images)),
                "wells": int(len(sequences)),
                "dense": dense,
                "feature_candidates": int(len(metadata)),
                "database": str(database),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

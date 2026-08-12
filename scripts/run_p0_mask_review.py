from __future__ import annotations

import argparse
import json
from pathlib import Path

from cellvision.config import load_config
from cellvision.v2_mask_review import create_mask_review_round


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run versioned P0 V2 inference and initialize mask review artifacts."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="V2 instance checkpoint; defaults to artifacts/v2/models/latest_instance_segmenter.pt",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="source integrated predictions; defaults to latest_integrated_predictions.csv",
    )
    parser.add_argument("--round-id", default=None)
    args = parser.parse_args()
    config = load_config(Path(args.config))
    result = create_mask_review_round(
        config,
        checkpoint_path=args.checkpoint,
        predictions_path=args.predictions,
        round_id=args.round_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from copy import deepcopy

from cellvision.config import load_config
from cellvision.train_v2_instance import train_v2_instance_segmenter
from cellvision.train_v2_temporal import train_v2_temporal_model


def main() -> None:
    config = load_config("configs/v2_training.yaml")
    scratch = deepcopy(config)
    scratch["v2_instance_segmentation"]["initial_checkpoint"] = ""
    scratch["v2_temporal_model"]["initial_checkpoint"] = ""
    instance_run = train_v2_instance_segmenter(scratch)
    temporal_run = train_v2_temporal_model(scratch)
    print(f"instance={instance_run}")
    print(f"temporal={temporal_run}")


if __name__ == "__main__":
    main()

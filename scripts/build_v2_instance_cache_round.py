from __future__ import annotations

from cellvision.config import load_config
from cellvision.v2_instance_dataset import build_v2_instance_cache


if __name__ == "__main__":
    config = load_config("configs/v2_training.yaml")
    print(build_v2_instance_cache(config), flush=True)

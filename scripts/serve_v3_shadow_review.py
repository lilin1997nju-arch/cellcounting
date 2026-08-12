from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from cellvision.temporal_shadow_review import create_shadow_review_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the standalone temporal Shadow review UI")
    parser.add_argument("--root", type=Path, default=Path("artifacts/v3_shadow_tests"), help="Directory containing Shadow run folders")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    uvicorn.run(create_shadow_review_app(args.root), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

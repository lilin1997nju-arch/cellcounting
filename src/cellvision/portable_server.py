"""Minimal normal-review entrypoint without importing compute CLI modules."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Cell Vision portable normal review")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()

    # These must be set before importing the project/review application so
    # development-only model and training routes are never imported.
    os.environ["CELLVISION_PRODUCTION"] = "1"
    os.environ["CELLVISION_PORTABLE_REVIEW"] = "1"
    os.environ["CELLVISION_DEVICE"] = "cpu"

    import uvicorn

    from .project_server import create_project_app

    uvicorn.run(
        create_project_app(args.manifest),
        host=args.host,
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()

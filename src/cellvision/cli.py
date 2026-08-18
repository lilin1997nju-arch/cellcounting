from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import uvicorn

from .active_learning import build_review_queue
from .audit import run_audit
from .config import load_config
from .manifest import build_manifest, split_sequences
from .session_index import write_session_group_manifest
from .review_server import create_app
from .project_server import create_project_app
from .project_catalog import ProjectCatalog, catalog_path_for_manifest
from .v2_instance_inference import infer_v2_instances, refinalize_v2_file
from .evaluate_v2 import write_v2_evaluation
from .v2_temporal_inference import infer_v2_temporal_evidence, refinalize_v2_temporal_noncell_labels
from .review_image_cache import precache_review_images
from .late_growth_inference import infer_late_growth
from .gated_screening import build_gated_plate_report
from .project_worker import ProjectTaskWorker
from .runtime import (
    SUPPORTED_DEVICE_REQUESTS,
    detect_compute_runtime,
    ensure_training_allowed,
    production_mode_enabled,
)


# Keep one stable project-review endpoint.  Starting a new task should reuse
# this service instead of incrementing ports and leaving old processes behind.
PROJECT_REVIEW_PORT = 8777


def _wells(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cellvision", description="Local multi-timepoint cell vision")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "build-manifest", "split", "build-review-queue"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default="configs/default.yaml")
    sessions = subparsers.add_parser(
        "parse-sessions",
        help="Parse a vendor sessions.idx into grouped acquisition/timepoint manifests",
    )
    sessions.add_argument("--root", required=True, help="Export root containing the SessionFolder paths")
    sessions.add_argument("--index", default="", help="Path to sessions.idx (defaults to <root>/sessions.idx)")
    sessions.add_argument(
        "--output",
        required=True,
        help="Output session-level CSV; a .groups.csv and .summary.json are created alongside it",
    )
    sessions.add_argument("--timepoint-origin", type=int, choices=(0, 1), default=0)
    if not production_mode_enabled():
        train = subparsers.add_parser("train")
        train.add_argument(
            "model", choices=["weak-segmenter", "morphology-classifier", "v2-instance-segmenter", "v3-temporal-pairwise-model"]
        )
        train.add_argument("--config", default="configs/default.yaml")
    infer_v2 = subparsers.add_parser("infer-v2")
    infer_v2.add_argument("--config", default="configs/default.yaml")
    infer_v2.add_argument("--checkpoint", required=True)
    evaluate_v2 = subparsers.add_parser("evaluate-v2")
    evaluate_v2.add_argument("--config", default="configs/a12_22_training.yaml")
    temporal_v2 = subparsers.add_parser("infer-v2-temporal")
    temporal_v2.add_argument("--config", default="configs/default.yaml")
    refinalize_v2 = subparsers.add_parser("refinalize-v2")
    refinalize_v2.add_argument("--config", default="configs/default.yaml")
    resolve_v2_noncell = subparsers.add_parser("resolve-v2-noncell")
    resolve_v2_noncell.add_argument("--config", default="configs/default.yaml")
    late_growth = subparsers.add_parser("infer-late-growth")
    late_growth.add_argument("--config", default="configs/ql2202_validation.yaml")
    late_growth.add_argument("--wells", default="")
    gated = subparsers.add_parser(
        "build-gated-screening",
        help="Build a Day14-gated early-compute queue and 96-well overview",
    )
    gated.add_argument("--config", default="configs/default.yaml")
    gated.add_argument("--day14-csv", required=True)
    gated.add_argument("--group-id", required=True)
    gated.add_argument("--output-dir", required=True)
    gated.add_argument("--early-screening-csv", default="")
    gated.add_argument("--sessions-csv", default="")
    gated.add_argument("--endpoint-day-label", default="Day14", help="Actual culture day used as the endpoint gate")
    gated.add_argument(
        "--skip-day7-localization",
        action="store_true",
        help="Do not locate representative Day7 dense regions",
    )
    precache = subparsers.add_parser("precache-review-images")
    precache.add_argument("--config", default="configs/default.yaml")
    precache.add_argument("--sizes", default="1400")
    review = subparsers.add_parser("review")
    review.add_argument("--config", default="configs/default.yaml")
    review.add_argument("--host", default=os.getenv("CELLVISION_HOST", "127.0.0.1"))
    review.add_argument("--port", type=int, default=int(os.getenv("CELLVISION_PORT", str(PROJECT_REVIEW_PORT))))
    review.add_argument("--allow-remote", action="store_true", help="Allow binding beyond loopback")
    project_review = subparsers.add_parser(
        "review-project",
        help="Serve a project-level hub and mount its completed plate review apps",
    )
    project_review.add_argument("--manifest", required=True, help="Project JSON manifest")
    project_review.add_argument("--host", default=os.getenv("CELLVISION_HOST", "127.0.0.1"))
    project_review.add_argument("--port", type=int, default=int(os.getenv("CELLVISION_PORT", str(PROJECT_REVIEW_PORT))))
    project_review.add_argument("--allow-remote", action="store_true", help="Allow binding beyond loopback")
    project_review.add_argument(
        "--no-worker",
        action="store_true",
        help="Do not start the adaptive project worker alongside the review service",
    )
    project_review.add_argument(
        "--access-log",
        action="store_true",
        help="Enable per-request Uvicorn access logs (disabled by default for the local UI)",
    )
    project_review.add_argument(
        "--background",
        action="store_true",
        help="On Windows, relaunch the project service in a hidden process and return",
    )
    project_review.add_argument(
        "--worker-device",
        choices=SUPPORTED_DEVICE_REQUESTS,
        default=os.getenv("CELLVISION_WORKER_DEVICE", "auto"),
        help="Worker device: auto detects CUDA and falls back to CPU",
    )
    project_review.add_argument("--worker-poll-seconds", type=float, default=2.0)
    project_worker = subparsers.add_parser(
        "project-worker",
        help="Run the adaptive project queue worker (CUDA when available, otherwise CPU)",
    )
    project_worker.add_argument("--manifest", required=True, help="Project JSON manifest that owns task_queue.json")
    project_worker.add_argument(
        "--device",
        choices=SUPPORTED_DEVICE_REQUESTS,
        default=os.getenv("CELLVISION_WORKER_DEVICE", "auto"),
        help="auto detects CUDA; cpu/cuda can be used as an explicit preference",
    )
    project_worker.add_argument("--worker-id", default="")
    project_worker.add_argument("--poll-seconds", type=float, default=2.0)
    project_worker.add_argument("--once", action="store_true", help="Claim at most one started task and exit")
    runtime_info = subparsers.add_parser(
        "runtime-info",
        help="Detect the effective CPU/CUDA inference runtime",
    )
    runtime_info.add_argument(
        "--device",
        choices=SUPPORTED_DEVICE_REQUESTS,
        default=os.getenv("CELLVISION_DEVICE", "auto"),
    )
    catalog_sync = subparsers.add_parser(
        "catalog-sync",
        help="Reconcile project_catalog.sqlite from all sibling project manifests",
    )
    catalog_sync.add_argument("--manifest", required=True, help="Any project JSON manifest in the collection")
    catalog_sync.add_argument("--force", action="store_true", help="Re-read unchanged manifests and reports")
    return parser


def _start_project_review_in_background(port: int) -> int:
    """Detach ``review-project`` from a visible Windows console.

    The browser polls the task endpoint regularly.  If the service is started
    directly from a PowerShell window, those access logs make that window look
    like a continuously running task.  This launcher keeps the service alive
    while redirecting diagnostics to the normal artifact log directory.
    """

    log_root = Path(os.getenv("CELLVISION_LOG_ROOT", "artifacts/logs"))
    if not log_root.is_absolute():
        log_root = Path.cwd() / log_root
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"review-project-{port}.stdout.log"
    stderr_path = log_root / f"review-project-{port}.stderr.log"
    child_args = [value for value in sys.argv[1:] if value != "--background"]
    command = [sys.executable, "-m", "cellvision", *child_args]
    creationflags = 0
    popen_options: dict[str, object] = {
        "cwd": str(Path.cwd()),
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
        creationflags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
        popen_options["creationflags"] = creationflags
    else:
        popen_options["start_new_session"] = True
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        popen_options["stdout"] = stdout
        popen_options["stderr"] = stderr
        process = subprocess.Popen(command, **popen_options)
    print(
        f"Cell Vision project server started in background (PID {process.pid}); "
        f"logs: {stdout_path} / {stderr_path}",
        flush=True,
    )
    return process.pid


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        try:
            ensure_training_allowed()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    if args.command == "parse-sessions":
        index = args.index or str(Path(args.root) / "sessions.idx")
        sessions, groups, summary_path = write_session_group_manifest(
            index,
            args.root,
            args.output,
            timepoint_origin=args.timepoint_origin,
        )
        print(json.dumps({
            "sessions": len(sessions),
            "groups": len(groups),
            "output": str(Path(args.output).resolve()),
            "group_output": str(Path(args.output).with_name(f"{Path(args.output).stem}.groups.csv").resolve()),
            "summary": str(summary_path),
        }, ensure_ascii=False, indent=2))
        return
    if args.command == "review-project":
        if args.host not in {"127.0.0.1", "localhost", "::1"} and not (
            args.allow_remote or os.getenv("CELLVISION_ALLOW_REMOTE") == "1"
        ):
            raise SystemExit("Remote binding requires --allow-remote or CELLVISION_ALLOW_REMOTE=1")
        if args.background:
            _start_project_review_in_background(args.port)
            return
        worker = None
        if not args.no_worker:
            worker = ProjectTaskWorker.from_manifest(
                args.manifest,
                requested_device=args.worker_device,
            )
            print(json.dumps({
                "event": "worker_started",
                **worker.runtime.as_dict(),
                "worker_id": worker.worker_id,
                "mode": "background",
            }, ensure_ascii=False), flush=True)
            worker.start_background(poll_seconds=args.worker_poll_seconds)
        try:
            uvicorn.run(
                create_project_app(args.manifest),
                host=args.host,
                port=args.port,
                access_log=bool(args.access_log),
                log_level="info" if args.access_log else "warning",
            )
        finally:
            if worker is not None:
                worker.stop()
        return
    if args.command == "project-worker":
        worker = ProjectTaskWorker.from_manifest(
            args.manifest,
            requested_device=args.device,
            worker_id=args.worker_id or None,
        )
        print(json.dumps({
            "event": "worker_started",
            **worker.runtime.as_dict(),
            "worker_id": worker.worker_id,
            "mode": "once" if args.once else "foreground",
        }, ensure_ascii=False), flush=True)
        if args.once:
            result = worker.run_once()
            print(json.dumps({"event": "worker_once_finished", "task": result}, ensure_ascii=False, indent=2), flush=True)
        else:
            worker.run_forever(poll_seconds=args.poll_seconds)
        return
    if args.command == "runtime-info":
        print(json.dumps(detect_compute_runtime(args.device).as_dict(), ensure_ascii=False, indent=2))
        return
    if args.command == "catalog-sync":
        manifest_path = Path(args.manifest).expanduser().resolve()
        catalog = ProjectCatalog(catalog_path_for_manifest(manifest_path))
        if args.force:
            paths = sorted(
                manifest_path.parent.parent.glob("*/project.json"),
                key=lambda item: item.as_posix().casefold(),
            )
            synced = sum(
                catalog.sync_manifest(path, force=True) is not None
                for path in paths
            )
        else:
            synced = catalog.reconcile_all(manifest_path)
        print(json.dumps({"synced_projects": synced, **catalog.status()}, ensure_ascii=False, indent=2))
        return
    config = load_config(args.config)
    if args.command == "audit":
        result = run_audit(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "build-manifest":
        images, sequences = build_manifest(config)
        print(f"images={len(images)} sequences={len(sequences)}")
    elif args.command == "split":
        print(json.dumps(split_sequences(config), ensure_ascii=False, indent=2))
    elif args.command == "build-review-queue":
        result = build_review_queue(config)
        print(f"review_candidates={len(result)}")
    elif args.command == "train":
        # Training modules are imported only for the development CLI.  The
        # production parser does not expose this command at all.
        from .train import train_weak_segmenter
        from .train_morphology import train_morphology_classifier
        from .train_v2_instance import train_v2_instance_segmenter
        from .temporal_pairwise import train_temporal_pairwise_model

        if args.model == "weak-segmenter":
            run_dir = train_weak_segmenter(config)
        elif args.model == "morphology-classifier":
            run_dir = train_morphology_classifier(config)
        elif args.model == "v2-instance-segmenter":
            run_dir = train_v2_instance_segmenter(config)
        elif args.model == "v3-temporal-pairwise-model":
            run_dir = train_temporal_pairwise_model(config)
        else:
            raise SystemExit(f"Unsupported training model: {args.model}")
        print(str(run_dir))
    elif args.command == "infer-v2":
        print(str(infer_v2_instances(config, args.checkpoint)))
    elif args.command == "evaluate-v2":
        print(str(write_v2_evaluation(config)))
    elif args.command == "infer-v2-temporal":
        print(str(infer_v2_temporal_evidence(config)))
    elif args.command == "refinalize-v2":
        print(str(refinalize_v2_file(config)))
    elif args.command == "resolve-v2-noncell":
        print(str(refinalize_v2_temporal_noncell_labels(config)))
    elif args.command == "infer-late-growth":
        selected = {well.upper() for well in _wells(args.wells)} or None
        print(str(infer_late_growth(config, selected)))
    elif args.command == "build-gated-screening":
        result = build_gated_plate_report(
            args.day14_csv,
            args.group_id,
            args.output_dir,
            early_screening_csv=args.early_screening_csv or None,
            sessions_csv=args.sessions_csv or None,
            locate_day7=not args.skip_day7_localization,
            endpoint_day_label=args.endpoint_day_label,
        )
        print(json.dumps({key: value for key, value in result.items() if key != "wells"}, ensure_ascii=False, indent=2))
    elif args.command == "precache-review-images":
        sizes = tuple(int(value.strip()) for value in args.sizes.split(",") if value.strip())
        print(json.dumps(precache_review_images(config, sizes), ensure_ascii=False))
    elif args.command == "review":
        if args.host not in {"127.0.0.1", "localhost", "::1"} and not (
            args.allow_remote or os.getenv("CELLVISION_ALLOW_REMOTE") == "1"
        ):
            raise SystemExit("Remote binding requires --allow-remote or CELLVISION_ALLOW_REMOTE=1")
        uvicorn.run(create_app(config), host=args.host, port=args.port)

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn

from .active_learning import build_review_queue
from .audit import run_audit
from .baseline import generate_baseline
from .config import load_config
from .infer import infer_smoke
from .manifest import build_manifest, split_sequences
from .session_index import write_session_group_manifest
from .review_server import create_app
from .project_server import create_project_app
from .train import train_weak_segmenter
from .train_morphology import train_morphology_classifier
from .train_v2_instance import train_v2_instance_segmenter
from .train_v2_temporal import train_v2_temporal_model
from .v2_instance_inference import infer_v2_instances, refinalize_v2_file
from .evaluate_v2 import write_v2_evaluation
from .v2_temporal_inference import infer_v2_temporal_evidence, refinalize_v2_temporal_noncell_labels
from .review_image_cache import precache_review_images
from .late_growth_inference import infer_late_growth
from .gated_screening import build_gated_plate_report


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
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--config", default="configs/default.yaml")
    baseline.add_argument("--wells", default="A1,B3,F12,G2,H6")
    train = subparsers.add_parser("train")
    train.add_argument(
        "model", choices=["weak-segmenter", "morphology-classifier", "v2-instance-segmenter", "v2-temporal-model"]
    )
    train.add_argument("--config", default="configs/default.yaml")
    infer = subparsers.add_parser("infer")
    infer.add_argument("--config", default="configs/model.yaml")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--wells", default="A1,B3,F12,G2,H6")
    infer_v2 = subparsers.add_parser("infer-v2")
    infer_v2.add_argument("--config", default="configs/default.yaml")
    infer_v2.add_argument("--checkpoint", required=True)
    evaluate_v2 = subparsers.add_parser("evaluate-v2")
    evaluate_v2.add_argument("--config", default="configs/a12_22_training.yaml")
    temporal_v2 = subparsers.add_parser("infer-v2-temporal")
    temporal_v2.add_argument("--config", default="configs/default.yaml")
    temporal_v2.add_argument("--checkpoint", required=True)
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
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
        uvicorn.run(create_project_app(args.manifest), host=args.host, port=args.port)
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
    elif args.command == "baseline":
        result = generate_baseline(config, _wells(args.wells))
        print(f"candidate_objects={len(result)}")
    elif args.command == "train":
        if args.model == "weak-segmenter":
            run_dir = train_weak_segmenter(config)
        elif args.model == "morphology-classifier":
            run_dir = train_morphology_classifier(config)
        elif args.model == "v2-instance-segmenter":
            run_dir = train_v2_instance_segmenter(config)
        else:
            run_dir = train_v2_temporal_model(config)
        print(str(run_dir))
    elif args.command == "infer":
        result = infer_smoke(config, args.checkpoint, _wells(args.wells))
        print(str(result))
    elif args.command == "infer-v2":
        print(str(infer_v2_instances(config, args.checkpoint)))
    elif args.command == "evaluate-v2":
        print(str(write_v2_evaluation(config)))
    elif args.command == "infer-v2-temporal":
        print(str(infer_v2_temporal_evidence(config, args.checkpoint)))
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

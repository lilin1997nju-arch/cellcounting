import pytest
from fastapi import FastAPI

from cellvision.cli import build_parser
from cellvision.runtime import (
    detect_compute_runtime,
    ensure_training_allowed,
    remove_development_routes,
)


def test_cpu_runtime_is_always_available():
    runtime = detect_compute_runtime("cpu")
    assert runtime.requested_device == "cpu"
    assert runtime.selected_device == "cpu"
    assert runtime.worker_kind == "cpu"
    assert runtime.cpu_count >= 1


def test_auto_runtime_selects_a_supported_backend():
    runtime = detect_compute_runtime("auto")
    assert runtime.selected_device in {"cpu", "cuda"}
    assert runtime.worker_kind == ("gpu" if runtime.selected_device == "cuda" else "cpu")
    assert runtime.as_dict()["label"]


def test_cuda_request_falls_back_when_cuda_is_unavailable():
    runtime = detect_compute_runtime("cuda")
    if not runtime.cuda_available:
        assert runtime.selected_device == "cpu"
        assert runtime.fallback_reason


def test_production_mode_blocks_training(monkeypatch):
    monkeypatch.setenv("CELLVISION_PRODUCTION", "1")
    try:
        ensure_training_allowed()
    except RuntimeError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("training guard did not reject production mode")


def test_production_mode_removes_training_and_specialist_review_routes(monkeypatch):
    monkeypatch.setenv("CELLVISION_PRODUCTION", "1")
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    for path in (
        "/teach",
        "/single-doublet-review",
        "/mask-review",
        "/api/teach-train",
        "/api/integrated-review-new-round",
        "/api/mask-review-rounds",
    ):
        app.add_api_route(path, lambda: {}, methods=["GET"])

    remove_development_routes(app)

    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/health" in paths
    assert "/teach" not in paths
    assert "/single-doublet-review" not in paths
    assert "/mask-review" not in paths
    assert "/api/teach-train" not in paths
    assert "/api/integrated-review-new-round" not in paths
    assert "/api/mask-review-rounds" not in paths


def test_production_cli_does_not_offer_train_command(monkeypatch):
    monkeypatch.setenv("CELLVISION_PRODUCTION", "1")

    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "morphology-classifier"])

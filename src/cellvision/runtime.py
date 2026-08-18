"""Runtime device detection and torch-device selection helpers.

The project is intended to run on both GPU workstations and CPU-only
machines.  Keep CUDA probing in one small module so the queue worker, CLI and
inference code report the same decision and use the same fallback policy.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


SUPPORTED_DEVICE_REQUESTS = ("auto", "cuda", "cpu")

# Development-only review surfaces must not be discoverable or callable in a
# production installation.  Keep this list central so both the project hub and
# the per-plate review app enforce the same boundary.
PRODUCTION_HIDDEN_ROUTE_PATHS = {
    "/teach",
    "/doublet-teach",
    "/single-doublet-review",
    "/integrated-review",
    "/mask-review",
    "/projects/{project_id}/single-doublet-review",
    "/projects/{project_id}/mask-review",
    "/api/auto-review-new-round",
}
PRODUCTION_HIDDEN_ROUTE_PREFIXES = (
    "/api/teach-",
    "/api/multiplicity-",
    "/api/integrated-review-",
    "/api/mask-review",
    "/api/mask-comparison",
)


def production_mode_enabled() -> bool:
    """Whether this process is running in compute-only production mode."""

    return os.getenv("CELLVISION_PRODUCTION", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def ensure_training_allowed() -> None:
    """Reject model training when the process is marked as production."""

    if production_mode_enabled():
        raise RuntimeError(
            "Model training is disabled in CELLVISION_PRODUCTION mode; "
            "run training in a separate development environment."
        )


def remove_development_routes(app: Any) -> None:
    """Remove training and specialist model-audit routes in production.

    Filtering the router makes these endpoints return 404 and removes them
    from OpenAPI, instead of merely displaying a disabled button or returning
    a late 403 after the training implementation has already been exposed.
    """

    if not production_mode_enabled():
        return

    def retained(route: Any) -> bool:
        path = str(getattr(route, "path", ""))
        if path in PRODUCTION_HIDDEN_ROUTE_PATHS:
            return False
        return not any(path.startswith(prefix) for prefix in PRODUCTION_HIDDEN_ROUTE_PREFIXES)

    app.router.routes[:] = [route for route in app.router.routes if retained(route)]


def _normalise_request(value: str | None) -> str:
    requested = str(value or os.getenv("CELLVISION_DEVICE", "auto")).strip().lower()
    if requested not in SUPPORTED_DEVICE_REQUESTS:
        raise ValueError(
            f"Unsupported compute device {requested!r}; "
            f"choose one of {', '.join(SUPPORTED_DEVICE_REQUESTS)}"
        )
    return requested


@dataclass(frozen=True)
class ComputeRuntime:
    """A serialisable snapshot of the worker's selected compute backend."""

    requested_device: str
    selected_device: str
    worker_kind: str
    torch_available: bool
    torch_version: str
    cuda_available: bool
    cuda_version: str
    gpu_count: int
    gpu_name: str
    cpu_count: int
    hostname: str
    fallback_reason: str
    detected_at: str

    @property
    def label(self) -> str:
        if self.selected_device == "cuda":
            return f"GPU worker ({self.gpu_name or 'CUDA'})"
        return "CPU worker"

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "selected_device": self.selected_device,
            "worker_kind": self.worker_kind,
            "label": self.label,
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "gpu_count": self.gpu_count,
            "gpu_name": self.gpu_name,
            "cpu_count": self.cpu_count,
            "hostname": self.hostname,
            "fallback_reason": self.fallback_reason,
            "detected_at": self.detected_at,
        }


def detect_compute_runtime(requested_device: str | None = None) -> ComputeRuntime:
    """Detect CUDA and choose ``cuda`` or ``cpu``.

    ``auto`` prefers CUDA only when PyTorch reports a usable CUDA runtime.  A
    requested CUDA backend also falls back to CPU rather than preventing the
    queue from running, which makes the same deployment command safe on a
    CPU-only workstation.
    """

    requested = _normalise_request(requested_device)
    torch_available = False
    torch_version = ""
    cuda_available = False
    cuda_version = ""
    gpu_count = 0
    gpu_name = ""
    probe_error = ""

    try:
        import torch

        torch_available = True
        torch_version = str(getattr(torch, "__version__", ""))
        try:
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                gpu_count = int(torch.cuda.device_count())
                if gpu_count > 0:
                    gpu_name = str(torch.cuda.get_device_name(0))
                cuda_version = str(getattr(torch.version, "cuda", "") or "")
                # is_available() can be true while the driver fails on the
                # first real allocation. Exercise one tiny tensor so auto
                # mode chooses CUDA only when the runtime is actually usable.
                torch.empty(1, device="cuda").add_(1).item()
                torch.cuda.synchronize()
        except Exception as exc:  # a broken driver must behave like no CUDA
            probe_error = f"CUDA 检测失败：{type(exc).__name__}: {exc}"
            cuda_available = False
            gpu_count = 0
            gpu_name = ""
            cuda_version = ""
    except Exception as exc:  # a CPU-only environment may not have torch yet
        probe_error = f"PyTorch 不可用：{type(exc).__name__}: {exc}"

    if requested == "cpu":
        selected = "cpu"
        fallback_reason = "按配置强制使用 CPU"
    elif cuda_available:
        selected = "cuda"
        fallback_reason = ""
    else:
        selected = "cpu"
        fallback_reason = probe_error or "未检测到可用 CUDA，自动切换 CPU"

    return ComputeRuntime(
        requested_device=requested,
        selected_device=selected,
        worker_kind="gpu" if selected == "cuda" else "cpu",
        torch_available=torch_available,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        cpu_count=max(1, int(os.cpu_count() or 1)),
        hostname=platform.node(),
        fallback_reason=fallback_reason,
        detected_at=datetime.now(timezone.utc).isoformat(),
    )


def select_torch_device(requested_device: str | None = None):
    """Return a torch device honoring ``CELLVISION_DEVICE`` when set."""

    import torch

    runtime = detect_compute_runtime(requested_device)
    return torch.device(runtime.selected_device)


def cuda_runtime_enabled(requested_device: str | None = None) -> bool:
    """Whether CUDA synchronisation should be used for this process."""

    return detect_compute_runtime(requested_device).selected_device == "cuda"

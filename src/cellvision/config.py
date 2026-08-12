from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


_PATH_ENV_OVERRIDES = {
    "CELLVISION_DATA_ROOT": ("paths", "data_root"),
    "CELLVISION_ARTIFACT_ROOT": ("paths", "artifact_root"),
    "CELLVISION_MODEL_ROOT": ("paths", "model_root"),
    "CELLVISION_DB_ROOT": ("paths", "db_root"),
    "CELLVISION_LOG_ROOT": ("paths", "log_root"),
    "CELLVISION_SHARED_MODEL_ROOT": ("late_growth", "shared_model_root"),
}


def _resolve_path(value: str, base: Path) -> str:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def _deep_merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Merge nested configuration dictionaries without dropping siblings."""

    merged = copy.deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _config_reference(path: Path, reference: str) -> Path:
    """Resolve a base config in both the repository and deployed layouts."""

    candidate = Path(os.path.expandvars(reference)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    beside_config = (path.parent / candidate).resolve()
    if beside_config.exists():
        return beside_config
    return (PROJECT_ROOT / candidate).resolve()


def _load_raw_config(config_path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    if config_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, config_path))
        raise ValueError(f"Circular base_config reference: {chain}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    if "base_config" in config:
        base_path = _config_reference(config_path, str(config["base_config"]))
        base = _load_raw_config(base_path, (*stack, config_path))
        child = {key: value for key, value in config.items() if key != "base_config"}
        config = _deep_merge(base, child)
    return config


def validate_config(config: dict[str, Any], *, require_experiment: bool = False) -> None:
    """Validate deployment-critical configuration keys early.

    Validation intentionally checks structure rather than filesystem existence:
    a server may start before a mounted data volume becomes available.
    """

    errors: list[str] = []
    paths = config.get("paths")
    if not isinstance(paths, dict):
        errors.append("paths must be a mapping")
    else:
        for key in ("data_root", "artifact_root"):
            if not str(paths.get(key, "")).strip():
                errors.append(f"paths.{key} is required")
    if require_experiment:
        experiment = config.get("experiment")
        if not isinstance(experiment, dict):
            errors.append("experiment must be a mapping")
        else:
            for key in ("experiment_id", "plate_id", "plate_rows", "plate_columns"):
                if key not in experiment:
                    errors.append(f"experiment.{key} is required")
    if errors:
        raise ValueError("Invalid configuration: " + "; ".join(errors))


def load_config(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = _load_raw_config(config_path)

    # Environment overrides are intentionally applied last so the same YAML
    # can run locally, in Docker, or on a lab server without rewriting files.
    for variable, (section, key) in _PATH_ENV_OVERRIDES.items():
        value = os.getenv(variable)
        if value:
            config.setdefault(section, {})[key] = value

    if validate:
        validate_config(config)
    paths = config.setdefault("paths", {})
    if "data_root" not in paths:
        paths["data_root"] = ""
    paths["data_root"] = _resolve_path(paths["data_root"], PROJECT_ROOT)
    paths["artifact_root"] = _resolve_path(paths.get("artifact_root", "artifacts"), PROJECT_ROOT)
    for key in ("model_root", "db_root", "log_root"):
        if key in paths and str(paths[key]).strip():
            paths[key] = _resolve_path(paths[key], PROJECT_ROOT)
    late_growth = config.get("late_growth")
    if isinstance(late_growth, dict) and str(late_growth.get("shared_model_root", "")).strip():
        late_growth["shared_model_root"] = _resolve_path(late_growth["shared_model_root"], PROJECT_ROOT)
    config.setdefault("runtime", {})
    host = os.getenv("CELLVISION_HOST")
    if host:
        config["runtime"]["host"] = host
    else:
        config["runtime"].setdefault("host", "127.0.0.1")
    port = os.getenv("CELLVISION_PORT")
    if port:
        try:
            config["runtime"]["port"] = int(port)
        except ValueError as exc:
            raise ValueError("CELLVISION_PORT must be an integer") from exc
    else:
        config["runtime"].setdefault("port", 8777)
    return config


def artifact_path(config: dict[str, Any], *parts: str) -> Path:
    path = Path(config["paths"]["artifact_root"]).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def is_validation_holdout(config: dict[str, Any], source_path: str | Path) -> bool:
    """Whether a source config is explicitly frozen for evaluation."""

    candidate = Path(source_path).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    configured = config.get("validation_holdout_sources", [])
    return any(
        candidate == Path(str(value)).expanduser().resolve()
        if Path(str(value)).expanduser().is_absolute()
        else candidate == (PROJECT_ROOT / str(value)).resolve()
        for value in configured
    )

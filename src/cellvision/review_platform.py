"""Installed lightweight platform for opening platform-independent .cvreview data."""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .review_data_package import DATA_PACKAGE_FORMAT, DATA_PACKAGE_MANIFEST, validate_review_data_package


def _platform_home() -> Path:
    configured = os.getenv("CELLVISION_REVIEW_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt" and os.getenv("LOCALAPPDATA"):
        return (Path(os.environ["LOCALAPPDATA"]) / "CellVisionReviewPlatform").resolve()
    return (Path.home() / "Library" / "Application Support" / "CellVisionReviewPlatform").resolve()


def _registry_path() -> Path:
    path = _platform_home() / "registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_registry() -> dict[str, Any]:
    path = _registry_path()
    if not path.is_file():
        return {"version": 1, "packages": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "packages": []}
    return value if isinstance(value, dict) else {"version": 1, "packages": []}


def register_review_package(package_root: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(package_root).expanduser().resolve()
    metadata_path = root / DATA_PACKAGE_MANIFEST
    metadata_stat = metadata_path.stat() if metadata_path.is_file() else None
    registry = _read_registry()
    packages = registry.get("packages") if isinstance(registry.get("packages"), list) else []
    # Opening a review package is latency-sensitive and the package can be
    # larger than a gigabyte.  Validate the manifest and required file sizes,
    # but do not re-read every image solely to recompute export-time hashes.
    metadata = validate_review_data_package(root, verify_hashes=False)
    record = {
        "package_id": str(metadata["package_id"]),
        "project_id": str(metadata["project_id"]),
        "project_name": str(metadata.get("project_name") or metadata["project_id"]),
        "package_path": str(root),
        "manifest_bytes": int(metadata_stat.st_size) if metadata_stat is not None else 0,
        "manifest_modified_ns": int(metadata_stat.st_mtime_ns) if metadata_stat is not None else 0,
        "hashes_verified": False,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "last_opened_at": datetime.now(timezone.utc).isoformat(),
    }
    packages = [
        item for item in packages
        if isinstance(item, dict)
        and str(item.get("package_id") or "") != record["package_id"]
        and str(item.get("package_path") or "") != record["package_path"]
        and str(item.get("project_id") or "") != record["project_id"]
    ]
    packages.append(record)
    registry = {"version": 1, "packages": packages}
    path = _registry_path()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return root / str(metadata["entrypoint"]), metadata


def write_review_hub_manifest() -> Path:
    """Write the lightweight hub manifest for every available imported package."""

    registry = _read_registry()
    packages = registry.get("packages") if isinstance(registry.get("packages"), list) else []
    manifests: list[str] = []
    for item in packages:
        if not isinstance(item, dict):
            continue
        package_root = Path(str(item.get("package_path") or "")).expanduser()
        metadata_path = package_root / DATA_PACKAGE_MANIFEST
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("format") != DATA_PACKAGE_FORMAT:
            continue
        entrypoint = package_root / str(metadata.get("entrypoint") or "")
        if entrypoint.is_file():
            manifests.append(str(entrypoint.resolve()))

    hub = _platform_home() / "hub" / "project.json"
    hub.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "project_id": "offline-review-hub",
        "project_name": "Cell Vision 离线审核平台",
        "root": ".",
        "portable_review": True,
        "review_hub": True,
        "project_manifest_paths": manifests,
        "plates": [],
    }
    temporary = hub.with_name(f".{hub.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(hub)
    return hub


def review_platform_status() -> dict[str, Any]:
    registry = _read_registry()
    packages = registry.get("packages") if isinstance(registry.get("packages"), list) else []
    items = []
    for item in packages:
        if not isinstance(item, dict):
            continue
        value = dict(item)
        value["available"] = (Path(str(item.get("package_path") or "")) / DATA_PACKAGE_MANIFEST).is_file()
        items.append(value)
    return {
        "enabled": True,
        "package_count": len(items),
        "available_count": sum(1 for item in items if item["available"]),
        "missing_count": sum(1 for item in items if not item["available"]),
        "packages": items,
    }


def choose_review_package() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    try:
        # The review platform normally runs as a hidden background process
        # while the browser owns the foreground.  Give the native chooser an
        # invisible, topmost owner so Windows does not place it behind the
        # browser window.
        root.overrideredirect(True)
        root.geometry("1x1+0+0")
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.0)
        except tk.TclError:
            pass
        root.deiconify()
        root.update_idletasks()
        root.lift()
        root.focus_force()
        root.update()
        if os.name == "nt":
            try:
                import ctypes

                ctypes.windll.user32.BringWindowToTop(root.winfo_id())
                ctypes.windll.user32.SetForegroundWindow(root.winfo_id())
            except (AttributeError, OSError):
                pass
        selected = filedialog.askdirectory(
            parent=root,
            title="选择要导入的 Cell Vision .cvreview 审核数据包",
            mustexist=True,
        )
    finally:
        root.destroy()
    return Path(selected).resolve() if selected else None


def _available_port(requested: int) -> int:
    for port in range(max(1024, requested), max(1024, requested) + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError("没有可用的本地审核服务端口")


def _open_browser_when_ready(port: int, project_id: str = "") -> None:
    for _ in range(80):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                target = f"/projects/{project_id}/" if project_id else "/"
                webbrowser.open(f"http://127.0.0.1:{port}{target}")
                return
        except OSError:
            time.sleep(0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cell Vision lightweight review platform")
    parser.add_argument("--package", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.package:
        register_review_package(Path(args.package).expanduser().resolve())
    manifest_path = write_review_hub_manifest()
    os.environ["CELLVISION_PRODUCTION"] = "1"
    os.environ["CELLVISION_PORTABLE_REVIEW"] = "1"
    os.environ["CELLVISION_REVIEW_PLATFORM"] = "1"
    os.environ["CELLVISION_DEVICE"] = "cpu"

    import uvicorn
    from .project_server import create_project_app

    port = _available_port(args.port)
    if not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(port,),
            daemon=True,
        ).start()
    uvicorn.run(
        create_project_app(manifest_path),
        host=args.host,
        port=port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()

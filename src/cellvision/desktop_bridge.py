"""Interactive per-user bridge for a machine-wide Cell Vision service."""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware


BRIDGE_PORT_START = 8790
BRIDGE_PORT_END = 8820


def _bridge_home() -> Path:
    configured = os.getenv("CELLVISION_DESKTOP_BRIDGE_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "CellVisionDesktopBridge"
    return Path.home() / ".cellvision-desktop-bridge"


def _choose_folder() -> str:
    if os.name == "nt":
        script = r'''Add-Type -AssemblyName System.Windows.Forms
$form = New-Object Windows.Forms.Form
$form.StartPosition = "CenterScreen"
$form.Size = New-Object Drawing.Size(1, 1)
$form.ShowInTaskbar = $false
$form.TopMost = $true
$dialog = New-Object Windows.Forms.FolderBrowserDialog
$dialog.Description = "选择包含 sessions.idx 的数据文件夹"
$dialog.ShowNewFolderButton = $false
try {
    $form.Show()
    $form.Activate()
    $form.BringToFront()
    if ($dialog.ShowDialog($form) -eq [Windows.Forms.DialogResult]::OK) {
        [Console]::OutputEncoding = [Text.Encoding]::UTF8
        Write-Output $dialog.SelectedPath
    }
} finally {
    $dialog.Dispose()
    $form.Dispose()
}'''
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-STA",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                encoded,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)),
            check=False,
        )
        selected = result.stdout.strip().splitlines()
        return str(Path(selected[-1]).resolve()) if result.returncode == 0 and selected else ""

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    try:
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
            title="选择包含 sessions.idx 的数据文件夹",
            mustexist=True,
        )
    finally:
        root.destroy()
    return str(Path(selected).resolve()) if selected else ""


def create_bridge_app(token: str, production_origin: str) -> FastAPI:
    app = FastAPI(title="Cell Vision Desktop Bridge", docs_url=None, redoc_url=None)
    parsed = urllib.parse.urlsplit(production_origin)
    port = parsed.port or 8777
    allowed_origins = sorted(
        {
            production_origin.rstrip("/"),
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
        }
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["X-CellVision-Bridge-Token", "Content-Type"],
    )

    def authorize(value: str) -> None:
        if not value or not secrets.compare_digest(value, token):
            raise HTTPException(status_code=403, detail="Invalid desktop bridge token")

    @app.get("/api/status")
    def status(x_cellvision_bridge_token: str = Header(default="")) -> dict[str, Any]:
        authorize(x_cellvision_bridge_token)
        return {"status": "ready", "pid": os.getpid()}

    @app.post("/api/browse-folder")
    def browse_folder(x_cellvision_bridge_token: str = Header(default="")) -> dict[str, str]:
        authorize(x_cellvision_bridge_token)
        selected = _choose_folder()
        return {
            "path": selected,
            "error": "" if selected else "未选择文件夹；也可以直接输入共享数据路径。",
        }

    return app


def _status(port: int, token: str, *, timeout: float = 0.6) -> bool:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/status",
        headers={"X-CellVision-Bridge-Token": token},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return payload.get("status") == "ready"


def _available_port() -> int:
    for port in range(BRIDGE_PORT_START, BRIDGE_PORT_END + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError("没有可用的 Cell Vision 桌面桥接端口")


def _browser_url(production_url: str, port: int, token: str) -> str:
    separator = "&" if "?" in production_url else "?"
    query = urllib.parse.urlencode(
        {"desktop_bridge_port": port, "desktop_bridge_token": token}
    )
    return f"{production_url}{separator}{query}"


def launch_bridge(production_url: str) -> dict[str, Any]:
    home = _bridge_home()
    home.mkdir(parents=True, exist_ok=True)
    registry_path = home / "bridge.json"
    if registry_path.is_file():
        try:
            existing = json.loads(registry_path.read_text(encoding="utf-8"))
            existing_port = int(existing["port"])
            existing_token = str(existing["token"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            existing_port = 0
            existing_token = ""
        if existing_port and _status(existing_port, existing_token):
            url = _browser_url(production_url, existing_port, existing_token)
            webbrowser.open(url)
            return {"status": "reused", "port": existing_port, "url": url}

    port = _available_port()
    token = secrets.token_urlsafe(32)
    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "cellvision.desktop_bridge",
        "--serve",
        "--port",
        str(port),
        "--token",
        token,
        "--production-url",
        production_url,
    ]
    creationflags = 0
    popen_options: dict[str, Any] = {
        "cwd": str(Path(sys.executable).resolve().parent),
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        creationflags |= int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
        creationflags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
        popen_options["creationflags"] = creationflags
    else:
        popen_options["start_new_session"] = True
    with (logs / "bridge.stdout.log").open("a", encoding="utf-8") as stdout, (
        logs / "bridge.stderr.log"
    ).open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, **popen_options)

    for _ in range(60):
        if _status(port, token):
            break
        if process.poll() is not None:
            raise RuntimeError("Cell Vision 桌面桥接进程启动失败，请查看用户日志目录。")
        time.sleep(0.1)
    else:
        process.terminate()
        raise RuntimeError("Cell Vision 桌面桥接进程启动超时。")

    registry_path.write_text(
        json.dumps({"pid": process.pid, "port": port, "token": token}, ensure_ascii=False),
        encoding="utf-8",
    )
    url = _browser_url(production_url, port, token)
    webbrowser.open(url)
    return {"status": "started", "pid": process.pid, "port": port, "url": url}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cell Vision per-user desktop bridge")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT_START)
    parser.add_argument("--token", default="")
    parser.add_argument("--production-url", default="http://127.0.0.1:8777/")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.serve:
        if not args.token:
            raise SystemExit("--token is required with --serve")
        uvicorn.run(
            create_bridge_app(args.token, args.production_url),
            host="127.0.0.1",
            port=args.port,
            access_log=False,
            log_level="warning",
        )
        return
    print(json.dumps(launch_bridge(args.production_url), ensure_ascii=False))


if __name__ == "__main__":
    main()

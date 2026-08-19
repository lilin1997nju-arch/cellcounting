from pathlib import Path

import pytest
from fastapi import HTTPException

from cellvision.desktop_bridge import _browser_url, create_bridge_app


ROOT = Path(__file__).parents[1]


def test_desktop_bridge_requires_its_per_user_token(monkeypatch):
    monkeypatch.setattr("cellvision.desktop_bridge._choose_folder", lambda: "D:\\SharedData")
    app = create_bridge_app("secret", "http://127.0.0.1:8777")
    endpoints = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}

    with pytest.raises(HTTPException, match="Invalid desktop bridge token"):
        endpoints["/api/status"]("")
    assert endpoints["/api/browse-folder"]("secret")["path"] == "D:\\SharedData"


def test_browser_url_carries_session_specific_bridge_credentials():
    url = _browser_url("http://127.0.0.1:8777/", 8793, "a token")

    assert "desktop_bridge_port=8793" in url
    assert "desktop_bridge_token=a+token" in url


def test_production_pages_use_the_per_user_desktop_bridge():
    for name in ("project-list.js", "project-dashboard.js"):
        script = (ROOT / "review-ui" / name).read_text(encoding="utf-8")
        assert "cellvision.desktopBridgePort" in script
        assert "X-CellVision-Bridge-Token" in script
        assert "browseProjectFolder()" in script

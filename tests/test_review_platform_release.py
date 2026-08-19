from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_windows_review_platform_installs_only_review_runtime():
    installer = (ROOT / "deploy" / "review_platform" / "windows" / "install_review_platform.ps1").read_text(encoding="utf-8")
    builder = (ROOT / "scripts" / "build_review_platform_windows.ps1").read_text(encoding="utf-8")
    opener = (ROOT / "deploy" / "review_platform" / "windows" / "open_review_platform.ps1").read_text(encoding="utf-8")

    assert "requirements-portable-review.txt" in installer
    assert "torch torchvision" not in installer
    assert "Find-Python312" in installer
    assert "Python 3.12 installation completed but python.exe could not be located." in installer
    assert '"torch-", "torchvision-"' in builder
    assert "cellvision.review_platform" in opener
    assert "ConvertTo-ProcessArgument" in opener


def test_macos_review_platform_has_separate_architecture_build_and_signing():
    builder = (ROOT / "scripts" / "build_review_platform_macos.sh").read_text(encoding="utf-8")

    assert '"arm64"' in builder
    assert '"x86_64"' in builder
    assert "PyInstaller" in builder
    assert "codesign" in builder
    assert "requirements-portable-review.txt" in builder

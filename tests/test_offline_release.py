from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_offline_release_builder_pins_git_and_bundles_runtime_assets():
    builder = (ROOT / "scripts" / "build_offline_release.ps1").read_text(encoding="utf-8")
    installer = (ROOT / "deploy" / "offline" / "install_offline.ps1").read_text(encoding="utf-8")
    launcher = (ROOT / "deploy" / "offline" / "Install-CellVision.cmd").read_text(encoding="utf-8")
    install_ui = (ROOT / "deploy" / "windows" / "install_ui.ps1").read_text(encoding="utf-8")

    assert "status --porcelain" in builder
    assert "git_commit" in builder
    assert "RELEASE_GIT_COMMIT.txt" in builder
    assert "SHA256SUMS.txt" in builder
    assert "python-$PythonVersion-amd64.exe" in builder
    assert '"torch==2.11.0"' in builder
    assert '"torchvision==0.26.0"' in builder
    assert "teaching_classifier.pt" in builder
    assert "multiplicity_classifier.pt" in builder
    assert "latest_instance_segmenter.pt" in builder
    assert "latest_temporal_evidence.pt" not in builder

    assert "Get-FileHash" in installer
    assert "-Offline" in installer
    assert "-Wheelhouse" in installer
    assert "Move-Item -LiteralPath $InstallRoot" in installer
    assert "install_ui.ps1" in launcher
    assert "-Mode production" in launcher
    assert "FolderBrowserDialog" in install_ui
    assert '"-InstallRoot", $script:selectedInstallRoot' in install_ui
    assert '"-StartAfterInstall"' in install_ui


def test_offline_setup_uses_no_index_and_three_model_contract():
    setup = (ROOT / "scripts" / "setup_production.ps1").read_text(encoding="utf-8")
    start = (ROOT / "scripts" / "start_production.ps1").read_text(encoding="utf-8")

    assert '--no-index", "--find-links", $Wheelhouse' in setup
    assert "if ($Offline)" in setup
    for script in (setup, start):
        assert "teaching_classifier.pt" in script
        assert "multiplicity_classifier.pt" in script
        assert "latest_instance_segmenter.pt" in script
        assert "latest_temporal_evidence.pt" not in script

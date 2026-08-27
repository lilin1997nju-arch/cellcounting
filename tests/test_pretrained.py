from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models import resnet18

from cellvision.pretrained import (
    file_sha256,
    load_resnet18_imagenet_extractor,
    resolve_resnet18_imagenet_checkpoint,
)


ROOT = Path(__file__).parents[1]


def test_resnet18_extractor_loads_local_state_without_network(monkeypatch, tmp_path):
    checkpoint = tmp_path / "resnet18.pth"
    torch.save(resnet18(weights=None).state_dict(), checkpoint)

    def reject_network(*_args, **_kwargs):
        raise AssertionError("network download must not be attempted")

    monkeypatch.setattr(
        "torchvision.models._api.load_state_dict_from_url",
        reject_network,
    )
    extractor = load_resnet18_imagenet_extractor(
        checkpoint,
        torch.device("cpu"),
        expected_sha256=file_sha256(checkpoint),
    )

    assert isinstance(extractor.fc, torch.nn.Identity)


def test_resnet18_extractor_rejects_corrupt_checkpoint_before_loading(tmp_path):
    checkpoint = tmp_path / "resnet18.pth"
    checkpoint.write_bytes(b"not a checkpoint")

    try:
        load_resnet18_imagenet_extractor(
            checkpoint,
            torch.device("cpu"),
            expected_sha256="0" * 64,
        )
    except RuntimeError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("corrupt checkpoint was accepted")


def test_resnet18_checkpoint_resolves_from_shared_model_bundle(tmp_path):
    checkpoint = tmp_path / "models" / "resnet18-f37072fd.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weight")

    resolved = resolve_resnet18_imagenet_checkpoint(
        {
            "paths": {"artifact_root": str(tmp_path / "workspace")},
            "late_growth": {"shared_model_root": str(tmp_path)},
        }
    )

    assert resolved == checkpoint.resolve()


def test_inference_sources_do_not_request_torchvision_downloads():
    sources = "\n".join(
        (ROOT / "src" / "cellvision" / filename).read_text(encoding="utf-8")
        for filename in ("teaching.py", "late_growth_inference.py", "pretrained.py")
    )

    assert "ResNet18_Weights.DEFAULT" not in sources
    assert "load_state_dict_from_url" not in sources

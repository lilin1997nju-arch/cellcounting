from pathlib import Path

from PIL import Image

from cellvision.decode import inspect_tiff


def test_tiff_decode(tmp_path: Path):
    path = tmp_path / "sample.tif"
    Image.new("L", (32, 24), 127).save(path)
    result = inspect_tiff(path)
    assert result["decode_status"] == "ok"
    assert result["width_px"] == 32
    assert result["height_px"] == 24
    assert result["bit_depth"] == 8


def test_unreadable_tiff_is_reported(tmp_path: Path):
    path = tmp_path / "bad.tif"
    path.write_bytes(b"not-a-tiff")
    result = inspect_tiff(path)
    assert result["decode_status"] == "error"
    assert result["decode_error"]


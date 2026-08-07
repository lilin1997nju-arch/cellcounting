$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"

& $python -m cellvision audit --config configs/default.yaml
& $python -m cellvision build-manifest --config configs/default.yaml
& $python -m cellvision split --config configs/default.yaml
& $python -m cellvision baseline --config configs/default.yaml --wells A1,B3,F12,G2,H6
& $python -m cellvision train weak-segmenter --config configs/model.yaml
& $python -m pytest


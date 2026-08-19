#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="$(cd "$(dirname "$0")/.." && pwd)"
REQUESTED_ARCH="${1:-$(uname -m)}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
CURRENT_ARCH="$(uname -m)"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This builder must run on macOS." >&2
  exit 1
fi
if [[ "$REQUESTED_ARCH" != "arm64" && "$REQUESTED_ARCH" != "x86_64" ]]; then
  echo "Architecture must be arm64 or x86_64." >&2
  exit 1
fi
if [[ "$CURRENT_ARCH" != "$REQUESTED_ARCH" ]]; then
  echo "Build $REQUESTED_ARCH on a matching macOS host (current: $CURRENT_ARCH)." >&2
  exit 1
fi

COMMIT="$(git -C "$REPOSITORY" rev-parse HEAD)"
SHORT="${COMMIT:0:10}"
BUILD_ROOT="$REPOSITORY/build/review-platform-$REQUESTED_ARCH"
DIST_ROOT="$REPOSITORY/release/CellVisionReviewPlatform-$SHORT-macos-$REQUESTED_ARCH"
"$PYTHON_BIN" -m venv "$BUILD_ROOT/venv"
"$BUILD_ROOT/venv/bin/python" -m pip install --upgrade pip setuptools wheel pyinstaller
"$BUILD_ROOT/venv/bin/python" -m pip install -r "$REPOSITORY/requirements-portable-review.txt"
PYTHONPATH="$REPOSITORY/src" "$BUILD_ROOT/venv/bin/python" -m PyInstaller \
  --noconfirm --clean --windowed --onedir \
  --name CellVisionReviewPlatform \
  --paths "$REPOSITORY/src" \
  --add-data "$REPOSITORY/review-ui:review-ui" \
  --add-data "$REPOSITORY/configs:configs" \
  --distpath "$DIST_ROOT" \
  --workpath "$BUILD_ROOT/work" \
  --specpath "$BUILD_ROOT" \
  "$REPOSITORY/scripts/review_platform_entry.py"

APP="$DIST_ROOT/CellVisionReviewPlatform.app"
IDENTITY="${CODESIGN_IDENTITY:--}"
codesign --force --deep --sign "$IDENTITY" "$APP"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$DIST_ROOT.zip"
echo "macOS review platform ready: $DIST_ROOT.zip"

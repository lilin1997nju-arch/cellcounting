"""Rebase copied task paths before starting the portable normal review UI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TEXT_SUFFIXES = {".json", ".yaml", ".yml", ".csv"}


def rebase(package_root: Path) -> Path:
    package_root = package_root.expanduser().resolve()
    metadata = json.loads((package_root / "PACKAGE.json").read_text(encoding="utf-8"))
    project_root = package_root / "project"
    data_root = package_root.parent
    replacements = {
        str(metadata["source_project_root"]): str(project_root),
        str(metadata["source_data_root"]): str(data_root),
    }
    replacements.update({key.replace("\\", "/"): value.replace("\\", "/") for key, value in list(replacements.items())})
    for path in project_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for source, destination in replacements.items():
            updated = updated.replace(source, destination)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
    manifest_path = project_root / "project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["portable_review"] = True
    manifest["root"] = str(data_root)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    print(rebase(args.package_root))


if __name__ == "__main__":
    main()

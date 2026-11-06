#!/usr/bin/env python3
"""Create a source archive from publication_files.txt, excluding private runtime assets."""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", default=str(ROOT.parent / "vacv_project_bundle.zip"))
args = parser.parse_args()
manifest = ROOT / "publication_files.txt"
source_files = []
for name in manifest.read_text(encoding="utf-8-sig").splitlines():
    if not name.strip():
        continue
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Invalid publication path: {name}")
    source_files.append(path)
if len(set(source_files)) != len(source_files):
    raise ValueError("Duplicate paths in publication_files.txt")
output = Path(args.output).resolve()
if output in source_files:
    raise ValueError("Archive output cannot replace a source file.")
with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in source_files:
        archive.write(path, arcname=(Path("vacv_project") / path.relative_to(ROOT)).as_posix())
print(f"Saved {args.output}: {Path(args.output).stat().st_size / 1024 ** 2:.2f} MiB")

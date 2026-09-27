#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def sha(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    files = sorted(p for p in analysis_dir.iterdir() if p.is_file())
    required = {
        "POWER_BBLR_REPLICATION_DIAGNOSTICS.csv.gz",
        "POWER_BBLR_CELL_SUMMARY.csv",
        "POWER_GLOBAL_SUMMARY.json",
    }
    if not required.issubset({p.name for p in files}):
        raise SystemExit("analysis directory is incomplete")

    outdir = (args.output_dir or analysis_dir.parent / "handbacks").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = outdir / f"existing_bblr_power_analysis_{stamp}.zip"
    sha_path = outdir / f"{zip_path.name}.sha256"

    manifest = []
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for path in files:
            z.write(path, arcname=path.name)
            manifest.append({
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            })
        z.writestr(
            "HANDOFF_MANIFEST.json",
            json.dumps({"schema": "BBLR_EXISTING_POWER_ANALYSIS_V1", "files": manifest}, indent=2) + "\n",
        )

    digest = sha(zip_path)
    sha_path.write_text(f"{digest}  {zip_path.name}\n")
    print(json.dumps({
        "status": "PASS",
        "zip": str(zip_path),
        "sha256_file": str(sha_path),
        "sha256": digest,
        "zip_mb": round(zip_path.stat().st_size / 2**20, 3),
    }, indent=2))


if __name__ == "__main__":
    main()

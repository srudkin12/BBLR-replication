#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd

def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser(
        description="Create a targeted BBLR shard bundle for a requested design cell."
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--dgp", required=True)
    p.add_argument("--design", required=True)
    p.add_argument("--n", required=True, type=int)
    p.add_argument("--noise", required=True, type=float)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    expected_path = run_dir / "EXPECTED_CHUNKS.csv"
    if not expected_path.exists():
        raise SystemExit(f"Missing {expected_path}")

    df = pd.read_csv(expected_path)
    mask = (
        df["dgp_code"].astype(str).eq(args.dgp)
        & df["design"].astype(str).eq(args.design)
        & df["n"].astype(int).eq(args.n)
        & (df["noise_sd"].astype(float) - args.noise).abs().lt(1e-12)
    )
    chosen = df.loc[mask].copy()
    if chosen.empty:
        raise SystemExit("No matching chunks found.")

    shard_paths = [run_dir / "shards" / fn for fn in chosen["filename"].tolist()]
    missing = [str(p) for p in shard_paths if not p.exists()]
    if missing:
        raise SystemExit("Missing shard files:\n" + "\n".join(missing))

    output_dir = (args.output_dir or run_dir.parent / "handbacks").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    noise_tag = str(args.noise).replace(".", "p")
    base = f"{run_dir.name}_{args.dgp}_{args.design}_n{args.n}_sd{noise_tag}_{stamp}"
    zip_path = output_dir / f"{base}.zip"
    sha_path = output_dir / f"{base}.zip.sha256"

    meta = {
        "schema": "BBLR_TARGETED_SHARD_HANDBACK_V1",
        "run_name": run_dir.name,
        "dgp": args.dgp,
        "design": args.design,
        "n": args.n,
        "noise_sd": args.noise,
        "chunks": len(shard_paths),
        "rep_start": int(chosen["rep_start"].min()),
        "rep_stop": int(chosen["rep_stop"].max()),
    }

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("TARGET.json", json.dumps(meta, indent=2) + "\n")
        z.writestr("EXPECTED_CHUNKS_SUBSET.csv", chosen.to_csv(index=False))
        for path in shard_paths:
            z.write(path, arcname=f"shards/{path.name}")

    digest = sha256_file(zip_path)
    sha_path.write_text(f"{digest}  {zip_path.name}\n")
    print(json.dumps({
        "status": "PASS",
        "zip": str(zip_path),
        "sha256_file": str(sha_path),
        "chunks": len(shard_paths),
        "zip_mb": round(zip_path.stat().st_size / 1024 / 1024, 3),
    }, indent=2))

if __name__ == "__main__":
    main()

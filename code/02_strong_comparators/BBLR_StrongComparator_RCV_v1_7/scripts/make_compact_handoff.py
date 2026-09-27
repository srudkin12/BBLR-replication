#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED

CORE_FILES = [
    "RUN_CONFIG.json",
    "COMPLETION_AUDIT.json",
    "EXPECTED_CHUNKS.csv",
    "CHUNK_MANIFEST.csv",
    "HPC_CONFIGURATION_SUMMARY.csv",
    "HPC_SELECTED_BY_Y.csv.gz",
    "HPC_SELECTED_METHOD_SUMMARY.csv",
    "HPC_ORACLE_SLOPE_BY_METHOD.csv.gz",
    "HPC_ORACLE_MEAN_BY_METHOD.csv.gz",
    "HPC_PARETO_CONFIGURATIONS.csv.gz",
    "PHASE_BBLR_VS_COMPARATORS.csv",
    "PHASE_PREDICTION_SELECTION_REGRET.csv.gz",
    "PHASE_SELECTION_REGRET_SUMMARY.csv",
    "PHASE_BBLR_TUNING_BOUNDARY_AUDIT.csv",
    "PHASE_FAMILY_SUMMARY.csv",
]

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
        description="Create a compact, auditable BBLR result handback without copying raw shards."
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--skip-shard-hashes",
        action="store_true",
        help="Skip per-shard SHA-256 calculation. Faster, but less audit-rich.",
    )
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    audit_path = run_dir / "COMPLETION_AUDIT.json"
    if not audit_path.exists():
        raise SystemExit(f"Missing completion audit: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS":
        raise SystemExit(
            f"Refusing compact handback because completion status is {audit.get('status')!r}, not PASS."
        )

    missing = [name for name in CORE_FILES if not (run_dir / name).exists()]
    if missing:
        raise SystemExit("Missing required summary files: " + ", ".join(missing))

    shards_dir = run_dir / "shards"
    shard_paths = sorted(p for p in shards_dir.glob("*.csv.gz") if p.is_file())
    expected = int(audit.get("expected_chunks", -1))
    present = len(shard_paths)
    if expected >= 0 and present != expected:
        raise SystemExit(f"Shard count mismatch: expected {expected}, found {present}")

    output_dir = (args.output_dir or run_dir.parent / "handbacks").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{run_dir.name}_compact_handback_{stamp}"
    zip_path = output_dir / f"{base}.zip"
    sha_path = output_dir / f"{base}.zip.sha256"

    shard_rows = []
    total_shard_bytes = 0
    for i, path in enumerate(shard_paths, start=1):
        size = path.stat().st_size
        total_shard_bytes += size
        digest = "" if args.skip_shard_hashes else sha256_file(path)
        shard_rows.append({
            "filename": path.name,
            "bytes": size,
            "sha256": digest,
        })
        if not args.skip_shard_hashes and (i % 250 == 0 or i == present):
            print(f"Hashed {i}/{present} shards")

    source_rows = []
    for name in CORE_FILES:
        path = run_dir / name
        source_rows.append({
            "filename": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    handoff_manifest = {
        "schema": "BBLR_COMPACT_HANDBACK_V1",
        "run_name": run_dir.name,
        "source_run_dir": str(run_dir),
        "completion_status": audit.get("status"),
        "expected_chunks": expected,
        "present_chunks": present,
        "raw_shard_bytes_retained_on_source_machine": total_shard_bytes,
        "raw_shards_included_in_handback": False,
        "per_shard_hashes_included": not args.skip_shard_hashes,
        "core_files": CORE_FILES,
        "purpose": (
            "Compact conversational/analysis handback. Raw shards remain authoritative "
            "on the source machine and can be supplied selectively if later diagnostics require them."
        ),
    }

    # Construct CSV payloads in memory; these are small relative to the shards.
    import io
    source_buf = io.StringIO()
    sw = csv.DictWriter(source_buf, fieldnames=["filename", "bytes", "sha256"])
    sw.writeheader()
    sw.writerows(source_rows)

    shard_buf = io.StringIO()
    shw = csv.DictWriter(shard_buf, fieldnames=["filename", "bytes", "sha256"])
    shw.writeheader()
    shw.writerows(shard_rows)

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for name in CORE_FILES:
            z.write(run_dir / name, arcname=name)
        z.writestr("SOURCE_FILE_SHA256.csv", source_buf.getvalue())
        z.writestr("SHARD_SHA256.csv", shard_buf.getvalue())
        z.writestr("HANDOFF_MANIFEST.json", json.dumps(handoff_manifest, indent=2) + "\n")

    digest = sha256_file(zip_path)
    sha_path.write_text(f"{digest}  {zip_path.name}\n")

    print(json.dumps({
        "status": "PASS",
        "zip": str(zip_path),
        "sha256_file": str(sha_path),
        "zip_mb": round(zip_path.stat().st_size / 1024 / 1024, 3),
        "raw_shards_mb_retained": round(total_shard_bytes / 1024 / 1024, 3),
        "shards": present,
    }, indent=2))

if __name__ == "__main__":
    main()

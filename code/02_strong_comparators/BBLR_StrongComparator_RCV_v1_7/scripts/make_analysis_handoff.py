#!/usr/bin/env python3
"""Create a compact support-regime analysis handback.

Raw configuration summaries, oracle-per-replication files and all Monte Carlo
shards remain on the source machine.  The handback contains the empirically
selected replication-level results plus robust/phase summaries and hash
manifests, which is sufficient for routine analysis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

CORE_FILES = [
    "RUN_CONFIG.json",
    "COMPLETION_AUDIT.json",
    "EXPECTED_CHUNKS.csv",
    "CHUNK_MANIFEST.csv",
    "HPC_SELECTED_BY_Y.csv.gz",
    "HPC_SELECTED_METHOD_SUMMARY.csv",
    "HPC_SELECTED_METHOD_ROBUST_SUMMARY.csv",
    "SUPPORT_PAIRWISE_REPLICATION.csv.gz",
    "SUPPORT_PAIRWISE_CELL_SUMMARY.csv",
    "SUPPORT_PHASE_MAP.csv",
    "SUPPORT_SELECTION_REGRET_SUMMARY.csv",
    "SUPPORT_BBLR_TUNING_BOUNDARY_AUDIT.csv",
    "SUPPORT_METHOD_TUNING_BOUNDARY_AUDIT.csv",
    "SUPPORT_SELECTED_PARETO_SUMMARY.csv",
]

def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--skip-shard-hashes", action="store_true")
    args = ap.parse_args()

    run = args.run_dir.resolve()
    audit_path = run/"COMPLETION_AUDIT.json"
    if not audit_path.exists():
        raise SystemExit(f"Missing {audit_path}")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS":
        raise SystemExit(f"Completion audit is not PASS: {audit.get('status')}")

    missing = [f for f in CORE_FILES if not (run/f).exists()]
    if missing:
        raise SystemExit("Missing required analysis files: " + ", ".join(missing))

    shards = sorted((run/"shards").glob("*.csv.gz"))
    expected = int(audit["expected_chunks"])
    if len(shards) != expected:
        raise SystemExit(f"Expected {expected} shards, found {len(shards)}")

    outdir = (args.output_dir or run.parent/"handbacks").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{run.name}_analysis_handback_{stamp}"
    zpath = outdir/f"{base}.zip"
    spath = outdir/f"{base}.zip.sha256"

    source_rows = []
    for f in CORE_FILES:
        p = run/f
        source_rows.append({"filename":f, "bytes":p.stat().st_size, "sha256":sha256_file(p)})

    shard_rows = []
    raw_bytes = 0
    for i,p in enumerate(shards,1):
        raw_bytes += p.stat().st_size
        digest = "" if args.skip_shard_hashes else sha256_file(p)
        shard_rows.append({"filename":p.name, "bytes":p.stat().st_size, "sha256":digest})
        if not args.skip_shard_hashes and (i % 500 == 0 or i == len(shards)):
            print(f"Hashed {i}/{len(shards)} shards")

    import io
    sb = io.StringIO()
    w = csv.DictWriter(sb, fieldnames=["filename","bytes","sha256"])
    w.writeheader(); w.writerows(source_rows)
    hb = io.StringIO()
    w = csv.DictWriter(hb, fieldnames=["filename","bytes","sha256"])
    w.writeheader(); w.writerows(shard_rows)

    manifest = {
        "schema":"BBLR_SUPPORT_REGIME_ANALYSIS_HANDBACK_V1",
        "run_name":run.name,
        "completion_status":"PASS",
        "expected_chunks":expected,
        "present_chunks":len(shards),
        "raw_shards_included":False,
        "raw_shard_bytes_retained_on_source_machine":raw_bytes,
        "per_shard_hashes_included":not args.skip_shard_hashes,
        "oracle_replication_files_included":False,
        "configuration_summary_included":False,
        "purpose":"Routine analysis handback; raw shards and large oracle/configuration files remain authoritative on source machine.",
    }

    with ZipFile(zpath,"w",compression=ZIP_DEFLATED,compresslevel=6) as z:
        for f in CORE_FILES:
            z.write(run/f, arcname=f)
        z.writestr("SOURCE_FILE_SHA256.csv", sb.getvalue())
        z.writestr("SHARD_SHA256.csv", hb.getvalue())
        z.writestr("HANDOFF_MANIFEST.json", json.dumps(manifest,indent=2)+"\n")

    digest = sha256_file(zpath)
    spath.write_text(f"{digest}  {zpath.name}\n")
    print(json.dumps({
        "status":"PASS",
        "zip":str(zpath),
        "sha256_file":str(spath),
        "zip_mb":round(zpath.stat().st_size/1024/1024,3),
        "raw_shards_mb_retained":round(raw_bytes/1024/1024,3),
        "shards":len(shards),
    },indent=2))

if __name__ == "__main__":
    main()

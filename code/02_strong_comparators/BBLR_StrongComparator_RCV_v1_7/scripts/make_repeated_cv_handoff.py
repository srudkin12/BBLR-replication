#!/usr/bin/env python3
"""Create compact BBLR repeated-CV handback plus SHA-256."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

CORE = [
    "RUN_CONFIG.json", "COMPLETION_AUDIT.json", "LANDMARK_CELLS.csv",
    "EXPECTED_CHUNKS.csv", "CHUNK_MANIFEST.csv", "RCV_SELECTOR_ROWS.csv.gz",
    "RCV_SELECTOR_SUMMARY.csv", "RCV_TUNING_GEOMETRY.csv.gz",
    "RCV_TUNING_GEOMETRY_SUMMARY.csv",
    "RCV1000_REPLICATION_DIAGNOSTICS.csv.gz",
    "RCV1000_CELL_OPERATING_SUMMARY.csv",
    "RCV1000_BLOCK_SUMMARY.csv",
    "RCV1000_EXPECTED_REGIME_SUMMARY.csv",
    "RCV1000_MASTER_CELL_SUMMARY.csv",
]

def sha(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True,type=Path)
    args=ap.parse_args(); run=args.run_dir.resolve()
    audit=json.loads((run/'COMPLETION_AUDIT.json').read_text())
    if audit.get('status')!='PASS': raise SystemExit('refusing handback: completion audit is not PASS')
    missing=[x for x in CORE if not (run/x).exists()]
    if missing: raise SystemExit('missing summary files: '+', '.join(missing))
    out=run.parent/'handbacks'; out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    zpath=out/f'{run.name}_repeated_cv_handback_{stamp}.zip'
    with ZipFile(zpath,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for name in CORE: z.write(run/name,arcname=name)
    digest=sha(zpath); spath=Path(str(zpath)+'.sha256')
    spath.write_text(f'{digest}  {zpath.name}\n')
    print(zpath); print(spath); print(digest)
if __name__=='__main__': main()

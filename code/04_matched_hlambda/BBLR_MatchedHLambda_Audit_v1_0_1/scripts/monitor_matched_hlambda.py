#!/usr/bin/env python3
"""Fast progress monitor; does not open completed replication shards."""
import argparse
from pathlib import Path
import pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',type=Path,required=True); a=ap.parse_args()
r=a.run_dir
if not (r/'EXPECTED_CHUNKS.csv').exists(): raise SystemExit('EXPECTED_CHUNKS.csv not found yet')
exp=pd.read_csv(r/'EXPECTED_CHUNKS.csv')
shard_dir=r/'replication_shards'
present={p.name for p in shard_dir.glob('*.csv.gz')} if shard_dir.exists() else set()
done=exp[exp['filename'].isin(present)].copy()
rows=int((done['stop']-done['start']).sum()) if len(done) else 0
total=int((exp['stop']-exp['start']).sum())
print(f'Run directory: {r}')
print(f'Chunks present: {len(done):,} / {len(exp):,} ({100*len(done)/len(exp):.2f}%)')
print(f'Rows complete:  {rows:,} / {total:,} ({100*rows/total:.2f}%)')
if (r/'COMPLETION_AUDIT.json').exists():
    print('\nCompletion audit:')
    print((r/'COMPLETION_AUDIT.json').read_text())

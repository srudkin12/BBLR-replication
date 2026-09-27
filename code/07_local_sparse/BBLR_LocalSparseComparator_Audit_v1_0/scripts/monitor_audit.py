#!/usr/bin/env python3
import argparse,re
from pathlib import Path
import pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--cells',default='data/AUDIT_CELLS.csv'); ap.add_argument('--reps',type=int,default=500); args=ap.parse_args()
rd=Path(args.run_dir); fs=sorted((rd/'shards').glob('*.csv.gz'))
cells=pd.read_csv(args.cells); expected=len(cells)*args.reps
seen=set(); rows=0
for f in fs:
    m=re.search(r'^(LS_\d+)__(\d+)_(\d+)\.csv\.gz$',f.name)
    if m:
        cid,lo,hi=m.group(1),int(m.group(2)),int(m.group(3))
        for r in range(lo,hi+1): seen.add((cid,r))
    try: rows += len(pd.read_csv(f,usecols=['cell_id','rep']))
    except Exception: pass
print(f'shards complete: {len(fs)}')
print(f'unique cell-replications represented: {len(seen)} / {expected}')
print(f'completion: {100*len(seen)/expected:.2f}%')
print(f'method rows written: {rows}')

#!/usr/bin/env python3
import argparse,json,re
from pathlib import Path
import pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--cells',default='data/PRODUCTION_CELLS.csv'); ap.add_argument('--reps',type=int,default=1000); args=ap.parse_args()
rd=Path(args.run_dir); fs=sorted((rd/'shards').glob('*.csv.gz'))
try:
    cells=pd.read_csv(args.cells)
    expected_cells=len(cells); expected=expected_cells*args.reps
except Exception:
    expected_cells=None; expected=None
print(f'shards complete: {len(fs)}')
seen=set(); method_rows=0
for f in fs:
    m=re.search(r'^(P2_\d+)__(\d+)_(\d+)\.csv\.gz$',f.name)
    if m:
        cid,lo,hi=m.group(1),int(m.group(2)),int(m.group(3))
        for r in range(lo,hi+1): seen.add((cid,r))
    try: method_rows += len(pd.read_csv(f,usecols=['cell_id','rep']))
    except Exception: pass
print(f'unique cell-replications represented: {len(seen)}')
if expected is not None:
    print(f'expected cell-replications: {expected}')
    print(f'completion: {100*len(seen)/expected:.2f}%')
print(f'method rows written: {method_rows}')
if (rd/'RUN_CONFIG.json').exists(): print((rd/'RUN_CONFIG.json').read_text())

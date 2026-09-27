#!/usr/bin/env python3
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import argparse, json, sys
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bblr.production import evaluate_rep


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cells',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--reps',type=int,default=1000)
    ap.add_argument('--jobs',type=int,default=80)
    ap.add_argument('--chunk-reps',type=int,default=2)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    cells=pd.read_csv(args.cells)
    out=Path(args.output_dir); shards=out/'shards'; shards.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for c in cells.to_dict('records'):
        cid=str(c['cell_id'])
        for lo in range(0,args.reps,args.chunk_reps):
            hi=min(args.reps,lo+args.chunk_reps)
            f=shards/f'{cid}__{lo:04d}_{hi-1:04d}.csv.gz'
            if args.resume and f.exists():
                continue
            tasks.append((c,lo,hi,f))
    def one(t):
        c,lo,hi,f=t
        z=pd.concat([evaluate_rep(c,r) for r in range(lo,hi)],ignore_index=True)
        tmp=f.with_suffix(f.suffix+'.tmp')
        z.to_csv(tmp,index=False,compression='gzip')
        os.replace(tmp,f)
        return str(f)
    Parallel(n_jobs=args.jobs,backend='loky',verbose=10,pre_dispatch='n_jobs')(delayed(one)(t) for t in tasks)
    cfg={
        'cells_file':str(Path(args.cells).name),
        'cells':len(cells),'reps':args.reps,
        'expected_cell_reps':len(cells)*args.reps,
        'chunk_reps':args.chunk_reps,'jobs_requested':args.jobs,
        'scientific_version':'production_v1.0',
        'exact_nesting_safeguard':'common min_points=max(d+5,15) for BBLR Stage 1, tuned LLR, and matched-radius LLR',
        'matched_radius_method':'LLR_MATCHED_BBLR_R'
    }
    (out/'RUN_CONFIG.json').write_text(json.dumps(cfg,indent=2))

if __name__=='__main__':
    main()

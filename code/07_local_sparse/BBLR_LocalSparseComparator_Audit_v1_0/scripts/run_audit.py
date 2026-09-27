#!/usr/bin/env python3
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import argparse, json, sys
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bblr.audit import evaluate_rep


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cells',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--reps',type=int,default=500)
    ap.add_argument('--jobs',type=int,default=80)
    ap.add_argument('--chunk-reps',type=int,default=1)
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
        'scientific_version':'local_sparse_comparator_audit_v1.0',
        'cells_file':Path(args.cells).name,'cells':len(cells),'reps':args.reps,
        'expected_cell_reps':len(cells)*args.reps,'jobs_requested':args.jobs,'chunk_reps':args.chunk_reps,
        'local_sparse_method':'LL_ALASSO',
        'local_sparse_description':'hard-ball local linear adaptive Lasso; unpenalized intercept; local slope standardization; adaptive weights from unpenalized local fit; radius and lambda fraction tuned by repeated outcome CV',
        'lambda_fractions':[0.0,0.025,0.05,0.10,0.20,0.40,0.70],
        'bblr_eps_over_r':[0.0,0.5,1.0,1.5,2.0],
    }
    (out/'RUN_CONFIG.json').write_text(json.dumps(cfg,indent=2))

if __name__=='__main__':
    main()

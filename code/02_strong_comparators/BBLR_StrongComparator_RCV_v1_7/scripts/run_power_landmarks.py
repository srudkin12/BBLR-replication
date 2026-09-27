#!/usr/bin/env python3
"""Run a non-Cartesian set of landmark BBLR design cells in parallel."""
from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed, parallel_config

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.run_univariate_hpc as core


def make_chunks(cells,reps,chunk_reps):
    out=[]; cid=0
    dmap={d:i for i,d in enumerate(sorted(cells.dgp.unique()))}
    xmap={d:i for i,d in enumerate(sorted(cells.design.unique()))}
    ns=sorted(cells.n_train.unique()); nmap={v:i for i,v in enumerate(ns)}
    ss=sorted(cells.noise_sd.unique()); smap={v:i for i,v in enumerate(ss)}
    for row in cells.itertuples(index=False):
        for start in range(0,reps,chunk_reps):
            out.append(core.ChunkSpec(
                cid,dmap[row.dgp],row.dgp,xmap[row.design],row.design,
                nmap[row.n_train],int(row.n_train),smap[row.noise_sd],float(row.noise_sd),
                start,min(start+chunk_reps,reps)))
            cid+=1
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cells',required=True,type=Path)
    ap.add_argument('--reps',type=int,default=300)
    ap.add_argument('--n-validation',type=int,default=0)
    ap.add_argument('--n-test',type=int,default=3000)
    ap.add_argument('--n-grid-eval',type=int,default=401)
    ap.add_argument('--r-grid',default='0.08,0.12,0.18,0.26,0.38,0.55,0.80,1.20,2.00,4.00')
    ap.add_argument('--eps-grid',default='0,0.04,0.08,0.12,0.18,0.26,0.38,0.55,0.80,1.20,2.00,4.00')
    ap.add_argument('--spline-lambda-grid',default='1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000')
    ap.add_argument('--knn-k-grid',default='15,25,40,60,90,130,180,250,350,500')
    ap.add_argument('--gap-factor-grid',default='16,20,30,inf')
    ap.add_argument('--component-min-points',type=int,default=15)
    ap.add_argument('--min-stage1',type=int,default=15)
    ap.add_argument('--min-stage2-grid',default='1')
    ap.add_argument('--guard-width',type=float,default=0.10)
    ap.add_argument('--comparators',default='LLR,AKLLR,AKLP2,SPLINE,BBLR,CAKLLR,CAKLP2,CSPLINE')
    ap.add_argument('--seed-offset',type=int,default=314159)
    ap.add_argument('--jobs',type=int,default=1)
    ap.add_argument('--chunk-reps',type=int,default=10)
    ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    cells=pd.read_csv(args.cells)
    req={'dgp','design','n_train','noise_sd'}
    if not req.issubset(cells.columns): raise SystemExit(f'cells must contain {sorted(req)}')
    if cells.duplicated(list(req)).any(): raise SystemExit('duplicate landmark cells')
    for d in cells.dgp:
        if d not in core.DGP_REGISTRY: raise SystemExit(f'unknown DGP {d}')
    for d in cells.design:
        if d not in core.SUPPORT_DESIGN_REGISTRY: raise SystemExit(f'unknown design {d}')
    comps=[x.upper() for x in core.parse_str_list(args.comparators)]
    cfg={
        'dgps':cells.dgp.unique().tolist(),'designs':cells.design.unique().tolist(),
        'n_grid':sorted(cells.n_train.unique().tolist()),'noise_grid':sorted(cells.noise_sd.unique().tolist()),
        'reps':args.reps,'n_validation':args.n_validation,'n_test':args.n_test,'n_grid_eval':args.n_grid_eval,
        'r_grid':core.parse_float_list(args.r_grid),'eps_grid':core.parse_float_list(args.eps_grid),
        'spline_lambda_grid':core.parse_float_list(args.spline_lambda_grid),
        'knn_k_grid':core.parse_int_list(args.knn_k_grid),'gap_factor_grid':core.parse_float_list(args.gap_factor_grid),
        'component_min_points':args.component_min_points,'min_stage1':args.min_stage1,
        'min_stage2_grid':core.parse_int_list(args.min_stage2_grid),'guard_width':args.guard_width,
        'comparators':comps,'seed_offset':args.seed_offset,'jobs':args.jobs,
        'chunk_reps':args.chunk_reps,'backend':'loky','resume':args.resume,
        'thread_env':{v:os.environ.get(v) for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        'cell_file':str(args.cells.resolve()),
    }
    out=args.output_dir.resolve(); shards=out/'shards'; shards.mkdir(parents=True,exist_ok=True)
    (out/'RUN_CONFIG.json').write_text(json.dumps(cfg,indent=2)+'\n')
    cells.to_csv(out/'LANDMARK_CELLS.csv',index=False)
    chunks=make_chunks(cells,args.reps,args.chunk_reps)
    pd.DataFrame([asdict(c)|{'filename':core.chunk_filename(c)} for c in chunks]).to_csv(out/'EXPECTED_CHUNKS.csv',index=False)
    t0=time.perf_counter()
    if args.jobs==1: res=[core.run_chunk(c,cfg,str(shards)) for c in chunks]
    else:
        with parallel_config(backend='loky',n_jobs=args.jobs,inner_max_num_threads=1):
            res=Parallel(verbose=10,batch_size=1,pre_dispatch='2*n_jobs')(
                delayed(core.run_chunk)(c,cfg,str(shards)) for c in chunks)
    elapsed=time.perf_counter()-t0
    pd.DataFrame(res).sort_values('chunk_id').to_csv(out/'CHUNK_MANIFEST.csv',index=False)
    missing=[core.chunk_filename(c) for c in chunks if not (shards/core.chunk_filename(c)).exists()]
    audit={'expected_chunks':len(chunks),'present_chunks':len(chunks)-len(missing),'missing_chunks':len(missing),
           'missing_files':missing[:100],'wall_seconds':elapsed,'jobs':args.jobs,'chunk_reps':args.chunk_reps,
           'status':'PASS' if not missing else 'FAIL'}
    (out/'COMPLETION_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__': main()

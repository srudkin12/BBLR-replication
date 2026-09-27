#!/usr/bin/env python3
"""HPC repeated-CV strong-comparator confirmation for univariate BBLR."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"

import argparse, json, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bblr.univariate import DGP_REGISTRY, SUPPORT_DESIGN_REGISTRY, generate_sample
from bblr.strong_comparator_cv import repeated_cv_risk_surface, select_minimum_risk, evaluate_selected
from scripts.run_univariate_hpc import stable_seed, parse_float_list, parse_int_list, parse_str_list

@dataclass(frozen=True)
class ChunkSpec:
    chunk_id:int; cell_id:int; label:str; dgp:str; design:str; n_train:int; noise_sd:float
    panel_block:str; expected_regime:str; reason:str; rep_start:int; rep_stop:int

def stem(s:ChunkSpec)->str:
    return f"chunk_{s.chunk_id:06d}_cell{s.cell_id:02d}_{s.dgp}_{s.design}_rep{s.rep_start:05d}-{s.rep_stop-1:05d}"

def make_chunks(cells:pd.DataFrame,reps:int,chunk_reps:int)->List[ChunkSpec]:
    out=[]; cid=0
    for cell_id,row in enumerate(cells.itertuples(index=False)):
        for start in range(0,reps,chunk_reps):
            out.append(ChunkSpec(
                cid,cell_id,str(row.label),str(row.dgp),str(row.design),int(row.n_train),float(row.noise_sd),
                str(getattr(row,'panel_block','')),str(getattr(row,'expected_regime','')),
                str(getattr(row,'reason','')),start,min(start+chunk_reps,reps)))
            cid+=1
    return out

def run_chunk(spec:ChunkSpec,cfg:Dict[str,object],out_dir:str)->Dict[str,object]:
    out=Path(out_dir); risk_dir=out/'risk_shards'; sel_dir=out/'selected_shards'
    risk_dir.mkdir(parents=True,exist_ok=True); sel_dir.mkdir(parents=True,exist_ok=True)
    rp=risk_dir/f"{stem(spec)}_risk.csv.gz"; sp=sel_dir/f"{stem(spec)}_selected.csv.gz"
    if bool(cfg['resume']) and rp.exists() and sp.exists() and rp.stat().st_size>50 and sp.stat().st_size>50:
        return {'chunk_id':spec.chunk_id,'status':'skipped','risk_path':str(rp),'selected_path':str(sp),'seconds':0.0}
    t0=time.perf_counter(); risks=[]; sels=[]
    for rep in range(spec.rep_start,spec.rep_stop):
        n_dev=int(round(spec.n_train*float(cfg['development_multiplier'])))
        x_dev,y_dev,_=generate_sample(spec.dgp,n_dev,spec.noise_sd,
            stable_seed(int(cfg['seed_offset']),spec.cell_id,rep,800),spec.design)
        x_test,y_test,_=generate_sample(spec.dgp,int(cfg['n_test']),spec.noise_sd,
            stable_seed(int(cfg['seed_offset']),spec.cell_id,rep,801),spec.design)
        seed=stable_seed(int(cfg['seed_offset']),spec.cell_id,rep,802)
        risk=repeated_cv_risk_surface(
            x_dev,y_dev,methods=cfg['methods'],r_grid=cfg['r_grid'],eps_grid=cfg['eps_grid'],
            k_grid=cfg['k_grid'],lambda_grid=cfg['lambda_grid'],gap_factor_grid=cfg['gap_factor_grid'],
            folds=int(cfg['folds']),repeats=int(cfg['repeats']),seed=seed,
            min_stage1=int(cfg['min_stage1']),min_stage2=int(cfg['min_stage2']),
            component_min_points=int(cfg['component_min_points']))
        meta={'cell_id':spec.cell_id,'label':spec.label,'panel_block':spec.panel_block,
              'expected_regime':spec.expected_regime,'reason':spec.reason,'dgp':spec.dgp,
              'design':spec.design,'n_train_legacy':spec.n_train,'n_development':n_dev,
              'noise_sd':spec.noise_sd,'rep':rep,'cv_folds':int(cfg['folds']),'cv_repeats':int(cfg['repeats'])}
        for k,v in meta.items(): risk[k]=v
        selected=select_minimum_risk(risk)
        ev=evaluate_selected(selected,x_dev,y_dev,x_test,y_test,dgp_code=spec.dgp,
            guard_width=float(cfg['guard_width']),n_grid=int(cfg['n_grid_eval']),
            min_stage1=int(cfg['min_stage1']),min_stage2=int(cfg['min_stage2']),
            component_min_points=int(cfg['component_min_points']))
        for k,v in meta.items(): ev[k]=v
        risks.append(risk); sels.append(ev)
    rdf=pd.concat(risks,ignore_index=True); sdf=pd.concat(sels,ignore_index=True)
    tr=rp.with_suffix(rp.suffix+f'.tmp.{os.getpid()}'); ts=sp.with_suffix(sp.suffix+f'.tmp.{os.getpid()}')
    rdf.to_csv(tr,index=False,compression='gzip'); sdf.to_csv(ts,index=False,compression='gzip')
    os.replace(tr,rp); os.replace(ts,sp)
    return {'chunk_id':spec.chunk_id,'status':'written','risk_path':str(rp),'selected_path':str(sp),
            'risk_rows':len(rdf),'selected_rows':len(sdf),'seconds':time.perf_counter()-t0}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cells',type=Path,default=ROOT/'STRONG_COMPARATOR_CELLS.csv')
    ap.add_argument('--reps',type=int,default=1000)
    ap.add_argument('--folds',type=int,default=5); ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--development-multiplier',type=float,default=2.0); ap.add_argument('--n-test',type=int,default=3000)
    ap.add_argument('--n-grid-eval',type=int,default=401)
    ap.add_argument('--r-grid',default='0.08,0.12,0.18,0.26,0.38,0.55,0.80,1.20,2.00,4.00')
    ap.add_argument('--eps-grid',default='0,0.04,0.08,0.12,0.18,0.26,0.38,0.55,0.80,1.20,2.00,4.00')
    ap.add_argument('--k-grid',default='15,25,40,60,90,130,180,250,350,500')
    ap.add_argument('--lambda-grid',default='1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000')
    ap.add_argument('--gap-factor-grid',default='16,20,30,inf')
    ap.add_argument('--methods',default='BBLR,LLR,AKLLR,AKLP2,SPLINE,CAKLLR,CAKLP2,CSPLINE')
    ap.add_argument('--min-stage1',type=int,default=15); ap.add_argument('--min-stage2',type=int,default=1)
    ap.add_argument('--component-min-points',type=int,default=15); ap.add_argument('--guard-width',type=float,default=0.10)
    ap.add_argument('--seed-offset',type=int,default=2718281); ap.add_argument('--jobs',type=int,default=1)
    ap.add_argument('--chunk-reps',type=int,default=5); ap.add_argument('--resume',action='store_true')
    ap.add_argument('--output-dir',required=True,type=Path)
    args=ap.parse_args()
    cells=pd.read_csv(args.cells)
    req={'label','dgp','design','n_train','noise_sd'}
    if not req.issubset(cells.columns): raise SystemExit(f'cells file must contain {sorted(req)}')
    for d in cells.dgp:
        if d not in DGP_REGISTRY: raise SystemExit(f'unknown DGP {d}')
    allowed=set(SUPPORT_DESIGN_REGISTRY)|{'uniform','normal','skewed','gap'}
    for d in cells.design:
        if d not in allowed: raise SystemExit(f'unknown support design {d}')
    cfg={'folds':args.folds,'repeats':args.repeats,'development_multiplier':args.development_multiplier,
         'n_test':args.n_test,'n_grid_eval':args.n_grid_eval,'r_grid':parse_float_list(args.r_grid),
         'eps_grid':parse_float_list(args.eps_grid),'k_grid':parse_int_list(args.k_grid),
         'lambda_grid':parse_float_list(args.lambda_grid),'gap_factor_grid':parse_float_list(args.gap_factor_grid),
         'methods':[m.upper() for m in parse_str_list(args.methods)],'min_stage1':args.min_stage1,
         'min_stage2':args.min_stage2,'component_min_points':args.component_min_points,
         'guard_width':args.guard_width,'seed_offset':args.seed_offset,'jobs':args.jobs,
         'chunk_reps':args.chunk_reps,'resume':args.resume,'cells_file':str(args.cells.resolve()),
         'thread_env':{v:os.environ.get(v) for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    (out/'RUN_CONFIG.json').write_text(json.dumps(cfg,indent=2)+'\n'); cells.to_csv(out/'STRONG_COMPARATOR_CELLS.csv',index=False)
    chunks=make_chunks(cells,args.reps,args.chunk_reps)
    pd.DataFrame([asdict(c)|{'risk_filename':f'{stem(c)}_risk.csv.gz','selected_filename':f'{stem(c)}_selected.csv.gz'} for c in chunks]).to_csv(out/'EXPECTED_CHUNKS.csv',index=False)
    t0=time.perf_counter()
    if args.jobs==1: results=[run_chunk(c,cfg,str(out)) for c in chunks]
    else:
        with parallel_config(backend='loky',n_jobs=args.jobs,inner_max_num_threads=1):
            results=Parallel(verbose=10,batch_size=1,pre_dispatch='2*n_jobs')(
                delayed(run_chunk)(c,cfg,str(out)) for c in chunks)
    elapsed=time.perf_counter()-t0
    pd.DataFrame(results).sort_values('chunk_id').to_csv(out/'CHUNK_MANIFEST.csv',index=False)
    missing=[]
    for c in chunks:
        if not (out/'risk_shards'/f'{stem(c)}_risk.csv.gz').exists() or not (out/'selected_shards'/f'{stem(c)}_selected.csv.gz').exists():
            missing.append(stem(c))
    audit={'expected_chunks':len(chunks),'present_chunks':len(chunks)-len(missing),'missing_chunks':len(missing),
           'missing_files':missing[:100],'wall_seconds':elapsed,'jobs':args.jobs,'chunk_reps':args.chunk_reps,
           'status':'PASS' if not missing else 'FAIL'}
    (out/'COMPLETION_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,indent=2))
if __name__=='__main__': main()

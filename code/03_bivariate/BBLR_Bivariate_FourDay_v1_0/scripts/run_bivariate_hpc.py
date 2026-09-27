#!/usr/bin/env python3
from __future__ import annotations
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[v]='1'
import argparse, json, time, sys
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel,delayed,parallel_config
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from bblr.bivariate import DGP_REGISTRY,SUPPORT_LABELS,generate_sample
from bblr.bivariate_cv import repeated_cv_risk_surface,select_minimum_risk,evaluate_selected

def stable_seed(base,cell,rep,stream):
    ss=np.random.SeedSequence([int(base),int(cell),int(rep),int(stream)]); return int(ss.generate_state(1)[0])
def fl(s): return [float(x) for x in str(s).split(',')]
def il(s): return [int(x) for x in str(s).split(',')]
def sl(s): return [x.strip() for x in str(s).split(',') if x.strip()]
@dataclass(frozen=True)
class Chunk:
    chunk_id:int; cell_id:int; label:str; dgp:str; design:str; n_train:int; noise_sd:float; block:str; expected_regime:str; rep_start:int; rep_stop:int

def stem(c): return f'chunk_{c.chunk_id:06d}_cell{c.cell_id:02d}_{c.dgp}_{c.design}_rep{c.rep_start:05d}-{c.rep_stop-1:05d}'
def chunks(cells,reps,chunk_reps):
    out=[]; cid=0
    for cell_id,row in enumerate(cells.itertuples(index=False)):
        for a in range(0,reps,chunk_reps):
            out.append(Chunk(cid,cell_id,str(row.label),str(row.dgp),str(row.design),int(row.n_train),float(row.noise_sd),str(row.block),str(row.expected_regime),a,min(a+chunk_reps,reps))); cid+=1
    return out

def run_chunk(c,cfg,outdir):
    out=Path(outdir); rd=out/'risk_shards'; sd=out/'selected_shards'; rd.mkdir(parents=True,exist_ok=True); sd.mkdir(parents=True,exist_ok=True)
    rp=rd/f'{stem(c)}_risk.csv.gz'; sp=sd/f'{stem(c)}_selected.csv.gz'
    if cfg['resume'] and rp.exists() and sp.exists() and rp.stat().st_size>50 and sp.stat().st_size>50:
        return {'chunk_id':c.chunk_id,'status':'skipped','seconds':0.0,'risk_path':str(rp),'selected_path':str(sp)}
    t0=time.perf_counter(); risks=[]; sels=[]
    for rep in range(c.rep_start,c.rep_stop):
        ndev=int(round(c.n_train*cfg['development_multiplier']))
        X,y,_=generate_sample(c.dgp,ndev,c.noise_sd,stable_seed(cfg['seed_offset'],c.cell_id,rep,900),c.design)
        Xt,yt,_=generate_sample(c.dgp,cfg['n_test'],c.noise_sd,stable_seed(cfg['seed_offset'],c.cell_id,rep,901),c.design)
        risk=repeated_cv_risk_surface(X,y,cfg['methods'],cfg['r_grid'],cfg['eps_grid'],cfg['k_grid'],cfg['knot_grid'],cfg['alpha_grid'],cfg['gap_grid'],cfg['folds'],cfg['repeats'],stable_seed(cfg['seed_offset'],c.cell_id,rep,902),cfg['min_stage1'])
        meta={'cell_id':c.cell_id,'label':c.label,'block':c.block,'expected_regime':c.expected_regime,'dgp':c.dgp,'design':c.design,'n_train_legacy':c.n_train,'n_development':ndev,'noise_sd':c.noise_sd,'rep':rep,'cv_folds':cfg['folds'],'cv_repeats':cfg['repeats']}
        for k,v in meta.items(): risk[k]=v
        sel=select_minimum_risk(risk)
        ev=evaluate_selected(sel,X,y,Xt,yt,c.dgp,min_stage1=cfg['min_stage1'],grid_n=cfg['grid_n'])
        for k,v in meta.items(): ev[k]=v
        risks.append(risk); sels.append(ev)
    rdf=pd.concat(risks,ignore_index=True); sdf=pd.concat(sels,ignore_index=True)
    tr=Path(str(rp)+f'.tmp.{os.getpid()}'); ts=Path(str(sp)+f'.tmp.{os.getpid()}')
    rdf.to_csv(tr,index=False,compression='gzip'); sdf.to_csv(ts,index=False,compression='gzip'); os.replace(tr,rp); os.replace(ts,sp)
    return {'chunk_id':c.chunk_id,'status':'written','seconds':time.perf_counter()-t0,'risk_rows':len(rdf),'selected_rows':len(sdf),'risk_path':str(rp),'selected_path':str(sp)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cells',type=Path,default=ROOT/'BIVARIATE_CELLS.csv'); ap.add_argument('--reps',type=int,default=600); ap.add_argument('--folds',type=int,default=5); ap.add_argument('--repeats',type=int,default=5); ap.add_argument('--development-multiplier',type=float,default=2.0); ap.add_argument('--n-test',type=int,default=2500); ap.add_argument('--grid-n',type=int,default=31)
    ap.add_argument('--r-grid',default='0.25,0.35,0.50,0.70,1.00,1.40,2.00,3.00,6.00'); ap.add_argument('--eps-grid',default='0,0.15,0.25,0.35,0.50,0.70,1.00,1.40,2.00,3.00,6.00'); ap.add_argument('--k-grid',default='30,50,80,120,180,260,380,550,900,1400,2500'); ap.add_argument('--knot-grid',default='4,6,8'); ap.add_argument('--alpha-grid',default='0.001,0.1,10,1000'); ap.add_argument('--gap-grid',default='4,6,inf')
    ap.add_argument('--methods',default='BBLR,LLR,LP2,AKLLR,AKLP2,TSPLINE,OLS,GQ2'); ap.add_argument('--min-stage1',type=int,default=10); ap.add_argument('--seed-offset',type=int,default=3141592); ap.add_argument('--jobs',type=int,default=1); ap.add_argument('--chunk-reps',type=int,default=5); ap.add_argument('--resume',action='store_true'); ap.add_argument('--output-dir',required=True,type=Path)
    a=ap.parse_args(); cells=pd.read_csv(a.cells)
    for d in cells.dgp:
        if d not in DGP_REGISTRY: raise SystemExit(f'unknown DGP {d}')
    for d in cells.design:
        if d not in SUPPORT_LABELS: raise SystemExit(f'unknown support {d}')
    cfg={'folds':a.folds,'repeats':a.repeats,'development_multiplier':a.development_multiplier,'n_test':a.n_test,'grid_n':a.grid_n,'r_grid':fl(a.r_grid),'eps_grid':fl(a.eps_grid),'k_grid':il(a.k_grid),'knot_grid':il(a.knot_grid),'alpha_grid':fl(a.alpha_grid),'gap_grid':fl(a.gap_grid),'methods':[x.upper() for x in sl(a.methods)],'min_stage1':a.min_stage1,'seed_offset':a.seed_offset,'jobs':a.jobs,'chunk_reps':a.chunk_reps,'resume':a.resume}
    out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); cells.to_csv(out/'BIVARIATE_CELLS.csv',index=False); (out/'RUN_CONFIG.json').write_text(json.dumps(cfg,indent=2)+'\n')
    cs=chunks(cells,a.reps,a.chunk_reps); pd.DataFrame([asdict(c)|{'risk_filename':f'{stem(c)}_risk.csv.gz','selected_filename':f'{stem(c)}_selected.csv.gz'} for c in cs]).to_csv(out/'EXPECTED_CHUNKS.csv',index=False)
    t=time.perf_counter()
    if a.jobs==1: res=[run_chunk(c,cfg,str(out)) for c in cs]
    else:
        with parallel_config(backend='loky',n_jobs=a.jobs,inner_max_num_threads=1): res=Parallel(verbose=10,batch_size=1,pre_dispatch='2*n_jobs')(delayed(run_chunk)(c,cfg,str(out)) for c in cs)
    elapsed=time.perf_counter()-t; pd.DataFrame(res).sort_values('chunk_id').to_csv(out/'CHUNK_MANIFEST.csv',index=False)
    missing=[]
    for c in cs:
        if not (out/'risk_shards'/f'{stem(c)}_risk.csv.gz').exists() or not (out/'selected_shards'/f'{stem(c)}_selected.csv.gz').exists(): missing.append(stem(c))
    audit={'expected_chunks':len(cs),'present_chunks':len(cs)-len(missing),'missing_chunks':len(missing),'missing_files':missing[:100],'wall_seconds':elapsed,'jobs':a.jobs,'chunk_reps':a.chunk_reps,'status':'PASS' if not missing else 'FAIL'}
    (out/'COMPLETION_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,indent=2))
if __name__=='__main__': main()

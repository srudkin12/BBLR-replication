#!/usr/bin/env python3
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
import argparse,json,sys,time
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np,pandas as pd
from joblib import Parallel,delayed,parallel_config
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bblr.univariate import DGP_REGISTRY,SUPPORT_DESIGN_REGISTRY,differentiable_mask,generate_sample
from bblr.univariate_fast import FastBBLR1D
from bblr.matched_kernel import MatchedHLocalLinear1D,gamma_lambda
from bblr.stable_matched_kernel import StableMatchedHLocalLinear1D
SEED_OFFSET=1618033;N_TEST=3000;N_GRID=401;GUARD_WIDTH=.10;MIN_STAGE1=15;MIN_STAGE2=1

def stable_seed(base,*parts):
    ss=np.random.SeedSequence([int(base),*map(int,parts)]);return int(ss.generate_state(1,dtype=np.uint32)[0])
def mse(a,t,m):
    ok=np.isfinite(a)&np.isfinite(t)&np.asarray(m,bool)
    if not np.any(ok):return np.nan,0
    d=a[ok]-t[ok];return float(np.mean(d*d)),int(ok.sum())
def support_mask(x,design):
    x=np.asarray(x,float)
    if design in SUPPORT_DESIGN_REGISTRY:
        s=SUPPORT_DESIGN_REGISTRY[design]
        if s.family=='hard_gap' and float(s.parameter_value)>0:return np.abs(x)>=float(s.parameter_value)-1e-14
    return np.ones(x.size,bool)

def evaluate(row):
    cid=int(row.cell_id);rep=int(row.rep);dgp_code=str(row.dgp);design=str(row.design);n=int(row.n_development);noise=float(row.noise_sd);r=float(row.r);eps=float(row.eps);lam=eps/r
    x,y,_=generate_sample(dgp_code,n,noise,stable_seed(SEED_OFFSET,cid,rep,700),design);xt,_,_=generate_sample(dgp_code,N_TEST,noise,stable_seed(SEED_OFFSET,cid,rep,701),design)
    dgp=DGP_REGISTRY[dgp_code];xg=np.linspace(dgp.support[0],dgp.support[1],N_GRID);tt=np.asarray(dgp.derivative(xt),float);tg=np.asarray(dgp.derivative(xg),float);mt=differentiable_mask(xt,dgp,GUARD_WIDTH);mg_full=differentiable_mask(xg,dgp,GUARD_WIDTH);mg=mg_full&support_mask(xg,design)
    b=FastBBLR1D(r,eps,MIN_STAGE1,MIN_STAGE2).fit(x,y);_,bs_t,_=b.predict_with_slope(xt);_,bs_g,_=b.predict_with_slope(xg)
    wn=MatchedHLocalLinear1D(r,lam,MIN_STAGE1,safe=True).fit(x,y);_,sn_t,dn_t=wn.predict_with_slope(xt);_,sn_g,dn_g=wn.predict_with_slope(xg)
    wc=StableMatchedHLocalLinear1D(r,lam,MIN_STAGE1).fit(x,y);sc_t,dc_t=wc.slopes(xt);sc_g,dc_g=wc.slopes(xg)
    bm_t,_=mse(bs_t,tt,mt);bm_g,_=mse(bs_g,tg,mg);bm_gfull,_=mse(bs_g,tg,mg_full)
    nn_t,_=mse(sn_t,tt,mt);nn_g,_=mse(sn_g,tg,mg);cc_t,_=mse(sc_t,tt,mt);cc_g,_=mse(sc_g,tg,mg)
    out=row.to_dict();out.update({'lambda':lam,'Gamma_lambda':gamma_lambda(lam),'lambda_stratum':'eps0' if eps==0 else ('lt1' if lam<1-1e-12 else ('eq1' if abs(lam-1)<=1e-12 else 'gt1')),'gamma_abs_lt_0p01':abs(gamma_lambda(lam))<.01,'bblr_test_mse':bm_t,'bblr_supported_grid_mse':bm_g,'wnormal_test_mse':nn_t,'wnormal_supported_grid_mse':nn_g,'wcenter_test_mse':cc_t,'wcenter_supported_grid_mse':cc_g,'normal_center_test_max_abs_slope_diff':float(np.nanmax(np.abs(sn_t-sc_t))),'normal_center_supported_grid_max_abs_slope_diff':float(np.nanmax(np.abs(sn_g[mg]-sc_g[mg]))) if np.any(mg) else np.nan,'normal_test_fallback_share':float(np.mean(dn_t['fallback'][mt])),'center_test_fallback_share':float(np.mean(dc_t['fallback'][mt])),'fallback_mask_disagreement_test_share':float(np.mean(dn_t['fallback'][mt]!=dc_t['fallback'][mt])),'archive_test_rmse_abs_diff':abs(np.sqrt(bm_t)-float(row.test_slope_rmse)),'archive_grid_rmse_abs_diff':abs(np.sqrt(bm_gfull)-float(row.grid_slope_rmse))})
    return out
@dataclass(frozen=True)
class Chunk:chunk_id:int;start:int;stop:int
def stem(c):return f"chunk_{c.chunk_id:05d}_rows{c.start:06d}-{c.stop-1:06d}"
def run_chunk(c,df,out,resume):
    sd=Path(out)/'replication_shards';sd.mkdir(parents=True,exist_ok=True);p=sd/(stem(c)+'.csv.gz')
    if resume and p.exists() and p.stat().st_size>100:return {'chunk_id':c.chunk_id,'status':'skipped'}
    t=time.perf_counter();ans=[evaluate(df.iloc[i]) for i in range(c.start,c.stop)];tmp=p.with_suffix(p.suffix+f'.tmp.{os.getpid()}');pd.DataFrame(ans).to_csv(tmp,index=False,compression='gzip');os.replace(tmp,p);return {'chunk_id':c.chunk_id,'status':'written','rows':len(ans),'seconds':time.perf_counter()-t}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--configs',type=Path,default=ROOT/'data/FROZEN_SELECTED_CONFIGS.csv.gz');ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--jobs',type=int,default=1);ap.add_argument('--chunk-rows',type=int,default=25);ap.add_argument('--resume',action='store_true');a=ap.parse_args();df=pd.read_csv(a.configs).sort_values(['cell_id','rep']).reset_index(drop=True);chunks=[Chunk(i,s,min(s+a.chunk_rows,len(df))) for i,s in enumerate(range(0,len(df),a.chunk_rows))];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);df.to_csv(out/'FROZEN_SELECTED_CONFIGS_USED.csv.gz',index=False,compression='gzip');pd.DataFrame([asdict(c)|{'filename':stem(c)+'.csv.gz'} for c in chunks]).to_csv(out/'EXPECTED_CHUNKS.csv',index=False);(out/'RUN_CONFIG.json').write_text(json.dumps({'rows':len(df),'jobs':a.jobs,'chunk_rows':a.chunk_rows,'selection':'frozen; no retuning','comparison':'same W_lambda estimator, normal vs centred arithmetic'},indent=2)+'\n');t=time.perf_counter()
    if a.jobs==1:res=[run_chunk(c,df,str(out),a.resume) for c in chunks]
    else:
        with parallel_config(backend='loky',n_jobs=a.jobs,inner_max_num_threads=1):res=Parallel(verbose=10,batch_size=1,pre_dispatch='2*n_jobs')(delayed(run_chunk)(c,df,str(out),a.resume) for c in chunks)
    pd.DataFrame(res).to_csv(out/'CHUNK_RUN_LOG.csv',index=False);missing=[stem(c)+'.csv.gz' for c in chunks if not (out/'replication_shards'/(stem(c)+'.csv.gz')).exists()];audit={'expected_rows':len(df),'expected_chunks':len(chunks),'missing_chunks':len(missing),'wall_seconds':time.perf_counter()-t,'status':'PASS' if not missing else 'FAIL'};(out/'COMPLETION_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()

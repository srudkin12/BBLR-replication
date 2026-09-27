from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from .config import SUPPORT,N_GRID,METHODS,BASE_SEED,CV_REPEATS,CV_FOLDS
from .dgp import generate,g_true
from .cv import split_schedule,tune_bblr,tune_llr,tune_adaptive,tune_spline
from .estimators import bblr_predict,llr_predict,adaptive_local_poly_predict,smoothing_spline_predict

def seed_for(sigma:float, rep:int, base_seed:int=BASE_SEED):
    # stable integer key independent of worker/task ordering
    sig=int(round(float(sigma)*1000))
    ss=np.random.SeedSequence([int(base_seed),sig,int(rep)])
    return int(ss.generate_state(1,dtype=np.uint32)[0])

def run_rep(sigma:float, rep:int, base_seed:int=BASE_SEED):
    t0=time.time(); seed=seed_for(sigma,rep,base_seed)
    x,y,xt,yt=generate(seed,float(sigma),1000,3000)
    splits=split_schedule(len(x),seed ^ 0xA5A5A5A5,CV_REPEATS,CV_FOLDS)
    grid=np.linspace(SUPPORT[0],SUPPORT[1],N_GRID)
    gt=g_true(xt); gg=g_true(grid)
    rows=[]
    # BBLR
    (r,e),cv=tune_bblr(x,y,splits)
    pt,dt,diag=bblr_predict(x,y,xt,r,e); pg,dg,_=bblr_predict(x,y,grid,r,e)
    rows.append(dict(method="BBLR",sigma=sigma,rep=rep,seed=seed,param1=r,param2=e,cv_mse=cv,
        outcome_rmse=float(np.sqrt(np.mean((yt-pt)**2))),deriv_rmse_test=float(np.sqrt(np.mean((gt-dt)**2))),
        deriv_rmse_grid=float(np.sqrt(np.mean((gg-dg)**2))),stage2_active=int(e>0),**diag))
    # LLR
    rL,cv=tune_llr(x,y,splits); pt,dt,diag=llr_predict(x,y,xt,rL); _,dg,_=llr_predict(x,y,grid,rL)
    rows.append(dict(method="LLR",sigma=sigma,rep=rep,seed=seed,param1=rL,param2=np.nan,cv_mse=cv,
        outcome_rmse=float(np.sqrt(np.mean((yt-pt)**2))),deriv_rmse_test=float(np.sqrt(np.mean((gt-dt)**2))),
        deriv_rmse_grid=float(np.sqrt(np.mean((gg-dg)**2))),stage2_active=0,**diag))
    # AKLLR
    k,cv=tune_adaptive(x,y,splits,degree=1); pt,dt=adaptive_local_poly_predict(x,y,xt,k,1); _,dg=adaptive_local_poly_predict(x,y,grid,k,1)
    rows.append(dict(method="AKLLR",sigma=sigma,rep=rep,seed=seed,param1=k,param2=np.nan,cv_mse=cv,
        outcome_rmse=float(np.sqrt(np.mean((yt-pt)**2))),deriv_rmse_test=float(np.sqrt(np.mean((gt-dt)**2))),
        deriv_rmse_grid=float(np.sqrt(np.mean((gg-dg)**2))),stage2_active=0,stage1_min=np.nan,stage1_fallback_share=np.nan))
    # AKLP2
    k,cv=tune_adaptive(x,y,splits,degree=2); pt,dt=adaptive_local_poly_predict(x,y,xt,k,2); _,dg=adaptive_local_poly_predict(x,y,grid,k,2)
    rows.append(dict(method="AKLP2",sigma=sigma,rep=rep,seed=seed,param1=k,param2=np.nan,cv_mse=cv,
        outcome_rmse=float(np.sqrt(np.mean((yt-pt)**2))),deriv_rmse_test=float(np.sqrt(np.mean((gt-dt)**2))),
        deriv_rmse_grid=float(np.sqrt(np.mean((gg-dg)**2))),stage2_active=0,stage1_min=np.nan,stage1_fallback_share=np.nan))
    # SPLINE
    lam,cv=tune_spline(x,y,splits); pt,dt=smoothing_spline_predict(x,y,xt,lam); _,dg=smoothing_spline_predict(x,y,grid,lam)
    rows.append(dict(method="SPLINE",sigma=sigma,rep=rep,seed=seed,param1=lam,param2=np.nan,cv_mse=cv,
        outcome_rmse=float(np.sqrt(np.mean((yt-pt)**2))),deriv_rmse_test=float(np.sqrt(np.mean((gt-dt)**2))),
        deriv_rmse_grid=float(np.sqrt(np.mean((gg-dg)**2))),stage2_active=0,stage1_min=np.nan,stage1_fallback_share=np.nan))
    for row in rows: row["elapsed_rep_sec"]=time.time()-t0
    return pd.DataFrame(rows)

def batch_path(outdir:Path,sigma:float,start:int,stop:int):
    tag=f"{sigma:.2f}".replace(".","p")
    return outdir/"raw_batches"/f"sigma_{tag}"/f"reps_{start:04d}_{stop-1:04d}.csv"

def run_batch(outdir:str,sigma:float,start:int,stop:int,base_seed:int=BASE_SEED):
    out=Path(outdir); path=batch_path(out,float(sigma),start,stop)
    if path.exists(): return str(path)
    frames=[]
    for rep in range(start,stop): frames.append(run_rep(float(sigma),rep,base_seed))
    df=pd.concat(frames,ignore_index=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".csv.tmp"); df.to_csv(tmp,index=False); os.replace(tmp,path)
    return str(path)

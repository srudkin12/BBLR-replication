#!/usr/bin/env python3
from __future__ import annotations
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v]='1'
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from uvlib.univariate import DGP_REGISTRY as UV_DGPS, generate_sample as uv_generate_sample, differentiable_mask
from uvlib.strong_comparator_cv import _predict_config as uv_predict_config
from bivlib.bivariate import DGP_REGISTRY as BIV_DGPS, generate_sample as biv_generate_sample
from bivlib.bivariate_cv import _fit_predict as biv_fit_predict


def stable_seed(base:int,*parts:int)->int:
    ss=np.random.SeedSequence([int(base),*map(int,parts)])
    return int(ss.generate_state(1,dtype=np.uint32)[0])

def rmse(a,b,mask=None):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if mask is not None:
        mask=np.asarray(mask,bool); a=a[mask]; b=b[mask]
    if a.size==0: return np.nan
    return float(np.sqrt(np.mean((a-b)**2)))

def mean_bias(a,b,mask=None):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if mask is not None:
        mask=np.asarray(mask,bool); a=a[mask]; b=b[mask]
    if a.size==0:return np.nan
    return float(np.mean(a-b))

def uv_origin_support_class(design:str)->str:
    d=str(design)
    if d.startswith('G') and d!='G00': return 'unsupported'
    if d in ('D10','D02'): return 'thin_positive'
    if d in ('D25','D50','D75'): return 'positive_depleted'
    if d=='gap': return 'weak_positive_mixture'
    return 'supported'

def uv_grid_support_mask(x,design:str):
    x=np.asarray(x,float); d=str(design)
    if d.startswith('G') and d!='G00':
        try:g=int(d[1:])/100.0
        except:g=0.0
        return np.abs(x)>=g
    return np.ones(len(x),bool)

def biv_origin_support_class(design:str)->str:
    d=str(design)
    if '_gap' in d or d=='hole_gap55': return 'unsupported'
    if d.endswith('soft02'): return 'thin_positive'
    if d.endswith('soft10'): return 'positive_depleted'
    if d.startswith('two_clusters'): return 'weak_positive_mixture'
    return 'supported'

def biv_grid_support_mask(X,design:str):
    X=np.asarray(X,float); d=str(design)
    if d=='vertical_gap40': return np.abs(X[:,0])>=.40
    if d=='vertical_gap70': return np.abs(X[:,0])>=.70
    z=np.abs((X[:,0]+X[:,1])/np.sqrt(2.0))
    if d=='diag_gap40': return z>=.40
    if d=='diag_gap70': return z>=.70
    if d=='hole_gap55': return np.sqrt(np.sum(X*X,axis=1))>=.55
    return np.ones(len(X),bool)

def uv_row_eval(rows:pd.DataFrame):
    first=rows.iloc[0]
    cell=int(first.cell_id); rep=int(first.rep)
    xdev,ydev,_=uv_generate_sample(str(first.dgp),int(first.n_development),float(first.noise_sd),stable_seed(int(first.seed_offset),cell,rep,int(first.stream_dev)),str(first.design))
    xt,yt,_=uv_generate_sample(str(first.dgp),int(first.n_test),float(first.noise_sd),stable_seed(int(first.seed_offset),cell,rep,int(first.stream_test)),str(first.design))
    dgp=UV_DGPS[str(first.dgp)]
    true_m=np.asarray(dgp.mean(xt),float); true_g=np.asarray(dgp.derivative(xt),float)
    mask=differentiable_mask(np.asarray(xt,float),dgp,float(first.guard_width))
    true_alpha=true_m-np.asarray(xt,float)*true_g
    grid=np.linspace(dgp.support[0],dgp.support[1],int(first.grid_n))
    gm=np.asarray(dgp.mean(grid),float); gg=np.asarray(dgp.derivative(grid),float); ga=gm-grid*gg
    gmask=differentiable_mask(grid,dgp,float(first.guard_width)); support=uv_grid_support_mask(grid,str(first.design)); gs=gmask & support
    origin=np.array([0.0]); alpha0=float(np.asarray(dgp.mean(origin),float)[0]); support_class=uv_origin_support_class(str(first.design))
    out=[]
    for rr in rows.itertuples(index=False):
        method=str(rr.method); base_method='LLR' if method=='LLR_MATCHED_R' else method
        cfg={}
        for k in ('r','eps','k','lam','gap_factor'):
            if hasattr(rr,k):
                v=getattr(rr,k)
                if pd.notna(v): cfg[k]=v
        try:
            p,g,diag=uv_predict_config(base_method,cfg,xdev,ydev,xt,min_stage1=int(rr.min_stage1),min_stage2=int(rr.min_stage2),component_min_points=int(getattr(rr,'component_min_points',15) if pd.notna(getattr(rr,'component_min_points',15)) else 15))
            pg,gg_hat,_=uv_predict_config(base_method,cfg,xdev,ydev,grid,min_stage1=int(rr.min_stage1),min_stage2=int(rr.min_stage2),component_min_points=int(getattr(rr,'component_min_points',15) if pd.notna(getattr(rr,'component_min_points',15)) else 15))
            p0,g0,_=uv_predict_config(base_method,cfg,xdev,ydev,origin,min_stage1=int(rr.min_stage1),min_stage2=int(rr.min_stage2),component_min_points=int(getattr(rr,'component_min_points',15) if pd.notna(getattr(rr,'component_min_points',15)) else 15))
            ah=p-np.asarray(xt,float)*g; agh=pg-grid*gg_hat
            rec=rr._asdict(); rec.update({
                'status':'ok','test_y_rmse_replay':rmse(p,yt),'test_mean_rmse_replay':rmse(p,true_m),'test_slope_rmse_replay':rmse(g,true_g,mask),
                'grid_slope_rmse_replay':rmse(gg_hat,gg,gmask),'test_tangent_alpha_rmse':rmse(ah,true_alpha,mask),'test_tangent_alpha_bias':mean_bias(ah,true_alpha,mask),
                'grid_tangent_alpha_rmse':rmse(agh,ga,gmask),'grid_tangent_alpha_rmse_supported':rmse(agh,ga,gs),
                'alpha0_true':alpha0,'alpha0_hat':float(p0[0]),'alpha0_signed_error':float(p0[0]-alpha0),'alpha0_abs_error':float(abs(p0[0]-alpha0)),
                'alpha0_sq_error':float((p0[0]-alpha0)**2),'origin_support_class':support_class,'origin_nearest_distance_dev':float(np.min(np.abs(xdev))),
                'test_n':int(len(xt)),'grid_supported_share':float(np.mean(support))})
        except Exception as e:
            rec=rr._asdict(); rec.update({'status':'fail','error':repr(e)})
        out.append(rec)
    return out

def biv_row_eval(rows:pd.DataFrame):
    first=rows.iloc[0]; cell=int(first.cell_id); rep=int(first.rep)
    X,y,_=biv_generate_sample(str(first.dgp),int(first.n_development),float(first.noise_sd),stable_seed(int(first.seed_offset),cell,rep,int(first.stream_dev)),str(first.design))
    Xt,yt,_=biv_generate_sample(str(first.dgp),int(first.n_test),float(first.noise_sd),stable_seed(int(first.seed_offset),cell,rep,int(first.stream_test)),str(first.design))
    d=BIV_DGPS[str(first.dgp)]; tm=np.asarray(d.mean(Xt),float); tg=np.asarray(d.gradient(Xt),float); ta=tm-np.sum(Xt*tg,axis=1)
    axis=np.linspace(-2,2,int(first.grid_n)); xx,yy=np.meshgrid(axis,axis); grid=np.column_stack([xx.ravel(),yy.ravel()])
    gm=np.asarray(d.mean(grid),float); gg=np.asarray(d.gradient(grid),float); ga=gm-np.sum(grid*gg,axis=1); sm=biv_grid_support_mask(grid,str(first.design))
    origin=np.zeros((1,2)); alpha0=float(np.asarray(d.mean(origin),float)[0]); support_class=biv_origin_support_class(str(first.design))
    out=[]
    for rr in rows.itertuples(index=False):
        method=str(rr.method); base_method='LLR' if method=='LLR_MATCHED_R' else method
        cfg={}
        for k in ('r','eps','k','n_knots','alpha','gap_factor'):
            if hasattr(rr,k):
                v=getattr(rr,k)
                if pd.notna(v):cfg[k]=v
        if method=='LLR_MATCHED_R': cfg={'r':float(rr.r)}
        try:
            p,g,diag=biv_fit_predict(base_method,cfg,X,y,Xt,min_stage1=int(rr.min_stage1))
            pg,gh,_=biv_fit_predict(base_method,cfg,X,y,grid,min_stage1=int(rr.min_stage1))
            p0,g0,_=biv_fit_predict(base_method,cfg,X,y,origin,min_stage1=int(rr.min_stage1))
            ah=p-np.sum(Xt*g,axis=1); agh=pg-np.sum(grid*gh,axis=1)
            rec=rr._asdict(); rec.update({
                'status':'ok','test_y_rmse_replay':rmse(p,yt),'test_mean_rmse_replay':rmse(p,tm),'test_gradient_rmse_replay':rmse(g,tg),
                'test_slope1_rmse_replay':rmse(g[:,0],tg[:,0]),'test_slope2_rmse_replay':rmse(g[:,1],tg[:,1]),'grid_gradient_rmse_replay':rmse(gh,gg),
                'test_tangent_alpha_rmse':rmse(ah,ta),'test_tangent_alpha_bias':mean_bias(ah,ta),'grid_tangent_alpha_rmse':rmse(agh,ga),
                'grid_tangent_alpha_rmse_supported':rmse(agh,ga,sm),'alpha0_true':alpha0,'alpha0_hat':float(p0[0]),
                'alpha0_signed_error':float(p0[0]-alpha0),'alpha0_abs_error':float(abs(p0[0]-alpha0)),'alpha0_sq_error':float((p0[0]-alpha0)**2),
                'origin_support_class':support_class,'origin_nearest_distance_dev':float(np.min(np.sqrt(np.sum(X*X,axis=1)))),'test_n':int(len(Xt)),'grid_supported_share':float(np.mean(sm))})
        except Exception as e:
            rec=rr._asdict(); rec.update({'status':'fail','error':repr(e)})
        out.append(rec)
    return out

def make_chunks(df,track,chunk_reps):
    keycols=['cell_id','rep'] if track!='biv101' else ['suite','cell_id','rep']
    uniq=df[keycols].drop_duplicates().sort_values(keycols,kind='mergesort')
    groups=[]
    if track=='biv101':
        for (suite,cell),g in uniq.groupby(['suite','cell_id'],sort=False):
            reps=g.rep.to_numpy(int)
            for i in range(0,len(reps),chunk_reps): groups.append((str(suite),int(cell),reps[i:i+chunk_reps].tolist()))
    else:
        for cell,g in uniq.groupby('cell_id',sort=False):
            reps=g.rep.to_numpy(int)
            for i in range(0,len(reps),chunk_reps): groups.append((int(cell),reps[i:i+chunk_reps].tolist()))
    return groups

def run_chunk(df,track,spec,outdir,idx):
    out=Path(outdir); sd=out/'shards';sd.mkdir(parents=True,exist_ok=True); p=sd/f'{track}_chunk_{idx:06d}.csv.gz'
    if p.exists() and p.stat().st_size>100:return {'chunk':idx,'status':'skipped','path':str(p)}
    if track=='biv101': suite,cell,reps=spec; sub=df[(df.suite==suite)&(df.cell_id==cell)&(df.rep.isin(reps))]
    else: cell,reps=spec; sub=df[(df.cell_id==cell)&(df.rep.isin(reps))]
    keycols=['suite','cell_id','rep'] if track=='biv101' else ['cell_id','rep']
    records=[]
    for _,g in sub.groupby(keycols,sort=False): records.extend(biv_row_eval(g) if track=='biv101' else uv_row_eval(g))
    res=pd.DataFrame(records); tmp=Path(str(p)+f'.tmp.{os.getpid()}');res.to_csv(tmp,index=False,compression='gzip');os.replace(tmp,p)
    return {'chunk':idx,'status':'written','rows':len(res),'path':str(p)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--track',choices=['uv92','uv27','biv101'],required=True);ap.add_argument('--configs',type=Path);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--jobs',type=int,default=1);ap.add_argument('--chunk-reps',type=int,default=5);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(); default={'uv92':ROOT/'data/UV92_FROZEN_CONFIGS.csv.gz','uv27':ROOT/'data/UV27_FROZEN_CONFIGS.csv.gz','biv101':ROOT/'data/BIV101_FROZEN_CONFIGS.csv.gz'}[a.track]
    path=a.configs or default;df=pd.read_csv(path);out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    specs=make_chunks(df,a.track,int(a.chunk_reps));
    cfg={'track':a.track,'configs':str(path.resolve()),'jobs':a.jobs,'chunk_reps':a.chunk_reps,'expected_rows':int(len(df)),'expected_chunks':int(len(specs))}
    (out/f'RUN_CONFIG_{a.track}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    t=time.perf_counter()
    if a.jobs==1: ans=[run_chunk(df,a.track,s,out,i) for i,s in enumerate(specs)]
    else:
        with parallel_config(backend='loky',n_jobs=a.jobs,inner_max_num_threads=1):
            ans=Parallel(verbose=10,batch_size=1,pre_dispatch='2*n_jobs')(delayed(run_chunk)(df,a.track,s,out,i) for i,s in enumerate(specs))
    man=pd.DataFrame(ans).sort_values('chunk');man.to_csv(out/f'CHUNK_MANIFEST_{a.track}.csv',index=False)
    present=sum((out/'shards'/f'{a.track}_chunk_{i:06d}.csv.gz').exists() for i in range(len(specs)))
    audit={'track':a.track,'expected_chunks':len(specs),'present_chunks':present,'expected_rows':len(df),'wall_seconds':time.perf_counter()-t,'status':'PASS' if present==len(specs) else 'FAIL'}
    (out/f'COMPLETION_AUDIT_{a.track}.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()

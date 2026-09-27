#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, math, os, platform, sys, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
MIN_N=15
METHODS=['BBLR','LLR','LP2','PSpline','Hinge']
POINTS=np.array([15.,18.,20.,22.,24.,28.,30.])
SEED=170917

def parse(s): return ast.literal_eval(str(s))

def local_fit1(x,y,c,r):
    dist=np.abs(x-c); idx=np.flatnonzero(dist<=r)
    if len(idx)<MIN_N: idx=np.argsort(dist)[:min(MIN_N,len(x))]
    z=x[idx]-c; X=np.column_stack([np.ones(len(idx)),z])
    b=np.linalg.lstsq(X,y[idx],rcond=None)[0]
    return float(b[0]),float(b[1])

def local_fit2(x,y,c,r):
    dist=np.abs(x-c); idx=np.flatnonzero(dist<=r)
    if len(idx)<MIN_N: idx=np.argsort(dist)[:min(MIN_N,len(x))]
    z=x[idx]-c; X=np.column_stack([np.ones(len(idx)),z,z*z])
    b=np.linalg.lstsq(X,y[idx],rcond=None)[0]
    return float(b[0]),float(b[1]),float(b[2])

def fit_field(x,y,r):
    a=np.empty(len(x)); b=np.empty(len(x))
    for j,c in enumerate(x): a[j],b[j]=local_fit1(x,y,c,r)
    return a,b

def bblr_predict(x,y,q,r,e,field=None):
    q=np.asarray(q,float); pred=[]; der=[]
    if e==0:
        for z in q:
            a,b=local_fit1(x,y,z,r); pred.append(a); der.append(b)
        return np.asarray(pred),np.asarray(der)
    if field is None: field=fit_field(x,y,r)
    a,b=field
    for z in q:
        idx=np.flatnonzero(np.abs(x-z)<=e)
        if len(idx)==0: idx=np.array([int(np.argmin(np.abs(x-z)))])
        pred.append(float(np.mean(a[idx]+b[idx]*(z-x[idx])))); der.append(float(np.mean(b[idx])))
    return np.asarray(pred),np.asarray(der)

def lp2_predict(x,y,q,r):
    vals=[local_fit2(x,y,z,r) for z in q]
    return np.array([v[0] for v in vals]),np.array([v[1] for v in vals])

def pspline_fit(x,y,k,lam):
    tr=SplineTransformer(n_knots=int(k),degree=3,knots='quantile',include_bias=True,extrapolation='linear')
    B=tr.fit_transform(np.asarray(x)[:,None]); p=B.shape[1]; D=np.diff(np.eye(p),n=2,axis=0)
    beta=np.linalg.solve(B.T@B+float(lam)*(D.T@D)+1e-10*np.eye(p),B.T@np.asarray(y))
    return tr,beta

def pspline_predict(fit,q):
    tr,b=fit; q=np.asarray(q,float); pred=tr.transform(q[:,None])@b; h=.02
    der=(tr.transform((q+h)[:,None])@b-tr.transform((q-h)[:,None])@b)/(2*h)
    return pred,der

def hinge_predict(x,y,q,c,h):
    X=np.column_stack([np.ones(len(x)),np.maximum(c-x,0),np.maximum(x-h,0)])
    b=np.linalg.lstsq(X,y,rcond=None)[0]; q=np.asarray(q,float)
    Z=np.column_stack([np.ones(len(q)),np.maximum(c-q,0),np.maximum(q-h,0)])
    d=np.zeros(len(q)); d[q<c]=-b[1]; d[q>h]=b[2]
    return Z@b,d

def fit_predict(method,param,x,y,q):
    if method=='BBLR': return bblr_predict(x,y,q,float(param[0]),float(param[1]))
    if method=='LLR': return bblr_predict(x,y,q,float(param),0.)
    if method=='LP2': return lp2_predict(x,y,q,float(param))
    if method=='PSpline': return pspline_predict(pspline_fit(x,y,*param),q)
    if method=='Hinge': return hinge_predict(x,y,q,*param)
    raise KeyError(method)

def crossing(grid,g):
    m=(grid>=18)&(grid<=30); x=grid[m]; y=g[m]; cr=np.nan
    for i in range(len(x)-1):
        if y[i]<=0<y[i+1]:
            den=y[i+1]-y[i]; cr=float(x[i]-y[i]*(x[i+1]-x[i])/den) if den else float(x[i]); break
    return cr,int(np.sum(np.sign(g[1:])*np.sign(g[:-1])<0)),float(np.sum(np.abs(np.diff(g))))

def boot_one(payload):
    seed,groups,x,y,grid,params=payload
    rng=np.random.default_rng(seed); ug=np.array(sorted(pd.unique(groups))); draw=rng.choice(ug,size=len(ug),replace=True)
    idx=np.concatenate([np.flatnonzero(groups==g) for g in draw]); xb=x[idx]; yb=y[idx]
    out={}
    for m,p in params.items(): out[m]=fit_predict(m,p,xb,yb,grid)[1]
    return out

def run_outcome(name,params,B,workers):
    d=pd.read_csv(ROOT/f'source_data/{name}_workdays.csv'); x=d.Temperature.to_numpy(float); y=d.y.to_numpy(float); groups=d.week_group.to_numpy()
    lo,hi=np.quantile(x,[.05,.95]); grid=np.linspace(lo,hi,300)
    od=OUT/name; od.mkdir(parents=True,exist_ok=True)
    curves={'Temperature':grid}
    for m,p in params.items():
        pr,de=fit_predict(m,p,x,y,grid); curves[m]=pr; curves[m+'_derivative']=de
    pd.DataFrame(curves).to_csv(od/'grouped_primary_curves.csv',index=False)
    payload=[(SEED+(0 if name=='peak' else 100000)+j,groups,x,y,grid,params) for j in range(B)]
    res=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut=[ex.submit(boot_one,p) for p in payload]
        for j,f in enumerate(as_completed(fut),1):
            res.append(f.result())
            if j%100==0: print(f'{name}/bootstrap {j}/{B}')
    long=[]; feat=[]
    for b,z in enumerate(res):
        for m,g in z.items():
            vals=np.interp(POINTS,grid,g)
            for T,v in zip(POINTS,vals): long.append(dict(replicate=b,method=m,temperature=T,derivative=v))
            cr,nchg,tv=crossing(grid,g); feat.append(dict(replicate=b,method=m,crossing=cr,sign_changes=nchg,total_variation=tv))
    ld=pd.DataFrame(long); fd=pd.DataFrame(feat); ld.to_csv(od/'week_block_bootstrap_points.csv',index=False); fd.to_csv(od/'week_block_bootstrap_features.csv',index=False)
    sm=ld.groupby(['method','temperature']).derivative.agg(mean='mean',sd='std',q025=lambda s:np.quantile(s,.025),q975=lambda s:np.quantile(s,.975)).reset_index()
    sm['p_negative']=ld.groupby(['method','temperature']).derivative.apply(lambda s:np.mean(s<0)).values
    sm['p_positive']=ld.groupby(['method','temperature']).derivative.apply(lambda s:np.mean(s>0)).values
    sm.to_csv(od/'week_block_bootstrap_point_summary.csv',index=False)
    fs=fd.groupby('method').agg(crossing_median=('crossing','median'),crossing_q025=('crossing',lambda s:np.nanquantile(s,.025)),crossing_q975=('crossing',lambda s:np.nanquantile(s,.975)),crossing_available_share=('crossing',lambda s:np.mean(np.isfinite(s))),median_sign_changes=('sign_changes','median'),median_total_variation=('total_variation','median')).reset_index()
    fs.to_csv(od/'week_block_bootstrap_feature_summary.csv',index=False)
    # figures created only in parent process
    import matplotlib; matplotlib.use('Agg',force=True); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.2,4.8));
    for m in METHODS: ax.plot(grid,curves[m+'_derivative'],label=m,lw=1.5)
    ax.axhline(0,lw=.8); ax.set_xlabel('Maximum daily temperature (°C)'); ax.set_ylabel('Local demand derivative (GW per °C)'); ax.legend(frameon=False,ncol=2); fig.tight_layout(); fig.savefig(od/'grouped_primary_derivatives.pdf'); fig.savefig(od/'grouped_primary_derivatives.png',dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.8)); offs=np.linspace(-.28,.28,len(METHODS))
    for off,m in zip(offs,METHODS):
        z=sm[sm.method==m].sort_values('temperature'); xx=z.temperature.to_numpy()+off
        ax.errorbar(xx,z['mean'],yerr=[z['mean']-z.q025,z.q975-z['mean']],fmt='o',ms=3,capsize=2,label=m)
    ax.axhline(0,lw=.8); ax.set_xlabel('Maximum daily temperature (°C)'); ax.set_ylabel('Week-block bootstrap derivative (GW per °C)'); ax.legend(frameon=False,ncol=2); fig.tight_layout(); fig.savefig(od/'grouped_primary_bootstrap_points.pdf'); fig.savefig(od/'grouped_primary_bootstrap_points.png',dpi=220); plt.close(fig)

def make_handback():
    hb=ROOT/'BBLR_Electricity_GroupedPrimary_v1_0_HANDback.zip'
    with zipfile.ZipFile(hb,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(OUT))
        for p in [ROOT/'DESIGN_NOTE.md',ROOT/'frozen_grouped_selection.csv']:
            z.write(p,p.name)
    h=hashlib.sha256(hb.read_bytes()).hexdigest(); (ROOT/(hb.name+'.sha256')).write_text(f'{h}  {hb.name}\n')
    print(hb); print(h)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=80); ap.add_argument('--bootstrap',type=int,default=1000); a=ap.parse_args()
    sel=pd.read_csv(ROOT/'frozen_grouped_selection.csv'); allp={}
    for outcome in ['peak','evening1800']:
        z=sel[sel.outcome==outcome]; params={r.method:parse(r.param) for _,r in z.iterrows()}; allp[outcome]=params
        run_outcome(outcome,params,a.bootstrap,min(a.workers,os.cpu_count() or 1))
    (OUT/'environment.txt').write_text(f'python={sys.version}\nplatform={platform.platform()}\nworkers={a.workers}\nbootstrap={a.bootstrap}\n')
    make_handback()
if __name__=='__main__': main()

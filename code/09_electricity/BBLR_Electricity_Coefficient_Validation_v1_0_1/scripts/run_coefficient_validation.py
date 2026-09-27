#!/usr/bin/env python3
from __future__ import annotations

import argparse, ast, hashlib, json, math, os, platform, shutil, sys, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Numerical workers must never initialise a GUI backend.  In particular,
# importing pyplot before ProcessPoolExecutor forks can inherit Tcl/Tk state
# into workers and cause Tcl_AsyncDelete/main-thread crashes on Linux.
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'source_data'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SEED = 270914
MIN_N = 15
BANDS = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
METHODS = ['BBLR', 'LLR', 'LP2', 'PSpline', 'Hinge']
POINTS = np.array([15.,18.,20.,22.,24.,28.,30.])
OUTCOMES = {
    'peak': {'column':'PeakDemand','label':'Daily peak demand'},
    'evening1800': {'column':'Demand1800','label':'18:00 demand'},
}


def import_preflight():
    sys.path.insert(0, str(ROOT/'scripts'))
    from preflight import load_and_construct
    return load_and_construct


def parse_param(s):
    return ast.literal_eval(str(s))


def load_frozen():
    p = pd.read_csv(ROOT/'frozen_selection_provenance.csv')
    out = {}
    for _,r in p.iterrows():
        out[(r.outcome, r.scheme, r.method)] = parse_param(r.param)
    return out, p


def local_fit1(x,y,center,r,min_n=MIN_N):
    dist=np.abs(x-center); idx=np.flatnonzero(dist<=r)
    safeguard=len(idx)<min_n
    if safeguard: idx=np.argsort(dist)[:min(min_n,len(x))]
    z=x[idx]-center; yy=y[idx]
    X=np.column_stack([np.ones(len(idx)),z])
    beta=np.linalg.lstsq(X,yy,rcond=None)[0]
    return float(beta[0]),float(beta[1]),safeguard


def local_fit2(x,y,center,r,min_n=MIN_N):
    dist=np.abs(x-center); idx=np.flatnonzero(dist<=r)
    if len(idx)<min_n: idx=np.argsort(dist)[:min(min_n,len(x))]
    z=x[idx]-center; yy=y[idx]
    X=np.column_stack([np.ones(len(idx)),z,z*z])
    beta=np.linalg.lstsq(X,yy,rcond=None)[0]
    return float(beta[0]),float(beta[1]),float(beta[2])


def fit_field(x,y,r):
    a=np.empty(len(x)); b=np.empty(len(x))
    for j,c in enumerate(x): a[j],b[j],_=local_fit1(x,y,c,r)
    return a,b


def bblr_derivative(xtr,ytr,xq,r,eps,field=None):
    q=np.atleast_1d(xq).astype(float)
    if eps==0:
        return np.array([local_fit1(xtr,ytr,z,r)[1] for z in q])
    if field is None: field=fit_field(xtr,ytr,r)
    _,b=field; out=[]
    for z in q:
        idx=np.flatnonzero(np.abs(xtr-z)<=eps)
        if len(idx)==0: idx=np.array([int(np.argmin(np.abs(xtr-z)))])
        out.append(float(np.mean(b[idx])))
    return np.asarray(out)


def lp2_derivative(xtr,ytr,xq,r):
    return np.array([local_fit2(xtr,ytr,z,r)[1] for z in np.atleast_1d(xq).astype(float)])


def pspline_fit(xtr,ytr,k,lam):
    tr=SplineTransformer(n_knots=int(k),degree=3,knots='quantile',include_bias=True,extrapolation='linear')
    B=tr.fit_transform(np.asarray(xtr)[:,None]); p=B.shape[1]
    D=np.diff(np.eye(p),n=2,axis=0)
    beta=np.linalg.solve(B.T@B+float(lam)*(D.T@D)+1e-10*np.eye(p),B.T@np.asarray(ytr))
    return tr,beta


def pspline_derivative(fit,xq):
    tr,beta=fit; q=np.atleast_1d(xq).astype(float); h=0.02
    return (tr.transform((q+h)[:,None])@beta-tr.transform((q-h)[:,None])@beta)/(2*h)


def hinge_fit(xtr,ytr,c,h):
    X=hinge_design(xtr,c,h); return np.linalg.lstsq(X,ytr,rcond=None)[0]


def hinge_design(x,c,h):
    x=np.asarray(x,float)
    return np.column_stack([np.ones(len(x)),np.maximum(c-x,0.0),np.maximum(x-h,0.0)])


def hinge_derivative(beta,xq,c,h):
    q=np.atleast_1d(xq).astype(float); d=np.zeros(len(q))
    d[q<c]=-beta[1]; d[q>h]=beta[2]
    return d


def fit_derivative_object(method,param,xtr,ytr):
    if method=='BBLR':
        r,e=param; return ('BBLR',float(r),float(e),fit_field(xtr,ytr,float(r)) if e>0 else None)
    if method=='LLR': return ('LLR',float(param),0.0,None)
    if method=='LP2': return ('LP2',float(param))
    if method=='PSpline': return ('PSpline',*param,pspline_fit(xtr,ytr,*param))
    if method=='Hinge': return ('Hinge',*param,hinge_fit(xtr,ytr,*param))
    if method=='BBLR_MatchedZero':
        r=float(param[0] if isinstance(param,tuple) else param); return ('BBLR_MatchedZero',r,0.0,None)
    raise KeyError(method)


def derivative(obj,xtr,ytr,xq):
    typ=obj[0]
    if typ in ('BBLR','LLR','BBLR_MatchedZero'):
        _,r,e,field=obj; return bblr_derivative(xtr,ytr,xq,r,e,field)
    if typ=='LP2': return lp2_derivative(xtr,ytr,xq,obj[1])
    if typ=='PSpline': return pspline_derivative(obj[-1],xq)
    if typ=='Hinge': return hinge_derivative(obj[-1],xq,obj[1],obj[2])
    raise KeyError(typ)


def random_splits(n,n_repeats,seed):
    rkf=RepeatedKFold(n_splits=5,n_repeats=n_repeats,random_state=seed)
    return [(j//5,j%5,tr,va) for j,(tr,va) in enumerate(rkf.split(np.arange(n)))]


def grouped_splits(groups,n_repeats,seed):
    groups=np.asarray(groups); ug=np.array(sorted(pd.unique(groups))); out=[]
    for rep in range(n_repeats):
        rng=np.random.default_rng(seed+rep); g=ug.copy(); rng.shuffle(g)
        chunks=np.array_split(g,5)
        for fold,vg in enumerate(chunks):
            va=np.flatnonzero(np.isin(groups,vg)); tr=np.flatnonzero(~np.isin(groups,vg))
            out.append((rep,fold,tr,va))
    return out


def eligible_pairs(xv,yv,lo,hi,band):
    blo,bhi=band; rows=[]; n=len(xv)
    for i in range(n-1):
        for j in range(i+1,n):
            if xv[i]==xv[j]: continue
            if xv[i] < xv[j]: a,b=i,j
            else: a,b=j,i
            x1,x2=float(xv[a]),float(xv[b]); dx=x2-x1
            if dx < blo or dx > bhi: continue
            if x1 < lo or x2 > hi: continue
            dy=float(yv[b]-yv[a]); mid=.5*(x1+x2)
            rows.append((x1,x2,mid,dx,dy))
    return rows


def equal_location_mean(loss,mid,lo,hi,nbins=10):
    edges=np.linspace(lo,hi,nbins+1); b=np.digitize(mid,edges[1:-1],right=False)
    vals=[]
    for k in range(nbins):
        z=loss[b==k]
        if len(z): vals.append(float(np.mean(z)))
    return float(np.mean(vals)) if vals else np.nan


def fold_audit(payload):
    outcome,scheme,rep,fold,tr,va,x,y,params,lo,hi=payload
    xtr,ytr=x[tr],y[tr]; xv,yv=x[va],y[va]
    objects={}
    for m in METHODS:
        objects[m]=fit_derivative_object(m,params[m],xtr,ytr)
    if params['BBLR'][1]>0:
        objects['BBLR_MatchedZero']=fit_derivative_object('BBLR_MatchedZero',params['BBLR'],xtr,ytr)
    methods=list(objects)
    rows=[]
    for band in BANDS:
        pairs=eligible_pairs(xv,yv,lo,hi,band)
        if not pairs: continue
        arr=np.asarray(pairs,float); x1,x2,mid,dx,dy=arr.T
        sec=dy/dx
        q=np.unique(np.concatenate([x1,mid,x2]))
        for m in methods:
            gd=derivative(objects[m],xtr,ytr,q)
            lookup={float(k):float(v) for k,v in zip(q,gd)}
            g1=np.array([lookup[float(z)] for z in x1]); gm=np.array([lookup[float(z)] for z in mid]); g2=np.array([lookup[float(z)] for z in x2])
            gsim=(g1+4*gm+g2)/6.0
            mid_loss=(sec-gm)**2
            sim_loss=(sec-gsim)**2
            contrast_mid=(dy-gm*dx)**2
            contrast_sim=(dy-gsim*dx)**2
            rows.append({
                'outcome':outcome,'scheme':scheme,'repeat':rep,'fold':fold,'method':m,
                'band':f'{band[0]:g}-{band[1]:g}','n_pairs':len(pairs),
                'midpoint_secant_mse_design':float(np.mean(mid_loss)),
                'simpson_secant_mse_design':float(np.mean(sim_loss)),
                'midpoint_contrast_mse_design':float(np.mean(contrast_mid)),
                'simpson_contrast_mse_design':float(np.mean(contrast_sim)),
                'midpoint_secant_mse_equal_location':equal_location_mean(mid_loss,mid,lo,hi),
                'simpson_secant_mse_equal_location':equal_location_mean(sim_loss,mid,lo,hi),
                'midpoint_contrast_mse_equal_location':equal_location_mean(contrast_mid,mid,lo,hi),
                'simpson_contrast_mse_equal_location':equal_location_mean(contrast_sim,mid,lo,hi),
            })
    return rows


def run_crossfit(outdir,outcome,scheme,splits,x,y,params,lo,hi,workers):
    payload=[(outcome,scheme,rep,fold,tr,va,x,y,params,lo,hi) for rep,fold,tr,va in splits]
    rows=[]
    if workers<=1:
        for j,p in enumerate(payload,1):
            rows.extend(fold_audit(p))
            if j%100==0: print(f'{outcome}/{scheme}: {j}/{len(payload)} folds')
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs=[ex.submit(fold_audit,p) for p in payload]
            for j,f in enumerate(as_completed(futs),1):
                rows.extend(f.result())
                if j%100==0: print(f'{outcome}/{scheme}: {j}/{len(futs)} folds')
    df=pd.DataFrame(rows)
    df.to_csv(outdir/f'coefficient_fold_losses_{scheme}.csv',index=False)
    # Fold-weighted within repeat: each fold contributes equally, preventing pair-rich folds from dominating repeat summaries.
    metrics=[c for c in df.columns if c.endswith('_design') or c.endswith('_equal_location')]
    rep=df.groupby(['outcome','scheme','repeat','method','band'],as_index=False)[metrics+['n_pairs']].mean()
    rep.to_csv(outdir/f'coefficient_repeat_losses_{scheme}.csv',index=False)
    summ=rep.groupby(['outcome','scheme','method','band'],as_index=False).agg(**{m:(m,'mean') for m in metrics},mean_pairs=('n_pairs','mean'))
    summ.to_csv(outdir/f'coefficient_loss_summary_{scheme}.csv',index=False)
    return df,rep,summ


def paired_vs_bblr(rep):
    metrics=['midpoint_secant_mse_design','simpson_secant_mse_design','midpoint_secant_mse_equal_location','simpson_secant_mse_equal_location']
    rows=[]
    for (outcome,scheme,band),g in rep.groupby(['outcome','scheme','band']):
        base=g[g.method=='BBLR'].set_index('repeat')
        for m in sorted(g.method.unique()):
            if m=='BBLR': continue
            q=g[g.method==m].set_index('repeat').join(base[metrics],rsuffix='_BBLR',how='inner')
            for met in metrics:
                d=q[met]-q[met+'_BBLR']
                rows.append({'outcome':outcome,'scheme':scheme,'band':band,'method':m,'metric':met,
                             'mean_method_minus_bblr':float(d.mean()),'resampling_se':float(d.std(ddof=1)/math.sqrt(len(d))),
                             'method_win_share':float(np.mean(d<0)),'n_repeats':int(len(d))})
    return pd.DataFrame(rows)


def bootstrap_one(payload):
    seed,groups,x,y,grid,params=payload
    rng=np.random.default_rng(seed); ug=np.array(sorted(pd.unique(groups))); draw=rng.choice(ug,size=len(ug),replace=True)
    idx=np.concatenate([np.flatnonzero(groups==g) for g in draw]); xb=x[idx]; yb=y[idx]
    row={}
    for m in METHODS:
        obj=fit_derivative_object(m,params[m],xb,yb); gg=derivative(obj,xb,yb,grid)
        row[m]=gg
    return row


def crossing_features(grid,g):
    # first negative-to-positive crossing in [18,30], linear interpolation
    mask=(grid>=18)&(grid<=30); x=grid[mask]; y=g[mask]
    cross=np.nan
    for i in range(len(x)-1):
        if y[i] <= 0 and y[i+1] > 0:
            den=y[i+1]-y[i]
            cross=float(x[i]-y[i]*(x[i+1]-x[i])/den) if den!=0 else float(x[i])
            break
    signs=np.sign(g); changes=int(np.sum(signs[1:]*signs[:-1] < 0))
    tv=float(np.sum(np.abs(np.diff(g))))
    return cross,changes,tv


def run_bootstrap(outdir,outcome,groups,x,y,params,lo,hi,B,workers):
    grid=np.linspace(lo,hi,300)
    payload=[(SEED+700000+j,groups,x,y,grid,params) for j in range(B)]
    res=[]
    if workers<=1:
        for j,p in enumerate(payload,1):
            res.append(bootstrap_one(p))
            if j%100==0: print(f'{outcome}/bootstrap: {j}/{B}')
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs=[ex.submit(bootstrap_one,p) for p in payload]
            for j,f in enumerate(as_completed(futs),1):
                res.append(f.result())
                if j%100==0: print(f'{outcome}/bootstrap: {j}/{B}')
    long=[]; feat=[]
    for b,z in enumerate(res):
        for m,g in z.items():
            gp=np.interp(POINTS,grid,g)
            for xx,v in zip(POINTS,gp): long.append({'replicate':b,'method':m,'temperature':xx,'derivative':v})
            cross,nchg,tv=crossing_features(grid,g)
            feat.append({'replicate':b,'method':m,'crossing_18_30':cross,'sign_changes_main_support':nchg,'total_variation':tv})
    ld=pd.DataFrame(long); fd=pd.DataFrame(feat)
    ld.to_csv(outdir/'coefficient_week_block_bootstrap_points.csv',index=False)
    fd.to_csv(outdir/'coefficient_week_block_bootstrap_features.csv',index=False)
    sm=ld.groupby(['method','temperature']).derivative.agg(['mean','std',lambda s:np.quantile(s,.025),lambda s:np.quantile(s,.975)]).reset_index()
    sm.columns=['method','temperature','mean','sd','q025','q975']
    sm['p_negative']=ld.groupby(['method','temperature']).derivative.apply(lambda s:np.mean(s<0)).values
    sm['p_positive']=ld.groupby(['method','temperature']).derivative.apply(lambda s:np.mean(s>0)).values
    sm.to_csv(outdir/'coefficient_bootstrap_point_summary.csv',index=False)
    fs=fd.groupby('method').agg(crossing_median=('crossing_18_30','median'),crossing_q025=('crossing_18_30',lambda s:np.nanquantile(s,.025)),crossing_q975=('crossing_18_30',lambda s:np.nanquantile(s,.975)),crossing_available_share=('crossing_18_30',lambda s:np.mean(np.isfinite(s))),median_sign_changes=('sign_changes_main_support','median'),median_total_variation=('total_variation','median')).reset_index()
    fs.to_csv(outdir/'coefficient_bootstrap_feature_summary.csv',index=False)
    return sm,fs


def agreement(outdir,x,y,params,lo,hi):
    grid=np.linspace(lo,hi,300); curves={}
    for m in METHODS:
        obj=fit_derivative_object(m,params[m],x,y); curves[m]=derivative(obj,x,y,grid)
    rows=[]
    for i,m1 in enumerate(METHODS):
        for m2 in METHODS[i+1:]:
            d=curves[m1]-curves[m2]
            rows.append({'method1':m1,'method2':m2,'rms_derivative_distance':float(np.sqrt(np.mean(d*d))),
                         'mean_abs_derivative_distance':float(np.mean(np.abs(d))),
                         'sign_agreement_share':float(np.mean(np.sign(curves[m1])==np.sign(curves[m2])))})
    pd.DataFrame(rows).to_csv(outdir/'full_sample_derivative_agreement.csv',index=False)
    pd.DataFrame({'Temperature':grid,**{m:curves[m] for m in METHODS}}).to_csv(outdir/'full_sample_derivative_curves.csv',index=False)
    return grid,curves


def make_fig(outdir,grid,curves,summary,outcome_label):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.2,4.8)); ax.axhline(0,lw=.8)
    for m in METHODS: ax.plot(grid,curves[m],lw=1.5,label=m)
    ax.set_xlabel('Maximum daily temperature (°C)'); ax.set_ylabel('Local demand derivative (GW per °C)'); ax.legend(frameon=False,ncol=2)
    fig.tight_layout(); fig.savefig(outdir/'fig_all_method_derivatives.pdf'); fig.savefig(outdir/'fig_all_method_derivatives.png',dpi=220); plt.close(fig)
    # Bootstrap point uncertainty for selected points
    fig,ax=plt.subplots(figsize=(7.2,4.8));
    offs=np.linspace(-.28,.28,len(METHODS))
    for off,m in zip(offs,METHODS):
        z=summary[summary.method==m].sort_values('temperature'); xx=z.temperature.to_numpy()+off
        ax.errorbar(xx,z['mean'],yerr=[z['mean']-z.q025,z.q975-z['mean']],fmt='o',ms=3,capsize=2,label=m)
    ax.axhline(0,lw=.8); ax.set_xlabel('Maximum daily temperature (°C)'); ax.set_ylabel('Week-block bootstrap derivative (GW per °C)'); ax.legend(frameon=False,ncol=2)
    fig.tight_layout(); fig.savefig(outdir/'fig_coefficient_bootstrap_points.pdf'); fig.savefig(outdir/'fig_coefficient_bootstrap_points.png',dpi=220); plt.close(fig)


def write_summary(outdir,outcome,scheme_summaries,paired,bootpts,bootfeat,agree):
    lines=[f"# Coefficient-validation audit: {OUTCOMES[outcome]['label']}",'',
           'All configurations are frozen from the 1,000-repetition outcome-CV production audit. No method is retuned on coefficient-validation loss.','']
    for scheme,s in scheme_summaries.items():
        lines += [f'## {scheme.replace("_"," ").title()} held-out contrasts','']
        for band in [f'{a:g}-{b:g}' for a,b in BANDS]:
            z=s[s.band==band].sort_values('simpson_secant_mse_design')
            lines.append(f'### Temperature separation {band} °C')
            for _,r in z.iterrows(): lines.append(f"- {r.method}: Simpson secant MSE {r.simpson_secant_mse_design:.6g}; midpoint secant MSE {r.midpoint_secant_mse_design:.6g}; equal-location Simpson MSE {r.simpson_secant_mse_equal_location:.6g}.")
            lines.append('')
    lines += ['## Week-block coefficient stability','']
    for m in METHODS:
        f=bootfeat[bootfeat.method==m].iloc[0]
        p18=bootpts[(bootpts.method==m)&(bootpts.temperature==18)].iloc[0]
        p28=bootpts[(bootpts.method==m)&(bootpts.temperature==28)].iloc[0]
        lines.append(f"- {m}: median heating-to-cooling crossing {f.crossing_median:.3g} °C (2.5--97.5% {f.crossing_q025:.3g}--{f.crossing_q975:.3g}; available in {f.crossing_available_share:.3f} of resamples); P[g(18)<0]={p18.p_negative:.3f}; P[g(28)>0]={p28.p_positive:.3f}; median total variation {f.median_total_variation:.4g}.")
    lines += ['','## Interpretation','',
              'Held-out secant losses are noisy empirical implications of the coefficient field, not observations of derivative truth. The Monte Carlo sections use direct derivative MSE because truth is known there; this audit exists only because real-data derivative truth is latent. Stability, smoothness, and agreement are reported separately from held-out contrast accuracy.']
    (outdir/'coefficient_validation_summary.md').write_text('\n'.join(lines)+'\n')


def make_handback():
    hb=ROOT/'BBLR_Electricity_Coefficient_Validation_v1_0_1_HANDback.zip'
    with zipfile.ZipFile(hb,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file(): z.write(p,arcname=str(p.relative_to(OUT)))
        for p in [ROOT/'DESIGN_NOTE.md',ROOT/'MANUSCRIPT_RATIONALE.md',ROOT/'INTERPRETATION_RULES.md',ROOT/'frozen_selection_provenance.csv']:
            z.write(p,arcname=p.name)
        for p in sorted(SRC.glob('construction_audit.csv')): z.write(p,arcname='source_data/'+p.name)
    h=hashlib.sha256(hb.read_bytes()).hexdigest(); (ROOT/(hb.name+'.sha256')).write_text(f'{h}  {hb.name}\n')
    print('Handback:',hb); print('SHA256:',h)


def validate_reuse_dir(srcdir: Path, outcome: str):
    required = [
        'coefficient_loss_summary_random.csv',
        'coefficient_loss_summary_grouped_week.csv',
        'paired_vs_bblr_random.csv',
        'paired_vs_bblr_grouped_week.csv',
        'coefficient_bootstrap_point_summary.csv',
        'coefficient_bootstrap_feature_summary.csv',
        'full_sample_derivative_agreement.csv',
        'full_sample_derivative_curves.csv',
        'coefficient_validation_summary.md',
    ]
    missing=[f for f in required if not (srcdir/f).is_file()]
    if missing:
        raise RuntimeError(f'Reuse directory for {outcome} is incomplete; missing: {missing}')


def copy_reused_outcome(srcdir: Path, outcome: str):
    validate_reuse_dir(srcdir, outcome)
    dest=OUT/outcome
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(srcdir, dest)
    print(f'Reused completed {outcome} outputs from {srcdir}')


def rebuild_combined_outputs():
    overview=[]; paired_frames=[]
    for outcome in ['peak','evening1800']:
        od=OUT/outcome
        if not od.is_dir(): continue
        for scheme in ['random','grouped_week']:
            sp=od/f'coefficient_loss_summary_{scheme}.csv'
            pp=od/f'paired_vs_bblr_{scheme}.csv'
            if sp.is_file():
                s=pd.read_csv(sp)
                for band in s.band.unique():
                    z=s[s.band==band].sort_values('simpson_secant_mse_design').iloc[0]
                    overview.append({'outcome':outcome,'scheme':scheme,'band':band,
                                     'best_method_simpson_design':z.method,
                                     'best_simpson_secant_mse_design':z.simpson_secant_mse_design})
            if pp.is_file(): paired_frames.append(pd.read_csv(pp))
    pd.DataFrame(overview).to_csv(OUT/'combined_coefficient_validation_overview.csv',index=False)
    if paired_frames:
        pd.concat(paired_frames,ignore_index=True).to_csv(OUT/'combined_paired_vs_bblr.csv',index=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--random-repeats',type=int,default=1000)
    ap.add_argument('--group-repeats',type=int,default=1000)
    ap.add_argument('--bootstrap',type=int,default=1000)
    ap.add_argument('--outcomes',nargs='+',choices=['peak','evening1800'],default=['peak','evening1800'],
                    help='Outcomes to recompute. Useful after an interrupted run.')
    ap.add_argument('--reuse-peak-dir',type=str,default=None,
                    help='Optional completed peak output directory to copy into this clean package before recomputing another outcome.')
    a=ap.parse_args(); workers=max(1,a.workers)
    if a.reuse_peak_dir:
        copy_reused_outcome(Path(a.reuse_peak_dir).expanduser().resolve(),'peak')
    load_and_construct=import_preflight(); _,daily,_,_=load_and_construct(write=True)
    frozen,prov=load_frozen()
    for oi,outcome in enumerate(a.outcomes):
        od=OUT/outcome
        if od.exists(): shutil.rmtree(od)
        od.mkdir(parents=True,exist_ok=True)
        w=daily[daily.WorkDay==1].copy().reset_index(drop=True)
        y=w[OUTCOMES[outcome]['column']].to_numpy(float); x=w.Temperature.to_numpy(float); groups=w.week_group.to_numpy()
        lo,hi=np.quantile(x,[.05,.95])
        scheme_summaries={}
        for si,scheme in enumerate(['random','grouped_week']):
            params={m:frozen[(outcome,scheme,m)] for m in METHODS}
            splits=random_splits(len(w),a.random_repeats,SEED+oi*100000+si*10000) if scheme=='random' else grouped_splits(groups,a.group_repeats,SEED+500000+oi*100000)
            _,rep,summ=run_crossfit(od,outcome,scheme,splits,x,y,params,lo,hi,workers)
            paired=paired_vs_bblr(rep); paired.to_csv(od/f'paired_vs_bblr_{scheme}.csv',index=False)
            scheme_summaries[scheme]=summ
        rparams={m:frozen[(outcome,'random',m)] for m in METHODS}
        grid,curves=agreement(od,x,y,rparams,lo,hi)
        bootpts,bootfeat=run_bootstrap(od,outcome,groups,x,y,rparams,lo,hi,a.bootstrap,workers)
        make_fig(od,grid,curves,bootpts,OUTCOMES[outcome]['label'])
        agree=pd.read_csv(od/'full_sample_derivative_agreement.csv')
        paired_local=[]
        for scheme in ['random','grouped_week']:
            pp=od/f'paired_vs_bblr_{scheme}.csv'
            if pp.is_file(): paired_local.append(pd.read_csv(pp))
        write_summary(od,outcome,scheme_summaries,pd.concat(paired_local,ignore_index=True) if paired_local else pd.DataFrame(),bootpts,bootfeat,agree)
    rebuild_combined_outputs()
    env={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'workers':workers,'random_repeats':a.random_repeats,'group_repeats':a.group_repeats,'bootstrap':a.bootstrap,'seed':SEED,'outcomes_recomputed':a.outcomes,'mplbackend':os.environ.get('MPLBACKEND')}
    (OUT/'environment.json').write_text(json.dumps(env,indent=2))
    make_handback()

if __name__=='__main__': main()

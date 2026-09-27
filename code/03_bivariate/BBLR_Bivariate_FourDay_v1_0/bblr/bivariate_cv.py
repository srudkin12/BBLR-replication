from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from .bivariate_models import BBLR2D,LocalPolynomial2D,TensorSpline2D,ComponentwiseTensorSpline2D,GlobalPoly2D
from .bivariate import DGP_REGISTRY

def _rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def _grad_rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

def fold_iterator(n,folds,repeats,seed):
    ss=np.random.SeedSequence(int(seed)); children=ss.spawn(int(repeats))
    for rr,ch in enumerate(children):
        rs=int(ch.generate_state(1)[0]); kf=KFold(n_splits=int(folds),shuffle=True,random_state=rs)
        for tr,va in kf.split(np.arange(n)): yield rr,tr,va

def _configs(method,r_grid,eps_grid,k_grid,knot_grid,alpha_grid,gap_grid):
    if method=='BBLR': return [{'r':r,'eps':e} for r in r_grid for e in eps_grid]
    if method in ('LLR','LP2'): return [{'r':r} for r in r_grid]
    if method in ('AKLLR','AKLP2'): return [{'k':k} for k in k_grid]
    if method=='TSPLINE': return [{'n_knots':k,'alpha':a} for k in knot_grid for a in alpha_grid]
    if method=='CTSPLINE': return [{'n_knots':k,'alpha':a,'gap_factor':g} for k in knot_grid[:2] for a in alpha_grid for g in gap_grid]
    if method in ('OLS','GQ2'): return [{}]
    raise ValueError(method)

def _fit_model(method,cfg,X,y,min_stage1=10):
    if method=='BBLR': return BBLR2D(cfg['r'],cfg['eps'],min_stage1=min_stage1).fit(X,y)
    if method=='LLR': return LocalPolynomial2D(1,r=cfg['r'],min_points=min_stage1).fit(X,y)
    if method=='LP2': return LocalPolynomial2D(2,r=cfg['r'],min_points=max(15,min_stage1)).fit(X,y)
    if method=='AKLLR': return LocalPolynomial2D(1,k=cfg['k'],min_points=min_stage1).fit(X,y)
    if method=='AKLP2': return LocalPolynomial2D(2,k=cfg['k'],min_points=max(15,min_stage1)).fit(X,y)
    if method=='TSPLINE': return TensorSpline2D(cfg['n_knots'],cfg['alpha']).fit(X,y)
    if method=='CTSPLINE': return ComponentwiseTensorSpline2D(cfg['n_knots'],cfg['alpha'],cfg['gap_factor']).fit(X,y)
    if method=='OLS': return GlobalPoly2D(1).fit(X,y)
    if method=='GQ2': return GlobalPoly2D(2).fit(X,y)
    raise ValueError(method)

def _fit_predict(method,cfg,X,y,Q,min_stage1=10):
    return _fit_model(method,cfg,X,y,min_stage1).predict_with_gradient(Q)

def _fit_predict_y(method,cfg,X,y,Q,min_stage1=10):
    m=_fit_model(method,cfg,X,y,min_stage1)
    if method in ('TSPLINE','CTSPLINE'):
        return np.asarray(m.predict(Q),float)
    return np.asarray(m.predict_with_gradient(Q)[0],float)

def repeated_cv_risk_surface(X,y,methods,r_grid,eps_grid,k_grid,knot_grid,alpha_grid,gap_grid,folds,repeats,seed,min_stage1=10):
    X=np.asarray(X,float); y=np.asarray(y,float); methods=[m.upper() for m in methods]
    configs={m:_configs(m,r_grid,eps_grid,k_grid,knot_grid,alpha_grid,gap_grid) for m in methods}
    sse={(m,i):0. for m in methods for i in range(len(configs[m]))}; nobs=0
    for _,tr,va in fold_iterator(len(X),folds,repeats,seed):
        Xt,yt=X[tr],y[tr]; Xv,yv=X[va],y[va]; nobs+=len(va)
        # BBLR grouped by r to reuse stage-1 fits across eps.
        if 'BBLR' in methods:
            for r in r_grid:
                base=BBLR2D(r,1.0,min_stage1=min_stage1).fit(Xt,yt)
                for i,cfg in enumerate(configs['BBLR']):
                    if cfg['r']!=r: continue
                    if cfg['eps']==0:
                        p,_,_=LocalPolynomial2D(1,r=r,min_points=min_stage1).fit(Xt,yt).predict_with_gradient(Xv)
                    else:
                        base.eps=float(cfg['eps']); p,_,_=base.predict_with_gradient(Xv)
                    sse[('BBLR',i)]+=float(np.sum((p-yv)**2))
        for m in methods:
            if m=='BBLR': continue
            for i,cfg in enumerate(configs[m]):
                try:
                    p=_fit_predict_y(m,cfg,Xt,yt,Xv,min_stage1=min_stage1)
                    sse[(m,i)]+=float(np.sum((p-yv)**2)) if np.all(np.isfinite(p)) else np.inf
                except Exception:
                    sse[(m,i)]=np.inf
    rows=[]
    for m in methods:
        for i,cfg in enumerate(configs[m]):
            risk=sse[(m,i)]; rows.append({'method':m,**cfg,'repeated_cv_rmse':float(np.sqrt(risk/nobs)) if np.isfinite(risk) else np.inf})
    return pd.DataFrame(rows)

def select_minimum_risk(surface):
    out=[]
    for m,g in surface.groupby('method',sort=False):
        z=g.copy()
        for c in ['eps','r','k','n_knots','alpha','gap_factor']:
            if c not in z: z[c]=np.nan
        z['_gap']=z.gap_factor.replace(np.inf,1e300)
        z=z.sort_values(['repeated_cv_rmse','eps','r','k','n_knots','alpha','_gap'],na_position='last',kind='mergesort')
        out.append(z.iloc[0].drop(labels=['_gap']).to_dict())
    return pd.DataFrame(out)

def evaluate_selected(selected,X,y,Xtest,ytest,dgp_code,min_stage1=10,grid_n=35):
    d=DGP_REGISTRY[dgp_code]; truth=d.gradient(Xtest); mean=d.mean(Xtest)
    axis=np.linspace(-2,2,int(grid_n)); xx,yy=np.meshgrid(axis,axis); grid=np.column_stack([xx.ravel(),yy.ravel()]); gtruth=d.gradient(grid)
    rows=[]
    for row in selected.to_dict('records'):
        cfg={k:row[k] for k in ['r','eps','k','n_knots','alpha','gap_factor'] if k in row and pd.notna(row[k])}
        try:
            p,g,diag=_fit_predict(row['method'],cfg,X,y,Xtest,min_stage1=min_stage1)
            _,gg,_=_fit_predict(row['method'],cfg,X,y,grid,min_stage1=min_stage1)
            out=dict(row); out.update({'test_y_rmse':_rmse(p,ytest),'test_mean_rmse':_rmse(p,mean),
                'test_gradient_rmse':_grad_rmse(g,truth),'test_slope1_rmse':_rmse(g[:,0],truth[:,0]),'test_slope2_rmse':_rmse(g[:,1],truth[:,1]),
                'grid_gradient_rmse':_grad_rmse(gg,gtruth),'n_support_components':float(diag.get('n_support_components',np.nan)) if isinstance(diag,dict) else np.nan})
        except Exception:
            out=dict(row); out.update({'test_y_rmse':np.nan,'test_mean_rmse':np.nan,'test_gradient_rmse':np.nan,'test_slope1_rmse':np.nan,'test_slope2_rmse':np.nan,'grid_gradient_rmse':np.nan,'n_support_components':np.nan})
        rows.append(out)
    return pd.DataFrame(rows)

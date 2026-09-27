from __future__ import annotations
import numpy as np
from .config import CV_REPEATS,CV_FOLDS,R_GRID,EPS_GRID,K_GRID,SPLINE_LAM_GRID
from .estimators import bblr_predict,llr_predict,adaptive_local_poly_predict,smoothing_spline_predict

def split_schedule(n:int, seed:int, repeats=CV_REPEATS, folds=CV_FOLDS):
    ss=np.random.SeedSequence(seed)
    children=ss.spawn(repeats)
    schedules=[]
    for child in children:
        rng=np.random.default_rng(child)
        idx=rng.permutation(n)
        parts=np.array_split(idx,folds)
        for f in range(folds):
            val=np.asarray(parts[f],int)
            train=np.concatenate([parts[j] for j in range(folds) if j!=f])
            schedules.append((train,val))
    return schedules

def _mean_loss(losses):
    return float(np.mean(losses))

def tune_bblr(x,y,splits):
    losses={(float(r),float(e)):[] for r in R_GRID for e in EPS_GRID}
    for tr,va in splits:
        xt,yt=x[tr],y[tr]; q=x[va]; target=y[va]
        for r in R_GRID:
            # stage1 centre fit is recomputed inside calls, acceptable and exact; query grids small.
            for e in EPS_GRID:
                p,_,_=bblr_predict(xt,yt,q,float(r),float(e))
                losses[(float(r),float(e))].append(np.mean((target-p)**2))
    best=min(losses,key=lambda z:(_mean_loss(losses[z]),z[0],z[1]))
    return best,_mean_loss(losses[best])

def tune_llr(x,y,splits):
    losses={float(r):[] for r in R_GRID}
    for tr,va in splits:
        for r in R_GRID:
            p,_,_=llr_predict(x[tr],y[tr],x[va],float(r))
            losses[float(r)].append(np.mean((y[va]-p)**2))
    best=min(losses,key=lambda z:(_mean_loss(losses[z]),z))
    return best,_mean_loss(losses[best])

def tune_adaptive(x,y,splits,degree):
    losses={int(k):[] for k in K_GRID}
    for tr,va in splits:
        for k in K_GRID:
            p,_=adaptive_local_poly_predict(x[tr],y[tr],x[va],int(k),degree=degree)
            losses[int(k)].append(np.mean((y[va]-p)**2))
    best=min(losses,key=lambda z:(_mean_loss(losses[z]),z))
    return best,_mean_loss(losses[best])

def tune_spline(x,y,splits):
    losses={float(l):[] for l in SPLINE_LAM_GRID}
    for tr,va in splits:
        for lam in SPLINE_LAM_GRID:
            try:
                p,_=smoothing_spline_predict(x[tr],y[tr],x[va],float(lam))
                val=float(np.mean((y[va]-p)**2))
            except Exception:
                val=float("inf")
            losses[float(lam)].append(val)
    best=min(losses,key=lambda z:(_mean_loss(losses[z]),z))
    return best,_mean_loss(losses[best])

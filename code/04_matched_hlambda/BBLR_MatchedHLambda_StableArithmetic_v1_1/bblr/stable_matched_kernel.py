"""Stable-arithmetic implementation of the exact matched-W local-linear slope.

Uses exactly the same W_lambda weights and exactly the same finite-sample branch
rules as the v1.0.1 safeguard estimator.  The only change is algebraic: when
the weighted fit is accepted, the slope is computed as centred weighted
covariance divided by centred weighted variance, avoiding cancellation in the
normal-equation formula.  This is mathematically the same estimator.
"""
from __future__ import annotations
import numpy as np
from numba import njit
from .matched_kernel import I_lambda, _w_scalar, _unweighted_knn_fit

@njit(cache=True)
def _predict_stable(xs,ys,q,r,lam,Ilam,min_points):
    nq=q.size
    slope=np.empty(nq,np.float64); fallback=np.zeros(nq,np.bool_); valid=np.ones(nq,np.bool_); count=np.zeros(nq,np.int64)
    sr=r*(1.0+lam)
    for i in range(nq):
        x0=q[i]; left=np.searchsorted(xs,x0-sr); right=np.searchsorted(xs,x0+sr,side='right')
        s0=0.0;s1=0.0;s2=0.0;t0=0.0;t1=0.0;cnt=0
        for j in range(left,right):
            z=xs[j]-x0; w=_w_scalar(z/r,lam,Ilam)
            if w>1e-14:
                y=ys[j];s0+=w;s1+=w*z;s2+=w*z*z;t0+=w*y;t1+=w*z*y;cnt+=1
        count[i]=cnt; det=s0*s2-s1*s1
        nonsingular=(cnt>=2 and s0>0.0 and abs(det)>1e-14*max(1.0,s0*s2))
        if cnt<min_points or not nonsingular:
            _,b=_unweighted_knn_fit(xs,ys,x0,min_points);slope[i]=b;fallback[i]=True
        else:
            zbar=s1/s0; ybar=t0/s0
            num=0.0; den=0.0
            for j in range(left,right):
                z=xs[j]-x0; w=_w_scalar(z/r,lam,Ilam)
                if w>1e-14:
                    dz=z-zbar; num+=w*dz*(ys[j]-ybar); den+=w*dz*dz
            if den<=0.0 or not np.isfinite(den):
                _,b=_unweighted_knn_fit(xs,ys,x0,min_points);slope[i]=b;fallback[i]=True
            else:
                slope[i]=num/den
    return slope,valid,fallback,count

class StableMatchedHLocalLinear1D:
    def __init__(self,r,lam,min_points=15):
        self.r=float(r);self.lam=float(lam);self.min_points=int(min_points);self.Ilam=I_lambda(self.lam)
    def fit(self,x,y):
        x=np.asarray(x,float).reshape(-1);y=np.asarray(y,float).reshape(-1);o=np.argsort(x,kind='mergesort');self.xs_=np.ascontiguousarray(x[o]);self.ys_=np.ascontiguousarray(y[o]);return self
    def slopes(self,xnew):
        q=np.ascontiguousarray(np.asarray(xnew,float).reshape(-1));s,v,f,c=_predict_stable(self.xs_,self.ys_,q,self.r,self.lam,self.Ilam,self.min_points);return s,{'valid':v,'fallback':f,'positive_weight_count':c}

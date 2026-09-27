from __future__ import annotations
import numpy as np
from scipy.interpolate import make_smoothing_spline
from .config import MIN_NEIGH

_EPS = np.finfo(float).eps

def _sorted_xy(x, y):
    o = np.argsort(x, kind="mergesort")
    return np.asarray(x, float)[o], np.asarray(y, float)[o]

def _prefix(z):
    z = np.asarray(z, float)
    return np.r_[0.0, np.cumsum(z)]

def _interval_indices(xs, q, radius):
    lo = np.searchsorted(xs, q-radius, side="left")
    hi = np.searchsorted(xs, q+radius, side="right")
    return lo, hi

def _ensure_min_neigh(xs, q, lo, hi, k=MIN_NEIGH):
    n = len(xs)
    if hi-lo >= k:
        return lo, hi
    # exactly k nearest points in 1D via local candidate window around insertion point
    pos = np.searchsorted(xs, q)
    a = max(0, pos-k)
    b = min(n, pos+k)
    cand = np.arange(a,b)
    take = cand[np.argpartition(np.abs(xs[cand]-q), min(k-1,len(cand)-1))[:k]]
    return int(take.min()), int(take.max()+1)

def _local_line_at_queries(x, y, q, r):
    xs, ys = _sorted_xy(x,y)
    n = len(xs)
    q = np.asarray(q,float)
    py = _prefix(ys); px = _prefix(xs); pxx = _prefix(xs*xs); pxy = _prefix(xs*ys)
    pred = np.empty(len(q)); slope = np.empty(len(q)); count=np.empty(len(q),int)
    for j,t in enumerate(q):
        lo,hi = _interval_indices(xs,t,r)
        lo,hi = _ensure_min_neigh(xs,t,lo,hi)
        nn=hi-lo; count[j]=nn
        sx=px[hi]-px[lo]; sy=py[hi]-py[lo]; sxx=pxx[hi]-pxx[lo]; sxy=pxy[hi]-pxy[lo]
        # centered OLS slope stable via raw sums
        den = sxx - sx*sx/nn
        if den <= 1e-14:
            idx=np.arange(lo,hi)
            xc=xs[idx]-t
            A=np.c_[np.ones(nn),xc]
            coef=np.linalg.lstsq(A,ys[idx],rcond=None)[0]
            pred[j]=coef[0]; slope[j]=coef[1]
        else:
            b=(sxy-sx*sy/nn)/den
            xbar=sx/nn; ybar=sy/nn
            pred[j]=ybar+b*(t-xbar); slope[j]=b
    return pred,slope,count

def fit_stage1_centres(x,y,r):
    xs,ys=_sorted_xy(x,y)
    pred,b,count=_local_line_at_queries(xs,ys,xs,r)
    # local affine map at centre: pred + b*(q-centre), so alpha0=pred-b*centre
    alpha0=pred-b*xs
    return xs,alpha0,b,count

def bblr_predict(x,y,q,r,eps):
    q=np.asarray(q,float)
    if eps == 0.0:
        p,g,c=_local_line_at_queries(x,y,q,r)
        return p,g,{"stage1_min":int(c.min()),"stage1_fallback_share":float(np.mean(c<=MIN_NEIGH))}
    xs,a0,b,c=fit_stage1_centres(x,y,r)
    pa=_prefix(a0); pb=_prefix(b)
    p=np.empty(len(q)); g=np.empty(len(q)); ncent=np.empty(len(q),int)
    for j,t in enumerate(q):
        lo,hi=_interval_indices(xs,t,eps)
        if hi<=lo:
            k=int(np.argmin(np.abs(xs-t))); lo=k; hi=k+1
        nn=hi-lo; ncent[j]=nn
        ma=(pa[hi]-pa[lo])/nn; mb=(pb[hi]-pb[lo])/nn
        p[j]=ma+mb*t; g[j]=mb
    return p,g,{"stage1_min":int(c.min()),"stage1_fallback_share":float(np.mean(c<=MIN_NEIGH)),"stage2_empty_share":float(np.mean(ncent==1))}

def llr_predict(x,y,q,r):
    p,g,c=_local_line_at_queries(x,y,q,r)
    return p,g,{"stage1_min":int(c.min()),"stage1_fallback_share":float(np.mean(c<=MIN_NEIGH))}

def _tricube_weights(d,h):
    if h <= 0:
        return np.ones_like(d)
    hp = np.nextafter(float(h), np.inf)
    z=np.clip(d/hp,0,None)
    w=np.maximum(1.0-z**3,0.0)**3
    return w

def adaptive_local_poly_predict(x,y,q,k,degree=1):
    x=np.asarray(x,float); y=np.asarray(y,float); q=np.asarray(q,float)
    k=min(int(k),len(x))
    pred=np.empty(len(q)); deriv=np.empty(len(q))
    for j,t in enumerate(q):
        d=np.abs(x-t)
        idx=np.argpartition(d,k-1)[:k]
        di=d[idx]
        # h = kth nearest distance within chosen set
        h=float(di.max())
        w=_tricube_weights(di,h)
        z=x[idx]-t
        if degree==1:
            A=np.c_[np.ones(k),z]
        else:
            A=np.c_[np.ones(k),z,z*z]
        sw=np.sqrt(w)
        coef=np.linalg.lstsq(A*sw[:,None],y[idx]*sw,rcond=None)[0]
        pred[j]=coef[0]; deriv[j]=coef[1]
    return pred,deriv

def smoothing_spline_predict(x,y,q,lam):
    xs,ys=_sorted_xy(x,y)
    # make_smoothing_spline requires strictly increasing x. Average exact ties defensively.
    ux, inv=np.unique(xs,return_inverse=True)
    if len(ux)<len(xs):
        sums=np.bincount(inv,weights=ys); cnt=np.bincount(inv)
        ys=sums/cnt; xs=ux
    spl=make_smoothing_spline(xs,ys,lam=float(lam))
    q=np.asarray(q,float)
    return np.asarray(spl(q),float), np.asarray(spl.derivative(1)(q),float)

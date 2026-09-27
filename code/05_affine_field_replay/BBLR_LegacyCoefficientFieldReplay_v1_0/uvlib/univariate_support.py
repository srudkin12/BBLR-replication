"""Support-aware one-dimensional comparators for the BBLR laboratory.

Support components are inferred from the *training X values only*.  No DGP or
known support label is used.  A component split occurs only when an adjacent
sorted-X gap is large relative to the typical observed spacing and both
resulting components contain a minimum number of observations.

The gap-factor is a tuning parameter.  ``np.inf`` means no splitting, which
nests the corresponding conventional estimator exactly (up to floating point).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.interpolate import make_smoothing_spline

from .univariate_fast import KNNLocalPolynomial1D

Array=np.ndarray


@dataclass(frozen=True)
class ComponentDiagnostics1D:
    n_components: int
    spacing_baseline: float
    max_observed_gap: float
    max_gap_ratio: float
    split_points: Tuple[float, ...]


class SupportComponents1D:
    """Infer disconnected/weakly connected support components from training X.

    Parameters
    ----------
    gap_factor:
        Adjacent spacing must exceed ``gap_factor * median_positive_spacing``
        to be eligible as a component boundary. ``np.inf`` disables splitting.
    min_component_points:
        A split is accepted only when every resulting component has at least
        this many training observations.
    max_components:
        Safety cap preventing pathological over-segmentation.
    """
    def __init__(self, gap_factor: float=20.0, min_component_points: int=15,
                 max_components: int=6):
        self.gap_factor=float(gap_factor)
        self.min_component_points=int(min_component_points)
        self.max_components=int(max_components)
        if self.gap_factor <= 0:
            raise ValueError('gap_factor must be positive')
        if self.min_component_points < 3:
            raise ValueError('min_component_points must be at least 3')
        if self.max_components < 1:
            raise ValueError('max_components must be at least 1')

    def fit(self, x: Array) -> 'SupportComponents1D':
        x=np.asarray(x,float).reshape(-1)
        if x.size < self.min_component_points:
            raise ValueError('sample smaller than min_component_points')
        if not np.all(np.isfinite(x)):
            raise ValueError('non-finite x')
        order=np.argsort(x,kind='mergesort')
        xs=x[order]
        gaps=np.diff(xs)
        pos=gaps[gaps>0]
        baseline=float(np.median(pos)) if pos.size else 0.0
        maxgap=float(np.max(gaps)) if gaps.size else 0.0
        ratio=(maxgap/baseline) if baseline>0 else np.inf

        accepted=[]
        if np.isfinite(self.gap_factor) and baseline>0 and gaps.size:
            candidate=np.flatnonzero(gaps >= self.gap_factor*baseline)
            # Add largest gaps first, accepting only splits that leave every
            # component sufficiently populated.
            candidate=sorted(candidate.tolist(), key=lambda i: gaps[i], reverse=True)
            for idx in candidate:
                trial=sorted(accepted+[idx])
                cuts=[-1]+trial+[len(xs)-1]
                sizes=[cuts[j+1]-cuts[j] for j in range(len(cuts)-1)]
                if min(sizes) >= self.min_component_points:
                    accepted=trial
                    if len(accepted)+1 >= self.max_components:
                        break

        mids=tuple(float((xs[i]+xs[i+1])/2.0) for i in accepted)
        labels_sorted=np.searchsorted(np.asarray(mids,float), xs, side='right')
        labels=np.empty_like(labels_sorted)
        labels[order]=labels_sorted

        self.x_=x
        self.x_sorted_=xs
        self.order_=order
        self.boundaries_=np.asarray(mids,float)
        self.labels_=labels
        self.labels_sorted_=labels_sorted
        self.n_components_=len(mids)+1
        self.diagnostics_=ComponentDiagnostics1D(
            n_components=self.n_components_, spacing_baseline=baseline,
            max_observed_gap=maxgap, max_gap_ratio=float(ratio), split_points=mids,
        )
        return self

    def component_of(self, x_new: Array) -> Array:
        q=np.asarray(x_new,float).reshape(-1)
        return np.searchsorted(self.boundaries_,q,side='right').astype(int)

    def indices(self, component: int) -> Array:
        return np.flatnonzero(self.labels_==int(component))

    def diagnostics_dict(self, n_query: int | None=None) -> Dict[str, Array | float | int]:
        d=self.diagnostics_
        return {
            'n_support_components': int(d.n_components),
            'spacing_baseline': float(d.spacing_baseline),
            'max_observed_gap': float(d.max_observed_gap),
            'max_gap_ratio': float(d.max_gap_ratio),
            'split_points': '|'.join(f'{v:.12g}' for v in d.split_points),
        }


class ComponentwiseSmoothingSpline1D:
    """Smoothing spline fitted separately within inferred support components."""
    def __init__(self, lam: float, gap_factor: float=20.0,
                 min_component_points: int=15):
        self.lam=float(lam)
        self.gap_factor=float(gap_factor)
        self.min_component_points=int(min_component_points)
        if self.lam < 0:
            raise ValueError('lam must be non-negative')

    def fit(self,x:Array,y:Array)->'ComponentwiseSmoothingSpline1D':
        x=np.asarray(x,float).reshape(-1); y=np.asarray(y,float).reshape(-1)
        if x.size!=y.size: raise ValueError('x/y length mismatch')
        self.detector_=SupportComponents1D(self.gap_factor,self.min_component_points).fit(x)
        self.models_=[]
        for c in range(self.detector_.n_components_):
            idx=self.detector_.indices(c)
            xs=x[idx]; ys=y[idx]
            order=np.argsort(xs,kind='mergesort'); xs=xs[order]; ys=ys[order]
            ux,inv,counts=np.unique(xs,return_inverse=True,return_counts=True)
            if len(ux)!=len(xs):
                ys=np.bincount(inv,weights=ys)/counts; xs=ux
            self.models_.append(make_smoothing_spline(xs,ys,lam=self.lam))
        return self

    def predict_with_slope(self,x_new:Array):
        q=np.asarray(x_new,float).reshape(-1)
        labels=self.detector_.component_of(q)
        pred=np.empty_like(q); slope=np.empty_like(q)
        for c,model in enumerate(self.models_):
            mask=labels==c
            if mask.any():
                pred[mask]=np.asarray(model(q[mask]),float)
                slope[mask]=np.asarray(model.derivative()(q[mask]),float)
        diag=self.detector_.diagnostics_dict()
        diag['query_component']=labels
        return pred,slope,diag

    def predict(self,x_new:Array)->Array:
        return self.predict_with_slope(x_new)[0]


class ComponentwiseKNNLocalPolynomial1D:
    """Adaptive kNN local polynomial constrained to inferred support components."""
    def __init__(self,k:int,degree:int=1,gap_factor:float=20.0,
                 min_component_points:int=15):
        self.k=int(k); self.degree=int(degree); self.gap_factor=float(gap_factor)
        self.min_component_points=int(min_component_points)

    def fit(self,x:Array,y:Array)->'ComponentwiseKNNLocalPolynomial1D':
        x=np.asarray(x,float).reshape(-1); y=np.asarray(y,float).reshape(-1)
        self.detector_=SupportComponents1D(self.gap_factor,self.min_component_points).fit(x)
        self.models_=[]
        for c in range(self.detector_.n_components_):
            idx=self.detector_.indices(c)
            model=KNNLocalPolynomial1D(self.k,self.degree).fit(x[idx],y[idx])
            self.models_.append(model)
        return self

    def predict_with_slope(self,x_new:Array):
        q=np.asarray(x_new,float).reshape(-1)
        labels=self.detector_.component_of(q)
        pred=np.empty_like(q); slope=np.empty_like(q)
        bandwidth=np.empty_like(q); counts=np.empty(q.size,dtype=int)
        fallback=np.zeros(q.size,dtype=bool)
        for c,model in enumerate(self.models_):
            mask=labels==c
            if mask.any():
                pp,bb,dd=model.predict_with_slope(q[mask])
                pred[mask]=pp; slope[mask]=bb
                bandwidth[mask]=dd['adaptive_bandwidth']
                counts[mask]=dd['neighbour_count']
                fallback[mask]=dd['fallback']
        diag=self.detector_.diagnostics_dict()
        diag.update({
            'query_component':labels,
            'adaptive_bandwidth':bandwidth,
            'neighbour_count':counts,
            'fallback':fallback,
        })
        return pred,slope,diag

    def predict(self,x_new:Array)->Array:
        return self.predict_with_slope(x_new)[0]

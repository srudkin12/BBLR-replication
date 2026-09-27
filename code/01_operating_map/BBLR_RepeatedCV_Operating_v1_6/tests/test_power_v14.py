import numpy as np
from bblr.univariate import BBLR1D, LocalLinear1D
from bblr.univariate_fast import FastBBLR1D, FastLocalLinear1D
from bblr.univariate_support import (
    SupportComponents1D, ComponentwiseSmoothingSpline1D,
    ComponentwiseKNNLocalPolynomial1D,
)


def test_bblr_eps0_exactly_nests_llr_reference():
    rng=np.random.default_rng(10); x=np.sort(rng.uniform(-2,2,160)); y=1+2*x+0.2*x*x+rng.normal(0,.1,160)
    q=np.linspace(-1.8,1.8,77)
    a,b,_=LocalLinear1D(.35,15).fit(x,y).predict_with_slope(q)
    c,d,diag=BBLR1D(.35,0.0,15,1).fit(x,y).predict_with_slope(q)
    assert np.allclose(a,c,rtol=0,atol=1e-12)
    assert np.allclose(b,d,rtol=0,atol=1e-12)
    assert diag['stage2_disabled'].all()


def test_bblr_eps0_exactly_nests_llr_fast():
    rng=np.random.default_rng(11); x=np.sort(rng.uniform(-2,2,200)); y=np.sin(x)+rng.normal(0,.1,200); q=np.linspace(-1.9,1.9,91)
    a,b,_=FastLocalLinear1D(.26,15).fit(x,y).predict_with_slope(q)
    c,d,diag=FastBBLR1D(.26,0.0,15,1).fit(x,y).predict_with_slope(q)
    assert np.allclose(a,c,rtol=0,atol=1e-12)
    assert np.allclose(b,d,rtol=0,atol=1e-12)
    assert diag['stage2_disabled'].all()


def test_support_detector_splits_clear_gap_not_uniform():
    rng=np.random.default_rng(20)
    xu=rng.uniform(-2,2,500)
    gu=SupportComponents1D(20,15).fit(xu)
    assert gu.n_components_==1
    x=np.r_[rng.uniform(-2,-.6,250),rng.uniform(.6,2,250)]
    gg=SupportComponents1D(20,15).fit(x)
    assert gg.n_components_==2
    assert len(gg.boundaries_)==1 and abs(gg.boundaries_[0])<.2


def test_inf_gap_factor_nests_global_adaptive_knn():
    rng=np.random.default_rng(30); x=np.sort(rng.uniform(-2,2,250)); y=x*x+rng.normal(0,.1,250); q=np.linspace(-1.8,1.8,61)
    from bblr.univariate_fast import KNNLocalPolynomial1D
    a,b,_=KNNLocalPolynomial1D(40,1).fit(x,y).predict_with_slope(q)
    c,d,diag=ComponentwiseKNNLocalPolynomial1D(40,1,np.inf,15).fit(x,y).predict_with_slope(q)
    assert diag['n_support_components']==1
    assert np.allclose(a,c,rtol=0,atol=1e-12)
    assert np.allclose(b,d,rtol=0,atol=1e-12)


def test_inf_gap_factor_nests_global_spline():
    from scipy.interpolate import make_smoothing_spline
    rng=np.random.default_rng(31); x=np.sort(rng.uniform(-2,2,200)); y=np.sin(x)+rng.normal(0,.1,200); q=np.linspace(-1.7,1.7,55)
    sp=make_smoothing_spline(x,y,lam=.01)
    a=np.asarray(sp(q)); b=np.asarray(sp.derivative()(q))
    c,d,diag=ComponentwiseSmoothingSpline1D(.01,np.inf,15).fit(x,y).predict_with_slope(q)
    assert diag['n_support_components']==1
    assert np.allclose(a,c,rtol=0,atol=1e-12)
    assert np.allclose(b,d,rtol=0,atol=1e-12)

import numpy as np
from bblr.matched_kernel import I_lambda,b_lambda,gamma_lambda,W_lambda,MatchedHLocalLinear1D
from bblr.univariate_fast import FastLocalLinear1D,FastBBLR1D


def trapz(y,x):
    return np.trapezoid(y,x)


def test_normalisation_and_moments():
    for lam in [0.0,0.25,0.75,1.0,1.5,3.0]:
        x=np.linspace(-(1+lam)-1e-6,(1+lam)+1e-6,200001)
        w=W_lambda(x,lam)
        assert np.nanmin(w) >= -1e-10
        assert abs(trapz(w,x)-1.0) < 2e-4
        mu2=trapz(x*x*w,x)
        assert abs(mu2-b_lambda(lam)) < 3e-4
        H=x*w/mu2
        assert abs(trapz(H,x)) < 2e-5
        assert abs(trapz(x*H,x)-1.0) < 5e-4


def test_special_values():
    assert abs(I_lambda(0.0)-3.0)<1e-14
    assert abs(I_lambda(1.0)-1.5)<1e-14
    assert abs(b_lambda(1.0)-2/3)<1e-14
    assert abs(gamma_lambda(1.0))<1e-14


def test_evenness_and_central_zero():
    x=np.linspace(0,2.5,1001)
    for lam in [0.3,1.0,2.0]:
        assert np.max(np.abs(W_lambda(x,lam)-W_lambda(-x,lam)))<1e-12
    z=np.linspace(-0.99,0.99,401)
    assert np.max(np.abs(W_lambda(z,2.0)))<1e-12


def test_lambda0_safe_matches_production_llr_and_bblr():
    rng=np.random.default_rng(123)
    x=np.sort(rng.uniform(-2,2,300)); y=1+2*x+rng.normal(0,.2,len(x)); q=np.linspace(-2,2,201); r=.12
    ll=FastLocalLinear1D(r,15).fit(x,y); _,bl,_=ll.predict_with_slope(q)
    bb=FastBBLR1D(r,0.0,15,1).fit(x,y); _,bbsl,_=bb.predict_with_slope(q)
    ww=MatchedHLocalLinear1D(r,0.0,15,safe=True).fit(x,y); _,bw,_=ww.predict_with_slope(q)
    assert np.max(np.abs(bl-bbsl))<1e-12
    assert np.max(np.abs(bl-bw))<1e-10


def test_affine_reproduction_strict_when_valid():
    rng=np.random.default_rng(44)
    x=np.sort(rng.uniform(-2,2,500)); y=3.0-1.75*x; q=np.linspace(-1.5,1.5,101)
    for lam in [0.25,1.0,2.0]:
        m=MatchedHLocalLinear1D(.25,lam,15,safe=False).fit(x,y)
        _,b,d=m.predict_with_slope(q)
        assert np.all(d['valid'])
        assert np.max(np.abs(b+1.75))<1e-10

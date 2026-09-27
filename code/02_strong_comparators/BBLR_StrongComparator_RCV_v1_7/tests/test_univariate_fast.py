import numpy as np

from bblr.univariate import LocalLinear1D, BBLR1D
from bblr.univariate_fast import FastLocalLinear1D, FastBBLR1D


def test_fast_llr_matches_reference():
    rng=np.random.default_rng(123)
    x=np.sort(rng.uniform(-2,2,120)); y=np.sin(x)+0.1*rng.normal(size=len(x))
    q=np.linspace(-2,2,81)
    ref=LocalLinear1D(r=.35,min_points=8).fit(x,y)
    fast=FastLocalLinear1D(r=.35,min_points=8).fit(x,y)
    pr,sr,dr=ref.predict_with_slope(q)
    pf,sf,df=fast.predict_with_slope(q)
    assert np.allclose(pf,pr,atol=1e-10,rtol=1e-10)
    assert np.allclose(sf,sr,atol=1e-10,rtol=1e-10)
    assert np.array_equal(df['neighbour_count'],dr['neighbour_count'])
    assert np.array_equal(df['fallback'],dr['fallback'])


def test_fast_bblr_matches_reference_mean():
    rng=np.random.default_rng(321)
    x=rng.uniform(-2,2,140); y=x+x*x+0.15*rng.normal(size=len(x))
    q=np.linspace(-2,2,91)
    ref=BBLR1D(r=.3,eps=.22,min_points_stage1=9,min_centres_stage2=3,aggregation='mean').fit(x,y)
    fast=FastBBLR1D(r=.3,eps=.22,min_points_stage1=9,min_centres_stage2=3).fit(x,y)
    pr,sr,dr=ref.predict_with_slope(q)
    pf,sf,df=fast.predict_with_slope(q)
    assert np.allclose(pf,pr,atol=1e-10,rtol=1e-10)
    assert np.allclose(sf,sr,atol=1e-10,rtol=1e-10)
    assert np.array_equal(df['stage2_centre_count'],dr['stage2_centre_count'])
    assert np.array_equal(df['stage2_fallback'],dr['stage2_fallback'])


def test_fast_noiseless_linear_exact():
    x=np.linspace(-2,2,101); y=1.5+2.75*x; q=np.linspace(-2,2,51)
    m=FastBBLR1D(.25,.2,min_points_stage1=7,min_centres_stage2=1).fit(x,y)
    p,s,_=m.predict_with_slope(q)
    assert np.max(np.abs(s-2.75))<1e-10
    assert np.max(np.abs(p-(1.5+2.75*q)))<1e-10

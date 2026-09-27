import numpy as np
from bblr.matched_kernel import MatchedHLocalLinear1D
from bblr.stable_matched_kernel import StableMatchedHLocalLinear1D

def test_centered_matches_normal_well_conditioned():
    rng=np.random.default_rng(42);x=np.sort(rng.uniform(-2,2,1000));y=1+2*x+rng.normal(0,.2,1000);q=np.linspace(-1.5,1.5,101)
    a=MatchedHLocalLinear1D(.4,1.0,15,safe=True).fit(x,y);_,sa,da=a.predict_with_slope(q)
    b=StableMatchedHLocalLinear1D(.4,1.0,15).fit(x,y);sb,db=b.slopes(q)
    assert np.max(np.abs(sa-sb))<1e-10
    assert np.array_equal(da['fallback'],db['fallback'])

def test_centered_matches_normal_lambda_gt1_regular_sample():
    rng=np.random.default_rng(7);x=np.sort(rng.uniform(-2,2,2000));y=x*x+rng.normal(0,.1,2000);q=np.linspace(-1,1,101)
    a=MatchedHLocalLinear1D(.5,2.0,15,safe=True).fit(x,y);_,sa,da=a.predict_with_slope(q)
    b=StableMatchedHLocalLinear1D(.5,2.0,15).fit(x,y);sb,db=b.slopes(q)
    ok=np.isfinite(sa)&np.isfinite(sb)
    assert np.max(np.abs(sa[ok]-sb[ok]))<1e-8
    assert np.array_equal(da['fallback'],db['fallback'])

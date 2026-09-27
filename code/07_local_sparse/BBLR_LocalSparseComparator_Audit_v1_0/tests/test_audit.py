import numpy as np
from bblr.local_sparse import AdaptiveLocalLinearLassoND
from bblr.nd_core import LocalPolynomialND
from bblr.audit import select_methods, _stage1_min_points
from bblr.dgps import generate_sample


def test_zero_penalty_exactly_matches_local_linear():
    X,y,_,_,_=generate_sample('transition',180,5,2,0.4,12345,'uniform')
    Q=X[:12]
    r=0.9; mp=_stage1_min_points(5)
    p1,g1,_=LocalPolynomialND(1,r=r,min_points=mp).fit(X,y).predict_with_gradient(Q)
    p2,g2,_=AdaptiveLocalLinearLassoND(r,0.0,min_points=mp).fit(X,y).predict_with_gradient(Q)
    assert np.max(np.abs(p1-p2)) < 1e-9
    assert np.max(np.abs(g1-g2)) < 1e-9


def test_lambda_one_zeroes_slopes_locally():
    X,y,_,_,_=generate_sample('linear',180,5,2,0.5,23456,'uniform')
    Q=X[:10]
    _,g,_=AdaptiveLocalLinearLassoND(1.0,1.0,min_points=_stage1_min_points(5)).fit(X,y).predict_with_gradient(Q)
    assert np.max(np.abs(g)) < 1e-8


def test_outcome_cv_selects_all_required_methods():
    X,y,_,_,_=generate_sample('smooth',120,3,2,0.5,34567,'uniform')
    sel=select_methods(X,y,folds=2,repeats=1,seed=999,eps_mults=(0.0,1.0),lambda_fracs=(0.0,0.1,0.4))
    assert {z['method'] for z in sel} == {'BBLR','LLR','LL_ALASSO','LASSO','OLS'}
    a=next(z for z in sel if z['method']=='LL_ALASSO')
    assert a['lambda_frac'] in (0.0,0.1,0.4)


def test_adaptive_lasso_path_shapes():
    X,y,_,_,_=generate_sample('transition',140,8,2,0.5,45678,'corr_nuisance')
    Q=X[:7]
    fr=(0.0,0.05,0.2,0.7)
    p,g,d=AdaptiveLocalLinearLassoND(1.4,0.1,min_points=_stage1_min_points(8)).fit(X,y).predict_path(Q,fr)
    for f in fr:
        assert p[f].shape==(7,)
        assert g[f].shape==(7,8)
        assert np.all(np.isfinite(p[f]))
        assert np.all(np.isfinite(g[f]))
    assert d['min_neighbours'] >= _stage1_min_points(8)

def test_d12_local_sparse_smoke():
    X,y,_,_,_=generate_sample('transition',180,12,2,0.5,56789,'corr_nuisance')
    Q=X[:3]
    p,g,_=AdaptiveLocalLinearLassoND(1.8,0.2,min_points=_stage1_min_points(12)).fit(X,y).predict_with_gradient(Q)
    assert p.shape==(3,)
    assert g.shape==(3,12)
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(g))

import numpy as np
from bblr.repeated_cv import evaluate_bblr_cv_surface, evaluate_full_bblr_grid, mark_selectors
from bblr.univariate import generate_sample


def test_repeated_cv_surface_and_selector_flags():
    x,y,_=generate_sample('U01',120,0.10,123,'uniform')
    cv=evaluate_bblr_cv_surface(x,y,r_grid=[0.3,0.6],eps_grid=[0.0,0.2],folds=3,repeats=2,seed=456,min_stage1=8,stability_grid=np.linspace(-2,2,21))
    assert len(cv)==4
    assert np.isfinite(cv[['single_split_rmse','cv5_rmse','repeated_cv_rmse','coefficient_instability']].to_numpy()).all()


def test_repeated_cv_truth_merge_and_marking():
    x,y,_=generate_sample('U01',140,0.10,111,'uniform')
    xt,yt,_=generate_sample('U01',300,0.10,222,'uniform')
    cv=evaluate_bblr_cv_surface(x,y,r_grid=[0.3,0.6],eps_grid=[0.0,0.2],folds=2,repeats=2,seed=333,min_stage1=8,stability_grid=np.linspace(-2,2,15))
    truth=evaluate_full_bblr_grid(x,y,xt,yt,dgp_code='U01',r_grid=[0.3,0.6],eps_grid=[0.0,0.2],min_stage1=8,x_grid=np.linspace(-2,2,31))
    g=mark_selectors(cv.merge(truth,on=['r','eps'],validate='one_to_one'))
    assert g.selected_single_split.sum()==1
    assert g.selected_cv5.sum()==1
    assert g.selected_repeated_cv.sum()==1
    assert g.oracle_slope.sum()==1
    assert g.selected_rcv_tol_0p005_stability.sum()==1


def test_eps_zero_remains_exact_llr_candidate():
    x,y,_=generate_sample('U01',100,0.05,777,'uniform')
    xt,yt,_=generate_sample('U01',200,0.05,778,'uniform')
    truth=evaluate_full_bblr_grid(x,y,xt,yt,dgp_code='U01',r_grid=[0.4],eps_grid=[0.0,0.2],min_stage1=8)
    assert (truth.eps==0.0).sum()==1
    assert np.isfinite(truth.loc[truth.eps==0.0,'test_slope_rmse']).all()

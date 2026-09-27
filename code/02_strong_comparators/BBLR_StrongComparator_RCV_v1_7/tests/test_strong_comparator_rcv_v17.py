import numpy as np
from bblr.strong_comparator_cv import repeated_cv_risk_surface, select_minimum_risk, evaluate_selected
from bblr.univariate import generate_sample


def _sample():
    x,y,_=generate_sample('U01',120,0.10,12345,'G00')
    return x,y


def test_bblr_eps0_repeated_cv_matches_llr():
    x,y=_sample()
    s=repeated_cv_risk_surface(x,y,methods=['BBLR','LLR'],r_grid=[0.3,0.6],eps_grid=[0,0.2],
        k_grid=[20],lambda_grid=[0.01],gap_factor_grid=[np.inf],folds=3,repeats=2,seed=99)
    b=s[(s.method=='BBLR') & (s.eps==0)].sort_values('r').repeated_cv_rmse.to_numpy()
    l=s[s.method=='LLR'].sort_values('r').repeated_cv_rmse.to_numpy()
    assert np.allclose(b,l,rtol=0,atol=1e-12)


def test_strong_comparator_surface_contains_all_methods():
    x,y=_sample()
    methods=['BBLR','LLR','AKLLR','AKLP2','SPLINE','CAKLLR','CAKLP2','CSPLINE']
    s=repeated_cv_risk_surface(x,y,methods=methods,r_grid=[0.4],eps_grid=[0,0.2],k_grid=[20],
        lambda_grid=[0.01],gap_factor_grid=[np.inf],folds=2,repeats=1,seed=101,
        component_min_points=10)
    assert set(s.method)==set(methods)
    assert np.isfinite(s.repeated_cv_rmse).all()


def test_selected_full_evaluation_is_finite_on_linear_dgp():
    x,y=_sample(); xt,yt,_=generate_sample('U01',150,0.10,54321,'G00')
    s=repeated_cv_risk_surface(x,y,methods=['BBLR','AKLLR','SPLINE'],r_grid=[0.4],eps_grid=[0,0.2],
        k_grid=[20],lambda_grid=[0.01],gap_factor_grid=[np.inf],folds=2,repeats=1,seed=101)
    sel=select_minimum_risk(s)
    ev=evaluate_selected(sel,x,y,xt,yt,dgp_code='U01',component_min_points=10)
    assert np.isfinite(ev.test_y_rmse).all()
    assert np.isfinite(ev.test_slope_rmse).all()

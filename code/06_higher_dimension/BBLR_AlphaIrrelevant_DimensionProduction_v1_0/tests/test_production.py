import numpy as np
from bblr.dgps import mean_gradient,tangent_alpha,reference_alpha_truth,generate_sample,ALPHA0
from bblr.nd_core import BBLRND,LocalPolynomialND
from bblr.production import _vector_rmse,_nuisance_metrics,_stage1_min_points,evaluate_rep,select_methods


def test_reference_alpha_is_half_for_all_dgps():
    cases=[('flat',5,0),('linear',5,2),('smooth',5,2),('transition',5,2),('highfreq',5,2),('linear_all',5,5),('smooth_all',5,5)]
    for code,d,a in cases:
        assert abs(reference_alpha_truth(code,d,a)-ALPHA0)<1e-12


def test_flat_truth_zero_gradient():
    X=np.array([[0.2,-0.4,0.7],[-0.5,0.1,0.0]])
    m,g=mean_gradient(X,'flat',0)
    assert np.allclose(m,ALPHA0)
    assert np.allclose(g,0)


def test_vector_rmse_matches_component_rmse_times_sqrt_d():
    rng=np.random.default_rng(4)
    e=rng.normal(size=(100,7))
    z=np.zeros_like(e)
    assert np.isclose(_vector_rmse(e,z),np.sqrt(np.mean(e*e))*np.sqrt(7))


def test_tangent_alpha_identity_for_exact_plane():
    rng=np.random.default_rng(8)
    X=rng.uniform(-1,1,(220,4)); beta=np.array([1.,-.75,0.,0.]); y=.5+X@beta
    Q=rng.uniform(-1,1,(40,4))
    p,g,_=BBLRND(.65,.5,min_stage1=_stage1_min_points(4)).fit(X,y).predict_with_gradient(Q)
    ah=p-np.sum(Q*g,axis=1)
    assert np.max(np.abs(ah-.5))<1e-8


def test_reference_alpha_estimation_exact_plane_at_origin():
    rng=np.random.default_rng(10)
    X=rng.uniform(-1,1,(220,4)); beta=np.array([1.,-.75,0.,0.]); y=.5+X@beta
    p,g,_=BBLRND(.65,.5,min_stage1=_stage1_min_points(4)).fit(X,y).predict_with_gradient(np.zeros((1,4)))
    assert abs(float(p[0])-.5)<1e-8


def test_nuisance_metrics_zero_and_dimension_normalisation():
    gh=np.column_stack([np.ones(25),-np.ones(25),np.zeros((25,3))])
    m=_nuisance_metrics(gh,2)
    assert m['nuisance_dim']==3
    assert m['nuisance_vector_rmse']==0
    assert m['nuisance_vector_rmse_per_sqrt_dim']==0
    assert m['nuisance_scaled_share_gt_010']==0
    rng=np.random.default_rng(5)
    gh=np.column_stack([np.zeros((50,2)),rng.normal(size=(50,4))])
    m=_nuisance_metrics(gh,2)
    assert np.isclose(m['nuisance_vector_rmse_per_sqrt_dim'],m['nuisance_component_rmse'])


def test_exact_nesting_common_safeguard_d2_and_d12():
    rng=np.random.default_rng(11)
    for d in (2,12):
        n=220 if d==2 else 320
        X=rng.uniform(-1,1,(n,d)); beta=np.zeros(d); beta[:2]=[1.,-.75]; y=.5+X@beta+rng.normal(0,.2,n)
        Q=rng.uniform(-1,1,(50,d)); r=0.8 if d==2 else 2.0; mp=_stage1_min_points(d)
        pb,gb,_=BBLRND(r,0.0,min_stage1=mp).fit(X,y).predict_with_gradient(Q)
        pl,gl,_=LocalPolynomialND(1,r=r,min_points=mp).fit(X,y).predict_with_gradient(Q)
        assert np.max(np.abs(pb-pl))<1e-12
        assert np.max(np.abs(gb-gl))<1e-12


def test_selected_matched_radius_equals_bblr_selected_radius():
    X,y,_,_,_=generate_sample('transition',140,3,2,.3,12345,'uniform')
    sel=select_methods(X,y,2,2,1,4567)
    b=next(z for z in sel if z['method']=='BBLR')
    m=next(z for z in sel if z['method']=='LLR_MATCHED_BBLR_R')
    assert np.isclose(b['r'],m['r'])
    assert np.isclose(b['eps'],m['matched_bblr_eps'])


def test_active_dim_zero_rep_runs_and_contains_matched_radius():
    cell=dict(cell_id='T0',track='flat_null',d=3,active_dim=0,dgp='flat',design='uniform',n_dev=120,n_test=100,noise_sd=.2,origin_supported=True,cv_folds=2,cv_repeats=1,base_seed=123,label='test')
    z=evaluate_rep(cell,0)
    assert 'BBLR' in set(z.method)
    assert 'LLR_MATCHED_BBLR_R' in set(z.method)
    assert 'MEAN_ONLY' in set(z.method)
    b=z[z.method=='BBLR'].iloc[0]
    assert np.isfinite(b.reference_alpha0_abs_error)
    assert np.isfinite(b.nuisance_vector_rmse)

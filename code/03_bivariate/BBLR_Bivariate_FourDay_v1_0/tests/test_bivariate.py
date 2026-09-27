import numpy as np
from bblr.bivariate import DGP_REGISTRY,generate_sample,sample_x,SUPPORT_LABELS
from bblr.bivariate_models import BBLR2D,LocalPolynomial2D,GlobalPoly2D

def test_registry_and_gradients_finite_difference():
    rng=np.random.default_rng(1); X=rng.uniform(-1.5,1.5,size=(20,2)); h=1e-6
    for d in DGP_REGISTRY.values():
        g=d.gradient(X); assert g.shape==(len(X),2); assert np.all(np.isfinite(g))
        for j in range(2):
            xp=X.copy(); xm=X.copy(); xp[:,j]+=h; xm[:,j]-=h
            fd=(d.mean(xp)-d.mean(xm))/(2*h)
            assert np.max(np.abs(fd-g[:,j]))<2e-4

def test_all_supports_generate():
    for design in SUPPORT_LABELS:
        X=sample_x(120,123,design); assert X.shape==(120,2); assert np.all(np.isfinite(X)); assert np.all((X>=-2)&(X<=2))

def test_noiseless_linear_local_and_bblr():
    rng=np.random.default_rng(2); X=rng.uniform(-2,2,size=(300,2)); y=X[:,0]+X[:,1]; Q=rng.uniform(-1,1,size=(40,2))
    p,g,_=LocalPolynomial2D(1,r=.6,min_points=10).fit(X,y).predict_with_gradient(Q)
    assert np.max(np.abs(g-1))<1e-10
    b=BBLR2D(.6,.4,min_stage1=10).fit(X,y); p,g,_=b.predict_with_gradient(Q); assert np.max(np.abs(g-1))<1e-10

def test_eps_zero_nests_llr():
    rng=np.random.default_rng(3); X=rng.uniform(-2,2,size=(250,2)); y=np.sin(X[:,0])+X[:,1]+rng.normal(0,.1,size=250); Q=rng.uniform(-1.5,1.5,size=(30,2))
    p1,g1,_=BBLR2D(.5,0,min_stage1=10).fit(X,y).predict_with_gradient(Q)
    p2,g2,_=LocalPolynomial2D(1,r=.5,min_points=10).fit(X,y).predict_with_gradient(Q)
    assert np.allclose(p1,p2); assert np.allclose(g1,g2)

def test_global_quadratic_exact():
    rng=np.random.default_rng(4); X=rng.uniform(-2,2,size=(200,2)); d=DGP_REGISTRY['B02']; y=d.mean(X); Q=rng.uniform(-1.5,1.5,size=(30,2)); _,g,_=GlobalPoly2D(2).fit(X,y).predict_with_gradient(Q)
    assert np.max(np.abs(g-d.gradient(Q)))<1e-10

def test_small_repeated_cv_surface_all_methods():
    from bblr.bivariate_cv import repeated_cv_risk_surface,select_minimum_risk,evaluate_selected
    X,y,_=generate_sample('B10',120,.2,999,'vertical_soft10')
    surf=repeated_cv_risk_surface(X,y,['BBLR','LLR','LP2','AKLLR','AKLP2','TSPLINE','CTSPLINE','OLS','GQ2'],
        r_grid=[.5],eps_grid=[0,.3],k_grid=[30],knot_grid=[4],alpha_grid=[.1],gap_grid=[6,np.inf],folds=2,repeats=1,seed=123,min_stage1=8)
    assert set(surf.method)=={'BBLR','LLR','LP2','AKLLR','AKLP2','TSPLINE','CTSPLINE','OLS','GQ2'}
    assert np.all(np.isfinite(surf.repeated_cv_rmse))
    sel=select_minimum_risk(surf)
    Xt,yt,_=generate_sample('B10',80,.2,1000,'vertical_soft10')
    ev=evaluate_selected(sel,X,y,Xt,yt,'B10',min_stage1=8,grid_n=9)
    assert len(ev)==9
    assert np.all(np.isfinite(ev.test_gradient_rmse))

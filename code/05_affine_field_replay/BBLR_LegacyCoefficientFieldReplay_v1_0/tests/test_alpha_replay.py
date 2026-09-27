import numpy as np
from uvlib.univariate import BBLR1D,DGP_REGISTRY
from bivlib.bivariate_models import BBLR2D

def test_univariate_bblr_tangent_intercept_identity():
 rng=np.random.default_rng(1);x=np.sort(rng.uniform(-2,2,100));y=0.5+x+0.2*x*x+rng.normal(0,.05,100);q=np.array([-.4,.2,.8])
 m=BBLR1D(.5,.4,min_points_stage1=10).fit(x,y);p,g,_=m.predict_with_slope(q)
 for i,z in enumerate(q):
  idx=m.index_.radius(z,m.eps)
  local_global_intercepts=m.stage1_level_[idx]-m.stage1_slope_[idx]*m.x_[idx]
  assert abs((p[i]-z*g[i])-np.mean(local_global_intercepts))<1e-10

def test_bivariate_bblr_tangent_intercept_identity():
 rng=np.random.default_rng(2);X=rng.uniform(-2,2,(120,2));y=.5+X[:,0]+.3*X[:,1]+rng.normal(0,.05,120);q=np.array([[.2,-.3]])
 m=BBLR2D(.8,.6,min_stage1=10).fit(X,y);p,g,_=m.predict_with_gradient(q)
 idx=np.asarray(m.tree.query_ball_point(q[0],m.eps),int)
 vals=m.a[idx]-np.sum(m.b[idx]*m.X[idx],axis=1)
 assert abs((p[0]-q[0]@g[0])-np.mean(vals))<1e-10

def test_true_alpha_definition():
 d=DGP_REGISTRY['U02'];x=np.array([-.5,.2]);a=d.mean(x)-x*d.derivative(x)
 assert np.allclose(a,-x*x)

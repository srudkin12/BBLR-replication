from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge

Array=np.ndarray

def _design(dx:Array, degree:int)->Array:
    dx=np.asarray(dx,float); n,d=dx.shape
    cols=[np.ones(n)]
    cols.extend(dx[:,j] for j in range(d))
    if degree>=2:
        for j in range(d):
            for k in range(j,d):
                cols.append(dx[:,j]*dx[:,k])
    return np.column_stack(cols)

def _fit_centered(X:Array,y:Array,center:Array,idx:Array,degree:int)->Array:
    A=_design(X[idx]-center,degree)
    return np.linalg.lstsq(A,y[idx],rcond=None)[0]

def quadratic_ncoef(d:int)->int:
    return 1+d+d*(d+1)//2

class LocalPolynomialND:
    def __init__(self,degree=1,r=None,k=None,min_points=None):
        self.degree=int(degree); self.r=None if r is None else float(r); self.k=None if k is None else int(k)
        self.min_points=min_points
    def fit(self,X,y):
        self.X=np.asarray(X,float); self.y=np.asarray(y,float); self.d=self.X.shape[1]
        base=(self.d+4 if self.degree==1 else quadratic_ncoef(self.d)+8)
        self.min_points=int(self.min_points or base)
        self.tree=cKDTree(self.X); return self
    def _idx(self,q):
        if self.k is not None:
            kk=min(max(int(self.k),self.min_points),len(self.X)); return np.atleast_1d(self.tree.query(q,k=kk)[1]).astype(int)
        idx=np.asarray(self.tree.query_ball_point(q,float(self.r)),int)
        if len(idx)<self.min_points:
            kk=min(self.min_points,len(self.X)); idx=np.atleast_1d(self.tree.query(q,k=kk)[1]).astype(int)
        return idx
    def predict_with_gradient(self,Q):
        Q=np.asarray(Q,float); pred=np.empty(len(Q)); grad=np.empty((len(Q),self.d)); counts=np.empty(len(Q),int)
        for i,q in enumerate(Q):
            idx=self._idx(q); c=_fit_centered(self.X,self.y,q,idx,self.degree)
            pred[i]=c[0]; grad[i]=c[1:1+self.d]; counts[i]=len(idx)
        return pred,grad,{'mean_neighbours':float(np.mean(counts)),'min_neighbours':int(np.min(counts))}

class BBLRND:
    def __init__(self,r,eps,min_stage1=None,min_stage2=1):
        self.r=float(r); self.eps=float(eps); self.min_stage1=min_stage1; self.min_stage2=int(min_stage2)
    def fit(self,X,y):
        self.X=np.asarray(X,float); self.y=np.asarray(y,float); self.d=self.X.shape[1]
        self.min_stage1=int(self.min_stage1 or max(self.d+5,15))
        self.tree=cKDTree(self.X)
        n=len(self.X); self.a=np.empty(n); self.b=np.empty((n,self.d)); self.counts=np.empty(n,int)
        for i,c in enumerate(self.X):
            idx=np.asarray(self.tree.query_ball_point(c,self.r),int)
            if len(idx)<self.min_stage1:
                idx=np.atleast_1d(self.tree.query(c,k=min(self.min_stage1,n))[1]).astype(int)
            cf=_fit_centered(self.X,self.y,c,idx,1)
            self.a[i]=cf[0]; self.b[i]=cf[1:1+self.d]; self.counts[i]=len(idx)
        return self
    def predict_with_gradient_eps(self,Q,eps):
        Q=np.asarray(Q,float); eps=float(eps)
        if eps==0:
            return LocalPolynomialND(1,r=self.r,min_points=self.min_stage1).fit(self.X,self.y).predict_with_gradient(Q)
        pred=np.empty(len(Q)); grad=np.empty((len(Q),self.d)); ns=np.empty(len(Q),int)
        for i,q in enumerate(Q):
            idx=np.asarray(self.tree.query_ball_point(q,eps),int)
            if len(idx)<self.min_stage2:
                idx=np.atleast_1d(self.tree.query(q,k=min(self.min_stage2,len(self.X)))[1]).astype(int)
            planes=self.a[idx]+np.sum(self.b[idx]*(q-self.X[idx]),axis=1)
            pred[i]=float(np.mean(planes)); grad[i]=np.mean(self.b[idx],axis=0); ns[i]=len(idx)
        return pred,grad,{'mean_stage1_neighbours':float(np.mean(self.counts)),'mean_stage2_centres':float(np.mean(ns))}
    def predict_with_gradient(self,Q):
        return self.predict_with_gradient_eps(Q,self.eps)

class GlobalLinearND:
    def fit(self,X,y):
        X=np.asarray(X,float); A=np.column_stack([np.ones(len(X)),X]); self.d=X.shape[1]
        self.coef=np.linalg.lstsq(A,np.asarray(y,float),rcond=None)[0]; return self
    def predict_with_gradient(self,Q):
        Q=np.asarray(Q,float); pred=self.coef[0]+Q@self.coef[1:]; grad=np.tile(self.coef[1:],(len(Q),1)); return pred,grad,{}

class RidgeLinearND:
    def __init__(self,alpha=1.0): self.alpha=float(alpha)
    def fit(self,X,y):
        self.X=np.asarray(X,float); self.d=self.X.shape[1]
        self.model=Ridge(alpha=self.alpha,fit_intercept=True,solver='auto').fit(self.X,np.asarray(y,float)); return self
    def predict_with_gradient(self,Q):
        Q=np.asarray(Q,float); pred=self.model.predict(Q); grad=np.tile(np.asarray(self.model.coef_,float),(len(Q),1)); return pred,grad,{}

class GlobalQuadraticND:
    def fit(self,X,y):
        self.X=np.asarray(X,float); self.d=self.X.shape[1]; self.coef=np.linalg.lstsq(_design(self.X,2),np.asarray(y,float),rcond=None)[0]; return self
    def predict_with_gradient(self,Q):
        Q=np.asarray(Q,float); pred=_design(Q,2)@self.coef; d=self.d
        grad=np.tile(self.coef[1:1+d],(len(Q),1)); pos=1+d
        for j in range(d):
            for k in range(j,d):
                c=self.coef[pos]; pos+=1
                if j==k: grad[:,j]+=2*c*Q[:,j]
                else:
                    grad[:,j]+=c*Q[:,k]; grad[:,k]+=c*Q[:,j]
        return pred,grad,{}

def radius_grid_from_knn(X:Array,k_targets)->list[float]:
    X=np.asarray(X,float); tree=cKDTree(X); vals=[]
    for k in k_targets:
        kk=min(max(2,int(k)),len(X)-1)
        dist=tree.query(X,k=kk+1)[0][:,kk]
        vals.append(float(np.median(dist)))
    out=[]
    for v in vals:
        if np.isfinite(v) and v>0 and not any(abs(v-z)<=1e-12 for z in out): out.append(v)
    return out

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.cluster import DBSCAN

Array=np.ndarray

def _design(dx,degree):
    if degree==1:
        return np.column_stack([np.ones(len(dx)),dx[:,0],dx[:,1]])
    return np.column_stack([np.ones(len(dx)),dx[:,0],dx[:,1],dx[:,0]**2,dx[:,0]*dx[:,1],dx[:,1]**2])

def _fit_centered(X,y,center,idx,degree):
    dx=X[idx]-center
    A=_design(dx,degree)
    coef=np.linalg.lstsq(A,y[idx],rcond=None)[0]
    return coef

class LocalPolynomial2D:
    def __init__(self,degree=1,r=None,k=None,min_points=None):
        self.degree=int(degree); self.r=r; self.k=k
        self.min_points=int(min_points or (8 if degree==1 else 15))
    def fit(self,X,y):
        self.X=np.asarray(X,float); self.y=np.asarray(y,float); self.tree=cKDTree(self.X); return self
    def _idx(self,q):
        if self.k is not None:
            kk=min(max(int(self.k),self.min_points),len(self.X)); return np.atleast_1d(self.tree.query(q,k=kk)[1])
        idx=np.asarray(self.tree.query_ball_point(q,float(self.r)),int)
        if len(idx)<self.min_points:
            kk=min(self.min_points,len(self.X)); idx=np.atleast_1d(self.tree.query(q,k=kk)[1])
        return idx
    def predict_with_gradient(self,Q):
        Q=np.asarray(Q,float); pred=np.empty(len(Q)); grad=np.empty((len(Q),2)); counts=np.empty(len(Q),int)
        for i,q in enumerate(Q):
            idx=self._idx(q); c=_fit_centered(self.X,self.y,q,idx,self.degree)
            pred[i]=c[0]; grad[i]=c[1:3]; counts[i]=len(idx)
        return pred,grad,{'mean_neighbours':float(np.mean(counts)),'min_neighbours':int(np.min(counts))}

class BBLR2D:
    def __init__(self,r,eps,min_stage1=8,min_stage2=1):
        self.r=float(r); self.eps=float(eps); self.min_stage1=int(min_stage1); self.min_stage2=int(min_stage2)
    def fit(self,X,y):
        self.X=np.asarray(X,float); self.y=np.asarray(y,float); self.tree=cKDTree(self.X)
        self.a=np.empty(len(self.X)); self.b=np.empty((len(self.X),2)); self.counts=np.empty(len(self.X),int)
        for i,c in enumerate(self.X):
            idx=np.asarray(self.tree.query_ball_point(c,self.r),int)
            if len(idx)<self.min_stage1:
                idx=np.atleast_1d(self.tree.query(c,k=min(self.min_stage1,len(self.X)))[1])
            cf=_fit_centered(self.X,self.y,c,idx,1); self.a[i]=cf[0]; self.b[i]=cf[1:3]; self.counts[i]=len(idx)
        return self
    def predict_with_gradient(self,Q):
        Q=np.asarray(Q,float)
        if self.eps==0:
            return LocalPolynomial2D(1,r=self.r,min_points=self.min_stage1).fit(self.X,self.y).predict_with_gradient(Q)
        pred=np.empty(len(Q)); grad=np.empty((len(Q),2)); ns=np.empty(len(Q),int)
        for i,q in enumerate(Q):
            idx=np.asarray(self.tree.query_ball_point(q,self.eps),int)
            if len(idx)<self.min_stage2:
                idx=np.atleast_1d(self.tree.query(q,k=min(self.min_stage2,len(self.X)))[1])
            planes=self.a[idx]+np.sum(self.b[idx]*(q-self.X[idx]),axis=1)
            pred[i]=float(np.mean(planes)); grad[i]=np.mean(self.b[idx],axis=0); ns[i]=len(idx)
        return pred,grad,{'mean_stage1_neighbours':float(np.mean(self.counts)),'mean_stage2_centres':float(np.mean(ns))}

class GlobalPoly2D:
    def __init__(self,degree=1): self.degree=int(degree)
    def _phi(self,X):
        X=np.asarray(X,float); x1,x2=X[:,0],X[:,1]
        if self.degree==1: return np.column_stack([np.ones(len(X)),x1,x2])
        return np.column_stack([np.ones(len(X)),x1,x2,x1*x1,x1*x2,x2*x2])
    def fit(self,X,y):
        self.coef=np.linalg.lstsq(self._phi(X),np.asarray(y,float),rcond=None)[0]; return self
    def predict_with_gradient(self,Q):
        Q=np.asarray(Q,float); pred=self._phi(Q)@self.coef
        if self.degree==1: grad=np.tile(self.coef[1:3],(len(Q),1))
        else:
            x1,x2=Q[:,0],Q[:,1]; c=self.coef
            grad=np.column_stack([c[1]+2*c[3]*x1+c[4]*x2,c[2]+c[4]*x1+2*c[5]*x2])
        return pred,grad,{}

class TensorSpline2D:
    def __init__(self,n_knots=6,alpha=.1,degree=3): self.n_knots=int(n_knots); self.alpha=float(alpha); self.degree=int(degree)
    def _basis_fit(self,X):
        self.s1=SplineTransformer(n_knots=self.n_knots,degree=self.degree,include_bias=True,extrapolation='linear')
        self.s2=SplineTransformer(n_knots=self.n_knots,degree=self.degree,include_bias=True,extrapolation='linear')
        b1=self.s1.fit_transform(X[:,[0]]); b2=self.s2.fit_transform(X[:,[1]])
        return np.einsum('ij,ik->ijk',b1,b2).reshape(len(X),-1)
    def _basis(self,X):
        b1=self.s1.transform(X[:,[0]]); b2=self.s2.transform(X[:,[1]])
        return np.einsum('ij,ik->ijk',b1,b2).reshape(len(X),-1)
    def fit(self,X,y):
        X=np.asarray(X,float); self.X=X; A=self._basis_fit(X)
        self.model=Ridge(alpha=self.alpha,fit_intercept=False,solver='cholesky').fit(A,np.asarray(y,float)); return self
    def predict(self,Q): return self.model.predict(self._basis(np.asarray(Q,float)))
    def predict_with_gradient(self,Q,h=1e-4):
        Q=np.asarray(Q,float); pred=self.predict(Q); grad=np.empty((len(Q),2))
        for j in range(2):
            qp=Q.copy(); qm=Q.copy(); qp[:,j]+=h; qm[:,j]-=h
            grad[:,j]=(self.predict(qp)-self.predict(qm))/(2*h)
        return pred,grad,{}

class ComponentwiseTensorSpline2D:
    def __init__(self,n_knots=6,alpha=.1,gap_factor=6.0,min_component=35):
        self.n_knots=int(n_knots); self.alpha=float(alpha); self.gap_factor=float(gap_factor); self.min_component=int(min_component)
    def fit(self,X,y):
        self.X=np.asarray(X,float); self.y=np.asarray(y,float); self.tree=cKDTree(self.X)
        if np.isinf(self.gap_factor): labels=np.zeros(len(self.X),int)
        else:
            nn=self.tree.query(self.X,k=2)[0][:,1]; scale=float(np.median(nn[nn>0])) if np.any(nn>0) else 1.0
            labels=DBSCAN(eps=self.gap_factor*scale,min_samples=1).fit_predict(self.X)
        # merge tiny components to nearest large component; if none, use one component
        ids,counts=np.unique(labels,return_counts=True); large=ids[counts>=self.min_component]
        if len(large)==0: labels=np.zeros(len(self.X),int); large=np.array([0])
        elif len(large)<len(ids):
            cent={lab:self.X[labels==lab].mean(axis=0) for lab in large}
            for lab in ids:
                if lab in large: continue
                pts=np.where(labels==lab)[0]; c=self.X[pts].mean(axis=0)
                tgt=min(large,key=lambda z:np.linalg.norm(c-cent[z])); labels[pts]=tgt
        # relabel sequential
        uniq=np.unique(labels); mp={z:i for i,z in enumerate(uniq)}; labels=np.array([mp[z] for z in labels])
        self.labels=labels; self.models={}
        for lab in np.unique(labels):
            idx=np.where(labels==lab)[0]
            self.models[int(lab)]=TensorSpline2D(self.n_knots,self.alpha).fit(self.X[idx],self.y[idx])
        return self
    def _query_labels(self,Q):
        idx=self.tree.query(np.asarray(Q,float),k=1)[1]; return self.labels[np.asarray(idx,int)]
    def predict_with_gradient(self,Q):
        Q=np.asarray(Q,float); labs=self._query_labels(Q); pred=np.empty(len(Q)); grad=np.empty((len(Q),2))
        for lab in np.unique(labs):
            idx=np.where(labs==lab)[0]; p,g,_=self.models[int(lab)].predict_with_gradient(Q[idx]); pred[idx]=p; grad[idx]=g
        return pred,grad,{'n_support_components':int(len(self.models))}
    def predict(self,Q):
        Q=np.asarray(Q,float); labs=self._query_labels(Q); pred=np.empty(len(Q))
        for lab in np.unique(labs):
            idx=np.where(labs==lab)[0]; pred[idx]=self.models[int(lab)].predict(Q[idx])
        return pred

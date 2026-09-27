from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import numpy as np

Array=np.ndarray

@dataclass(frozen=True)
class BivariateDGP:
    code:str
    label:str
    mean:Callable[[Array],Array]
    gradient:Callable[[Array],Array]
    family:str
    notes:str=''
    support:Tuple[float,float]=(-2.0,2.0)

def _lc(z):
    z=np.asarray(z,float)
    return np.logaddexp(z,-z)-np.log(2.0)

def _lin(x): return x[:,0]+x[:,1]
def _gconst(x): return np.column_stack([np.ones(len(x)),np.ones(len(x))])

def _quadratic(a=.30):
    def m(x): return x[:,0]+x[:,1]+a*(x[:,0]**2+x[:,1]**2)
    def g(x): return np.column_stack([1+2*a*x[:,0],1+2*a*x[:,1]])
    return m,g

def _interaction(a=.50):
    def m(x): return x[:,0]+x[:,1]+a*x[:,0]*x[:,1]
    def g(x): return np.column_stack([1+a*x[:,1],1+a*x[:,0]])
    return m,g

def _cubic(a=.15):
    def m(x): return x[:,0]+x[:,1]+a*x[:,0]**3
    def g(x): return np.column_stack([1+3*a*x[:,0]**2,np.ones(len(x))])
    return m,g

def _sin_axis(cycles=1.0,amp=1.0):
    omega=cycles*np.pi/2.0
    def m(x): return x[:,0]+x[:,1]+(amp/omega)*np.sin(omega*x[:,0])
    def g(x): return np.column_stack([1+amp*np.cos(omega*x[:,0]),np.ones(len(x))])
    return m,g

def _sin_diag(cycles=2.0,amp=1.0):
    omega=cycles*np.pi/2.0
    s=np.sqrt(2.0)
    def m(x):
        z=(x[:,0]+x[:,1])/s
        return x[:,0]+x[:,1]+(amp/omega)*np.sin(omega*z)
    def g(x):
        z=(x[:,0]+x[:,1])/s
        d=(amp/s)*np.cos(omega*z)
        return np.column_stack([1+d,1+d])
    return m,g

def _transition_axis(delta=.15,amp=1.0,center=0.0):
    def m(x): return x[:,0]+x[:,1]+amp*delta*_lc((x[:,0]-center)/delta)
    def g(x): return np.column_stack([1+amp*np.tanh((x[:,0]-center)/delta),np.ones(len(x))])
    return m,g

def _transition_diag(delta=.15,amp=1.0,center=0.0):
    s=np.sqrt(2.0)
    def m(x):
        z=(x[:,0]+x[:,1])/s-center
        return x[:,0]+x[:,1]+amp*delta*_lc(z/delta)
    def g(x):
        z=(x[:,0]+x[:,1])/s-center
        d=(amp/s)*np.tanh(z/delta)
        return np.column_stack([1+d,1+d])
    return m,g

def _bump(width=.30,center=(0.0,0.0),deriv_scale=1.0):
    c=np.asarray(center,float); A=deriv_scale*width*np.sqrt(np.e)
    def m(x):
        u=x-c; q=np.sum(u*u,axis=1)
        return x[:,0]+x[:,1]+A*np.exp(-0.5*q/(width*width))
    def g(x):
        u=x-c; q=np.sum(u*u,axis=1); e=np.exp(-0.5*q/(width*width))
        d=-(A/(width*width))*u*e[:,None]
        return np.column_stack([np.ones(len(x)),np.ones(len(x))])+d
    return m,g

def _two_trans(delta=.15,amp=.70):
    def m(x):
        z1=(x[:,0]+.65)/delta; z2=(x[:,0]-.65)/delta
        return x[:,0]+x[:,1]+amp*delta*(_lc(z1)-_lc(z2))
    def g(x):
        d=amp*(np.tanh((x[:,0]+.65)/delta)-np.tanh((x[:,0]-.65)/delta))
        return np.column_stack([1+d,np.ones(len(x))])
    return m,g

def _mixed():
    omega=np.pi
    delta=.30
    def m(x):
        return x[:,0]+x[:,1]+.35*x[:,0]*x[:,1]+(.55/omega)*np.sin(omega*x[:,0])+.45*delta*_lc(x[:,1]/delta)
    def g(x):
        g1=1+.35*x[:,1]+.55*np.cos(omega*x[:,0])
        g2=1+.35*x[:,0]+.45*np.tanh(x[:,1]/delta)
        return np.column_stack([g1,g2])
    return m,g

_dq=_quadratic(); _di=_interaction(); _dc=_cubic();
_s1=_sin_axis(1); _s4=_sin_axis(4); _sd2=_sin_diag(2); _sd4=_sin_diag(4)
_tb=_transition_axis(.60); _ts=_transition_axis(.15); _td=_transition_diag(.15)
_bb=_bump(.60); _bn=_bump(.22); _bo=_bump(.30,(.85,-.70)); _tt=_two_trans(); _mx=_mixed()

DGP_REGISTRY:Dict[str,BivariateDGP]={
 'B01':BivariateDGP('B01','Linear plane',_lin,_gconst,'linear_control'),
 'B02':BivariateDGP('B02','Separable quadratic',*_dq,'polynomial'),
 'B03':BivariateDGP('B03','Bilinear interaction',*_di,'interaction'),
 'B04':BivariateDGP('B04','Cubic in x1',*_dc,'polynomial'),
 'B05':BivariateDGP('B05','Low-frequency x1 sinusoid',*_s1,'frequency'),
 'B06':BivariateDGP('B06','High-frequency x1 sinusoid',*_s4,'frequency'),
 'B07':BivariateDGP('B07','Diagonal sinusoid',*_sd2,'frequency_interaction'),
 'B08':BivariateDGP('B08','High-frequency diagonal sinusoid',*_sd4,'frequency_interaction'),
 'B09':BivariateDGP('B09','Broad vertical slope transition',*_tb,'transition'),
 'B10':BivariateDGP('B10','Sharp vertical slope transition',*_ts,'transition'),
 'B11':BivariateDGP('B11','Sharp diagonal slope transition',*_td,'transition_interaction'),
 'B12':BivariateDGP('B12','Broad central bump',*_bb,'local_feature'),
 'B13':BivariateDGP('B13','Narrow central bump',*_bn,'local_feature'),
 'B14':BivariateDGP('B14','Off-centre local bump',*_bo,'local_feature'),
 'B15':BivariateDGP('B15','Two vertical slope transitions',*_tt,'multiple_regimes'),
 'B16':BivariateDGP('B16','Mixed interaction and anisotropy',*_mx,'mixed'),
}

SUPPORT_LABELS={
 'uniform':'Uniform square',
 'normal_corr':'Correlated centre-dense',
 'skewed':'Skewed independent',
 'vertical_soft50':'Vertical centre retention 50%',
 'vertical_soft10':'Vertical centre retention 10%',
 'vertical_soft02':'Vertical centre retention 2%',
 'vertical_gap40':'Vertical hard gap 0.40',
 'vertical_gap70':'Vertical hard gap 0.70',
 'diag_soft10':'Diagonal retention 10%',
 'diag_soft02':'Diagonal retention 2%',
 'diag_gap40':'Diagonal hard gap 0.40',
 'diag_gap70':'Diagonal hard gap 0.70',
 'hole_soft10':'Central-hole retention 10%',
 'hole_soft02':'Central-hole retention 2%',
 'hole_gap55':'Central hard hole 0.55',
 'two_clusters_x':'Two x-separated clusters',
 'two_clusters_diag':'Two diagonal clusters',
 'manifold_diag':'Near-diagonal manifold',
}

def _accept_rejection(rng,n,proposal,accept,batch=4096):
    out=[]; total=0; attempts=0
    while total<n:
        x=proposal(max(batch,2*(n-total)))
        p=np.asarray(accept(x),float)
        keep=rng.random(len(x))<p
        a=x[keep]
        if len(a): out.append(a); total+=len(a)
        attempts+=1
        if attempts>10000: raise RuntimeError('support sampler failed')
    return np.vstack(out)[:n]

def sample_x(n:int,seed:int,design:str)->Array:
    rng=np.random.default_rng(seed); lo,hi=-2.,2.
    if design=='uniform': return rng.uniform(lo,hi,size=(n,2))
    if design=='skewed': return lo+(hi-lo)*rng.beta(2.0,5.0,size=(n,2))
    if design=='normal_corr':
        cov=np.array([[1,.75],[.75,1.]])
        return _accept_rejection(rng,n,lambda m:rng.multivariate_normal([0,0],cov,size=m),
            lambda x:np.all((x>=lo)&(x<=hi),axis=1).astype(float))
    if design.startswith('vertical_soft'):
        retain={'vertical_soft50':.50,'vertical_soft10':.10,'vertical_soft02':.02}[design]
        return _accept_rejection(rng,n,lambda m:rng.uniform(lo,hi,size=(m,2)),
            lambda x:np.where(np.abs(x[:,0])<.35,retain,1.0))
    if design.startswith('vertical_gap'):
        g={'vertical_gap40':.40,'vertical_gap70':.70}[design]
        return _accept_rejection(rng,n,lambda m:rng.uniform(lo,hi,size=(m,2)),
            lambda x:(np.abs(x[:,0])>=g).astype(float))
    if design.startswith('diag_'):
        def z(x): return np.abs((x[:,0]+x[:,1])/np.sqrt(2.0))
        if design=='diag_soft10': retain=.10; return _accept_rejection(rng,n,lambda m:rng.uniform(lo,hi,size=(m,2)),lambda x:np.where(z(x)<.35,retain,1.0))
        if design=='diag_soft02': retain=.02; return _accept_rejection(rng,n,lambda m:rng.uniform(lo,hi,size=(m,2)),lambda x:np.where(z(x)<.35,retain,1.0))
        g={'diag_gap40':.40,'diag_gap70':.70}[design]
        return _accept_rejection(rng,n,lambda m:rng.uniform(lo,hi,size=(m,2)),lambda x:(z(x)>=g).astype(float))
    if design.startswith('hole_'):
        rad=lambda x:np.sqrt(np.sum(x*x,axis=1))
        if design=='hole_soft10': return _accept_rejection(rng,n,lambda m:rng.uniform(lo,hi,size=(m,2)),lambda x:np.where(rad(x)<.55,.10,1.0))
        if design=='hole_soft02': return _accept_rejection(rng,n,lambda m:rng.uniform(lo,hi,size=(m,2)),lambda x:np.where(rad(x)<.55,.02,1.0))
        return _accept_rejection(rng,n,lambda m:rng.uniform(lo,hi,size=(m,2)),lambda x:(rad(x)>=.55).astype(float))
    if design in ('two_clusters_x','two_clusters_diag'):
        means=[np.array([-1.,0.]),np.array([1.,0.])] if design=='two_clusters_x' else [np.array([-.85,-.85]),np.array([.85,.85])]
        def prop(m):
            lab=rng.integers(0,2,size=m); x=np.empty((m,2))
            for j in (0,1):
                idx=lab==j; x[idx]=rng.normal(means[j],.34,size=(idx.sum(),2))
            return x
        return _accept_rejection(rng,n,prop,lambda x:np.all((x>=lo)&(x<=hi),axis=1).astype(float))
    if design=='manifold_diag':
        def prop(m):
            x1=rng.uniform(-1.8,1.8,size=m); x2=x1+rng.normal(0,.22,size=m)
            return np.column_stack([x1,x2])
        return _accept_rejection(rng,n,prop,lambda x:np.all((x>=lo)&(x<=hi),axis=1).astype(float))
    raise ValueError(f'unknown design {design}')

def generate_sample(dgp_code,n,noise_sd,seed,design):
    X=sample_x(int(n),int(seed),design)
    d=DGP_REGISTRY[dgp_code]
    mu=np.asarray(d.mean(X),float)
    rng=np.random.default_rng(int(seed)+987654321)
    y=mu+rng.normal(0,float(noise_sd),size=len(X))
    return X,y,mu

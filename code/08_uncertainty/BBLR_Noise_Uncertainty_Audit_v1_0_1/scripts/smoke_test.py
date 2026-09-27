#!/usr/bin/env python3
from pathlib import Path
import sys, tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from bblr_noise_audit.dgp import generate,g_true
from bblr_noise_audit.estimators import bblr_predict,llr_predict,adaptive_local_poly_predict,smoothing_spline_predict

def main():
    x,y,xt,yt=generate(123,0.25,120,80)
    q=np.linspace(-1.8,1.8,31)
    p0,g0,_=bblr_predict(x,y,q,.38,0.0)
    pl,gl,_=llr_predict(x,y,q,.38)
    assert np.max(np.abs(p0-pl))<1e-12 and np.max(np.abs(g0-gl))<1e-12
    p,g,_=bblr_predict(x,y,q,.38,.26); assert np.all(np.isfinite(p)) and np.all(np.isfinite(g))
    p,g=adaptive_local_poly_predict(x,y,q,25,2); assert np.all(np.isfinite(g))
    p,g=smoothing_spline_predict(x,y,q,1e-2); assert np.all(np.isfinite(g))
    print("exact_zero_branch_error",float(np.max(np.abs(g0-gl))))
    print("PASS_NOISE_UNCERTAINTY_SMOKE")
if __name__=='__main__': main()

#!/usr/bin/env python3
from pathlib import Path
import sys,platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, scipy, matplotlib
from bblr_noise_audit.config import *
from bblr_noise_audit.dgp import m_true,g_true

def main():
    assert len(NOISE_GRID)==9 and np.allclose(NOISE_GRID,np.arange(.10,.501,.05))
    x=np.linspace(-2,2,1001); h=1e-6
    num=(m_true(x+h)-m_true(x-h))/(2*h)
    err=float(np.max(np.abs(num-g_true(x))))
    assert err<1e-6, err
    assert 0.25 in NOISE_GRID and 0.10 in NOISE_GRID and 0.50 in NOISE_GRID
    print("BBLR_NOISE_UNCERTAINTY_PREFLIGHT")
    print(f"python={platform.python_version()} numpy={np.__version__} scipy={scipy.__version__} pandas={pd.__version__} matplotlib={matplotlib.__version__}")
    print(f"noise_grid={NOISE_GRID.tolist()}")
    print(f"n_dev={N_DEV} n_test={N_TEST} grid={N_GRID} cv={CV_REPEATS}x{CV_FOLDS}")
    print(f"r_grid={R_GRID.tolist()}")
    print(f"eps_grid={EPS_GRID.tolist()}")
    print(f"k_grid={K_GRID.tolist()}")
    print(f"spline_lam_grid={SPLINE_LAM_GRID.tolist()}")
    print(f"analytic_derivative_check={err:.3g}")
    print("PASS_FOR_NOISE_UNCERTAINTY_PRODUCTION")
if __name__=='__main__': main()

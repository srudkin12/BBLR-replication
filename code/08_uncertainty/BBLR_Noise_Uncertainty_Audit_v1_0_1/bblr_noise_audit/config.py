from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np

# Avoid nested BLAS/OpenMP parallelism under ProcessPoolExecutor.
for _k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_k, "1")

NOISE_GRID = np.array([0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50], dtype=float)
R_GRID = np.array([0.08,0.12,0.18,0.26,0.38,0.55,0.80,1.20,2.00,4.00], dtype=float)
EPS_GRID = np.array([0.00,0.04,0.08,0.12,0.18,0.26,0.38,0.55,0.80,1.20,2.00,4.00], dtype=float)
K_GRID = np.array([15,25,40,60,90,130,180,250,350,500], dtype=int)
# Manuscript documents spline tuning from 1e-6 through 1e3. The decimal log grid is frozen here.
SPLINE_LAM_GRID = np.logspace(-6,3,10)
N_DEV = 1000
N_TEST = 3000
N_GRID = 401
CV_REPEATS = 5
CV_FOLDS = 5
MIN_NEIGH = 15
SUPPORT = (-2.0, 2.0)
FREQ_C = 4.0
OMEGA = FREQ_C * np.pi / 2.0  # 2*pi
BASE_SEED = 2026091404
METHODS = ("BBLR","LLR","AKLLR","AKLP2","SPLINE")

@dataclass(frozen=True)
class Design:
    sigma: float
    rep: int
    n_dev: int = N_DEV
    n_test: int = N_TEST

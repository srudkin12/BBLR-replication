from __future__ import annotations
import numpy as np
from .config import SUPPORT, OMEGA

def m_true(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x + np.sin(OMEGA*x)/OMEGA

def g_true(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 + np.cos(OMEGA*x)

def generate(seed: int, sigma: float, n_dev: int, n_test: int):
    rng = np.random.default_rng(seed)
    lo, hi = SUPPORT
    x_dev = rng.uniform(lo, hi, size=n_dev)
    y_dev = m_true(x_dev) + rng.normal(0.0, sigma, size=n_dev)
    x_test = rng.uniform(lo, hi, size=n_test)
    y_test = m_true(x_test) + rng.normal(0.0, sigma, size=n_test)
    return x_dev, y_dev, x_test, y_test

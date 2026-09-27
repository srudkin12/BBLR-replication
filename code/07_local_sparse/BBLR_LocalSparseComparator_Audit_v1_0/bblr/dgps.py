from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray
ALPHA0 = 0.50

@dataclass(frozen=True)
class DGP:
    code: str
    label: str
    family: str
    constant_tangent_alpha: bool


def _beta_active(active_dim: int) -> Array:
    active_dim = int(active_dim)
    if active_dim <= 0:
        return np.zeros(0, dtype=float)
    base = np.array([1.0, -0.75], float)
    if active_dim <= 2:
        return base[:active_dim]
    # Keep total linear-gradient magnitude stable across all-active dimension designs.
    b = np.array([1.0 if j % 2 == 0 else -0.75 for j in range(active_dim)], float)
    return b / np.linalg.norm(b) * 1.25


def mean_gradient(X: Array, code: str, active_dim: int):
    X = np.asarray(X, float)
    n, d = X.shape
    a = int(active_dim)
    if a < 0 or a > d:
        raise ValueError(f"active_dim={a} incompatible with d={d}")

    if code == "flat":
        return np.full(n, ALPHA0, dtype=float), np.zeros((n, d), dtype=float)

    beta = _beta_active(a)
    g = np.zeros((n, d), dtype=float)
    if a:
        g[:, :a] = beta[None, :]
    m = np.full(n, ALPHA0, dtype=float)
    if a:
        m = m + X[:, :a] @ beta

    if code in ("linear", "linear_all"):
        return m, g

    if a < 1:
        raise ValueError(f"DGP {code} requires active_dim >= 1")

    if code == "smooth":
        omega = np.pi
        m = m + (0.55 / omega) * np.sin(omega * X[:, 0])
        g[:, 0] += 0.55 * np.cos(omega * X[:, 0])
        if a >= 2:
            m = m + (0.35 / omega) * np.sin(omega * X[:, 1])
            g[:, 1] += 0.35 * np.cos(omega * X[:, 1])
        return m, g

    if code == "transition":
        delta = 0.18
        z = X[:, 0] / delta
        m = m + 0.80 * delta * np.log(np.cosh(z))
        g[:, 0] += 0.80 * np.tanh(z)
        return m, g

    if code == "highfreq":
        omega = 4 * np.pi
        m = m + (0.75 / omega) * np.sin(omega * X[:, 0])
        g[:, 0] += 0.75 * np.cos(omega * X[:, 0])
        return m, g

    if code == "smooth_all":
        amp = 0.50 / np.sqrt(a)
        omega = np.pi
        for j in range(a):
            m = m + (amp / omega) * np.sin(omega * X[:, j])
            g[:, j] += amp * np.cos(omega * X[:, j])
        return m, g

    raise ValueError(code)


def tangent_alpha(X: Array, mean: Array, grad: Array) -> Array:
    return np.asarray(mean, float) - np.sum(np.asarray(X, float) * np.asarray(grad, float), axis=1)


def reference_alpha_truth(code: str, d: int, active_dim: int) -> float:
    X0 = np.zeros((1, int(d)), dtype=float)
    m0, _ = mean_gradient(X0, code, int(active_dim))
    return float(m0[0])


def _accept_rejection(rng, n, proposal, accept, batch=4096):
    out = []
    total = 0
    tries = 0
    while total < n:
        x = proposal(max(batch, 2 * (n - total)))
        p = np.asarray(accept(x), float)
        keep = rng.random(len(x)) < p
        y = x[keep]
        if len(y):
            out.append(y)
            total += len(y)
        tries += 1
        if tries > 10000:
            raise RuntimeError("support sampler failed")
    return np.vstack(out)[:n]


def sample_x(n: int, d: int, seed: int, design: str) -> Array:
    rng = np.random.default_rng(int(seed))
    n = int(n)
    d = int(d)
    if design == "uniform":
        return rng.uniform(-1, 1, size=(n, d))
    if design == "shifted_uniform":
        return rng.uniform(0, 2, size=(n, d))
    if design == "corr_nuisance":
        if d < 2:
            return rng.uniform(-1, 1, size=(n, d))
        z0 = rng.normal(size=n)
        Z = np.empty((n, d))
        Z[:, 0] = z0
        rho = 0.72
        for j in range(1, d):
            Z[:, j] = rho * z0 + np.sqrt(1 - rho * rho) * rng.normal(size=n)
        return np.tanh(Z / 1.2)
    if design in ("aligned_soft10", "orthogonal_soft10"):
        axis = 0 if design == "aligned_soft10" else (2 if d >= 3 else 1)
        return _accept_rejection(
            rng,
            n,
            lambda m: rng.uniform(-1, 1, size=(m, d)),
            lambda x: np.where(np.abs(x[:, axis]) < 0.25, 0.10, 1.0),
        )
    raise ValueError(design)


def generate_sample(code, n, d, active_dim, noise_sd, seed, design):
    X = sample_x(n, d, seed, design)
    mu, g = mean_gradient(X, code, active_dim)
    rng = np.random.default_rng(int(seed) + 987654321)
    y = mu + rng.normal(0, float(noise_sd), size=len(X))
    return X, y, mu, g, tangent_alpha(X, mu, g)

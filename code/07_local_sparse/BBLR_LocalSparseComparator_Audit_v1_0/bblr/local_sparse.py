from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

Array = np.ndarray


def _soft_threshold(x: float, lam: float) -> float:
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


def _weighted_lasso_cd(Z: Array, y: Array, lam: float, weights: Array, init: Array | None = None,
                       tol: float = 1e-8, max_iter: int = 2000) -> Array:
    """Solve 0.5/n ||y-Zb||^2 + lam * sum_j weights_j |b_j|.

    Z and y are already centered; columns of Z are locally standardized.
    The intercept is therefore unpenalized and reconstructed outside this solver.
    """
    Z = np.asarray(Z, float)
    y = np.asarray(y, float)
    n, d = Z.shape
    if d == 0:
        return np.zeros(0, dtype=float)
    b = np.zeros(d, dtype=float) if init is None else np.asarray(init, float).copy()
    v = np.mean(Z * Z, axis=0)
    good = v > 1e-14
    b[~good] = 0.0
    res = y - Z @ b
    for _ in range(int(max_iter)):
        max_delta = 0.0
        max_abs = float(np.max(np.abs(b))) if d else 0.0
        for j in range(d):
            if not good[j]:
                continue
            zj = Z[:, j]
            old = b[j]
            res += zj * old
            rho = float(np.mean(zj * res))
            new = _soft_threshold(rho, float(lam) * float(weights[j])) / float(v[j])
            b[j] = new
            res -= zj * new
            max_delta = max(max_delta, abs(new - old))
        if max_delta <= float(tol) * (1.0 + max_abs):
            break
    return b


class AdaptiveLocalLinearLassoND:
    """Hard-ball local-linear adaptive Lasso with an unpenalized intercept.

    At each query q, the local model is y ~= a(q) + b(q)'(X-q). The slopes are
    standardized within the local neighborhood before penalization. Adaptive
    weights are based on the unpenalized local-linear slope in standardized
    coordinates. The tuning parameter is specified as a fraction of the
    query-specific weighted lambda_max; lambda_frac=0 exactly returns the
    local-linear least-squares fit under the same neighborhood safeguard.
    """

    def __init__(self, r: float, lambda_frac: float = 0.10, min_points: int | None = None,
                 adaptive_gamma: float = 1.0, tau_frac: float = 0.05,
                 tol: float = 1e-8, max_iter: int = 2000):
        self.r = float(r)
        self.lambda_frac = float(lambda_frac)
        self.min_points = min_points
        self.adaptive_gamma = float(adaptive_gamma)
        self.tau_frac = float(tau_frac)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def fit(self, X: Array, y: Array):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        self.d = self.X.shape[1]
        self.min_points = int(self.min_points or max(self.d + 5, 15))
        self.tree = cKDTree(self.X)
        return self

    def _idx(self, q: Array) -> Array:
        idx = np.asarray(self.tree.query_ball_point(q, self.r), int)
        if len(idx) < self.min_points:
            kk = min(self.min_points, len(self.X))
            idx = np.atleast_1d(self.tree.query(q, k=kk)[1]).astype(int)
        return idx

    def _local_components(self, q: Array):
        idx = self._idx(q)
        D = self.X[idx] - q
        y = self.y[idx]
        dm = np.mean(D, axis=0)
        ym = float(np.mean(y))
        Z = D - dm
        yc = y - ym
        scale = np.sqrt(np.mean(Z * Z, axis=0))
        scale = np.where(scale > 1e-10, scale, 1.0)
        Zs = Z / scale
        # Initial unpenalized local-linear slope in standardized coordinates.
        gamma0 = np.linalg.lstsq(Zs, yc, rcond=None)[0]
        yscale = max(float(np.std(yc)), 1e-8)
        tau = self.tau_frac * yscale + 1e-10
        weights = 1.0 / np.power(np.abs(gamma0) + tau, self.adaptive_gamma)
        corr = np.abs((Zs.T @ yc) / max(len(yc), 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            candidate = np.where(weights > 0, corr / weights, 0.0)
        lambda_max = float(np.max(candidate)) if len(candidate) else 0.0
        return idx, D, dm, ym, Zs, yc, scale, gamma0, weights, lambda_max

    def _fit_one(self, q: Array, lambda_frac: float):
        idx, D, dm, ym, Zs, yc, scale, gamma0, weights, lambda_max = self._local_components(q)
        frac = float(lambda_frac)
        if frac <= 0.0 or lambda_max <= 0.0:
            gamma = gamma0
        else:
            gamma = _weighted_lasso_cd(
                Zs, yc, frac * lambda_max, weights,
                init=np.zeros(self.d), tol=self.tol, max_iter=self.max_iter,
            )
        b = gamma / scale
        a = ym - float(dm @ b)
        b[np.abs(b) < 1e-12] = 0.0
        return a, b, len(idx), lambda_max

    def _fit_path_one(self, q: Array, fractions):
        idx, D, dm, ym, Zs, yc, scale, gamma0, weights, lambda_max = self._local_components(q)
        fracs = [float(x) for x in fractions]
        out = {}
        # Solve positive penalties from strongest to weakest using warm starts.
        positives = sorted({f for f in fracs if f > 0}, reverse=True)
        gamma = np.zeros(self.d, dtype=float)
        for f in positives:
            if lambda_max <= 0:
                gamma = gamma0.copy()
            else:
                gamma = _weighted_lasso_cd(
                    Zs, yc, f * lambda_max, weights,
                    init=gamma, tol=self.tol, max_iter=self.max_iter,
                )
            b = gamma / scale
            b[np.abs(b) < 1e-12] = 0.0
            out[f] = (ym - float(dm @ b), b.copy())
        if any(f <= 0 for f in fracs):
            b0 = gamma0 / scale
            b0[np.abs(b0) < 1e-12] = 0.0
            out[0.0] = (ym - float(dm @ b0), b0.copy())
        return out, len(idx), lambda_max

    def predict_with_gradient(self, Q: Array):
        Q = np.asarray(Q, float)
        pred = np.empty(len(Q), float)
        grad = np.empty((len(Q), self.d), float)
        counts = np.empty(len(Q), int)
        lmax = np.empty(len(Q), float)
        for i, q in enumerate(Q):
            a, b, nloc, lm = self._fit_one(q, self.lambda_frac)
            pred[i] = a
            grad[i] = b
            counts[i] = nloc
            lmax[i] = lm
        return pred, grad, {
            'mean_neighbours': float(np.mean(counts)),
            'min_neighbours': int(np.min(counts)),
            'median_lambda_max': float(np.median(lmax)),
        }

    def predict_path(self, Q: Array, fractions):
        """Return prediction/gradient arrays for all fractions with one local path per query."""
        Q = np.asarray(Q, float)
        fracs = [float(x) for x in fractions]
        pred = {f: np.empty(len(Q), float) for f in fracs}
        grad = {f: np.empty((len(Q), self.d), float) for f in fracs}
        counts = np.empty(len(Q), int)
        lmax = np.empty(len(Q), float)
        for i, q in enumerate(Q):
            path, nloc, lm = self._fit_path_one(q, fracs)
            counts[i] = nloc
            lmax[i] = lm
            for f in fracs:
                key = 0.0 if f <= 0 else f
                a, b = path[key]
                pred[f][i] = a
                grad[f][i] = b
        diag = {
            'mean_neighbours': float(np.mean(counts)),
            'min_neighbours': int(np.min(counts)),
            'median_lambda_max': float(np.median(lmax)),
        }
        return pred, grad, diag

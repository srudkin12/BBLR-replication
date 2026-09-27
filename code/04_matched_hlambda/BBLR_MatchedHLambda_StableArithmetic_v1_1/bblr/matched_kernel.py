"""Faithful matched-H_lambda one-stage local-linear derivative estimator.

The kernel is defined exactly from BBLR Paper 1 v0.9:

    H_lambda(t) = int L(v) K1(t-lambda v) dv,
    W_lambda(t) = I_lambda^{-1} H_lambda(t) / t,

with the continuous extension at t=0.  L is the uniform Stage-2 kernel and
K1(t)=3t/2 on |t|<=1 is the Stage-1 derivative equivalent kernel.

Two finite-sample modes are exposed:

* ``strict``: use W_lambda literally whenever the weighted local-linear design
  is nonsingular; otherwise return NaN and flag the query as invalid.
* ``safe``: use the same W_lambda fit when at least ``min_points`` positive
  kernel weights are available and the moment matrix is nonsingular; otherwise
  fall back to the same k-nearest-neighbour unweighted local-linear safeguard
  used by production BBLR Stage 1.

The strict estimator is the primary mechanism comparator.  The safeguarded
version is a secondary finite-sample diagnostic, not a redefinition of W_lambda.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numba import njit

Array = np.ndarray


def I_lambda(lam: float) -> float:
    """Normalising integral I_lambda from Proposition 3.7.

    The lambda=0 value is the continuous limit, giving the uniform Stage-1
    kernel W_0(t)=1/2 1{|t|<=1}.
    """
    lam = float(lam)
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    if lam == 0.0:
        return 3.0
    if abs(lam - 1.0) <= 1e-12:
        return 1.5
    if lam < 1.0:
        val = 2.0 * lam + (1.0 - lam * lam) * math.log((1.0 + lam) / (1.0 - lam))
        return 3.0 * val / (4.0 * lam)
    val = 2.0 * lam - (lam * lam - 1.0) * math.log((lam + 1.0) / (lam - 1.0))
    return 3.0 * val / (4.0 * lam)


def b_lambda(lam: float) -> float:
    return 1.0 / I_lambda(lam)


def gamma_lambda(lam: float) -> float:
    b = b_lambda(lam)
    return (3.0 * b - 1.0 - float(lam) ** 2) / 6.0


@njit(cache=True)
def _w_scalar(t: float, lam: float, Ilam: float) -> float:
    at = abs(t)
    if lam == 0.0:
        return 0.5 if at <= 1.0 else 0.0
    if at > 1.0 + lam:
        return 0.0
    if at < 1e-12:
        if lam < 1.0 - 1e-12:
            return 1.5 / Ilam
        if lam > 1.0 + 1e-12:
            return 0.0
        return 0.75 / Ilam
    a = max(-1.0, (t - 1.0) / lam)
    b = min(1.0, (t + 1.0) / lam)
    if a >= b:
        return 0.0
    H = 0.75 * (t * (b - a) - 0.5 * lam * (b * b - a * a))
    w = H / (Ilam * t)
    if w < 0.0 and w > -1e-13:
        w = 0.0
    return w


def W_lambda(t: Array | float, lam: float) -> Array:
    """Evaluate W_lambda vectorwise."""
    arr = np.asarray(t, dtype=float)
    I = I_lambda(lam)
    flat = arr.reshape(-1)
    out = np.empty_like(flat)
    for i, v in enumerate(flat):
        out[i] = _w_scalar(float(v), float(lam), float(I))
    return out.reshape(arr.shape)


@njit(cache=True)
def _knn_positions_sorted(xs: Array, q: float, k: int) -> Array:
    n = xs.size
    k = min(max(k, 1), n)
    p = np.searchsorted(xs, q)
    lo = p - 1
    hi = p
    out = np.empty(k, dtype=np.int64)
    m = 0
    while m < k:
        if lo < 0:
            out[m] = hi; hi += 1
        elif hi >= n:
            out[m] = lo; lo -= 1
        else:
            dl = abs(xs[lo] - q)
            dr = abs(xs[hi] - q)
            if dl <= dr:
                out[m] = lo; lo -= 1
            else:
                out[m] = hi; hi += 1
        m += 1
    return out


@njit(cache=True)
def _unweighted_knn_fit(xs: Array, ys: Array, q: float, k: int) -> Tuple[float, float]:
    pos = _knn_positions_sorted(xs, q, k)
    n = float(pos.size)
    s1 = 0.0; s2 = 0.0; t0 = 0.0; t1 = 0.0
    for p in pos:
        z = xs[p] - q
        y = ys[p]
        s1 += z; s2 += z * z; t0 += y; t1 += z * y
    det = n * s2 - s1 * s1
    if abs(det) <= 1e-14 * max(1.0, n * s2):
        return t0 / n, 0.0
    a = (t0 * s2 - t1 * s1) / det
    b = (n * t1 - s1 * t0) / det
    return a, b


@njit(cache=True)
def _predict_core(xs: Array, ys: Array, q: Array, r: float, lam: float,
                  Ilam: float, min_points: int, safe: bool):
    nq = q.size
    pred = np.empty(nq, dtype=np.float64)
    slope = np.empty(nq, dtype=np.float64)
    valid = np.ones(nq, dtype=np.bool_)
    fallback = np.zeros(nq, dtype=np.bool_)
    positive_count = np.zeros(nq, dtype=np.int64)
    support_radius = r * (1.0 + lam)

    for i in range(nq):
        x0 = q[i]
        left = np.searchsorted(xs, x0 - support_radius)
        right = np.searchsorted(xs, x0 + support_radius, side='right')
        s0 = 0.0; s1 = 0.0; s2 = 0.0; t0 = 0.0; t1 = 0.0
        cnt = 0
        for j in range(left, right):
            z = xs[j] - x0
            t = z / r
            w = _w_scalar(t, lam, Ilam)
            if w > 1e-14:
                y = ys[j]
                s0 += w
                s1 += w * z
                s2 += w * z * z
                t0 += w * y
                t1 += w * z * y
                cnt += 1
        positive_count[i] = cnt
        det = s0 * s2 - s1 * s1
        nonsingular = (cnt >= 2 and s0 > 0.0 and abs(det) > 1e-14 * max(1.0, s0 * s2))
        if safe and (cnt < min_points or not nonsingular):
            a, b = _unweighted_knn_fit(xs, ys, x0, min_points)
            pred[i] = a; slope[i] = b; fallback[i] = True
        elif nonsingular:
            pred[i] = (t0 * s2 - t1 * s1) / det
            slope[i] = (s0 * t1 - s1 * t0) / det
        else:
            pred[i] = np.nan; slope[i] = np.nan; valid[i] = False
    return pred, slope, valid, fallback, positive_count


@dataclass
class MatchedKernelDiagnostics:
    valid: Array
    fallback: Array
    positive_weight_count: Array


class MatchedHLocalLinear1D:
    """One-stage local-linear estimator using the matched W_lambda kernel."""

    def __init__(self, r: float, lam: float, min_points: int = 15, safe: bool = False):
        self.r = float(r)
        self.lam = float(lam)
        self.min_points = int(min_points)
        self.safe = bool(safe)
        if self.r <= 0:
            raise ValueError("r must be positive")
        if self.lam < 0:
            raise ValueError("lambda must be non-negative")
        if self.min_points < 2:
            raise ValueError("min_points must be at least 2")
        self.Ilam = I_lambda(self.lam)

    def fit(self, x: Array, y: Array) -> "MatchedHLocalLinear1D":
        x = np.asarray(x, float).reshape(-1)
        y = np.asarray(y, float).reshape(-1)
        if x.size != y.size or x.size == 0:
            raise ValueError("x/y length mismatch or empty sample")
        order = np.argsort(x, kind="mergesort")
        self.xs_ = np.ascontiguousarray(x[order])
        self.ys_ = np.ascontiguousarray(y[order])
        return self

    def predict_with_slope(self, x_new: Array):
        q = np.ascontiguousarray(np.asarray(x_new, float).reshape(-1))
        pred, slope, valid, fallback, cnt = _predict_core(
            self.xs_, self.ys_, q, self.r, self.lam, self.Ilam,
            self.min_points, self.safe,
        )
        return pred, slope, {
            "valid": valid,
            "fallback": fallback,
            "positive_weight_count": cnt,
        }

    def predict(self, x_new: Array) -> Array:
        return self.predict_with_slope(x_new)[0]

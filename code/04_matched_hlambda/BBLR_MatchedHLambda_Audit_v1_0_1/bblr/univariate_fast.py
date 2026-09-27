"""Fast one-dimensional estimators for the BBLR Monte Carlo laboratory.

The reference implementations in :mod:`bblr.univariate` are intentionally
transparent and loop over local least-squares fits.  This module preserves the
same estimands but exploits one-dimensional ordering and prefix sums for the
uniform-ball local-linear and mean-aggregation BBLR calculations.

The fast classes are used by the HPC runner.  They are tested against the
reference implementations on identical data.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

Array = np.ndarray


def _prefix(v: Array) -> Array:
    v = np.asarray(v, dtype=float)
    return np.concatenate(([0.0], np.cumsum(v, dtype=float)))


def _range_sum(pref: Array, left: Array, right: Array) -> Array:
    return pref[right] - pref[left]


class _SortedData1D:
    def _fit_sorted(self, x: Array, y: Array) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.size != y.size:
            raise ValueError("x and y must have the same length")
        if x.size == 0:
            raise ValueError("empty sample")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("x/y contain non-finite values")
        order = np.argsort(x, kind="mergesort")
        self.x_sorted_ = x[order]
        self.y_sorted_ = y[order]
        self.order_ = order
        self.n_ = x.size
        self.px_ = _prefix(self.x_sorted_)
        self.px2_ = _prefix(self.x_sorted_ ** 2)
        self.py_ = _prefix(self.y_sorted_)
        self.pxy_ = _prefix(self.x_sorted_ * self.y_sorted_)

    def _bounds(self, q: Array, radius: float) -> Tuple[Array, Array, Array]:
        q = np.asarray(q, dtype=float).reshape(-1)
        left = np.searchsorted(self.x_sorted_, q - radius, side="left")
        right = np.searchsorted(self.x_sorted_, q + radius, side="right")
        return left, right, right - left

    def _knn_sorted_positions(self, q: float, k: int) -> Array:
        """Return positions in sorted x for exact 1D kNN, deterministic on ties."""
        k = min(max(int(k), 1), self.n_)
        xs = self.x_sorted_
        p = int(np.searchsorted(xs, q, side="left"))
        lo = p - 1
        hi = p
        out = []
        while len(out) < k:
            if lo < 0:
                out.append(hi); hi += 1; continue
            if hi >= self.n_:
                out.append(lo); lo -= 1; continue
            dl = abs(xs[lo] - q)
            dr = abs(xs[hi] - q)
            if dl <= dr:  # stable left-first tie break
                out.append(lo); lo -= 1
            else:
                out.append(hi); hi += 1
        return np.asarray(out, dtype=int)

    def _local_linear_from_bounds(self, q: Array, left: Array, right: Array) -> Tuple[Array, Array, Array]:
        q = np.asarray(q, dtype=float).reshape(-1)
        n = (right - left).astype(float)
        sx = _range_sum(self.px_, left, right)
        sx2 = _range_sum(self.px2_, left, right)
        sy = _range_sum(self.py_, left, right)
        sxy = _range_sum(self.pxy_, left, right)
        # centred moments around q
        s1 = sx - n * q
        s2 = sx2 - 2.0 * q * sx + n * q * q
        t0 = sy
        t1 = sxy - q * sy
        det = n * s2 - s1 * s1
        good = np.abs(det) > 1e-14
        a = np.full_like(q, np.nan, dtype=float)
        b = np.full_like(q, np.nan, dtype=float)
        a[good] = (t0[good] * s2[good] - t1[good] * s1[good]) / det[good]
        b[good] = (n[good] * t1[good] - s1[good] * t0[good]) / det[good]
        return a, b, good

    def _local_linear_knn(self, q: float, k: int) -> Tuple[float, float, int]:
        pos = self._knn_sorted_positions(q, k)
        z = self.x_sorted_[pos] - q
        y = self.y_sorted_[pos]
        n = float(len(pos))
        s1 = float(np.sum(z)); s2 = float(np.sum(z*z))
        t0 = float(np.sum(y)); t1 = float(np.sum(z*y))
        det = n*s2 - s1*s1
        if abs(det) <= 1e-14:
            # Degenerate x support: prediction remains defined, slope is set to
            # zero rather than allowing numerical explosion.
            return float(np.mean(y)), 0.0, len(pos)
        a = (t0*s2 - t1*s1)/det
        b = (n*t1 - s1*t0)/det
        return float(a), float(b), len(pos)


class FastLocalLinear1D(_SortedData1D):
    """Uniform-ball local-linear regression using vectorised prefix sums."""

    def __init__(self, r: float, min_points: int = 15):
        self.r = float(r)
        self.min_points = int(min_points)

    def fit(self, x: Array, y: Array) -> "FastLocalLinear1D":
        if self.r <= 0:
            raise ValueError("r must be positive")
        if self.min_points < 2:
            raise ValueError("min_points must be at least 2")
        self._fit_sorted(x, y)
        return self

    def predict_with_slope(self, x_new: Array) -> Tuple[Array, Array, Dict[str, Array]]:
        q = np.asarray(x_new, dtype=float).reshape(-1)
        left, right, counts = self._bounds(q, self.r)
        fallback = counts < self.min_points
        # Compute radius fits where there are enough points.
        a, b, good = self._local_linear_from_bounds(q, left, right)
        fallback = fallback | (~good)
        for i in np.flatnonzero(fallback):
            ai, bi, ci = self._local_linear_knn(float(q[i]), self.min_points)
            a[i] = ai; b[i] = bi; counts[i] = ci
        return a, b, {"neighbour_count": counts.astype(int), "fallback": fallback}

    def predict(self, x_new: Array) -> Array:
        return self.predict_with_slope(x_new)[0]


class FastBBLR1D(_SortedData1D):
    """Fast mean-aggregation two-stage BBLR in one dimension.

    Stage 1 uses the same exact uniform-ball local-linear estimator as
    :class:`FastLocalLinear1D`.  Stage 2 exploits

        mean[a_k + b_k (x0-x_k)]
        = mean[a_k - b_k x_k] + x0 mean[b_k]

    so each epsilon query is a pair of prefix-sum operations.
    """

    def __init__(self, r: float, eps: float, min_points_stage1: int = 15,
                 min_centres_stage2: int = 1):
        self.r = float(r)
        self.eps = float(eps)
        self.min_points_stage1 = int(min_points_stage1)
        self.min_centres_stage2 = int(min_centres_stage2)

    def fit(self, x: Array, y: Array) -> "FastBBLR1D":
        if self.r <= 0 or self.eps < 0:
            raise ValueError("r must be positive and eps non-negative")
        if self.min_points_stage1 < 2 or self.min_centres_stage2 < 1:
            raise ValueError("invalid minimum neighbour count")
        # Keep sorted data/prefix sums for Stage 1.
        self._fit_sorted(x, y)
        ll = FastLocalLinear1D(self.r, self.min_points_stage1).fit(self.x_sorted_, self.y_sorted_)
        self.stage1_query_estimator_ = ll
        level, slope, diag = ll.predict_with_slope(self.x_sorted_)
        self.stage1_level_sorted_ = level
        self.stage1_slope_sorted_ = slope
        self.stage1_count_sorted_ = diag["neighbour_count"]
        self.stage1_fallback_sorted_ = diag["fallback"]
        intercept0 = level - slope * self.x_sorted_
        self.pc0_ = _prefix(intercept0)
        self.pb_ = _prefix(slope)
        return self

    def set_eps(self, eps: float, min_centres_stage2: int | None = None) -> "FastBBLR1D":
        if eps < 0:
            raise ValueError("eps must be non-negative")
        self.eps = float(eps)
        if min_centres_stage2 is not None:
            if int(min_centres_stage2) < 1:
                raise ValueError("min_centres_stage2 must be at least 1")
            self.min_centres_stage2 = int(min_centres_stage2)
        return self

    def predict_with_slope(self, x_new: Array) -> Tuple[Array, Array, Dict[str, Array]]:
        q = np.asarray(x_new, dtype=float).reshape(-1)

        # eps=0 is defined as no second-stage smoothing.  This makes LLR an
        # exact nested special case of BBLR: BBLR(r,0) == LLR(r).
        if self.eps == 0.0:
            pred, slope, qdiag = self.stage1_query_estimator_.predict_with_slope(q)
            return pred, slope, {
                "stage2_centre_count": np.zeros(q.size, dtype=int),
                "stage2_fallback": np.zeros(q.size, dtype=bool),
                "stage2_disabled": np.ones(q.size, dtype=bool),
                "stage1_query_count": qdiag["neighbour_count"],
                "stage1_query_fallback": qdiag["fallback"],
                "stage1_count": self.stage1_count_sorted_.copy(),
                "stage1_fallback": self.stage1_fallback_sorted_.copy(),
            }

        left, right, counts = self._bounds(q, self.eps)
        fallback = counts < self.min_centres_stage2
        n = counts.astype(float)
        sum_c0 = _range_sum(self.pc0_, left, right)
        sum_b = _range_sum(self.pb_, left, right)
        pred = np.full_like(q, np.nan, dtype=float)
        slope = np.full_like(q, np.nan, dtype=float)
        good = ~fallback
        pred[good] = sum_c0[good] / n[good] + q[good] * (sum_b[good] / n[good])
        slope[good] = sum_b[good] / n[good]
        for i in np.flatnonzero(fallback):
            pos = self._knn_sorted_positions(float(q[i]), self.min_centres_stage2)
            b = self.stage1_slope_sorted_[pos]
            c0 = self.stage1_level_sorted_[pos] - b * self.x_sorted_[pos]
            slope[i] = float(np.mean(b))
            pred[i] = float(np.mean(c0) + q[i] * slope[i])
            counts[i] = len(pos)
        diag = {
            "stage2_centre_count": counts.astype(int),
            "stage2_fallback": fallback,
            "stage1_count": self.stage1_count_sorted_.copy(),
            "stage1_fallback": self.stage1_fallback_sorted_.copy(),
        }
        return pred, slope, diag

    def predict(self, x_new: Array) -> Array:
        return self.predict_with_slope(x_new)[0]


class TricubeLocalLinear1D:
    """Fixed-bandwidth tricube-weighted local-linear comparator.

    This is deliberately included because BBLR should be compared not only
    with a hard-ball local line but also with conventional smooth local
    weighting.  If fewer than ``min_points`` fall inside the bandwidth, exact
    1D kNN is used and the effective bandwidth expands to the furthest selected
    point.
    """

    def __init__(self, h: float, min_points: int = 15):
        self.h = float(h); self.min_points = int(min_points)

    def fit(self, x: Array, y: Array) -> "TricubeLocalLinear1D":
        if self.h <= 0 or self.min_points < 2:
            raise ValueError("invalid h/min_points")
        x = np.asarray(x, float).reshape(-1); y = np.asarray(y, float).reshape(-1)
        order = np.argsort(x, kind="mergesort")
        self.x_ = x[order]; self.y_ = y[order]; self.n_ = len(x)
        return self

    def _knn_positions(self, q: float, k: int) -> Array:
        p = int(np.searchsorted(self.x_, q, side="left")); lo=p-1; hi=p; out=[]
        k=min(max(int(k),1), self.n_)
        while len(out)<k:
            if lo<0: out.append(hi); hi+=1; continue
            if hi>=self.n_: out.append(lo); lo-=1; continue
            dl=abs(self.x_[lo]-q); dr=abs(self.x_[hi]-q)
            if dl<=dr: out.append(lo); lo-=1
            else: out.append(hi); hi+=1
        return np.asarray(out,int)

    def predict_with_slope(self, x_new: Array):
        q=np.asarray(x_new,float).reshape(-1)
        pred=np.empty_like(q); slope=np.empty_like(q)
        counts=np.empty(q.size,int); fallback=np.zeros(q.size,bool)
        for i,x0 in enumerate(q):
            left=np.searchsorted(self.x_,x0-self.h,'left'); right=np.searchsorted(self.x_,x0+self.h,'right')
            pos=np.arange(left,right,dtype=int)
            fb=len(pos)<self.min_points
            if fb:
                pos=self._knn_positions(float(x0),self.min_points)
            z=self.x_[pos]-x0
            scale=self.h
            if fb and len(z):
                scale=max(scale,float(np.max(np.abs(z)))*(1.0+1e-12))
            u=np.abs(z)/scale
            w=np.where(u<1.0,(1.0-u**3)**3,0.0)
            # Guard against all-zero weights due to pathological numerical ties.
            if np.sum(w)<=1e-14:
                w=np.ones_like(z)
            s0=float(np.sum(w)); s1=float(np.sum(w*z)); s2=float(np.sum(w*z*z))
            t0=float(np.sum(w*self.y_[pos])); t1=float(np.sum(w*z*self.y_[pos]))
            det=s0*s2-s1*s1
            if abs(det)<=1e-14:
                pred[i]=t0/s0; slope[i]=0.0
            else:
                pred[i]=(t0*s2-t1*s1)/det
                slope[i]=(s0*t1-s1*t0)/det
            counts[i]=len(pos); fallback[i]=fb
        return pred,slope,{"neighbour_count":counts,"fallback":fallback}

    def predict(self,x_new:Array)->Array:
        return self.predict_with_slope(x_new)[0]


class KNNLocalPolynomial1D:
    """Adaptive k-nearest-neighbour tricube local polynomial estimator.

    The neighbour count k is global but the effective bandwidth is query-specific:
    it expands in sparse regions and contracts in dense regions.  Degree 1 gives
    adaptive local-linear regression (AKLLR); degree 2 gives adaptive local
    quadratic regression (AKLP2).
    """

    def __init__(self, k: int, degree: int = 1):
        self.k = int(k)
        self.degree = int(degree)
        if self.degree < 1:
            raise ValueError("degree must be at least 1")

    def fit(self, x: Array, y: Array) -> "KNNLocalPolynomial1D":
        x = np.asarray(x, float).reshape(-1)
        y = np.asarray(y, float).reshape(-1)
        if x.size != y.size or x.size == 0:
            raise ValueError("invalid x/y")
        if self.k < self.degree + 2:
            raise ValueError("k must be at least degree+2")
        order = np.argsort(x, kind="mergesort")
        self.x_ = x[order]
        self.y_ = y[order]
        self.n_ = x.size
        self.k_eff_ = min(self.k, self.n_)
        return self

    def _knn_positions(self, q: float) -> Array:
        k = self.k_eff_
        p = int(np.searchsorted(self.x_, q, side="left"))
        lo = p - 1
        hi = p
        out = []
        while len(out) < k:
            if lo < 0:
                out.append(hi); hi += 1; continue
            if hi >= self.n_:
                out.append(lo); lo -= 1; continue
            dl = abs(self.x_[lo] - q)
            dr = abs(self.x_[hi] - q)
            if dl <= dr:
                out.append(lo); lo -= 1
            else:
                out.append(hi); hi += 1
        return np.asarray(out, dtype=int)

    def predict_with_slope(self, x_new: Array):
        q = np.asarray(x_new, float).reshape(-1)
        pred = np.empty_like(q)
        slope = np.empty_like(q)
        counts = np.full(q.size, self.k_eff_, dtype=int)
        bandwidth = np.empty_like(q)
        fallback = np.zeros(q.size, dtype=bool)

        for i, x0 in enumerate(q):
            pos = self._knn_positions(float(x0))
            z = self.x_[pos] - x0
            h = float(np.max(np.abs(z))) if len(z) else 0.0
            if h <= 1e-14:
                pred[i] = float(np.mean(self.y_[pos]))
                slope[i] = 0.0
                bandwidth[i] = 0.0
                fallback[i] = True
                continue

            # Slightly enlarge h so the furthest neighbour receives a small
            # positive weight rather than an exact zero at u=1.
            h_eff = h * (1.0 + 1e-10)
            u = np.abs(z) / h_eff
            w = np.where(u < 1.0, (1.0 - u**3)**3, 0.0)
            if np.sum(w) <= 1e-14:
                w = np.ones_like(z)
                fallback[i] = True

            A = np.column_stack([z**j for j in range(self.degree + 1)])
            sw = np.sqrt(w)
            Aw = A * sw[:, None]
            yw = self.y_[pos] * sw
            coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
            pred[i] = float(coef[0])
            slope[i] = float(coef[1])
            bandwidth[i] = h

        return pred, slope, {
            "neighbour_count": counts,
            "fallback": fallback,
            "adaptive_bandwidth": bandwidth,
        }

    def predict(self, x_new: Array) -> Array:
        return self.predict_with_slope(x_new)[0]

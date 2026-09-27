"""Clean one-dimensional BBLR laboratory.

The original repository mixed estimator development, neighbour-search
optimisation, DGP generation and manuscript tables.  This module deliberately
keeps the 1D problem small and auditable.  It provides:

* exact sorted-array radius and k-nearest-neighbour searches in 1D;
* a transparent two-stage BBLR1D estimator;
* a single-stage local-linear comparator using the same Stage-1 radius;
* a registry of smooth and non-smooth DGPs with analytical derivatives;
* support designs that separate functional-form difficulty from X geometry.

No confidence intervals are produced here.  The purpose of this module is to
establish finite-sample behaviour and derivative recovery before inference is
revisited.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


class SortedRadiusIndex1D:
    """Exact radius / kNN index for scalar covariates."""

    def __init__(self, x: Array):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size == 0:
            raise ValueError("x must contain at least one observation")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non-finite values")
        self.order = np.argsort(x, kind="mergesort")
        self.x_sorted = x[self.order]
        self.n = x.size

    def radius(self, q: float, r: float) -> Array:
        if r < 0:
            raise ValueError("r must be non-negative")
        left = np.searchsorted(self.x_sorted, q - r, side="left")
        right = np.searchsorted(self.x_sorted, q + r, side="right")
        return self.order[left:right]

    def knn(self, q: float, k: int) -> Array:
        k = int(k)
        if k < 1:
            raise ValueError("k must be at least 1")
        k = min(k, self.n)
        # For the 1D development version, exactness is more important than an
        # elaborate micro-optimisation.  Stable sorting also makes ties
        # deterministic.
        d = np.abs(self.x_sorted - q)
        pos = np.argsort(d, kind="mergesort")[:k]
        return self.order[pos]

    def radius_or_knn(self, q: float, r: float, min_points: int) -> Tuple[Array, bool]:
        idx = self.radius(q, r)
        fallback = len(idx) < min_points
        if fallback:
            idx = self.knn(q, min_points)
        return idx, fallback


def _local_linear_centered(x: Array, y: Array, centre: float) -> Tuple[float, float]:
    """Fit y = alpha + beta * (x-centre) using least squares."""
    z = np.asarray(x, dtype=float).reshape(-1) - centre
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


class LocalLinear1D:
    """Single-stage radius local-linear regression, used as the key comparator."""

    def __init__(self, r: float, min_points: int = 15):
        self.r = float(r)
        self.min_points = int(min_points)

    def fit(self, x: Array, y: Array) -> "LocalLinear1D":
        self.x_ = np.asarray(x, dtype=float).reshape(-1)
        self.y_ = np.asarray(y, dtype=float).reshape(-1)
        if len(self.x_) != len(self.y_):
            raise ValueError("x and y must have the same length")
        if self.min_points < 2:
            raise ValueError("min_points must be at least 2 for a local line")
        self.index_ = SortedRadiusIndex1D(self.x_)
        return self

    def predict_with_slope(self, x_new: Array) -> Tuple[Array, Array, Dict[str, Array]]:
        q = np.asarray(x_new, dtype=float).reshape(-1)
        pred = np.empty_like(q)
        slope = np.empty_like(q)
        counts = np.empty(q.size, dtype=int)
        fallback = np.zeros(q.size, dtype=bool)
        for i, x0 in enumerate(q):
            idx, fb = self.index_.radius_or_knn(x0, self.r, self.min_points)
            a, b = _local_linear_centered(self.x_[idx], self.y_[idx], x0)
            pred[i] = a
            slope[i] = b
            counts[i] = len(idx)
            fallback[i] = fb
        return pred, slope, {"neighbour_count": counts, "fallback": fallback}

    def predict(self, x_new: Array) -> Array:
        return self.predict_with_slope(x_new)[0]


class LocalPolynomial1D:
    """Single-stage centred local polynomial comparator.

    The coefficient on (x-x0) is the derivative estimate at x0.  Degree 2 is
    particularly useful for asking whether BBLR's second smoothing stage is
    doing more than a conventional local-quadratic bias reduction.
    """

    def __init__(self, r: float, degree: int = 2, min_points: int = 15):
        self.r = float(r)
        self.degree = int(degree)
        self.min_points = int(min_points)
        if self.degree < 1:
            raise ValueError("degree must be at least 1")

    def fit(self, x: Array, y: Array) -> "LocalPolynomial1D":
        self.x_ = np.asarray(x, dtype=float).reshape(-1)
        self.y_ = np.asarray(y, dtype=float).reshape(-1)
        if len(self.x_) != len(self.y_):
            raise ValueError("x and y must have the same length")
        if self.min_points < self.degree + 1:
            raise ValueError("min_points must be at least degree+1")
        self.index_ = SortedRadiusIndex1D(self.x_)
        return self

    def predict_with_slope(self, x_new: Array) -> Tuple[Array, Array, Dict[str, Array]]:
        q = np.asarray(x_new, dtype=float).reshape(-1)
        pred = np.empty_like(q)
        slope = np.empty_like(q)
        counts = np.empty(q.size, dtype=int)
        fallback = np.zeros(q.size, dtype=bool)
        for i, x0 in enumerate(q):
            idx, fb = self.index_.radius_or_knn(x0, self.r, self.min_points)
            z = self.x_[idx] - x0
            A = np.column_stack([z**j for j in range(self.degree + 1)])
            coef, *_ = np.linalg.lstsq(A, self.y_[idx], rcond=None)
            pred[i] = coef[0]
            slope[i] = coef[1]
            counts[i] = len(idx)
            fallback[i] = fb
        return pred, slope, {"neighbour_count": counts, "fallback": fallback}

    def predict(self, x_new: Array) -> Array:
        return self.predict_with_slope(x_new)[0]


class BBLR1D:
    """Transparent two-stage Ball-Based Linear Regression for one covariate.

    Stage 1
    -------
    At every training location x_k, fit a local line inside radius r:
        y ~= a_k + b_k (x - x_k).

    Stage 2
    -------
    At a query x_0, collect training centres inside radius eps and aggregate
    their local slopes.  Prediction is the corresponding aggregation of each
    local plane evaluated at x_0.  Mean and median aggregation are supported.

    The two radii are intentionally distinct.  Setting r == eps reproduces the
    single-scale specification used in much of the legacy code; allowing them
    to differ exposes the bias-variance role of the second smoothing stage.
    """

    def __init__(
        self,
        r: float,
        eps: float,
        min_points_stage1: int = 15,
        min_centres_stage2: int = 1,
        aggregation: str = "mean",
    ):
        self.r = float(r)
        self.eps = float(eps)
        self.min_points_stage1 = int(min_points_stage1)
        self.min_centres_stage2 = int(min_centres_stage2)
        self.aggregation = aggregation

    def fit(self, x: Array, y: Array) -> "BBLR1D":
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if self.r <= 0 or self.eps < 0:
            raise ValueError("r must be positive and eps must be non-negative")
        if self.min_points_stage1 < 2:
            raise ValueError("min_points_stage1 must be at least 2")
        if self.min_centres_stage2 < 1:
            raise ValueError("min_centres_stage2 must be at least 1")
        if self.aggregation not in {"mean", "median"}:
            raise ValueError("aggregation must be 'mean' or 'median'")

        self.x_ = x
        self.y_ = y
        self.index_ = SortedRadiusIndex1D(x)
        n = len(x)
        self.stage1_level_ = np.empty(n)
        self.stage1_slope_ = np.empty(n)
        self.stage1_count_ = np.empty(n, dtype=int)
        self.stage1_fallback_ = np.zeros(n, dtype=bool)

        for k, centre in enumerate(x):
            idx, fb = self.index_.radius_or_knn(centre, self.r, self.min_points_stage1)
            a, b = _local_linear_centered(x[idx], y[idx], centre)
            self.stage1_level_[k] = a
            self.stage1_slope_[k] = b
            self.stage1_count_[k] = len(idx)
            self.stage1_fallback_[k] = fb
        return self

    def _aggregate(self, values: Array) -> float:
        if self.aggregation == "mean":
            return float(np.mean(values))
        return float(np.median(values))

    def predict_with_slope(self, x_new: Array) -> Tuple[Array, Array, Dict[str, Array]]:
        q = np.asarray(x_new, dtype=float).reshape(-1)

        # Exact nesting convention: eps=0 switches Stage 2 off.  The resulting
        # estimator is the Stage-1 local-linear regression evaluated directly
        # at each query, not a nearest-centre approximation.
        if self.eps == 0.0:
            ll = LocalLinear1D(self.r, self.min_points_stage1).fit(self.x_, self.y_)
            pred, slope, diag = ll.predict_with_slope(q)
            return pred, slope, {
                "stage2_centre_count": np.zeros(q.size, dtype=int),
                "stage2_fallback": np.zeros(q.size, dtype=bool),
                "stage2_disabled": np.ones(q.size, dtype=bool),
                "stage1_query_count": diag["neighbour_count"],
                "stage1_query_fallback": diag["fallback"],
                "stage1_count": self.stage1_count_.copy(),
                "stage1_fallback": self.stage1_fallback_.copy(),
            }

        pred = np.empty_like(q)
        slope = np.empty_like(q)
        counts = np.empty(q.size, dtype=int)
        fallback = np.zeros(q.size, dtype=bool)

        for i, x0 in enumerate(q):
            idx = self.index_.radius(x0, self.eps)
            fb = len(idx) < self.min_centres_stage2
            if fb:
                idx = self.index_.knn(x0, self.min_centres_stage2)

            local_predictions = self.stage1_level_[idx] + self.stage1_slope_[idx] * (x0 - self.x_[idx])
            pred[i] = self._aggregate(local_predictions)
            slope[i] = self._aggregate(self.stage1_slope_[idx])
            counts[i] = len(idx)
            fallback[i] = fb

        diagnostics = {
            "stage2_centre_count": counts,
            "stage2_fallback": fallback,
            "stage1_count": self.stage1_count_.copy(),
            "stage1_fallback": self.stage1_fallback_.copy(),
        }
        return pred, slope, diagnostics

    def predict(self, x_new: Array) -> Array:
        return self.predict_with_slope(x_new)[0]


@dataclass(frozen=True)
class UnivariateDGP:
    code: str
    label: str
    mean: Callable[[Array], Array]
    derivative: Callable[[Array], Array]
    support: Tuple[float, float] = (-2.0, 2.0)
    nondifferentiable_points: Tuple[float, ...] = ()
    notes: str = ""
    family: str = "legacy_anchor"
    parameter_name: str = ""
    parameter_value: float = np.nan
    secondary_parameter_name: str = ""
    secondary_parameter_value: float = np.nan


def _step_mean(x: Array) -> Array:
    cuts = np.array([-1.0, 0.0, 1.0])
    values = np.array([-0.5, 0.75, -0.25, 1.0])
    return values[np.digitize(x, cuts)]


DGP_REGISTRY: Dict[str, UnivariateDGP] = {
    "U01": UnivariateDGP("U01", "Linear", lambda x: x, lambda x: np.ones_like(x)),
    "U02": UnivariateDGP("U02", "Quadratic", lambda x: x**2, lambda x: 2*x),
    "U03": UnivariateDGP("U03", "Cubic", lambda x: 0.5*x**3, lambda x: 1.5*x**2),
    "U04": UnivariateDGP("U04", "Sinusoid", lambda x: 2*np.sin(np.pi*x), lambda x: 2*np.pi*np.cos(np.pi*x)),
    "U05": UnivariateDGP("U05", "Linear plus sine", lambda x: x + np.sin(x), lambda x: 1 + np.cos(x)),
    "U06": UnivariateDGP("U06", "Mild exponential", lambda x: np.exp(0.6*x), lambda x: 0.6*np.exp(0.6*x)),
    "U07": UnivariateDGP("U07", "Smooth transition", lambda x: np.tanh(2*x), lambda x: 2/np.cosh(2*x)**2),
    "U08": UnivariateDGP(
        "U08", "Local Gaussian bump",
        lambda x: x + 1.5*np.exp(-4*(x-0.5)**2),
        lambda x: 1 - 12*(x-0.5)*np.exp(-4*(x-0.5)**2),
    ),
    "U09": UnivariateDGP(
        "U09", "Absolute-value kink", lambda x: np.abs(x), lambda x: np.sign(x),
        nondifferentiable_points=(0.0,), notes="Derivative undefined at x=0."
    ),
    "U10": UnivariateDGP(
        "U10", "Slope kink",
        lambda x: np.where(x < 0, -1.0*x, 2.0*x),
        lambda x: np.where(x < 0, -1.0, 2.0),
        nondifferentiable_points=(0.0,), notes="Derivative undefined at x=0."
    ),
    "U11": UnivariateDGP(
        "U11", "Level jump",
        lambda x: x + 1.5*(x >= 0),
        lambda x: np.ones_like(x),
        nondifferentiable_points=(0.0,), notes="Mean is discontinuous at x=0; score derivative away from the jump."
    ),
    "U12": UnivariateDGP(
        "U12", "Step function", _step_mean, lambda x: np.zeros_like(x),
        nondifferentiable_points=(-1.0, 0.0, 1.0),
        notes="Derivative is zero between jumps and undefined at each jump."
    ),

}


# ---------------------------------------------------------------------------
# Controlled smooth DGP families for the BBLR phase-map experiment.
#
# These families deliberately vary one geometric property at a time.  The
# primary objective is not to accumulate arbitrary named functions but to map
# where second-stage coefficient smoothing helps or harms derivative recovery.
# ---------------------------------------------------------------------------

def _log_cosh(z: Array) -> Array:
    """Numerically stable log(cosh(z))."""
    z = np.asarray(z, dtype=float)
    return np.logaddexp(z, -z) - np.log(2.0)


def _register_controlled_families(registry: Dict[str, UnivariateDGP]) -> None:
    # Q: curvature/amplitude ladder.  a=0 would reduce to the linear anchor.
    for i, a in enumerate((0.05, 0.10, 0.20, 0.40, 0.80, 1.20), start=1):
        code = f"Q{i:02d}"
        registry[code] = UnivariateDGP(
            code,
            f"Quadratic curvature a={a:g}",
            lambda x, a=a: x + a*x**2,
            lambda x, a=a: 1.0 + 2.0*a*x,
            family="quadratic_curvature",
            parameter_name="a",
            parameter_value=float(a),
            notes="Controlled curvature ladder: m(x)=x+a*x^2.",
        )

    # F: frequency ladder holding the amplitude of slope heterogeneity fixed.
    # Over support [-2,2], cycles=c corresponds to omega=c*pi/2.
    slope_amp = 1.0
    for i, cycles in enumerate((0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00), start=1):
        omega = cycles*np.pi/2.0
        code = f"F{i:02d}"
        registry[code] = UnivariateDGP(
            code,
            f"Slope-frequency cycles={cycles:g}",
            lambda x, omega=omega, A=slope_amp: x + (A/omega)*np.sin(omega*x),
            lambda x, omega=omega, A=slope_amp: 1.0 + A*np.cos(omega*x),
            family="slope_frequency",
            parameter_name="cycles",
            parameter_value=float(cycles),
            secondary_parameter_name="slope_amplitude",
            secondary_parameter_value=float(slope_amp),
            notes="Slope amplitude fixed while spatial frequency changes.",
        )

    # A: amplitude ladder holding spatial frequency fixed at one cycle over support.
    cycles = 1.0
    omega = cycles*np.pi/2.0
    for i, amp in enumerate((0.25, 0.50, 1.00, 1.50, 2.00, 3.00), start=1):
        code = f"A{i:02d}"
        registry[code] = UnivariateDGP(
            code,
            f"Slope-amplitude A={amp:g}",
            lambda x, omega=omega, A=amp: x + (A/omega)*np.sin(omega*x),
            lambda x, omega=omega, A=amp: 1.0 + A*np.cos(omega*x),
            family="slope_amplitude",
            parameter_name="slope_amplitude",
            parameter_value=float(amp),
            secondary_parameter_name="cycles",
            secondary_parameter_value=float(cycles),
            notes="Spatial frequency fixed while slope-heterogeneity amplitude changes.",
        )

    # T: smooth change in marginal effect.  As delta decreases the transition
    # becomes sharper but the two limiting slopes (0 and 2) remain unchanged.
    amp = 1.0
    for i, delta in enumerate((1.20, 0.80, 0.50, 0.30, 0.18, 0.10), start=1):
        code = f"T{i:02d}"
        registry[code] = UnivariateDGP(
            code,
            f"Marginal-effect transition delta={delta:g}",
            lambda x, d=delta, a=amp: x + a*d*_log_cosh(x/d),
            lambda x, d=delta, a=amp: 1.0 + a*np.tanh(x/d),
            family="slope_transition_width",
            parameter_name="delta",
            parameter_value=float(delta),
            secondary_parameter_name="slope_shift",
            secondary_parameter_value=float(amp),
            notes="Asymptotic slopes fixed; transition width changes.",
        )

    # B: local Gaussian bump.  Set the level amplitude so the maximum absolute
    # derivative perturbation equals D regardless of width.
    D = 1.0
    centre = 0.0
    for i, width in enumerate((1.20, 0.80, 0.50, 0.30, 0.18, 0.10), start=1):
        level_amp = D*width*np.sqrt(np.e)
        code = f"B{i:02d}"
        registry[code] = UnivariateDGP(
            code,
            f"Local bump width={width:g}",
            lambda x, w=width, c=centre, A=level_amp: x + A*np.exp(-0.5*((x-c)/w)**2),
            lambda x, w=width, c=centre, A=level_amp: 1.0 - A*(x-c)/(w*w)*np.exp(-0.5*((x-c)/w)**2),
            family="local_bump_width",
            parameter_name="width",
            parameter_value=float(width),
            secondary_parameter_name="max_slope_perturbation",
            secondary_parameter_value=float(D),
            notes="Localisation changes while maximum slope perturbation is held fixed.",
        )

    # L: move the same local feature through the support.  Combined with normal,
    # skewed and gap X designs this tests alignment between function geometry
    # and sampling density.
    width = 0.30
    D = 1.0
    level_amp = D*width*np.sqrt(np.e)
    for i, centre in enumerate((-1.50, -0.75, 0.00, 0.75, 1.50), start=1):
        code = f"L{i:02d}"
        registry[code] = UnivariateDGP(
            code,
            f"Local bump centre={centre:g}",
            lambda x, w=width, c=centre, A=level_amp: x + A*np.exp(-0.5*((x-c)/w)**2),
            lambda x, w=width, c=centre, A=level_amp: 1.0 - A*(x-c)/(w*w)*np.exp(-0.5*((x-c)/w)**2),
            family="local_bump_location",
            parameter_name="centre",
            parameter_value=float(centre),
            secondary_parameter_name="width",
            secondary_parameter_value=float(width),
            notes="Fixed local feature translated through the support.",
        )


_register_controlled_families(DGP_REGISTRY)


def _register_support_regime_dgps(registry: Dict[str, UnivariateDGP]) -> None:
    """Controlled smooth marginal-effect transitions for support-regime mapping.

    m'(x) = 1 + tanh((x-c)/delta), so the two limiting slopes are 0 and 2.
    Width delta and location c vary independently.
    """
    widths = (0.80, 0.40, 0.20, 0.10, 0.05)
    centres = (-0.80, -0.40, 0.00, 0.40, 0.80)
    for wi, delta in enumerate(widths, start=1):
        for ci, centre in enumerate(centres, start=1):
            code = f"R{wi:02d}C{ci:02d}"
            registry[code] = UnivariateDGP(
                code=code,
                label=f"Transition width={delta:g}, centre={centre:g}",
                mean=lambda x, d=delta, c=centre: x + d*_log_cosh((x-c)/d),
                derivative=lambda x, d=delta, c=centre: 1.0 + np.tanh((x-c)/d),
                family="support_transition",
                parameter_name="delta",
                parameter_value=float(delta),
                secondary_parameter_name="centre",
                secondary_parameter_value=float(centre),
                notes="Smooth slope-regime transition with limiting slopes 0 and 2.",
            )


_register_support_regime_dgps(DGP_REGISTRY)


@dataclass(frozen=True)
class SupportDesign1D:
    code: str
    label: str
    family: str
    parameter_name: str = ""
    parameter_value: float = np.nan
    secondary_parameter_name: str = ""
    secondary_parameter_value: float = np.nan
    notes: str = ""


SUPPORT_DESIGN_REGISTRY: Dict[str, SupportDesign1D] = {
    # Hard central gaps.  G00 is exactly uniform support.
    "G00": SupportDesign1D("G00", "Uniform support", "hard_gap",
                           "gap_halfwidth", 0.00),
    "G10": SupportDesign1D("G10", "Hard gap half-width 0.10", "hard_gap",
                           "gap_halfwidth", 0.10),
    "G25": SupportDesign1D("G25", "Hard gap half-width 0.25", "hard_gap",
                           "gap_halfwidth", 0.25),
    "G40": SupportDesign1D("G40", "Hard gap half-width 0.40", "hard_gap",
                           "gap_halfwidth", 0.40),
    "G60": SupportDesign1D("G60", "Hard gap half-width 0.60", "hard_gap",
                           "gap_halfwidth", 0.60),
    "G80": SupportDesign1D("G80", "Hard gap half-width 0.80", "hard_gap",
                           "gap_halfwidth", 0.80),
    # Soft gaps: uniform base density with central-window depletion.
    "D75": SupportDesign1D("D75", "Soft central retention 0.75", "soft_gap",
                           "central_retention", 0.75, "window_halfwidth", 0.40),
    "D50": SupportDesign1D("D50", "Soft central retention 0.50", "soft_gap",
                           "central_retention", 0.50, "window_halfwidth", 0.40),
    "D25": SupportDesign1D("D25", "Soft central retention 0.25", "soft_gap",
                           "central_retention", 0.25, "window_halfwidth", 0.40),
    "D10": SupportDesign1D("D10", "Soft central retention 0.10", "soft_gap",
                           "central_retention", 0.10, "window_halfwidth", 0.40),
    "D02": SupportDesign1D("D02", "Soft central retention 0.02", "soft_gap",
                           "central_retention", 0.02, "window_halfwidth", 0.40),
}


def support_design_metadata(code: str) -> SupportDesign1D:
    if code in SUPPORT_DESIGN_REGISTRY:
        return SUPPORT_DESIGN_REGISTRY[code]
    # Legacy designs remain available for backwards-compatible diagnostics.
    return SupportDesign1D(code, code, "legacy")


def _truncated_normal(rng: np.random.Generator, n: int, lo: float, hi: float, sd: float = 0.8) -> Array:
    chunks = []
    total = 0
    while total < n:
        z = rng.normal(0.0, sd, size=max(2*n, 100))
        z = z[(z >= lo) & (z <= hi)]
        chunks.append(z)
        total += len(z)
    return np.concatenate(chunks)[:n]


def generate_x(n: int, design: str, rng: np.random.Generator, support: Tuple[float, float] = (-2, 2)) -> Array:
    lo, hi = support

    # Controlled support-regime designs.
    if design in SUPPORT_DESIGN_REGISTRY:
        spec = SUPPORT_DESIGN_REGISTRY[design]
        if spec.family == "hard_gap":
            g = float(spec.parameter_value)
            if g < 0 or g >= min(abs(lo), abs(hi)):
                raise ValueError(f"invalid hard gap half-width {g}")
            side = rng.random(n) < 0.5
            x = np.empty(n)
            x[side] = rng.uniform(lo, -g, int(side.sum()))
            x[~side] = rng.uniform(g, hi, int((~side).sum()))
            return x

        if spec.family == "soft_gap":
            retention = float(spec.parameter_value)
            window = float(spec.secondary_parameter_value)
            chunks = []
            total = 0
            while total < n:
                cand = rng.uniform(lo, hi, max(2*n, 200))
                keep_prob = np.where(np.abs(cand) <= window, retention, 1.0)
                keep = rng.random(cand.size) <= keep_prob
                accepted = cand[keep]
                chunks.append(accepted)
                total += accepted.size
            return np.concatenate(chunks)[:n]

    # Legacy support designs retained for reproducibility.
    if design == "uniform":
        return rng.uniform(lo, hi, n)
    if design == "normal":
        return _truncated_normal(rng, n, lo, hi)
    if design == "skewed":
        return lo + (hi-lo) * rng.beta(2.0, 5.0, n)
    if design == "gap":
        choose = rng.random(n) < 0.5
        x = np.empty(n)
        x[choose] = rng.normal(-1.0, 0.28, choose.sum())
        x[~choose] = rng.normal(1.0, 0.28, (~choose).sum())
        bad = (x < lo) | (x > hi)
        while np.any(bad):
            m = bad.sum()
            side = rng.random(m) < 0.5
            z = np.empty(m)
            z[side] = rng.normal(-1.0, 0.28, side.sum())
            z[~side] = rng.normal(1.0, 0.28, (~side).sum())
            x[bad] = z
            bad = (x < lo) | (x > hi)
        return x
    raise ValueError(f"Unknown x design: {design}")

def generate_sample(
    dgp_code: str,
    n: int,
    noise_sd: float,
    seed: int,
    design: str = "uniform",
) -> Tuple[Array, Array, Array]:
    dgp = DGP_REGISTRY[dgp_code]
    rng = np.random.default_rng(seed)
    x = generate_x(n, design, rng, dgp.support)
    mu = dgp.mean(x)
    y = mu + rng.normal(0.0, noise_sd, n)
    return x, y, mu


def differentiable_mask(x: Array, dgp: UnivariateDGP, guard_width: float = 0.1) -> Array:
    x = np.asarray(x, dtype=float)
    mask = np.ones(x.size, dtype=bool)
    for point in dgp.nondifferentiable_points:
        mask &= np.abs(x - point) > guard_width
    return mask


def boundary_mask(x: Array, support: Tuple[float, float], fraction: float = 0.1) -> Array:
    lo, hi = support
    width = fraction * (hi-lo)
    x = np.asarray(x, dtype=float)
    return (x <= lo + width) | (x >= hi - width)


def rmse(est: Array, truth: Array, mask: Optional[Array] = None) -> float:
    est = np.asarray(est, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if mask is not None:
        est = est[mask]
        truth = truth[mask]
    return float(np.sqrt(np.mean((est-truth)**2)))

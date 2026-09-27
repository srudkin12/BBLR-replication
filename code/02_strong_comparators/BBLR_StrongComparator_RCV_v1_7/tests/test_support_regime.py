import numpy as np

from bblr.univariate import (
    DGP_REGISTRY, SUPPORT_DESIGN_REGISTRY, generate_x,
)
from bblr.univariate_fast import KNNLocalPolynomial1D


def test_support_regime_dgp_derivatives_against_finite_difference():
    x = np.linspace(-1.6, 1.6, 101)
    h = 1e-6
    for code in ["R01C01","R03C03","R05C05"]:
        d = DGP_REGISTRY[code]
        numerical = (d.mean(x+h)-d.mean(x-h))/(2*h)
        assert np.allclose(numerical, d.derivative(x), rtol=2e-5, atol=2e-5)


def test_hard_gap_support_contains_no_central_points():
    rng = np.random.default_rng(123)
    x = generate_x(5000, "G60", rng)
    assert np.all((x <= -0.60) | (x >= 0.60))
    assert x.min() >= -2.0 and x.max() <= 2.0


def test_g00_is_uniform_support_range():
    rng = np.random.default_rng(321)
    x = generate_x(10000, "G00", rng)
    assert np.all(x >= -2.0) and np.all(x <= 2.0)
    assert abs(np.mean(x)) < 0.08


def test_soft_gap_reduces_central_mass_monotonically():
    masses = []
    for code in ["D75","D50","D25","D10","D02"]:
        rng = np.random.default_rng(777)
        x = generate_x(30000, code, rng)
        masses.append(np.mean(np.abs(x) <= 0.40))
    assert all(a > b for a,b in zip(masses, masses[1:]))


def test_adaptive_knn_local_linear_exact_on_noiseless_line():
    x = np.linspace(-2,2,151)
    y = 1.5 + 2.25*x
    q = np.array([-1.7,-0.9,0.0,0.8,1.6])
    fit = KNNLocalPolynomial1D(k=25, degree=1).fit(x,y)
    pred, slope, diag = fit.predict_with_slope(q)
    assert np.allclose(pred, 1.5 + 2.25*q, atol=1e-10)
    assert np.allclose(slope, 2.25, atol=1e-10)
    assert np.all(diag["neighbour_count"] == 25)


def test_adaptive_knn_local_quadratic_exact_on_noiseless_quadratic():
    x = np.linspace(-2,2,201)
    y = 0.5 + 1.25*x + 0.8*x*x
    q = np.array([-1.5,-0.5,0.25,1.25])
    fit = KNNLocalPolynomial1D(k=35, degree=2).fit(x,y)
    pred, slope, _ = fit.predict_with_slope(q)
    assert np.allclose(pred, 0.5 + 1.25*q + 0.8*q*q, atol=1e-9)
    assert np.allclose(slope, 1.25 + 1.6*q, atol=1e-8)


def test_support_registry_expected_codes_present():
    expected = {"G00","G10","G25","G40","G60","G80","D75","D50","D25","D10","D02"}
    assert expected.issubset(SUPPORT_DESIGN_REGISTRY)

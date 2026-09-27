import numpy as np

from bblr.univariate import (
    BBLR1D, DGP_REGISTRY, LocalLinear1D, SortedRadiusIndex1D,
    differentiable_mask,
)


def test_radius_search_matches_bruteforce():
    x = np.array([-2.0, -1.0, -0.1, 0.0, 0.4, 1.0, 2.0])
    idx = SortedRadiusIndex1D(x)
    for q in [-1.3, -1.0, 0.0, 0.55, 1.6]:
        for r in [0.0, 0.2, 0.6, 2.5]:
            got = np.sort(idx.radius(q, r))
            expected = np.flatnonzero(np.abs(x-q) <= r + 1e-12)
            assert np.array_equal(got, expected)


def test_knn_matches_distance_sort():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    idx = SortedRadiusIndex1D(x)
    got = idx.knn(0.7, 3)
    expected = np.argsort(np.abs(x-0.7), kind="mergesort")[:3]
    assert np.array_equal(got, expected)


def test_noiseless_linear_local_linear_recovers_exact_slope():
    x = np.linspace(-2, 2, 101)
    y = 3.0 + 1.75*x
    q = np.linspace(-1.8, 1.8, 41)
    model = LocalLinear1D(r=0.35, min_points=5).fit(x, y)
    pred, slope, _ = model.predict_with_slope(q)
    assert np.max(np.abs(slope-1.75)) < 1e-10
    assert np.max(np.abs(pred-(3.0+1.75*q))) < 1e-10


def test_noiseless_linear_bblr_recovers_exact_slope():
    x = np.linspace(-2, 2, 101)
    y = -0.4 + 2.25*x
    q = np.linspace(-1.8, 1.8, 41)
    model = BBLR1D(r=0.35, eps=0.25, min_points_stage1=5).fit(x, y)
    pred, slope, _ = model.predict_with_slope(q)
    assert np.max(np.abs(slope-2.25)) < 1e-10
    assert np.max(np.abs(pred-(-0.4+2.25*q))) < 1e-10


def test_guard_band_excludes_kink():
    dgp = DGP_REGISTRY["U10"]
    x = np.array([-0.2, -0.05, 0.0, 0.05, 0.2])
    got = differentiable_mask(x, dgp, guard_width=0.1)
    assert np.array_equal(got, np.array([True, False, False, False, True]))


def test_analytical_derivatives_match_finite_differences_away_from_breaks():
    h = 1e-6
    grid = np.linspace(-1.8, 1.8, 37)
    for dgp in DGP_REGISTRY.values():
        mask = differentiable_mask(grid, dgp, guard_width=0.12)
        x = grid[mask]
        numeric = (dgp.mean(x+h) - dgp.mean(x-h)) / (2*h)
        analytic = dgp.derivative(x)
        assert np.max(np.abs(numeric-analytic)) < 2e-4, dgp.code

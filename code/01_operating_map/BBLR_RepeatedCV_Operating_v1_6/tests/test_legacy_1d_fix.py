import numpy as np

from bblr.bblr2 import build_snn_model, BallBasedLinearRegression


def test_legacy_radius_index_1d_native():
    x = np.array([-2., -1., 0., 1., 2.]).reshape(-1, 1)
    model = build_snn_model(x)
    got = np.sort(model.query_radius(np.array([-1.0]), 0.6))
    assert np.array_equal(got, np.array([1]))
    got = np.sort(model.query_radius(np.array([1.0]), 0.6))
    assert np.array_equal(got, np.array([3]))


def test_legacy_bblr_runs_in_1d_and_recovers_linear_slope():
    x = np.linspace(-2, 2, 80).reshape(-1, 1)
    y = 1 + 2*x[:, 0]
    q = np.linspace(-1.5, 1.5, 15).reshape(-1, 1)
    model = BallBasedLinearRegression(eps=0.35, r=0.35, min_points=5, ci=False)
    model.fit(x, y)
    model.predict(q)
    assert np.max(np.abs(model.coefs[:, 0] - 2.0)) < 1e-8

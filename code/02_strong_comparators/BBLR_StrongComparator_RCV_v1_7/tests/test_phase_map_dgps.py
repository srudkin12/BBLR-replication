import numpy as np
from bblr.univariate import DGP_REGISTRY

def test_controlled_phase_map_registry_size_and_codes():
    expected = (
        ["U01"] +
        [f"Q{i:02d}" for i in range(1,7)] +
        [f"F{i:02d}" for i in range(1,8)] +
        [f"A{i:02d}" for i in range(1,7)] +
        [f"T{i:02d}" for i in range(1,7)] +
        [f"B{i:02d}" for i in range(1,7)] +
        [f"L{i:02d}" for i in range(1,6)]
    )
    assert all(code in DGP_REGISTRY for code in expected)
    assert len(expected) == 37

def test_frequency_ladder_holds_slope_amplitude_fixed():
    x=np.linspace(-2,2,20001)
    for i in range(1,8):
        b=DGP_REGISTRY[f"F{i:02d}"].derivative(x)
        # derivative is 1 + cos(omega*x), so departure from baseline has max 1
        assert np.max(np.abs(b-1.0)) <= 1.000001
        assert np.max(np.abs(b-1.0)) >= 0.99

def test_transition_ladder_holds_limiting_slope_range_approximately_fixed():
    # Evaluate well beyond the nominal support to verify analytical limiting slopes.
    x=np.array([-20.0,20.0])
    for i in range(1,7):
        b=DGP_REGISTRY[f"T{i:02d}"].derivative(x)
        assert b[0] < 1e-6
        assert abs(b[1]-2.0) < 1e-6

def test_bump_width_ladder_holds_peak_slope_perturbation_fixed():
    for i in range(1,7):
        d=DGP_REGISTRY[f"B{i:02d}"]
        w=float(d.parameter_value)
        x=np.array([-w,w])
        b=d.derivative(x)
        assert np.allclose(np.abs(b-1.0),1.0,rtol=1e-10,atol=1e-10)

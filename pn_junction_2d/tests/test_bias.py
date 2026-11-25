import numpy as np
import pytest

from config import DeviceConfig, SolverConfig
from geometry import cosine
from postprocess import bias_diagnostics, current_conservation_error, ideality_factor, terminal_current
from solver import Device, run_voltage_sweep, solve_bias, solve_equilibrium


@pytest.fixture(scope="module")
def planar_sweep():
    dev = Device(DeviceConfig())
    V = np.concatenate([np.arange(0.05, 0.651, 0.05), [-0.5, -1.0]])
    states = run_voltage_sweep(dev, V)
    return dev, states


def test_zero_bias_current(planar_sweep):
    dev, states = planar_sweep
    I0 = terminal_current(dev, states[0.0])
    I_fwd = terminal_current(dev, states[0.3])
    assert abs(I0) < 1e-6 * abs(I_fwd)
    report = bias_diagnostics(dev, states[0.0])
    assert np.isnan(report["relative_current_deviation"])
    assert report["max_current_deviation_A_per_cm"] < 1e-20


def test_bias_diagnostics_match_face_currents(planar_sweep):
    dev, states = planar_sweep
    report = bias_diagnostics(dev, states[-1.0])
    np.testing.assert_allclose(
        [report["left_current_A_per_cm"], report["right_current_A_per_cm"]],
        terminal_current(dev, states[-1.0]), rtol=1e-3, atol=0,
    )
    assert report["poisson_residual_scaled"] < 1e-8
    assert report["electron_continuity_residual_cm-3_s-1"] < 1e6
    assert report["hole_continuity_residual_cm-3_s-1"] < 1e6


def test_current_conservation(planar_sweep):
    dev, states = planar_sweep
    for V, st in states.items():
        if V == 0.0:
            continue
        err = current_conservation_error(dev, st)
        assert err < 1e-3, f"V={V}: conservation error {err:.2e}"


def test_forward_exponential_and_reverse_saturation(planar_sweep):
    dev, states = planar_sweep
    Vs = np.array([v for v in states if v > 0])
    Is = np.array([terminal_current(dev, states[v]) for v in Vs])
    assert np.all(Is > 0) and np.all(np.diff(Is) > 0)
    nid = ideality_factor(Vs, Is, dev.cfg.VT)
    print(f"ideality factor = {nid:.3f}")
    assert 0.9 < nid < 2.1     # reported, loose sanity bound
    Ir1, Ir2 = (terminal_current(dev, states[v]) for v in (-0.5, -1.0))
    assert Ir1 < 0 and Ir2 < 0
    assert abs(Ir2 / Ir1) < 2.0   # weakly bias dependent (SRH generation grows with W)


def test_curved_reverse_bias_conservation():
    cfg = DeviceConfig()
    dev = Device(cfg, cosine(cfg.Lx / 2, 0.25e-4, cfg.Ly))
    st = run_voltage_sweep(dev, [-1.0])[-1.0]
    assert current_conservation_error(dev, st) < 1e-3

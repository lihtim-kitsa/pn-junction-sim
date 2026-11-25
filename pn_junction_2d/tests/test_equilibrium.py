import numpy as np
import pytest

from config import DeviceConfig, UM
from geometry import net_doping
from postprocess import (abrupt_depletion_width, abrupt_peak_field, built_in_potential,
                         depletion_width, electric_field, space_charge,
                         symmetric_equilibrium_peak_field)
from solver import Device, solve_equilibrium, solve_equilibrium_1d


@pytest.fixture(scope="module")
def eq():
    dev = Device(DeviceConfig())
    return dev, solve_equilibrium(dev)


def test_newton_converges_fast(eq):
    dev, st = eq
    assert st.iterations < 15


def test_built_in_potential(eq):
    dev, st = eq
    psi = st.psi * dev.sc.VT
    vbi = psi[dev.mesh.right][0] - psi[dev.mesh.left][0]
    assert abs(vbi / built_in_potential(dev.cfg) - 1) < 1e-3


def test_rows_identical(eq):
    dev, st = eq
    P = dev.reshape(st.psi)
    assert np.max(np.abs(P - P[0])) < 1e-10


def test_matches_fine_1d_reference(eq):
    dev, st = eq
    cfg = dev.cfg
    x1, psi1 = solve_equilibrium_1d(cfg, 4 * (cfg.Nx - 1) + 1, dev.w)
    row = dev.reshape(st.psi)[dev.mesh.Ny // 2] * dev.sc.VT
    err = np.max(np.abs(row - np.interp(dev.mesh.x, x1, psi1)))
    assert err < 0.01 * built_in_potential(cfg)


def test_charge_neutral_overall(eq):
    dev, st = eq
    rho = space_charge(dev, st)
    total_abs = np.sum(np.abs(rho))
    assert abs(np.sum(rho)) < 1e-3 * total_abs


def test_field_and_width_vs_abrupt(eq):
    dev, st = eq
    cfg = dev.cfg
    Emax = electric_field(dev, st)[4].max()
    W = depletion_width(dev, st)
    # Same-w comparison against the fine 1D reference: should agree closely.
    x1, psi1 = solve_equilibrium_1d(cfg, 4 * (cfg.Nx - 1) + 1, dev.w)
    E1 = np.max(np.abs(np.diff(psi1) / np.diff(x1)))
    assert abs(Emax / E1 - 1) < 0.02
    # Finite-w discrepancy vs abrupt formula (reported; tanh removes ~w ln2 of charge)
    dE = Emax / abrupt_peak_field(cfg) - 1
    print(f"w = {dev.w * 1e4:.4f} um: Emax/abrupt - 1 = {dE:+.3f}, "
          f"W/abrupt - 1 = {W / abrupt_depletion_width(cfg) - 1:+.3f}")
    assert -0.25 < dE < 0
    assert abs(W / abrupt_depletion_width(cfg) - 1) < 0.05


def test_w_convergence_1d():
    """As w -> 0 (resolved), the 1D peak field approaches the abrupt formula (O(w))."""
    cfg = DeviceConfig(Nx=8001)
    Eab = symmetric_equilibrium_peak_field(cfg)
    ws = np.array([0.08, 0.04, 0.02, 0.01]) * UM
    Es = []
    for w in ws:
        x, psi = solve_equilibrium_1d(cfg, cfg.Nx, w)
        Es.append(np.max(np.abs(np.diff(psi) / np.diff(x))))
    errs = np.abs(np.array(Es) / Eab - 1)
    print("w [um]:", ws * 1e4, " |Emax/abrupt - 1|:", np.round(errs, 4))
    assert np.all(np.diff(errs) < 0)
    E_extrap = 2 * Es[-1] - Es[-2]          # Richardson, first-order in w
    assert abs(E_extrap / Eab - 1) < 0.01


def test_asymmetric_junction_contacts_and_doping_zero():
    cfg = DeviceConfig(NA=4e16, ND=1e16, Nx=101, Ny=21)
    dev = Device(cfg)
    st = solve_equilibrium(dev)
    expected = cfg.VT * np.log(cfg.NA * cfg.ND / cfg.ni**2)
    vbi = cfg.VT * (st.psi[dev.mesh.right][0] - st.psi[dev.mesh.left][0])
    assert vbi == pytest.approx(expected, rel=1e-9)
    assert dev.sc.C0 * (dev.pL - dev.nL) == pytest.approx(cfg.NA)
    assert dev.sc.C0 * (dev.nR - dev.pR) == pytest.approx(cfg.ND)
    x_zero = cfg.Lx / 2 + 0.5 * dev.w * np.log(cfg.NA / cfg.ND)
    assert net_doping(cfg, x_zero, 0.0, dev.curve, dev.w) == pytest.approx(0, abs=100)
    with pytest.raises(ValueError, match="symmetric"):
        symmetric_equilibrium_peak_field(cfg)


def test_abrupt_width_rejects_zero_or_negative_barrier():
    cfg = DeviceConfig()
    with pytest.raises(ValueError, match="V < V_bi"):
        abrupt_depletion_width(cfg, built_in_potential(cfg))

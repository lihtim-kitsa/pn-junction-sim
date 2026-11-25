import warnings

import numpy as np

from config import DeviceConfig
from geometry import cosine, net_doping, planar
from numerics import Scaling, bernoulli
from solver import Device


def test_bernoulli_zero():
    assert bernoulli(0.0) == 1.0


def test_bernoulli_identity_and_finite():
    z = np.concatenate([np.linspace(-1000, 1000, 20001), np.logspace(-8, 3, 500),
                        -np.logspace(-8, 3, 500), [1e-4, -1e-4, 500, -500, 500.0001]])
    b, bm = bernoulli(z), bernoulli(-z)
    assert np.all(np.isfinite(b))
    assert np.all(b >= 0)
    assert np.all(np.abs((b - bm) + z) <= 1e-12 * np.maximum(1, np.abs(z)))


def test_bernoulli_continuity_at_switch():
    for z0 in (1e-4, 500.0):
        lo, hi = bernoulli(np.array([z0 * (1 - 1e-12)])), bernoulli(np.array([z0 * (1 + 1e-12)]))
        assert abs(lo[0] - hi[0]) < 1e-12 * max(1, z0)


def test_scale_roundtrip():
    sc = Scaling(DeviceConfig())
    rng = np.random.default_rng(0)
    for kind in ("psi", "n", "p", "x", "y", "mu", "tau", "J", "R", "E"):
        v = rng.random(5) * 10
        assert np.allclose(sc.unscale(kind, sc.scale(kind, v)), v, rtol=1e-14)
    assert np.isclose(sc.LD * 1e4, 0.0409, rtol=5e-3)          # Debye length in um


def test_width_rule_and_wall_warning():
    cfg = DeviceConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Device(cfg, cosine(cfg.Lx / 2, 0.4e-4, cfg.Ly))       # cosine: no wall warning
    try:
        Device(cfg, cosine(cfg.Lx / 2, 0.4e-4, cfg.Ly), w=0.05e-4)
        raise AssertionError("expected w-rule violation")
    except ValueError:
        pass


def test_signed_distance_follows_the_cosine_normal():
    cfg = DeviceConfig()
    curve = cosine(cfg.Lx / 2, 0.25e-4, cfg.Ly)
    y = np.array([0.0, 0.5e-4, 1.0e-4, cfg.Ly])
    offset = np.array([-0.01e-4, 0.02e-4, -0.03e-4, 0.01e-4])
    slope = curve.df(y)
    dx = offset / np.hypot(1, slope)
    actual = curve.signed_distance(curve.f(y) + dx, y - slope * dx, cfg.Ly)
    np.testing.assert_allclose(actual, offset, rtol=1e-8, atol=1e-14)


def test_planar_doping_profiles_are_identical():
    cfg = DeviceConfig(Nx=31, Ny=11)
    curve = planar(cfg.Lx / 2)
    dev = Device(cfg, curve)
    np.testing.assert_array_equal(
        net_doping(cfg, dev.mesh.X, dev.mesh.Y, curve, dev.w, "signed_distance"),
        dev.Nnet_phys,
    )

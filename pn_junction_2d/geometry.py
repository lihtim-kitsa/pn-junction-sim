"""Junction curves x = f(y) and the smooth tanh doping profile."""
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from config import DeviceConfig


@dataclass
class Curve:
    """Junction interface x = f(y) with analytic first and second derivatives (cm units)."""
    name: str
    f: Callable
    df: Callable
    d2f: Callable

    def curvature(self, y):
        return np.abs(self.d2f(y)) / (1.0 + self.df(y) ** 2) ** 1.5

    def signed_distance(self, X, Y, Ly, samples=1001):
        """Approximate closest-point distance to the interface, positive on the n side.

        Find the nearest sampled point, then minimize squared distance in the
        two adjacent sample intervals by vectorized golden-section search.
        The sampling should resolve each bend in the interface.
        """
        if self.name == "planar":
            return X - self.f(Y)
        y = np.linspace(0.0, Ly, samples)
        tree = cKDTree(np.column_stack((self.f(y), y)))
        _, i = tree.query(np.column_stack((X.ravel(), Y.ravel())))
        i = i.reshape(X.shape)
        lo = y[np.maximum(i - 1, 0)]
        hi = y[np.minimum(i + 1, samples - 1)]
        ratio = (np.sqrt(5.0) - 1.0) / 2.0

        def distance_sq(t):
            return (X - self.f(t)) ** 2 + (Y - t) ** 2

        for _ in range(28):
            left = hi - ratio * (hi - lo)
            right = lo + ratio * (hi - lo)
            keep_left = distance_sq(left) < distance_sq(right)
            lo, hi = np.where(keep_left, lo, left), np.where(keep_left, right, hi)
        return np.sign(X - self.f(Y)) * np.sqrt(distance_sq((lo + hi) / 2.0))


def planar(x0: float) -> Curve:
    return Curve("planar", lambda y: np.full_like(np.asarray(y, float), x0),
                 lambda y: np.zeros_like(np.asarray(y, float)),
                 lambda y: np.zeros_like(np.asarray(y, float)))


def cosine(x0: float, A: float, Ly: float, m: int = 1) -> Curve:
    """x0 + A cos(2 pi m y / Ly): f' = 0 on both Neumann walls."""
    k = 2.0 * np.pi * m / Ly
    return Curve(f"cosine(A={A * 1e4:g}um)",
                 lambda y: x0 + A * np.cos(k * np.asarray(y)),
                 lambda y: -A * k * np.sin(k * np.asarray(y)),
                 lambda y: -A * k**2 * np.cos(k * np.asarray(y)))


def parabolic(x0: float, c: float, Ly: float) -> Curve:
    yc = Ly / 2.0
    return Curve("parabolic",
                 lambda y: x0 + c * (np.asarray(y) - yc) ** 2,
                 lambda y: 2.0 * c * (np.asarray(y) - yc),
                 lambda y: np.full_like(np.asarray(y, float), 2.0 * c))


def circular_arc(xc: float, yc: float, R: float) -> Curve:
    """x = xc - sqrt(R^2 - (y - yc)^2); requires |y - yc| < R over the domain."""
    def s(y):
        d = R**2 - (np.asarray(y) - yc) ** 2
        if np.any(d <= 0):
            raise ValueError("circular arc: |y - yc| must be < R everywhere")
        return np.sqrt(d)
    return Curve("circular_arc",
                 lambda y: xc - s(y),
                 lambda y: (np.asarray(y) - yc) / s(y),
                 lambda y: R**2 / s(y) ** 3)


def min_transition_width(cfg: DeviceConfig, curve: Curve, n_sample: int = 2001,
                         profile: str = "horizontal") -> float:
    """Smallest admissible tanh scale w: 2.5*max(dx,dy)*sqrt(1 + max f'^2)."""
    if profile == "signed_distance":
        return 2.5 * max(cfg.dx, cfg.dy)
    y = np.linspace(0.0, cfg.Ly, n_sample)
    slope = np.max(np.abs(curve.df(y)))
    return 2.5 * max(cfg.dx, cfg.dy) * np.sqrt(1.0 + slope**2)


def check_walls(cfg: DeviceConfig, curve: Curve, tol: float = 0.05) -> None:
    for yw in (0.0, cfg.Ly):
        s = float(np.abs(curve.df(np.array([yw]))[0]))
        if s > tol:
            warnings.warn(f"{curve.name}: |f'| = {s:.3f} at y = {yw * 1e4:g} um; the junction "
                          "meets a zero-flux wall at an angle (mirror-symmetry kink).")


def net_doping(cfg: DeviceConfig, X, Y, curve: Curve, w: float, profile="horizontal"):
    """Net ionized doping [cm^-3] from a horizontal or normal transition."""
    if profile == "horizontal":
        distance = X - curve.f(Y)
    elif profile == "signed_distance":
        distance = curve.signed_distance(X, Y, cfg.Ly)
    else:
        raise ValueError(f"Unknown doping profile: {profile}")
    return 0.5 * (cfg.ND - cfg.NA) + 0.5 * (cfg.ND + cfg.NA) * np.tanh(distance / w)

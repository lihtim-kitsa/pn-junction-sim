"""Equilibrium Newton-Raphson, Scharfetter-Gummel continuity and Gummel iteration.

All linear algebra is done on scaled (dimensionless) variables.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from config import DeviceConfig, SolverConfig
from geometry import Curve, check_walls, min_transition_width, net_doping, planar
from numerics import (Mesh, Scaling, apply_dirichlet_rows, bernoulli,
                      divergence_operator, laplacian)


class ConvergenceError(RuntimeError):
    pass


# --- Device: geometry + doping + boundary values in scaled form -------------------
class Device:
    def __init__(self, cfg: DeviceConfig = DeviceConfig(), curve: Optional[Curve] = None,
                 w: Optional[float] = None, check: bool = True, profile: str = "horizontal"):
        self.cfg = cfg
        self.sc = sc = Scaling(cfg)
        self.mesh = mesh = Mesh(cfg, sc)
        self.curve = curve if curve is not None else planar(cfg.Lx / 2)
        self.profile = profile
        w_min = min_transition_width(cfg, self.curve, profile=profile)
        self.w = w if w is not None else w_min
        if not np.isfinite(self.w) or self.w <= 0:
            raise ValueError("The doping transition scale w must be finite and positive")
        if check:
            if self.w < w_min * (1 - 1e-9):
                raise ValueError(f"w = {self.w * 1e4:.4f} um < required "
                                 f"{w_min * 1e4:.4f} um for the {profile} profile")
            check_walls(cfg, self.curve)
        self.Nnet_phys = net_doping(cfg, mesh.X, mesh.Y, self.curve, self.w, profile)
        self.N = sc.scale("N", self.Nnet_phys).ravel()
        self.ni = sc.scale("ni", cfg.ni)
        self.mu_n = cfg.mu_n / sc.mu0
        self.mu_p = cfg.mu_p / sc.mu0
        self.tau_n = sc.scale("tau", cfg.tau_n)
        self.tau_p = sc.scale("tau", cfg.tau_p)
        # Contact values from exact charge neutrality (physical, then scaled)
        pL = cfg.NA / 2 + np.sqrt((cfg.NA / 2) ** 2 + cfg.ni**2)
        nR = cfg.ND / 2 + np.sqrt((cfg.ND / 2) ** 2 + cfg.ni**2)
        self.pL, self.nL = sc.scale("p", pL), sc.scale("n", cfg.ni**2 / pL)
        self.nR, self.pR = sc.scale("n", nR), sc.scale("p", cfg.ni**2 / nR)
        self.psiL0 = -np.log(pL / cfg.ni)
        self.psiR = np.log(nR / cfg.ni)
        self.L = laplacian(mesh)
        self.L_bc = apply_dirichlet_rows(self.L, mesh.dirichlet)

    # boundary vectors
    def psi_bc(self, V: float) -> np.ndarray:
        b = np.zeros(self.mesh.N)
        b[self.mesh.left] = self.psiL0 + V / self.sc.VT
        b[self.mesh.right] = self.psiR
        return b

    def n_bc(self) -> np.ndarray:
        b = np.zeros(self.mesh.N)
        b[self.mesh.left], b[self.mesh.right] = self.nL, self.nR
        return b

    def p_bc(self) -> np.ndarray:
        b = np.zeros(self.mesh.N)
        b[self.mesh.left], b[self.mesh.right] = self.pL, self.pR
        return b

    def reshape(self, u):
        return np.asarray(u).reshape(self.mesh.Ny, self.mesh.Nx)


@dataclass
class State:
    psi: np.ndarray
    n: np.ndarray
    p: np.ndarray
    V: float = 0.0
    iterations: int = 0
    history: list = field(default_factory=list)
    roundoff_limited: bool = False
    dn: Optional[np.ndarray] = None
    dp: Optional[np.ndarray] = None

    def copy(self) -> "State":
        return State(
            self.psi.copy(), self.n.copy(), self.p.copy(), self.V, self.iterations,
            list(self.history), self.roundoff_limited,
            None if self.dn is None else self.dn.copy(),
            None if self.dp is None else self.dp.copy(),
        )


# --- Poisson ---------------------------------------------------------------------
def poisson_residual(dev: Device, psi, n, p, V: float) -> np.ndarray:
    """F = L psi - n + p + N on interior nodes, psi - psi_bc on contacts."""
    F = dev.L @ psi - n + p + dev.N
    d = dev.mesh.dirichlet
    F[d] = psi[d] - dev.psi_bc(V)[d]
    return F


def poisson_jacobian(dev: Device, n, p) -> sp.csc_matrix:
    """L - diag(n + p) on interior rows, identity on contact rows."""
    return (dev.L_bc - sp.diags((n + p) * dev.mesh.interior)).tocsc()


def solve_equilibrium(dev: Device, scfg: SolverConfig = SolverConfig(), verbose=False) -> State:
    """Nonlinear Poisson with Boltzmann carriers, Newton-Raphson from the neutral guess."""
    psi = np.arcsinh(dev.N / (2 * dev.ni))
    d = dev.mesh.dirichlet
    psi[d] = dev.psi_bc(0.0)[d]
    hist = []
    for it in range(1, scfg.newton_max_iter + 1):
        n, p = dev.ni * np.exp(psi), dev.ni * np.exp(-psi)
        F = poisson_residual(dev, psi, n, p, 0.0)
        dpsi = spsolve(poisson_jacobian(dev, n, p), -F)
        dpsi = np.clip(dpsi, -scfg.newton_clip, scfg.newton_clip)
        psi += dpsi
        step, res = np.max(np.abs(dpsi)), np.max(np.abs(F))
        hist.append((step, res))
        if verbose:
            print(f"  Newton {it:2d}: max|dpsi| = {step:.3e}  max|F| = {res:.3e}")
        if step < scfg.newton_tol:
            n, p = dev.ni * np.exp(psi), dev.ni * np.exp(-psi)
            return State(psi, n, p, 0.0, it, hist)
    raise ConvergenceError("equilibrium Newton did not converge")


# --- Scharfetter-Gummel continuity operators ------------------------------------------
def assemble_electron_matrix(dev: Device, psi) -> sp.csr_matrix:
    """A_n = -div_h J_n (no BC rows). Positive diagonal, non-positive off-diagonals."""
    zx, zy = dev.mesh.face_dpsi(psi)
    mu = dev.mu_n
    D = divergence_operator(dev.mesh, mu * bernoulli(zx), mu * bernoulli(-zx),
                            mu * bernoulli(zy), mu * bernoulli(-zy))
    return -D


def assemble_hole_matrix(dev: Device, psi) -> sp.csr_matrix:
    """A_p = +div_h J_p (no BC rows). Positive diagonal, non-positive off-diagonals."""
    zx, zy = dev.mesh.face_dpsi(psi)
    mu = dev.mu_p
    D = divergence_operator(dev.mesh, -mu * bernoulli(-zx), -mu * bernoulli(zx),
                            -mu * bernoulli(-zy), -mu * bernoulli(zy))
    return D


def srh_denominator(dev: Device, n, p):
    return dev.tau_p * (n + dev.ni) + dev.tau_n * (p + dev.ni)


def compute_srh(dev: Device, n, p):
    """Scaled SRH net recombination rate R~."""
    return (n * p - dev.ni**2) / srh_denominator(dev, n, p)


def _solve_continuity(A: sp.csr_matrix, bc: np.ndarray, d: np.ndarray, other: np.ndarray,
                      Dn: np.ndarray, ni: float, ref: Optional[np.ndarray] = None,
                      enable_srh: bool = True):
    """Solve semi-implicit continuity system, optionally shifted by a zero-flux Boltzmann ref."""
    if enable_srh:
        A = A + sp.diags(other / Dn)
        rhs = (ni**2 - (ref * other if ref is not None else 0.0)) / Dn
    else:
        rhs = np.zeros_like(bc)
    rhs[d] = bc[d] - (ref[d] if ref is not None else 0.0)
    sol = spsolve(apply_dirichlet_rows(A, d), rhs)
    if ref is not None:
        return ref + sol, sol
    return sol, None


def solve_electrons(dev: Device, psi, n_old, p_old, enable_srh=True):
    """(A_n + diag(p/D)) n = ni^2/D, with Dirichlet contact rows."""
    Dn = srh_denominator(dev, n_old, p_old) if enable_srh else None
    n, _ = _solve_continuity(assemble_electron_matrix(dev, psi), dev.n_bc(),
                             dev.mesh.dirichlet, p_old, Dn, dev.ni, None, enable_srh)
    return n


def solve_holes(dev: Device, psi, n_old, p_old, enable_srh=True):
    """(A_p + diag(n/D)) p = ni^2/D, with Dirichlet contact rows."""
    Dn = srh_denominator(dev, n_old, p_old) if enable_srh else None
    p, _ = _solve_continuity(assemble_hole_matrix(dev, psi), dev.p_bc(),
                             dev.mesh.dirichlet, n_old, Dn, dev.ni, None, enable_srh)
    return p


# --- Gummel iteration -----------------------------------------------------------------
def quasi_fermi(dev: Device, psi, n, p):
    """Scaled quasi-Fermi potentials phi_n, phi_p."""
    return psi - np.log(n / dev.ni), psi + np.log(p / dev.ni)


def solve_bias(dev: Device, V: float, init: State, scfg: SolverConfig = SolverConfig(),
               verbose=False) -> State:
    """Gummel decoupled iteration at bias V (applied to the left p-contact)."""
    psi, n, p = init.psi.copy(), init.n.copy(), init.p.copy()
    d, interior = dev.mesh.dirichlet, dev.mesh.interior
    n[d], p[d] = dev.n_bc()[d], dev.p_bc()[d]
    hist = []
    omega = 1.0
    prev_res = np.inf
    # Subtract zero-flux Boltzmann reference for reverse / low-forward bias to avoid
    # O(1) majority-bulk floating-point cancellation in small reverse/low-bias currents.
    use_delta = V <= 0.30
    dn_arr = dp_arr = None
    for it in range(1, scfg.gummel_max_iter + 1):
        psi0 = psi.copy()
        phin0, phip0 = quasi_fermi(dev, psi, n, p)
        # (A) Poisson with fixed quasi-Fermi levels
        for _ in range(scfg.gummel_inner_newton):
            F = poisson_residual(dev, psi, n, p, V)
            dpsi = spsolve(poisson_jacobian(dev, n, p), -F)
            dpsi[interior] = np.clip(dpsi[interior], -scfg.gummel_clip, scfg.gummel_clip)
            dpsi[interior] *= omega
            psi += dpsi
            n[interior] *= np.exp(dpsi[interior])
            p[interior] *= np.exp(-dpsi[interior])
            if np.max(np.abs(dpsi)) < 0.1 * scfg.tol_psi:
                break
        # (B, C) continuity with semi-implicit SRH
        n_ref = dev.ni * np.exp(psi) if use_delta else None
        p_ref = dev.ni * np.exp(-psi + V / dev.sc.VT) if use_delta else None
        Dn = srh_denominator(dev, n, p) if scfg.enable_srh else None
        n_new, dn_arr = _solve_continuity(assemble_electron_matrix(dev, psi), dev.n_bc(),
                                          d, p, Dn, dev.ni, n_ref, scfg.enable_srh)
        Dn = srh_denominator(dev, n_new, p) if scfg.enable_srh else None
        p_new, dp_arr = _solve_continuity(assemble_hole_matrix(dev, psi), dev.p_bc(),
                                          d, n_new, Dn, dev.ni, p_ref, scfg.enable_srh)
        if not (np.all(np.isfinite(n_new)) and np.all(np.isfinite(p_new))
                and np.all(n_new > 0) and np.all(p_new > 0)):
            raise ConvergenceError(f"non-physical carriers at V = {V:.3f} V (iter {it})")
        n, p = n_new, p_new
        phin, phip = quasi_fermi(dev, psi, n, p)
        e_psi = np.max(np.abs(psi - psi0))
        e_n = np.max(np.abs(phin - phin0))
        e_p = np.max(np.abs(phip - phip0))
        res = np.max(np.abs(poisson_residual(dev, psi, n, p, V)))
        hist.append((e_psi, e_n, e_p, res))
        if verbose:
            print(f"  Gummel {it:3d}: dpsi={e_psi:.2e} dphin={e_n:.2e} dphip={e_p:.2e} res={res:.2e}")
        if res > 1.5 * prev_res and it > 3:
            omega = 0.5
        prev_res = res
        psi_ok = e_psi < scfg.tol_psi and res < scfg.tol_poisson_res
        e_phi = max(e_n, e_p)
        if psi_ok and e_phi < scfg.tol_phi:
            return State(psi, n, p, V, it, hist, dn=dn_arr, dp=dp_arr)
        # Round-off floor: quasi-Fermi levels of ~1e-12 (scaled) minority carriers next to
        # the contacts jitter at the LU precision limit. Accept if psi/residual converged,
        # the jitter is below tol_phi_floor and has stopped decreasing.
        if psi_ok and e_phi < scfg.tol_phi_floor and it >= 6:
            recent = [max(h[1], h[2]) for h in hist[-6:]]
            if min(recent[3:]) > 0.5 * min(recent[:3]):
                return State(psi, n, p, V, it, hist, roundoff_limited=True, dn=dn_arr, dp=dp_arr)
    raise ConvergenceError(f"Gummel did not converge at V = {V:.3f} V")


def ramp(dev: Device, start: State, V_target: float, dV: float,
         scfg: SolverConfig = SolverConfig(), verbose=False) -> State:
    """Step from start.V to V_target, halving the step on failure."""
    state, V = start, start.V
    step = abs(dV)
    sgn = np.sign(V_target - V)
    while abs(V_target - V) > 1e-12:
        h = min(step, abs(V_target - V))
        try:
            state = solve_bias(dev, V + sgn * h, state, scfg)
            V = state.V
            if verbose:
                print(f"  V = {V:+.4f} V converged in {state.iterations} Gummel iterations")
            step = min(abs(dV), 2 * step)
        except ConvergenceError:
            step /= 2
            if step < scfg.dV_min:
                raise
            if verbose:
                print(f"  step failed, halving to {step:.4f} V")
    return state


def run_voltage_sweep(dev: Device, voltages, eq: Optional[State] = None,
                      scfg: SolverConfig = SolverConfig(), verbose=False):
    """Solve at each voltage in `voltages` (sorted outward from 0 in both directions).

    Returns a dict V -> State, including V = 0 solved through the Gummel path.
    """
    eq = eq if eq is not None else solve_equilibrium(dev, scfg)
    voltages = np.unique(np.round(np.asarray(voltages, float), 10))
    out = {}
    s0 = solve_bias(dev, 0.0, eq, scfg)
    out[0.0] = s0
    for branch, dV in ((voltages[voltages > 0], scfg.dV_forward),
                       (voltages[voltages < 0][::-1], scfg.dV_reverse)):
        state = s0
        for V in branch:
            state = ramp(dev, state, float(V), dV, scfg, verbose)
            out[float(V)] = state
    return dict(sorted(out.items()))


# --- 1D reference (independent tridiagonal Newton) ------------------------------------
def solve_equilibrium_1d(cfg: DeviceConfig, Nx: int, w: float, x0: Optional[float] = None,
                         tol: float = 1e-12, max_iter: int = 100):
    """Planar 1D nonlinear Poisson on a uniform grid. Returns (x [cm], psi [V])."""
    from scipy.linalg import solve_banded
    sc = Scaling(cfg)
    x0 = cfg.Lx / 2 if x0 is None else x0
    x = np.linspace(0, cfg.Lx, Nx)
    h = (x[1] - x[0]) / sc.LD
    Nn = (0.5 * (cfg.ND - cfg.NA) + 0.5 * (cfg.ND + cfg.NA) * np.tanh((x - x0) / w)) / sc.C0
    ni = cfg.ni / sc.C0
    pL = cfg.NA / 2 + np.sqrt((cfg.NA / 2) ** 2 + cfg.ni**2)
    nR = cfg.ND / 2 + np.sqrt((cfg.ND / 2) ** 2 + cfg.ni**2)
    psi = np.arcsinh(Nn / (2 * ni))
    psi[0], psi[-1] = -np.log(pL / cfg.ni), np.log(nR / cfg.ni)
    for _ in range(max_iter):
        F = np.zeros(Nx)
        F[1:-1] = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / h**2 - ni * np.exp(psi[1:-1]) \
            + ni * np.exp(-psi[1:-1]) + Nn[1:-1]
        ab = np.zeros((3, Nx))
        ab[1, :] = 1.0
        ab[1, 1:-1] = -2 / h**2 - ni * (np.exp(psi[1:-1]) + np.exp(-psi[1:-1]))
        ab[0, 2:] = 1 / h**2          # super-diagonal (row i, col i+1) for interior rows
        ab[2, :-2] = 1 / h**2         # sub-diagonal (row i, col i-1) for interior rows
        dpsi = np.clip(solve_banded((1, 1), ab, -F), -2, 2)
        psi += dpsi
        if np.max(np.abs(dpsi)) < tol:
            return x, psi * sc.VT
    raise ConvergenceError("1D Newton did not converge")

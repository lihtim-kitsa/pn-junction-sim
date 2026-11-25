"""Rescaling, electric field, SG currents, analytical references and plots."""
import numpy as np
import matplotlib.pyplot as plt

from config import Q
from numerics import bernoulli
from solver import (Device, State, assemble_electron_matrix, assemble_hole_matrix,
                    compute_srh, poisson_residual)


# --- Fields and currents ----------------------------------------------------------
def physical(dev: Device, st: State):
    """Return psi [V], n, p [cm^-3] as (Ny, Nx) arrays."""
    sc = dev.sc
    return (dev.reshape(sc.unscale("psi", st.psi)), dev.reshape(sc.unscale("n", st.n)),
            dev.reshape(sc.unscale("p", st.p)))


def electric_field(dev: Device, st: State):
    """Face-centred FVM gradients averaged to cell centres.

    Returns (xc, yc, Ex, Ey, |E|) with fields in V/cm, shape (Ny-1, Nx-1).
    """
    m = dev.mesh
    psi = dev.reshape(st.psi) * dev.sc.VT
    Ex_f = -(psi[:, 1:] - psi[:, :-1]) / m.dx_phys        # (Ny, Nx-1)
    Ey_f = -(psi[1:, :] - psi[:-1, :]) / m.dy_phys        # (Ny-1, Nx)
    Ex = 0.5 * (Ex_f[1:, :] + Ex_f[:-1, :])
    Ey = 0.5 * (Ey_f[:, 1:] + Ey_f[:, :-1])
    xc, yc = 0.5 * (m.x[1:] + m.x[:-1]), 0.5 * (m.y[1:] + m.y[:-1])
    return xc, yc, Ex, Ey, np.hypot(Ex, Ey)


def face_currents_x(dev: Device, st: State):
    """SG current densities on x faces, J_n,x and J_p,x [A/cm^2], shape (Ny, Nx-1)."""
    psi = dev.reshape(st.psi)
    # Using dn, dp directly when available omits the zero-flux Boltzmann reference and avoids cancellation
    n = dev.reshape(st.dn if st.dn is not None else st.n)
    p = dev.reshape(st.dp if st.dp is not None else st.p)
    z = psi[:, 1:] - psi[:, :-1]
    h = dev.mesh.dx
    Jn = dev.mu_n / h * (bernoulli(z) * n[:, 1:] - bernoulli(-z) * n[:, :-1])
    Jp = -dev.mu_p / h * (bernoulli(-z) * p[:, 1:] - bernoulli(z) * p[:, :-1])
    return dev.sc.unscale("J", Jn), dev.sc.unscale("J", Jp)


def column_currents(dev: Device, st: State):
    """Total current I(x_{i+1/2}) per unit depth [A/cm], trapezoid weights in y."""
    Jn, Jp = face_currents_x(dev, st)
    return dev.mesh.wy_phys @ (Jn + Jp)


def terminal_current(dev: Device, st: State) -> float:
    return float(np.mean(column_currents(dev, st)))


def current_conservation_error(dev: Device, st: State, floor=1e-20) -> float:
    """Relative variation about the mean, undefined below the current floor."""
    I = column_currents(dev, st)
    mean = I.mean()
    return float(np.max(np.abs(I - mean)) / abs(mean)) if abs(mean) > floor else np.nan


def bias_diagnostics(dev: Device, st: State, enable_srh=True):
    """Export contact currents, local equation residuals, and the peak field."""
    I = column_currents(dev, st)
    mean = I.mean()
    spread = np.max(np.abs(I - mean))
    mask = dev.mesh.interior
    R = compute_srh(dev, st.n, st.p) if enable_srh else 0.0
    # Use the same zero-flux reference subtraction as current post-processing.
    n = st.n if st.dn is None else st.dn
    p = st.p if st.dp is None else st.dp
    rn = assemble_electron_matrix(dev, st.psi) @ n + R
    rp = assemble_hole_matrix(dev, st.psi) @ p + R
    xc, yc, _, _, field = electric_field(dev, st)
    j, i = np.unravel_index(field.argmax(), field.shape)
    return {
        "bias_V": st.V,
        "left_current_A_per_cm": I[0],
        "right_current_A_per_cm": I[-1],
        "mean_current_A_per_cm": mean,
        "max_current_deviation_A_per_cm": spread,
        "relative_current_deviation": spread / abs(mean) if abs(mean) > 1e-20 else np.nan,
        "poisson_residual_scaled": np.max(np.abs(poisson_residual(dev, st.psi, st.n, st.p, st.V))),
        "electron_continuity_residual_cm-3_s-1": dev.sc.R0 * np.max(np.abs(rn[mask])),
        "hole_continuity_residual_cm-3_s-1": dev.sc.R0 * np.max(np.abs(rp[mask])),
        "peak_field_V_per_cm": field[j, i],
        "peak_x_um": xc[i] * 1e4,
        "peak_y_um": yc[j] * 1e4,
        "gummel_iterations": st.iterations,
        "roundoff_limited": st.roundoff_limited,
    }


def space_charge(dev: Device, st: State):
    """rho/q = p - n + N_net  [cm^-3], shape (Ny, Nx)."""
    _, n, p = physical(dev, st)
    return p - n + dev.Nnet_phys


def depletion_width(dev: Device, st: State, row: int = None):
    """W = distance between the points where p = NA/2 (p-side) and n = ND/2 (n-side) [cm].

    (An integrated-charge width would just restate E_max via Gauss's law, so it is not an
    independent check under tanh smoothing.)
    """
    row = dev.mesh.Ny // 2 if row is None else row
    _, n, p = physical(dev, st)
    x = dev.mesh.x
    n, p = n[row], p[row]
    n_half, p_half = dev.cfg.ND / 2, dev.cfg.NA / 2
    n_edges = np.flatnonzero((n[:-1] < n_half) & (n[1:] >= n_half))
    p_edges = np.flatnonzero((p[:-1] >= p_half) & (p[1:] < p_half))
    if not (len(n_edges) and len(p_edges)):
        return np.nan  # No pair of threshold crossings in the device.
    i_n, i_p = n_edges[0], p_edges[-1]
    xn = np.interp(n_half, n[i_n:i_n + 2], x[i_n:i_n + 2])
    xp = np.interp(p_half, p[i_p:i_p + 2][::-1], x[i_p:i_p + 2][::-1])
    return xn - xp if xn >= xp else np.nan


# --- Analytical abrupt-junction references --------------------------------------------
def built_in_potential(cfg) -> float:
    """Contact potential difference from Boltzmann statistics and charge neutrality."""
    return cfg.VT * (np.arcsinh(cfg.NA / (2 * cfg.ni)) + np.arcsinh(cfg.ND / (2 * cfg.ni)))


def abrupt_depletion_width(cfg, V: float = 0.0) -> float:
    """Classical abrupt depletion width [cm] for a positive junction barrier."""
    barrier = built_in_potential(cfg) - V
    if not np.isfinite(barrier) or barrier <= 0:
        raise ValueError("Abrupt depletion approximation requires V < V_bi")
    return np.sqrt(2 * cfg.eps * barrier / Q * (1 / cfg.NA + 1 / cfg.ND))


def abrupt_peak_field(cfg, V: float = 0.0) -> float:
    """Classical abrupt depletion peak field [V/cm]."""
    return 2 * (built_in_potential(cfg) - V) / abrupt_depletion_width(cfg, V)


def symmetric_equilibrium_peak_field(cfg) -> float:
    """Abrupt Poisson–Boltzmann peak field [V/cm] for symmetric doping.

    The first integral on either side includes the change in mobile charge.
    For N >> ni this is the familiar 2 V_T correction to the full barrier.
    """
    if cfg.NA != cfg.ND or cfg.NA <= 0:
        raise ValueError("Equilibrium field reference requires symmetric positive doping")
    u = cfg.NA / (2 * cfg.ni)
    # hypot(1, u) - 1 written without subtractive cancellation near u = 0.
    mobile = 2 * cfg.ni * u * (u / (np.hypot(1, u) + 1))
    integral = cfg.NA * built_in_potential(cfg) / 2 - cfg.VT * mobile
    return np.sqrt(2 * Q * integral / cfg.eps)


def ideality_factor(V, I, VT, vmin=0.25, vmax=0.55):
    V, I = np.asarray(V), np.abs(np.asarray(I))
    sel = (V >= vmin) & (V <= vmax)
    slope = np.polyfit(V[sel], np.log(I[sel]), 1)[0]
    return 1.0 / (slope * VT)


# --- Plots ---------------------------------------------------------------------
def plot_dashboard(dev: Device, st: State, title: str = None, fname: str = None):
    m = dev.mesh
    um = 1e4
    psi, n, p = physical(dev, st)
    xc, yc, Ex, Ey, Emag = electric_field(dev, st)
    yy = np.linspace(0, dev.cfg.Ly, 400)
    fy = dev.curve.f(yy)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    panels = [
        (ax[0, 0], m.x, m.y, psi, r"$\psi$ [V]", "viridis"),
        (ax[1, 0], m.x, m.y, np.log10(n), r"$\log_{10} n$ [cm$^{-3}$]", "Blues"),
        (ax[1, 1], m.x, m.y, np.log10(p), r"$\log_{10} p$ [cm$^{-3}$]", "Reds"),
    ]
    for a, x, y, Z, lab, cmap in panels:
        pc = a.pcolormesh(x * um, y * um, Z, shading="auto", cmap=cmap)
        fig.colorbar(pc, ax=a, label=lab)
        a.plot(fy * um, yy * um, "w--", lw=1.2)
        a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel(r"$y$ [$\mu$m]")
        a.set_aspect("equal")

    # Colored unit-vector quiver plot for E = (Ex, Ey)
    ax_E = ax[0, 1]
    nx_q = 34
    ny_q = max(4, round(nx_q * dev.cfg.Ly / dev.cfg.Lx))
    sx = max(1, round(len(xc) / nx_q))
    sy = max(1, round(len(yc) / ny_q))
    ix = slice(sx // 2, None, sx)
    iy = slice(sy // 2, None, sy)
    Xq, Yq = np.meshgrid(xc[ix] * um, yc[iy] * um)
    Ex_q, Ey_q, Em_q = Ex[iy, ix], Ey[iy, ix], Emag[iy, ix]
    Emax = max(float(Emag.max()), 1e-30)
    valid = Em_q > 1e-3 * Emax
    Ux = Ex_q[valid] / Em_q[valid]
    Uy = Ey_q[valid] / Em_q[valid]
    dx_q = (xc[1] - xc[0]) * sx * um
    dy_q = (yc[1] - yc[0]) * sy * um
    arrow_len = 0.75 * min(dx_q, dy_q)

    ax_E.set_facecolor("#11131a")
    ax_E.plot(fy * um, yy * um, "w--", lw=1.1, alpha=0.65, zorder=2)
    q = ax_E.quiver(
        Xq[valid], Yq[valid], Ux, Uy, Em_q[valid],
        cmap="plasma", angles="xy", scale_units="xy", scale=1.0 / arrow_len,
        pivot="mid", width=0.0042, headwidth=3.8, headlength=4.2, headaxislength=3.6,
        zorder=3,
    )
    q.set_clim(0.0, Emax)
    fig.colorbar(q, ax=ax_E, label=r"$|\mathbf{E}|$ [V/cm]")
    ax_E.set_xlabel(r"$x$ [$\mu$m]"); ax_E.set_ylabel(r"$y$ [$\mu$m]")
    ax_E.set_aspect("equal")
    ax_E.set_xlim(0, dev.cfg.Lx * um); ax_E.set_ylim(0, dev.cfg.Ly * um)

    fig.suptitle(title or f"{dev.curve.name}, $V_{{app}}$ = {st.V:+.2f} V")
    if fname:
        fig.savefig(fname, dpi=130)
    return fig


def plot_iv(V, I, VT=None, fname: str = None):
    V, I = np.asarray(V), np.asarray(I)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for mask, label in ((V < 0, "Reverse"), (V > 0, "Forward")):
        ax.semilogy(V[mask], np.abs(I[mask]), "o-", ms=4, label=label)
    if np.any(V == 0):
        i0 = np.abs(I[V == 0][0])
        ax.text(0.03, 0.97, f"Equilibrium |I| = {i0:.2e} A/cm",
                transform=ax.transAxes, va="top")
    ax.legend(loc="lower right")
    ax.set_xlabel(r"$V_{app}$ [V]"); ax.set_ylabel(r"$|I|$ [A/cm]")
    ax.grid(True, which="both", alpha=0.3)
    if VT is not None and np.sum(V > 0.3) >= 3:
        nid = ideality_factor(V, I, VT)
        ax.set_title(f"I-V (fitted ideality n = {nid:.3f}, 0.25-0.55 V)")
    if fname:
        fig.savefig(fname, dpi=130)
    return fig


def plot_srh_generation(dev: Device, st: State, fname=None):
    """Plot log10 of positive net SRH generation in reverse bias."""
    mesh = dev.mesh
    generation = dev.reshape(-dev.sc.R0 * compute_srh(dev, st.n, st.p))
    values = np.ma.log10(np.ma.masked_less_equal(generation, 0))
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    im = ax.pcolormesh(mesh.x * 1e4, mesh.y * 1e4, values, shading="auto")
    ax.set(xlabel="x [µm]", ylabel="y [µm]", aspect="equal",
           title=f"Net SRH generation at {st.V:+g} V")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}(-R_{\mathrm{SRH}}/[\mathrm{cm}^{-3}\,\mathrm{s}^{-1}])$")
    if fname:
        fig.savefig(fname, dpi=150)
    return fig

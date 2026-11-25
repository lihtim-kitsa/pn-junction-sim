"""Script runner: planar I-V sweep, dashboards and curvature (field-crowding) study.

Usage:
    python main.py                # everything, figures into ./figures
    python main.py --iv           # planar I-V sweep only
    python main.py --curvature    # curvature study only
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from config import DeviceConfig, SolverConfig, UM
from geometry import cosine
from postprocess import (abrupt_depletion_width, bias_diagnostics, electric_field,
                         ideality_factor, plot_dashboard, plot_iv, plot_srh_generation,
                         terminal_current)
from solver import Device, State, run_voltage_sweep, solve_bias

FIG = Path(__file__).resolve().parent / "figures"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_domain(cfg: DeviceConfig, V_min: float):
    W = abrupt_depletion_width(cfg, V_min)
    ok = cfg.Lx >= 3 * W
    print(f"Domain check at {V_min:+.2f} V: W_d = {W * 1e4:.3f} um, Lx/W_d = {cfg.Lx / W:.2f}"
          f" -> {'OK' if ok else 'TOO NARROW'}")
    if not ok:
        raise ValueError("Lx must be >= 3 * W_d at the most negative bias")


def iv_study(cfg=DeviceConfig(), scfg=SolverConfig()):
    V_fwd = np.round(np.arange(0.05, 0.651, 0.05), 3)
    V_rev = np.round(np.arange(-0.2, -3.001, -0.2), 3)
    check_domain(cfg, V_rev.min())
    dev = Device(cfg)
    print(f"Planar device: w = {dev.w * 1e4:.4f} um, w/dx = {dev.w / cfg.dx:.2f}, "
          f"w/W_d(0) = {dev.w / abrupt_depletion_width(cfg):.3f}")
    t = time.time()
    states = run_voltage_sweep(dev, np.concatenate([V_fwd, V_rev]), scfg=scfg)
    print(f"Sweep done in {time.time() - t:.1f} s")
    V = np.array(list(states))
    I = np.array([terminal_current(dev, s) for s in states.values()])
    diagnostics = [bias_diagnostics(dev, st, scfg.enable_srh) for st in states.values()]
    print(f"{'V [V]':>8} {'I [A/cm]':>12} {'max |ΔI| [A/cm]':>16} {'rel. dev.':>10} {'Gummel it':>9}")
    for v, i, row in zip(V, I, diagnostics):
        relative = row["relative_current_deviation"]
        relative_text = f"{relative:.2e}" if np.isfinite(relative) else "—"
        print(f"{v:8.2f} {i:12.4e} {row['max_current_deviation_A_per_cm']:16.2e} "
              f"{relative_text:>10} {row['gummel_iterations']:9d}")
    print(f"Fitted ideality factor (0.25-0.55 V): {ideality_factor(V, I, cfg.VT):.3f}")
    FIG.mkdir(exist_ok=True, parents=True)
    write_csv(FIG / "iv_diagnostics.csv", diagnostics)
    plot_iv(V, I, cfg.VT, FIG / "iv_planar.png")
    for v in (0.0, 0.6, -3.0):
        plot_dashboard(dev, states[v], fname=FIG / f"dashboard_planar_{v:+.1f}V.png")
    if scfg.enable_srh:
        plot_srh_generation(dev, states[-3.0], fname=FIG / "srh_generation_reverse.png")
    plt.close("all")
    return dev, states


PROFILES = {"horizontal": "Horizontal tanh", "signed_distance": "Signed-distance tanh"}


def refine_state(coarse, state, fine, scfg):
    """Interpolate a converged state to initialize a new solve on a finer grid."""
    points = np.column_stack((fine.mesh.Y.ravel(), fine.mesh.X.ravel()))
    axes = coarse.mesh.y, coarse.mesh.x
    psi = RegularGridInterpolator(axes, coarse.reshape(state.psi))(points)
    n, p = (
        np.exp(RegularGridInterpolator(axes, np.log(coarse.reshape(carrier)))(points))
        for carrier in (state.n, state.p)
    )
    return solve_bias(fine, state.V, State(psi, n, p, state.V), scfg)


def plot_field_comparison(cases, V, w, normal_extent):
    """Compare field maps and the peak sampled along each junction normal."""
    fields = {profile: electric_field(dev, state) for profile, (dev, state) in cases.items()}
    vmax = max(field[4].max() for field in fields.values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    fig_n, ax_n = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax_k = ax_n.twinx()
    for ax, (profile, (dev, _)) in zip(axes, cases.items()):
        xc, yc, _, _, E = fields[profile]
        y = np.linspace(0, dev.cfg.Ly, 401)
        x = dev.curve.f(y)
        j, i = np.unravel_index(E.argmax(), E.shape)
        im = ax.pcolormesh(xc / UM, yc / UM, E / 1e3, shading="nearest", vmin=0, vmax=vmax / 1e3)
        ax.plot(x / UM, y / UM, "w--", linewidth=1)
        tip_y = np.linspace(0, dev.cfg.Ly, 3)
        ax.scatter(dev.curve.f(tip_y) / UM, tip_y / UM, marker="o", facecolors="none",
                   edgecolors="white", label="Curvature extrema")
        ax.scatter(xc[i] / UM, yc[j] / UM, marker="x", color="cyan", label="Field maximum")
        ax.set(xlabel="x [µm]", ylabel="y [µm]", title=PROFILES[profile], aspect="equal")
        ax.legend(loc="upper right", fontsize=7)

        # Normals point from p to n. Mirror the field across the symmetric walls.
        slope = dev.curve.df(y)
        offsets = np.linspace(-normal_extent, normal_extent, 201)
        dx = offsets[None, :] / np.sqrt(1 + slope[:, None] ** 2)
        yy = y[:, None] - slope[:, None] * dx
        yy = dev.cfg.Ly - np.abs(yy % (2 * dev.cfg.Ly) - dev.cfg.Ly)
        field = RegularGridInterpolator(
            (np.r_[0, yc, dev.cfg.Ly], xc), np.pad(E, ((1, 1), (0, 0)), mode="edge"),
            bounds_error=False, fill_value=np.nan,
        )
        normal_peak = np.nanmax(field((yy, x[:, None] + dx)), axis=1)
        arc = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))] / UM
        ax_n.plot(arc, normal_peak / 1e3, label=PROFILES[profile])
    ax_k.plot(arc, dev.curve.curvature(y) * UM, "--", color="0.5", label="Curvature magnitude")
    fig.colorbar(im, ax=axes, label="Field magnitude [kV/cm]")
    fig.suptitle(f"Field magnitude at {V:+g} V and w = {w / UM:g} µm")
    ax_n.set(xlabel="Junction arc length [µm]", ylabel="Peak along normal [kV/cm]")
    ax_k.set(ylabel="Curvature magnitude [1/µm]")
    ax_n.legend(handles=ax_n.lines + ax_k.lines, loc="best", fontsize=8)
    fig.savefig(FIG / "curvature_field_comparison.png", dpi=150)
    fig_n.savefig(FIG / "curvature_normal_profiles.png", dpi=150)
    plt.close(fig)
    plt.close(fig_n)


def curvature_study(cfg=DeviceConfig(), scfg=SolverConfig(), V=-2.0,
                    amplitudes=(0.0, 0.1, 0.25, 0.4), w=0.11 * UM, refine=True):
    check_domain(cfg, V)
    FIG.mkdir(exist_ok=True, parents=True)
    rows, largest = [], {}
    peak_amplitude = max(amplitudes)
    fine_cfg = cfg.with_(Nx=2 * cfg.Nx - 1, Ny=2 * cfg.Ny - 1)

    def record(A, kappa, dev, st, mesh):
        xc, yc, _, _, E = electric_field(dev, st)
        j, i = np.unravel_index(E.argmax(), E.shape)
        peak = E[j, i]
        rows.append((dev.profile, mesh, dev.cfg.Nx, dev.cfg.Ny, A, kappa, peak,
                     xc[i] / UM, yc[j] / UM, terminal_current(dev, st)))
        print(f"{dev.profile:15s} {mesh:7s} A={A:.2f} µm  "
              f"|E|max={peak / 1e3:.3f} kV/cm at ({xc[i] / UM:.3f}, {yc[j] / UM:.3f}) µm")

    for A in amplitudes:
        curve = cosine(cfg.Lx / 2, A * UM, cfg.Ly)
        kappa = abs(A) * (2 * np.pi / (cfg.Ly / UM)) ** 2
        for profile in PROFILES:
            dev = Device(cfg, curve, w=w, profile=profile)
            st = run_voltage_sweep(dev, [V], scfg=scfg)[round(V, 10)]
            record(A, kappa, dev, st, "base")
            if A == peak_amplitude:
                largest[profile] = dev, st
            if refine and A in (0.0, peak_amplitude):
                fine = Device(fine_cfg, curve, w=w, profile=profile)
                record(A, kappa, fine, refine_state(dev, st, fine, scfg), "refined")
            if profile == "horizontal":
                fig = plot_dashboard(dev, st, fname=FIG / f"dashboard_cos_A{A:.2f}_{V:+.1f}V.png")
                plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for profile, label in PROFILES.items():
        base = [row for row in rows if row[:2] == (profile, "base")]
        fine = [row for row in rows if row[:2] == (profile, "refined")]
        line, = ax.plot([row[5] for row in base], [row[6] / 1e3 for row in base], "o-", label=label)
        ax.scatter([row[5] for row in fine], [row[6] / 1e3 for row in fine],
                   marker="x", color=line.get_color(), zorder=3)
    if refine:
        ax.scatter([], [], marker="x", color="0.2", label="Refined grid")
    ax.set(xlabel="Maximum interface curvature [1/µm]", ylabel="Peak field magnitude [kV/cm]",
           title=f"Cosine amplitude sweep at {V:+g} V and w = {w / UM:g} µm")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(FIG / "emax_vs_curvature.png", dpi=150)
    plt.close(fig)
    plot_field_comparison(largest, V, w, abrupt_depletion_width(cfg, V))
    columns = ("profile", "mesh", "nx", "ny", "amplitude_um", "max_curvature_per_um",
               "peak_field_V_per_cm", "peak_x_um", "peak_y_um", "current_A_per_cm")
    write_csv(FIG / "curvature_diagnostics.csv", [
        dict(zip(columns, row)) | {"bias_V": V, "transition_scale_um": w / UM}
        for row in rows
    ])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", action="store_true")
    ap.add_argument("--curvature", action="store_true")
    a = ap.parse_args()
    run_all = not (a.iv or a.curvature)
    if a.iv or run_all:
        iv_study()
    if a.curvature or run_all:
        curvature_study()

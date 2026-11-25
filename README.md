---
title: 2D Curved PN Junction Drift-Diffusion Solver
tags: [computational-physics, semiconductors, pde, fvm, drift-diffusion]
date: 2026-11-24
---

# 2D Curved PN Junction: Mathematical Formulation & Numerical Solver

## Quick start

From the repository root, use Python 3.12 or later.

~~~sh
python -m pip install -r pn_junction_2d/requirements.txt
python -m pytest -q pn_junction_2d/tests
python pn_junction_2d/main.py
~~~

Use `--iv` for the planar sweep or `--curvature` for the cosine comparison.
The full run includes refinement to a \(401\times161\) grid and may take
several minutes. Open the notebook in `pn_junction_2d/` with the kernel working
directory set to that folder. It imports the same solver and study functions.
Use [task_fixed.md](task_fixed.md) for the current implementation plan. The
files in `archives/` preserve earlier drafts.

| Output in `pn_junction_2d/figures/` | Contents |
| --- | --- |
| `iv_planar.png` and `dashboard_planar_*.png` | Forward and reverse current branches, plus field and carrier plots |
| `srh_generation_reverse.png` | Positive net SRH generation at −3 V |
| `iv_diagnostics.csv` | Bias, contact and mean currents, absolute and relative current variation, three equation residuals, peak field and location, and solver status |
| `emax_vs_curvature.png` and `curvature_diagnostics.csv` | Both doping profiles at each amplitude and refined-grid endpoint checks |
| `curvature_field_comparison.png` and `curvature_normal_profiles.png` | Shared-scale field maps and peak field along normals versus junction arc length |

Both contact currents are signed in the \(+x\) direction and reported per
unit out-of-plane depth. At or below \(10^{-20}\) A/cm, the relative deviation
is undefined and the absolute deviation remains available. The continuity
residuals are in cm\(^{-3}\) s\(^{-1}\).

This document details the complete physical model, mathematical derivations, and numerical architecture for simulating a 2D Silicon PN junction with a variable junction curve $x = f(y)$ under applied bias. The solver uses the **Finite Volume Method (FVM)** on a structured Cartesian grid, **Scharfetter-Gummel** flux discretization, and **Gummel's decoupled iteration**.

---

## 1. Variable Definitions & Nomenclature

All physical equations use standard semiconductor $\text{cm}$-based units. The default material parameters below use a textbook Silicon model at $T = 300\text{ K}$; $n_i=10^{10}\,\text{cm}^{-3}$ is an approximation.

### 1.1 Physical Constants & Material Parameters

| Symbol | Name / Meaning | Value (Si, 300 K) | Units |
| :--- | :--- | :--- | :--- |
| $q$ | Elementary charge | $1.602176634 \times 10^{-19}$ | $\text{C}$ |
| $k_B$ | Boltzmann constant | $1.380649 \times 10^{-23}$ | $\text{J/K}$ |
| $T$ | Absolute temperature | $300$ | $\text{K}$ |
| $V_T$ | Thermal voltage ($k_B T / q$) | $\approx 0.02585$ | $\text{V}$ |
| $\varepsilon_0$ | Vacuum permittivity | $8.854187817 \times 10^{-14}$ | $\text{F/cm}$ |
| $\varepsilon_r$ | Relative permittivity of Silicon | $11.7$ | dimensionless |
| $\varepsilon$ | Absolute permittivity ($\varepsilon_r \varepsilon_0$) | $\approx 1.0359 \times 10^{-12}$ | $\text{F/cm}$ |
| $n_i$ | Intrinsic carrier concentration | $1.0 \times 10^{10}$ | $\text{cm}^{-3}$ |
| $\mu_n, \; \mu_p$ | Electron and hole mobilities | $1400, \; 450$ | $\text{cm}^2/(\text{V}\cdot\text{s})$ |
| $D_n, \; D_p$ | Electron and hole diffusion coefficients | $\mu_n V_T, \; \mu_p V_T$ | $\text{cm}^2/\text{s}$ |
| $\tau_n, \; \tau_p$ | Electron and hole SRH lifetimes | $1.0 \times 10^{-6}$ | $\text{s}$ |

### 1.2 Geometry, Doping & Grid Variables

| Symbol | Name / Meaning | Units / Notes |
| :--- | :--- | :--- |
| $L_x, \; L_y$ | Physical domain width and height | $\text{cm}$ (e.g., $2\text{ }\mu\text{m} = 2 \times 10^{-4}\text{ cm}$) |
| $N_x, \; N_y$ | Number of grid nodes along $x$ and $y$ | dimensionless integers |
| $(i, j)$ | Discrete grid indices | $i \in [0, N_x-1]$, $j \in [0, N_y-1]$ |
| $\Delta x, \; \Delta y$ | Uniform grid spacing in $x$ and $y$ | $\text{cm}$ |
| $\Omega_{i,j}, \; \partial\Omega_{i,j}$ | Control volume cell around $(i,j)$ and its boundary | Area $\Delta x \Delta y$ |
| $x = f(y)$ | Junction interface curve | User-defined (planar, sinusoidal, parabolic, etc.) |
| $w$ | Scale parameter of the $\tanh$ doping profile | $\text{cm}$ (see the grid resolution rule below) |
| $N_A, \; N_D$ | Acceptor (p-type) and donor (n-type) doping | $\text{cm}^{-3}$ (e.g., $10^{16}\text{ cm}^{-3}$) |
| $N_{\text{net}}(x,y)$ | Net ionized doping concentration ($N_D - N_A$) | $\text{cm}^{-3}$ |

### 1.3 Primary State & Field Variables

| Symbol | Name / Meaning | Units |
| :--- | :--- | :--- |
| $\psi(x,y)$ | Electrostatic potential (referenced to intrinsic Fermi level) | $\text{V}$ |
| $\mathbf{E}(x,y)$ | Electric field vector ($-\nabla \psi$) | $\text{V/cm}$ |
| $n(x,y), \; p(x,y)$ | Electron and hole concentrations | $\text{cm}^{-3}$ |
| $\rho(x,y)$ | Net space-charge density | $\text{C/cm}^3$ |
| $\mathbf{J}_n, \; \mathbf{J}_p$ | Electron and hole current density vectors | $\text{A/cm}^2$ |
| $R(x,y)$ | Net Shockley-Read-Hall (SRH) recombination rate | $\text{cm}^{-3}\text{s}^{-1}$ |
| $V_{\text{app}}$ | External voltage applied to the p-contact (n-contact grounded) | $\text{V}$ |

### 1.4 Scaling & Scharfetter-Gummel Auxiliary Variables

| Symbol | Definition | Meaning |
| :--- | :--- | :--- |
| $C_0$ | $\max(N_A, N_D)$ | Reference concentration for scaling ($\text{cm}^{-3}$) |
| $L_D$ | $\sqrt{\frac{\varepsilon V_T}{q C_0}}$ | Extrinsic Debye length ($\text{cm}$) |
| $\mu_0, \; t_0$ | $\mu_n, \; \frac{L_D^2}{\mu_0 V_T}$ | Reference mobility and time scale |
| $\tilde{\psi}, \tilde{n}, \tilde{p}, \tilde{x}, \tilde{y}$ | $\psi/V_T, \; n/C_0, \; p/C_0, \; x/L_D, \; y/L_D$ | Dimensionless state and spatial variables |
| $z$ | $\tilde{\psi}_{i+1,j} - \tilde{\psi}_{i,j}$ | Dimensionless potential drop across one grid step |
| $B(z)$ | $\frac{z}{e^z - 1}$ | Bernoulli function (weights drift vs. diffusion) |

---

## 2. Governing Equations (Continuum Model)

In a continuum semiconductor model, microscopic scattering events (mean free time $\tau_c$, mean free path) are averaged into macroscopic mobilities ($\mu = q\tau_c / m^*$). The steady-state 2D device is governed by three coupled partial differential equations (PDEs) for $(\psi, n, p)$.

### 2.1 Poisson's Equation (Gauss's Law + Charge Balance)
Starting from Gauss's law $\nabla \cdot (\varepsilon \mathbf{E}) = \rho$ with $\mathbf{E} = -\nabla \psi$ and uniform permittivity $\varepsilon$:

$$\nabla^2 \psi = \frac{\partial^2 \psi}{\partial x^2} + \frac{\partial^2 \psi}{\partial y^2} = -\frac{q}{\varepsilon} \Big( p - n + N_{\text{net}}(x,y) \Big)$$

* **Mobile charges:** Holes ($+p$) and electrons ($-n$).
* **Fixed lattice charges:** Ionized donors ($+N_D$) and ionized acceptors ($-N_A$).

### 2.2 Steady-State Continuity Equations
Conservation of electrons and holes under generation/recombination:

$$\nabla \cdot \mathbf{J}_n = q R, \qquad \nabla \cdot \mathbf{J}_p = -q R$$

### 2.3 Drift-Diffusion Current Densities
Carrier transport is the sum of **drift** (driven by $\mathbf{E} = -\nabla\psi$) and **diffusion** (driven by concentration gradients):

$$\mathbf{J}_n = -q \mu_n n \nabla \psi + q D_n \nabla n$$
$$\mathbf{J}_p = -q \mu_p p \nabla \psi - q D_p \nabla p$$

The diffusion coefficients are linked to mobilities via the **Einstein relation**:

$$D_n = \mu_n V_T, \qquad D_p = \mu_p V_T$$

### 2.4 Shockley-Read-Hall (SRH) Recombination
For mid-gap traps in indirect-bandgap Silicon:

$$R(n, p) = \frac{np - n_i^2}{\tau_p (n + n_i) + \tau_n (p + n_i)}$$

* In thermal equilibrium ($np = n_i^2$), $R = 0$.
* Under forward bias ($np > n_i^2$), $R > 0$ (net recombination).
* Under reverse bias ($np < n_i^2$), $R < 0$ (net thermal generation).

---

## 3. Representing the Variable Junction Curve

On a structured Cartesian grid, a step-function jump from $-N_A$ to $+N_D$ along a curved line $x = f(y)$ introduces jagged "staircasing" artifacts. To represent arbitrary curves smoothly, define the net doping profile using a hyperbolic tangent transition of width $w$:

$$N_{\text{net}}(x,y) = \frac{N_D - N_A}{2} + \frac{N_D + N_A}{2} \tanh\left(\frac{x - f(y)}{w}\right)$$

* **Planar benchmark:** $f(y) = L_x / 2$
* **Cosine interface:** $f(y) = L_x / 2 + A \cos(2\pi m y / L_y)$. Use cosine rather than sine so that $f'(0)=f'(L_y)=0$, which matches the mirror symmetry implied by the zero-flux walls.
* **Parabolic:** $f(y) = x_0 + c (y - L_y/2)^2$
* **Circular arc:** $f(y) = x_c - \sqrt{R^2 - (y-y_c)^2}$, for $|y-y_c|<R$

For any graph $x=f(y)$, the local curvature is
$$\kappa(y)=\frac{|f''(y)|}{\left[1+(f'(y))^2\right]^{3/2}},\qquad R_c(y)=\frac{1}{\kappa(y)}.$$

**Grid resolution rule:** Because the tanh argument is measured along $x$, the transition width normal to the junction is $w/\sqrt{1+f'^2}$. Choose $w \ge 2.5\max(\Delta x,\Delta y)\sqrt{1+\max f'^2}$ and also require $w \ll W_d$ for the benchmark of interest. Here $w$ is the tanh scale parameter, not the full physical transition width. The solver should report $w/\Delta x$ and the resulting transition width when validating grid resolution.

The signed-distance profile replaces $x-f(y)$ by the closest-point distance
to the curve, signed positive on the n side. It holds the transition scale
fixed along the local normal and uses the grid floor
$w\ge 2.5\max(\Delta x,\Delta y)$. For unequal doping, $f(y)$ denotes the
midpoint of the grading between $-N_A$ and $N_D$. The zero of $N_{\rm net}$
is at the signed coordinate $(w/2)\ln(N_A/N_D)$ in either profile.

---

## 4. Non-Dimensionalization (Scaling)

Solving in raw SI or CGS units causes severe floating-point overflow (e.g., $\exp(\psi / V_T)$) and ill-conditioned matrices. Before assembling any linear system, scale all quantities using $V_T$, $C_0 = \max(N_A, N_D)$, $L_D = \sqrt{\frac{\varepsilon V_T}{q C_0}}$, $\mu_0 = \mu_n$, and $t_0 = \frac{L_D^2}{\mu_0 V_T}$:

$$\tilde{\psi} = \frac{\psi}{V_T}, \quad \tilde{n} = \frac{n}{C_0}, \quad \tilde{p} = \frac{p}{C_0}, \quad \tilde{N}_{\text{net}} = \frac{N_{\text{net}}}{C_0}, \quad \tilde{x} = \frac{x}{L_D}, \quad \tilde{y} = \frac{y}{L_D}$$

$$\tilde{\mu}_n = 1, \quad \tilde{\mu}_p = \frac{\mu_p}{\mu_n}, \quad \tilde{\tau}_{n,p} = \frac{\tau_{n,p}}{t_0}, \quad \tilde{\mathbf{J}} = \frac{\mathbf{J}}{q C_0 \mu_0 V_T / L_D}, \quad \tilde R = \frac{R\,t_0}{C_0}, \quad \tilde n_i=\frac{n_i}{C_0}$$

Under this scaling, the governing equations become dimensionless and clean:

1. **Poisson:** $\tilde{\nabla}^2 \tilde{\psi} = \tilde{n} - \tilde{p} - \tilde{N}_{\text{net}}$
2. **Continuity:** $\tilde{\nabla} \cdot \tilde{\mathbf{J}}_n = \tilde{R}$, $\quad \tilde{\nabla} \cdot \tilde{\mathbf{J}}_p = -\tilde{R}$
3. **Currents:** $\tilde{\mathbf{J}}_n = \tilde{\mu}_n (-\tilde{n}\tilde{\nabla}\tilde{\psi} + \tilde{\nabla}\tilde{n})$, $\quad \tilde{\mathbf{J}}_p = \tilde{\mu}_p (-\tilde{p}\tilde{\nabla}\tilde{\psi} - \tilde{\nabla}\tilde{p})$

---

## 5. Boundary Conditions

Assume the **p-region is on the left** ($x = 0$) and the **n-region is on the right** ($x = L_x$).

### 5.1 Ohmic Contacts ($x = 0$ and $x = L_x$) — Dirichlet BCs
Contacts enforce local charge neutrality ($\rho = 0$) and mass-action equilibrium ($np = n_i^2$).

* **Left (p-contact, biased at $V_{\text{app}}$):**
  $$p_L = \frac{N_A}{2} + \sqrt{\left(\frac{N_A}{2}\right)^2 + n_i^2}, \qquad n_L = \frac{n_i^2}{p_L}$$
  $$\tilde{\psi}_L = -\ln\left(\frac{p_L}{n_i}\right) + \frac{V_{\text{app}}}{V_T}$$

* **Right (n-contact, grounded at $0\text{ V}$):**
  $$n_R = \frac{N_D}{2} + \sqrt{\left(\frac{N_D}{2}\right)^2 + n_i^2}, \qquad p_R = \frac{n_i^2}{n_R}$$
  $$\tilde{\psi}_R = +\ln\left(\frac{n_R}{n_i}\right)$$

*(Convention: $V_{\text{app}} > 0$ is forward bias; $V_{\text{app}} < 0$ is reverse bias. Ensure $L_x$ is at least $3\times$ the maximum reverse-bias depletion width. For the default $10^{16}\,\text{cm}^{-3}$ symmetric junction and a $-3$ V design sweep, use a domain of at least about $4\,\mu$m in $x$ rather than the original $2\,\mu$m.)*

### 5.2 Top & Bottom Walls ($y = 0$ and $y = L_y$) — Homogeneous Neumann BCs
No electric flux or carrier current crosses the insulating/symmetric top and bottom boundaries:

$$\frac{\partial \tilde{\psi}}{\partial \tilde{y}} = 0, \qquad \frac{\partial \tilde{n}}{\partial \tilde{y}} = 0, \qquad \frac{\partial \tilde{p}}{\partial \tilde{y}} = 0$$

**Discrete treatment:** Nodes on $j=0$ and $j=N_y-1$ own half control volumes of height $\Delta y/2$. Drop the outer face flux and divide the inner $y$-flux by $\Delta y/2$ (equivalently, use a mirror ghost node), for Poisson and for both continuity equations. Quantities integrated over $y$ (such as the terminal current) must use matching trapezoid weights.

---

## 6. Spatial Discretization: FVM & Scharfetter-Gummel

### 6.1 Finite Volume Method (FVM)
Around each node $(i, j)$, draw a rectangular control volume $\Omega_{i,j}$ bounded by cell midpoints $x_{i \pm 1/2}$ and $y_{j \pm 1/2}$. Integrating $\tilde{\nabla} \cdot \tilde{\mathbf{J}}_n = \tilde{R}$ over $\Omega_{i,j}$ and applying the **Divergence Theorem** converts the divergence into a sum of fluxes across the four cell faces:

$$\frac{\tilde{J}_{n, x, i+1/2, j} - \tilde{J}_{n, x, i-1/2, j}}{\Delta \tilde{x}} + \frac{\tilde{J}_{n, y, i, j+1/2} - \tilde{J}_{n, y, i, j-1/2}}{\Delta \tilde{y}} = \tilde{R}_{i,j}$$

Because a shared internal face has one numerical flux used by both neighboring control volumes, the internal face contributions cancel exactly in the assembled discrete balance. Global current conservation is therefore limited only by nonlinear/linear solver residuals and boundary conditions. For Poisson's equation on a uniform grid with constant $\varepsilon$, FVM reduces to the standard 5-point Laplacian stencil.

### 6.2 Scharfetter-Gummel Flux Discretization
Naive central differences for $\mathbf{J}_n$ assume $n(x)$ is linear between nodes. Because $n \propto e^{\tilde{\psi}}$ varies exponentially, central differences cause severe numerical oscillations and negative densities whenever $|\tilde{\psi}_{i+1} - \tilde{\psi}_i| > 2$.

Scharfetter-Gummel assumes **constant current** and **constant electric field** (linear $\tilde{\psi}$) across the interval $[x_i, x_{i+1}]$, then solves the resulting 1st-order ODE analytically using an integrating factor. Defining $z = \Delta\tilde{\psi}_x = \tilde{\psi}_{i+1,j} - \tilde{\psi}_{i,j}$ and the **Bernoulli function**:

$$B(z) = \frac{z}{e^z - 1}$$

the exact face fluxes across $(i+1/2, j)$ are:

$$\tilde{J}_{n, i+1/2, j} = \frac{\tilde{\mu}_n}{\Delta \tilde{x}} \Big[ B(z)\,\tilde{n}_{i+1,j} - B(-z)\,\tilde{n}_{i,j} \Big]$$

$$\tilde{J}_{p, i+1/2, j} = -\frac{\tilde{\mu}_p}{\Delta \tilde{x}} \Big[ B(-z)\,\tilde{p}_{i+1,j} - B(z)\,\tilde{p}_{i,j} \Big]$$

Identical formulas apply to the $y$-faces $(i, j \pm 1/2)$ with $\Delta \tilde{\psi}_y$ and $\Delta \tilde{y}$.

**Continuity-matrix sign convention:** Define
$$A_n\equiv-\nabla_h\cdot\tilde{\mathbf J}_n,\qquad A_p\equiv+\nabla_h\cdot\tilde{\mathbf J}_p.$$ 
The discrete continuity equations are therefore
$$A_n\tilde n=-\tilde R,\qquad A_p\tilde p=-\tilde R.$$ 
This is exactly equivalent to the continuum equations. Dirichlet contact rows are replaced by identity rows with the prescribed carrier value on the right-hand side.

**Numerical Stability of $B(z)$:**
* For $|z| < 10^{-4}$: use Taylor series $B(z) \approx 1 - \frac{z}{2} + \frac{z^2}{12} - \frac{z^4}{720}$ to avoid $0/0$.
* For $z > 500$: $B(z) = 0$.
* For $z < -500$: $B(z) = -z$.

Because $B(z)>0$ for finite $z$, the SG coefficients have the sign structure needed for monotone linear continuity operators. In this implementation define $A_n=-\nabla_h\!\cdot\mathbf J_n$ and $A_p=+\nabla_h\!\cdot\mathbf J_p$; both have positive diagonal and non-positive off-diagonal entries and are M-matrix-type operators under the stated Dirichlet/zero-flux boundary conditions. The corresponding equations are $A_n n=-R$ and $A_p p=-R$.

**Semi-implicit SRH:** Write $R=(np-n_i^2)/D$ with $D=\tau_p(n+n_i)+\tau_n(p+n_i)$ lagged. The electron system becomes $(A_n+\text{diag}(p/D))\,n=n_i^2/D$ and the hole system $(A_p+\text{diag}(n/D))\,p=n_i^2/D$. Both remain M-matrices with non-negative right-hand sides, so carrier positivity is guaranteed even under forward bias.

---

## 7. Numerical Solution Strategy

### 7.1 Stage 1: Thermal Equilibrium ($V_{\text{app}} = 0$)
At zero bias, $n$ and $p$ obey exact Boltzmann statistics ($\tilde{n} = \tilde{n}_i e^{\tilde{\psi}}$, $\tilde{p} = \tilde{n}_i e^{-\tilde{\psi}}$), reducing the system to a single nonlinear PDE:

$$F(\tilde{\psi}) = L\tilde{\psi} - \tilde{n}_i e^{\tilde{\psi}} + \tilde{n}_i e^{-\tilde{\psi}} + \tilde{N}_{\text{net}} = 0$$

where $L$ is the sparse 2D 5-point Laplacian matrix. For a Dirichlet node $k$, use the residual $F_k=\tilde\psi_k-\tilde\psi_{BC,k}$ and Jacobian row $J_{kk}=1$; do not set the Dirichlet RHS to zero unless the prescribed boundary potential is zero. Solve via **Newton-Raphson**:
1. **Initial guess:** Charge-neutral potential $\tilde{\psi}^{(0)} = \text{arcsinh}\left(\frac{\tilde{N}_{\text{net}}}{2\tilde{n}_i}\right)$.
2. **Jacobian:** $J = L - \text{diag}\left(\tilde{n}_i (e^{\tilde{\psi}} + e^{-\tilde{\psi}})\right)$.
3. **Update:** Solve $J \Delta\tilde{\psi} = -F(\tilde{\psi})$ via `spsolve`, clip large steps (`np.clip(dpsi, -2.0, 2.0)`), and iterate until $\max|\Delta\tilde{\psi}| < 10^{-10}$ while also checking the nonlinear residual norm.

### 7.2 Stage 2: Applied Bias (Gummel's Method + Voltage Ramping)
Start from the converged equilibrium state $(\tilde{\psi}_0, \tilde{n}_0, \tilde{p}_0)$ and step $V_{\text{app}}$ incrementally (e.g., $\Delta V = 0.05\text{ V}$ forward, $-0.2\text{ V}$ reverse), warm-starting each bias point from the previous solution.

At each $V_{\text{app}}$, repeat the **Gummel outer loop** until the potential and carrier updates are converged. Measure carrier convergence via quasi-Fermi potentials, $\tilde\phi_n=\tilde\psi-\ln(\tilde n/\tilde n_i)$ and $\tilde\phi_p=\tilde\psi+\ln(\tilde p/\tilde n_i)$: require $\|\Delta\tilde\psi\|_\infty<10^{-6}$, $\|\Delta\tilde\phi_n\|_\infty<10^{-6}$, $\|\Delta\tilde\phi_p\|_\infty<10^{-6}$. A norm relative to $\|\tilde n\|_\infty$ would be dominated by majority carriers and blind to minority-carrier errors, which set the current.

1. **Stabilized Poisson Update (Fixed Quasi-Fermi Levels):**
   Instead of holding $n,p$ strictly constant, linearize their dependence on $\tilde\psi$ while the carrier quasi-Fermi levels are held fixed during this Newton-like Poisson correction. Locally, $\partial\tilde n/\partial\tilde\psi=\tilde n$ and $\partial\tilde p/\partial\tilde\psi=-\tilde p$, giving the stabilizing Jacobian term $-(\tilde n+\tilde p)$:
   $$\Big( L - \text{diag}(\tilde{n} + \tilde{p}) \Big) \Delta\tilde{\psi} = -\Big( L\tilde{\psi} - \tilde{n} + \tilde{p} + \tilde{N}_{\text{net}} \Big)$$
   Update $\tilde{\psi} \leftarrow \tilde{\psi} + \omega \Delta\tilde{\psi}$ (with damping $\omega \in (0, 1]$ if needed). If several inner Newton steps are taken, refresh $\tilde n\leftarrow\tilde n e^{\omega\Delta\tilde\psi}$, $\tilde p\leftarrow\tilde p e^{-\omega\Delta\tilde\psi}$ between them so the quasi-Fermi levels truly stay fixed.
2. **Recombination Update:** Compute $\tilde{R}(\tilde{n}, \tilde{p})$ using the SRH formula.
3. **Electron Continuity:** Assemble $A_n(\tilde\psi)=-\nabla_h\cdot\mathbf J_n$ and solve $A_n\tilde{\mathbf n}=-\tilde R$.
4. **Hole Continuity:** Assemble $A_p(\tilde\psi)=+\nabla_h\cdot\mathbf J_p$ and solve $A_p\tilde{\mathbf p}=-\tilde R$.

*(Numerical note: Gummel iteration can become slower or fail to converge as the forward bias is increased. The $0.65\,$V default sweep limit is a conservative numerical choice for this implementation, not a universal physical threshold for high injection.)*

---

## 8. Post-Processing, Verification & Physics Outputs

Once converged at a target bias, rescale all arrays back to physical units:

* **Potential Heatmap:** $\psi(x,y) = V_T \tilde{\psi}$ (in $\text{V}$).
* **Electric Field Heatmap:** Compute $E_x = -\frac{\partial \psi}{\partial x}$ and $E_y = -\frac{\partial \psi}{\partial y}$ using the same face-centered gradients used by the FVM discretization for quantitative field measurements, then average both components to cell centres before forming $|\mathbf E|=\sqrt{E_x^2+E_y^2}$ in $\text{V/cm}$. `np.gradient` may be used only for smooth visualization/streamlines.
* **Carrier Concentration Heatmaps:** Plot $\log_{10}(n)$ and $\log_{10}(p)$ (in $\text{cm}^{-3}$), since concentrations span ~12 orders of magnitude.
* **Total Current & $I\text{–}V$ Curve:** Evaluate face fluxes $J_{n,x}$ and $J_{p,x}$ using the Scharfetter-Gummel formula and integrate across a vertical slice with trapezoid weights $w_j$ ($\Delta y/2$ at $j=0,N_y-1$, $\Delta y$ otherwise). SG is exact for its local constant-field/constant-current cell assumption, not an exact representation of the continuum solution:
  $$I = \sum_{j} w_j \left( J_{n, x, i+1/2, j} + J_{p, x, i+1/2, j} \right) \qquad [\text{A/cm}]$$

### Recommended Verification & Study Benchmarks
1. **1D Planar Benchmark ($A = 0$).** Neutral Boltzmann contacts give
   $V_{bi}=V_T[\operatorname{asinh}(N_A/2n_i)+\operatorname{asinh}(N_D/2n_i)]$,
   approximately $V_T\ln(N_AN_D/n_i^2)$ for $N_A,N_D\gg n_i$.
   For $V<V_{bi}$, the classical abrupt approximation gives
   $W_d^2=(2\varepsilon/q)(N_A^{-1}+N_D^{-1})(V_{bi}-V)$ and
   $E_{\max}=2(V_{bi}-V)/W_d$. The simulated half-density width uses the
   $n=N_D/2$ and $p=N_A/2$ crossings. The Poisson–Boltzmann first integral
   gives a separate peak-field reference for symmetric abrupt doping at
   equilibrium. For $N_A=N_D\gg n_i$, this corresponds to subtracting
   $2V_T$ from the full barrier. The tests approach that limit by reducing
   $w$ at fine resolution.
2. **Current Conservation Check:** Confirm $I(x)$ is constant across all columns $i$ to within $0.1\%$ after nonlinear convergence. FVM provides exact cancellation of shared internal face fluxes; the reported tolerance reflects solver and boundary residuals.
3. **Curvature study.** Hold $\lambda=L_y$, $w=0.11\,\mu\mathrm{m}$ and
   $V=-2$ V fixed across both profiles. Varying $A$ changes slope as well
   as $\kappa_{\max}=A(2\pi/\lambda)^2$. Compare the planar and largest
   amplitudes on the refined grid at the same physical $w$. Use the field
   map to check where the field maximum lies relative to the curvature extrema.

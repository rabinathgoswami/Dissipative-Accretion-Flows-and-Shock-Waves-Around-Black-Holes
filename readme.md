# Dissipative Accretion Flows and Shock Waves Around Black Holes


Numerical solvers and simulation pipelines for modeling steady-state, viscous, magnetized, dissipative transonic accretion flows and shock wave topologies in the vicinity of black holes.

---

## Overview

Matter accreting onto a black hole must transit from subsonic velocities at large distances to supersonic velocities before crossing the event horizon. This transition occurs at one or more **critical points (sonic points)** where the governing hydrodynamic equations become singular. In dissipative regimes—where viscosity, magnetic fields, and radiative processes act—the flow topology can exhibit multiple critical points, allowing the formation of standing or oscillating **shock waves**.

This repository provides an end-to-end Python framework to:
1. Analytically formulate and numerically solve the critical point conditions for dissipative accretion flows.
2. Classify critical points as **Saddle (X-type)**, **Spiral/Center (O-type)**, or **Nodal** points using eigenvalue and discriminant analysis of the velocity gradient quadratic.
3. Compute globally consistent transonic trajectories across radial grids $x = r/r_g$.
4. Perform parameter sweeps across specific energy ($\varepsilon$), specific angular momentum ($\lambda$), Shakura-Sunyaev viscosity parameter ($\alpha_\Pi$), and plasma magnetic parameter ($\beta$).
5. Generate publication-quality phase portraits, velocity gradients, Mach number profiles, and sound speed distributions in both PNG and PDF formats.

---


## Repository Structure

```text
Dissipative-Accretion-Flows-and-Shock-Waves-Around-Black-Holes/
├── parameters.py                    # Dataclass containing physical constants and flow parameters
├── libraries.py                     # Centralized imports (NumPy, Matplotlib)
├── functions.py                     # Analytical equations, root finders, and critical point solver
├── test.py                          # Unit verification script for the critical point solver
│
├── data_given_eps.py                # Data generator for fixed specific energy (eps)
├── data_given_lambda.py             # Data generator for fixed specific angular momentum (lambda)
├── data_param_sweep_given_eps.py    # Parameter sweep generator (alpha_pi, beta) at fixed eps
├── data_param_sweep_given_lambda.py # Parameter sweep generator (alpha_pi, beta) at fixed lambda
│
├── plot_data_given_eps.py           # Multi-panel phase plotter for fixed eps datasets
├── plot_data_given_lambda.py        # Multi-panel phase plotter for fixed lambda datasets
├── plot_param_sweep_given_eps.py    # Visualizer for eps-fixed parameter sweeps
├── plot_param_sweep_given_lambda.py # Visualizer for lambda-fixed parameter sweeps
├── plot_param_sweep_all_eps.py      # Aggregate visualizer across all energy configurations
├── plot_data_sweep_all_lambda.py    # Aggregate visualizer across all angular momentum sweeps
│
├── data/                            # Raw computed tabular datasets (.dat)
│   ├── eps_given/                   # Radial profiles for fixed energy
│   ├── lam_given/                   # Radial profiles for fixed angular momentum
│   ├── param_eps_given/             # Parameter sweep tables at fixed energy
│   └── param_given/                 # Parameter sweep tables at fixed lambda
│
└── output/                          # Publication-quality figures (PNG and PDF)
    ├── given_eps_outputs/           # Plots for fixed energy runs
    ├── given_lambda_outputs/        # Plots for fixed lambda runs
    ├── param_eps_given_outputs/     # Plots for energy parameter sweeps
    └── param_given_outputs/         # Plots for lambda parameter sweeps
```

---

## Installation & Dependencies

The codebase requires **Python 3.8+** and standard scientific computing packages:

```bash
# Clone the repository
git clone https://github.com/rabinathgoswami/Dissipative-Accretion-Flows-and-Shock-Waves-Around-Black-Holes.git
cd Dissipative-Accretion-Flows-and-Shock-Waves-Around-Black-Holes

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install numpy matplotlib
```

---

## Core Architecture & Modules

### 1. `parameters.py`
Encapsulates all astrophysical constants (in CGS units) and dimensionless flow parameters using a clean Python `dataclass`:
- **Astrophysical Constants:** $G$, $M_\odot$, $c$, $m_e$, $e$.
- **Thermodynamic & Flow Parameters:**
  - `gamma`: Ratio of specific heats ($\gamma = 4/3$ for relativistic or radiation-dominated plasma).
  - `alpha_pi`: Viscosity parameter ($\alpha_\Pi$).
  - `beta`: Magnetic plasma parameter ($p_{\text{mag}} / p_{\text{tot}}$ or related ratio).
  - `m_dot`: Dimensionless accretion rate ($\dot{m}$ in units of Eddington rate).
  - `I_n`, `g`, `mu`: Flow geometric factors and mean molecular weight ($\mu = 0.6$).

### 2. `functions.py`
The computational heart of the repository (~476 lines):
- **Numerical Root Finding:**
  - `bisection_root(f, lo, hi, tol, max_iter)`: Robust interval bisection solver.
  - `all_roots(f, lo, hi, num_samples)`: Grid search scanning for sign transitions followed by local root refinement.
- **Physical Formulations:**
  - `m_xc(p)`: Evaluates sonic Mach number $M_{xc}$ at the critical radius.
  - `a_xc_equation(...)`: Root equation for sound speed $a_{xc}$ at critical points.
  - `a_x_from_eps(p, x_in, eps, lambda_in, M_xc)`: Inverts Bernoulli's energy relation to calculate $a(x)$.
  - `dvdx_equation(...)`: Computes quadratic coefficients $(m_1, m_2, m_3)$ for velocity gradients via l'Hôpital's rule.
  - `solve_dvdx_quadratic(m1, m2, m3)`: Computes real and complex roots for $dv/dx$.
  - `master_critical_point_solver_eps(p, x_in, eps)`: Solves the self-consistent critical point system given radius $x_{in}$ and specific energy $\varepsilon$.


---

## Parameters & Physical Constants

Default parameters configured in `parameters.py`:

| Parameter | Symbol | Default Value | Physical Meaning |
| :--- | :--- | :--- | :--- |
| `gamma` | $\gamma$ | `4/3` | Adiabatic index (relativistic/radiation plasma) |
| `alpha_pi` | $\alpha_\Pi$ | `0.001` | Viscosity parameter ($\Pi$-component) |
| `beta` | $\beta$ | `0.01` | Plasma magnetic parameter ($p_{\text{mag}}/p_{\text{tot}}$) |
| `m_dot` | $\dot{m}$ | `0.233` | Dimensionless accretion rate ($\dot{M}/\dot{M}_{\text{Edd}}$) |
| `mu` | $\mu$ | `0.6` | Mean molecular weight (ionized hydrogen/helium) |
| `I_n` | $I_n$ | `1.0` | Vertical integration structure factor |
| `g` | $g$ | `1.5` | Geometric vertical factor |
| `c` | $c$ | $2.99792 \times 10^{10} \text{ cm s}^{-1}$ | Speed of light in vacuum |
| `G` | $G$ | $6.67430 \times 10^{-8} \text{ cm}^3 \text{ g}^{-1} \text{ s}^{-2}$ | Gravitational constant |
| `M_sun` | $M_\odot$ | $1.989 \times 10^{33} \text{ g}$ | Solar mass |

---

## Data & Output Formats

### Data Files (`data/`)
The generated `.dat` files contain space-separated numerical columns corresponding to:
1. `x_in`: Radial distance $x = r/r_g$
2. `M_xc`: Sonic Mach number
3. `a_xc`: Dimensionless sound speed
4. `v_xc`: Dimensionless radial velocity
5. `dvdx_1`, `dvdx_2`: Velocity gradient roots from the l'Hôpital quadratic equation
6. `lambda_in`: Specific angular momentum
7. `eps`: Specific energy
8. Classification flag / discriminant properties


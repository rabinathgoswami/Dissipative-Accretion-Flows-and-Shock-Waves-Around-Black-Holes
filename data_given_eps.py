import numpy as np
from pathlib import Path
from dataclasses import replace

from parameters import p
from functions import master_critical_point_solver_eps

# ==========================================
# NOTES ON EDITING eps_values
# ==========================================
# 1. Every eps in the list is recomputed and its files overwritten, not
#    just newly added ones. To generate only a new value, set the list to
#    just that value, e.g. eps_values = [0.008]. Folders for eps values
#    not in the list are left untouched.
#
# 2. For each (x_in, eps) the solver searches for the angular momentum lambda
#    on [0.9, 2.5] and the sound speed on [0.001, 1]. If no root exists in
#    those ranges the point is counted as "failed" and skipped. Many failed
#    points for an eps value is normal: it means no critical point with that
#    energy exists at those x_in. If failed = 1000 for a row, the eps value is
#    outside the range where critical points exist for the current parameters.
#
# 3. Negative eps values are written with an "m" (for minus) in names, e.g.
#    eps = -0.04 -> folder "epsm0o04", files "nodalm0o04.dat".


# ==========================================
# SETTINGS
# ==========================================

# (alpha_pi, beta) combinations to generate. Add or remove pairs freely.
param_combos = [
    (0.0, 0.0),
    (0.0, 0.1),
    (0.1, 0.0),
    (0.001, 0.01),   # your original defaults from parameters.py
]

# Any list or numpy array, e.g. np.round(np.arange(-0.04, 0.0101, 0.01), 6)
eps_values = [-0.04, -0.02, 0.0, 0.005, 0.01]

x_space = np.geomspace(2.0, 100.0, 1000)

base_dir = Path("data") / "eps_given"
root_tol = 1e-8

# Column order of every .dat file (8 columns)
header = (
    "x_in eps lambda_in M_xc v_xc a_xc "
    "dvdx_root_1 dvdx_root_2"
)
N_COLS = 8


# ==========================================
# HELPERS
# ==========================================

def tag(value):
    """1.6 -> '1o6', 0.001 -> '0o001', 0 -> '0o0', -0.04 -> 'm0o04'."""
    value = float(value)
    prefix = "m" if value < 0 else ""
    s = f"{round(abs(value), 6):.6f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return prefix + s.replace(".", "o")


def save_data(output_dir, filename, data):
    # reshape(-1, N_COLS) lets an empty dataset still write a valid file with a header
    np.savetxt(
        output_dir / filename,
        np.asarray(data, dtype=float).reshape(-1, N_COLS),
        header=header,
        fmt="%.12e",
    )
    print(f"    {filename}: {len(data)} rows saved")


def generate_for_eps(p_case, eps_in, case_dir):
    eps_tag = tag(eps_in)
    output_dir = case_dir / f"eps{eps_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    nodal_data, saddle_data, spiral_data = [], [], []
    failed_points = 0

    for x_in in x_space:
        try:
            (lambda_roots, M_xc, a_xc, v_xc, dvdx_roots,
             lambda_val, eps_out, a_from_eps) = master_critical_point_solver_eps(p_case, x_in, eps_in)
        except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
            failed_points += 1
            continue

        root1, root2 = dvdx_roots
        row = [x_in, eps_out, lambda_val, M_xc, v_xc, a_xc, root1, root2]

        both_real_nonzero = (
            not np.isnan(root1) and not np.isnan(root2)
            and abs(root1) > root_tol and abs(root2) > root_tol
        )

        if both_real_nonzero and root1 * root2 > 0:
            nodal_data.append(row)       # same sign
        elif both_real_nonzero and root1 * root2 < 0:
            saddle_data.append(row)      # opposite sign
        else:
            spiral_data.append(row)      # everything else

    print(f"  eps = {eps_in}  ->  {output_dir}")
    save_data(output_dir, f"nodal{eps_tag}.dat", nodal_data)
    save_data(output_dir, f"saddle{eps_tag}.dat", saddle_data)
    save_data(output_dir, f"spiral{eps_tag}.dat", spiral_data)

    return len(nodal_data), len(saddle_data), len(spiral_data), failed_points


# ==========================================
# MAIN LOOP
# ==========================================

summary = []

for alpha_pi, beta in param_combos:
    # Copy of the parameters with only alpha_pi and _beta changed
    p_case = replace(p, alpha_pi=alpha_pi, _beta=beta)

    case_name = f"alpha{tag(alpha_pi)}_beta{tag(beta)}"
    case_dir = base_dir / case_name
    print(f"\n##### {case_name}  (alpha_pi={alpha_pi}, beta={beta}) #####")

    for eps in eps_values:
        counts = generate_for_eps(p_case, eps, case_dir)
        summary.append((alpha_pi, beta, eps, *counts))

print("\n====================== DATA GENERATION SUMMARY ======================")
print(f"{'alpha_pi':>9} {'beta':>7} {'eps':>8} {'nodal':>7} {'saddle':>7} {'spiral':>7} {'failed':>7}")
for a, b, e, n, s, sp, f in summary:
    print(f"{a:>9} {b:>7} {e:>8} {n:>7} {s:>7} {sp:>7} {f:>7}")
print(f"Total x_in per case: {len(x_space)}")
print(f"Output base folder : {base_dir.resolve()}")
print("=====================================================================")
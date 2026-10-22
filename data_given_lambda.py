import numpy as np
from pathlib import Path
from dataclasses import replace

from parameters import p
from functions import master_critical_point_solver

# ==========================================
# NOTES ON EDITING lambda_values
# ==========================================
# 1. Every lambda in the list is recomputed and its files overwritten, not
#    just newly added ones. To generate only a new value, set the list to
#    just that value, e.g. lambda_values = [1.85]. Folders for lambda values
#    not in the list are left untouched.
#
# 2. The new lambda must work with master_critical_point_solver. It searches
#    for the sound speed on [0.001, 1], so a lambda with no root in that range
#    fails at every x_in. The summary table then shows failed = 1000 for that
#    row, and its .dat files contain only the header. This means the value is
#    outside the range where critical points exist for the current parameters.


# ==========================================
# SETTINGS
# ==========================================

# (alpha_pi, beta) combinations to generate. Add or remove pairs freely.
param_combos = [
    (0.0, 0.0),
    (0.0, 0.1),
    (0.1, 0.0),
    (0.001, 0.01)   # your original defaults from parameters.py
]

# Any list or numpy array, e.g. np.round(np.arange(1.5, 1.91, 0.05), 6)
lambda_values = [1.6, 1.65, 1.7, 1.75, 1.8]

x_space = np.geomspace(2.0, 100.0, 1000)

base_dir = Path("data") / "lam_given"
root_tol = 1e-8

header = (
    "x_in lambda_in M_xc v_xc a_xc N D "
    "dvdx_root_1 dvdx_root_2 eps discriminant"
)


# ==========================================
# HELPERS
# ==========================================

def tag(value):
    """1.6 -> '1o6', 0.001 -> '0o001', 0 -> '0o0' (used in folder/file names)."""
    s = f"{round(float(value), 6):.6f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s.replace(".", "o")


def save_data(output_dir, filename, data):
    # reshape(-1, 11) lets an empty dataset still write a valid file with a header
    np.savetxt(
        output_dir / filename,
        np.asarray(data, dtype=float).reshape(-1, 11),
        header=header,
        fmt="%.12e",
    )
    print(f"    {filename}: {len(data)} rows saved")


def generate_for_lambda(p_case, lambda_in, case_dir):
    lam_tag = tag(lambda_in)
    output_dir = case_dir / f"lambda{lam_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    nodal_data, saddle_data, spiral_data = [], [], []
    failed_points = 0

    for x_in in x_space:
        try:
            (M_xc, v_xc, a_xc, N, D,
             dvdx_roots, eps, discriminant) = master_critical_point_solver(p_case, x_in, lambda_in)
        except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
            failed_points += 1
            continue

        root1, root2 = dvdx_roots
        row = [x_in, lambda_in, M_xc, v_xc, a_xc, N, D,
               root1, root2, eps, discriminant]

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

    print(f"  lambda_in = {lambda_in}  ->  {output_dir}")
    save_data(output_dir, f"nodal{lam_tag}.dat", nodal_data)
    save_data(output_dir, f"saddle{lam_tag}.dat", saddle_data)
    save_data(output_dir, f"spiral{lam_tag}.dat", spiral_data)

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

    for lam in lambda_values:
        counts = generate_for_lambda(p_case, lam, case_dir)
        summary.append((alpha_pi, beta, lam, *counts))

print("\n====================== DATA GENERATION SUMMARY ======================")
print(f"{'alpha_pi':>9} {'beta':>7} {'lambda':>7} {'nodal':>7} {'saddle':>7} {'spiral':>7} {'failed':>7}")
for a, b, lam, n, s, sp, f in summary:
    print(f"{a:>9} {b:>7} {lam:>7} {n:>7} {s:>7} {sp:>7} {f:>7}")
print(f"Total x_in per case: {len(x_space)}")
print(f"Output base folder : {base_dir.resolve()}")
print("=====================================================================")
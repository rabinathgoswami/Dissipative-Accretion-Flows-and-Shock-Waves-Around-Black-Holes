import numpy as np
from pathlib import Path

from parameters import p
from functions import master_critical_point_solver


# ==========================================
# SETTINGS
# ==========================================

# Any list or numpy array works, e.g. [1.6, 1.65, 1.7] or np.round(np.arange(1.5, 1.91, 0.05), 6)
lambda_values = [1.6, 1.65, 1.7, 1.75, 1.8]

x_space = np.geomspace(2.0, 100.0, 1000)

base_dir = Path("data")
root_tol = 1e-8

header = (
    "x_in lambda_in M_xc v_xc a_xc N D "
    "dvdx_root_1 dvdx_root_2 eps discriminant"
)


# ==========================================
# HELPERS
# ==========================================

def lambda_tag(lam):
    """1.6 -> '1o6', 1.65 -> '1o65', 2.0 -> '2o0' (matches your folder names)."""
    s = f"{round(float(lam), 6):.6f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s.replace(".", "o")


def save_data(output_dir, filename, data):
    # reshape(-1, 11) makes an empty dataset still write a valid file with a header
    np.savetxt(
        output_dir / filename,
        np.asarray(data, dtype=float).reshape(-1, 11),
        header=header,
        fmt="%.12e",
    )
    print(f"  {filename}: {len(data)} rows saved")


def generate_for_lambda(lambda_in):
    tag = lambda_tag(lambda_in)
    output_dir = base_dir / f"lambda{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    nodal_data, saddle_data, spiral_data = [], [], []
    failed_points = 0

    for x_in in x_space:
        try:
            (M_xc, v_xc, a_xc, N, D,
             dvdx_roots, eps, discriminant) = master_critical_point_solver(p, x_in, lambda_in)
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

    print(f"lambda_in = {lambda_in}  ->  {output_dir}")
    save_data(output_dir, f"nodal{tag}.dat", nodal_data)
    save_data(output_dir, f"saddle{tag}.dat", saddle_data)
    save_data(output_dir, f"spiral{tag}.dat", spiral_data)

    return len(nodal_data), len(saddle_data), len(spiral_data), failed_points


# ==========================================
# MAIN LOOP
# ==========================================

summary = []
for lam in lambda_values:
    counts = generate_for_lambda(lam)
    summary.append((lam, *counts))

print("\n=============== DATA GENERATION SUMMARY ===============")
print(f"{'lambda':>8} {'nodal':>8} {'saddle':>8} {'spiral':>8} {'failed':>8}")
for lam, n, s, sp, f in summary:
    print(f"{lam:>8} {n:>8} {s:>8} {sp:>8} {f:>8}")
print(f"Total x_in per lambda: {len(x_space)}")
print(f"Output base folder   : {base_dir.resolve()}")
print("========================================================")
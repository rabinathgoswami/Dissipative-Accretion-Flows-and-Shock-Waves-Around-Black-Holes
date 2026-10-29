import numpy as np
from pathlib import Path
from dataclasses import replace

from parameters import p
from functions import master_critical_point_solver

# ==========================================
# NOTES ON EDITING THE SWEEP
# ==========================================
# 1. SWEEP_PARAM chooses which parameter is varied ("alpha_pi" or "beta"). The OTHER one
#    stays fixed at FIXED_BETA or FIXED_ALPHA_PI. Lambda is fixed at lambda_in. Only the
#    fixed value of the parameter that is NOT being swept is used; the other is ignored.
#
# 2. Every value in sweep_values is recomputed and its files overwritten. Folders for values
#    not in the list are left untouched. To generate only one new value, set the list to
#    just that value, e.g. sweep_values = [0.05].
#
# 3. A value for which no critical point exists (sound-speed root not found on [0.001, 1])
#    fails at every x_in: the summary then shows failed = 1000 for that row and its .dat
#    files contain only the header.


# ==========================================
# SETTINGS
# ==========================================

SWEEP_PARAM = "beta"              # "alpha_pi" or "beta"  <- the parameter that is toggled

# Values of the swept parameter. Any list or numpy array, e.g. np.round(np.arange(0.0, 0.11, 0.02), 6)
sweep_values = [0.0,0.2,0.4,0.6,0.8,1.0]

FIXED_ALPHA_PI = 0.00               # used when SWEEP_PARAM == "beta"
FIXED_BETA = 0.0                    # used when SWEEP_PARAM == "alpha_pi"
lambda_in = 1.7                       # fixed angular momentum

x_space = np.geomspace(2.0, 100.0, 1000)

base_dir = Path("data") / "param_given"
root_tol = 1e-8

# Column order of every .dat file (same as the lam_given files)
header = (
    "x_in lambda_in M_xc v_xc a_xc N D "
    "dvdx_root_1 dvdx_root_2 eps discriminant"
)
N_COLS = 11


# ==========================================
# HELPERS
# ==========================================

# name used in folders, and the field name in the Parameters dataclass
PARAM_INFO = {
    "alpha_pi": dict(short="alpha", field="alpha_pi"),
    "beta":     dict(short="beta",  field="_beta"),
}


def tag(value):
    """1.6 -> '1o6', 0.001 -> '0o001', 0 -> '0o0' (used in folder/file names)."""
    s = f"{round(float(value), 6):.6f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s.replace(".", "o")


def save_data(output_dir, filename, data):
    # reshape(-1, N_COLS) lets an empty dataset still write a valid file with a header
    np.savetxt(
        output_dir / filename,
        np.asarray(data, dtype=float).reshape(-1, N_COLS),
        header=header,
        fmt="%.12e",
    )
    print(f"    {filename}: {len(data)} rows saved")


def generate_for_value(p_case, value_tag, case_dir):
    output_dir = case_dir
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

    save_data(output_dir, f"nodal{value_tag}.dat", nodal_data)
    save_data(output_dir, f"saddle{value_tag}.dat", saddle_data)
    save_data(output_dir, f"spiral{value_tag}.dat", spiral_data)

    return len(nodal_data), len(saddle_data), len(spiral_data), failed_points


# ==========================================
# MAIN LOOP
# ==========================================

if SWEEP_PARAM not in PARAM_INFO:
    raise ValueError(f'SWEEP_PARAM must be "alpha_pi" or "beta", got {SWEEP_PARAM!r}')

swept = PARAM_INFO[SWEEP_PARAM]
if SWEEP_PARAM == "alpha_pi":
    fixed_name, fixed_value = "beta", FIXED_BETA
    fixed = PARAM_INFO["beta"]
else:
    fixed_name, fixed_value = "alpha_pi", FIXED_ALPHA_PI
    fixed = PARAM_INFO["alpha_pi"]

# e.g. data/param_given/alpha_sweep_beta0o01_lambda1o6/
sweep_dir = base_dir / f"{swept['short']}_sweep_{fixed['short']}{tag(fixed_value)}_lambda{tag(lambda_in)}"

print(f"Sweeping {SWEEP_PARAM}  |  fixed {fixed_name} = {fixed_value}  |  lambda_in = {lambda_in}")
print(f"Output: {sweep_dir}")

summary = []

for value in sweep_values:
    # Copy of the parameters with the swept and the fixed parameter set
    p_case = replace(p, **{swept["field"]: value, fixed["field"]: fixed_value})

    value_tag = tag(value)
    case_dir = sweep_dir / f"{swept['short']}{value_tag}"
    print(f"\n##### {SWEEP_PARAM} = {value}  ->  {case_dir} #####")

    counts = generate_for_value(p_case, value_tag, case_dir)
    summary.append((value, *counts))

print("\n====================== DATA GENERATION SUMMARY ======================")
print(f"Swept: {SWEEP_PARAM}   Fixed: {fixed_name} = {fixed_value}, lambda_in = {lambda_in}")
print(f"{SWEEP_PARAM:>10} {'nodal':>7} {'saddle':>7} {'spiral':>7} {'failed':>7}")
for v, n, s, sp, f in summary:
    print(f"{v:>10} {n:>7} {s:>7} {sp:>7} {f:>7}")
print(f"Total x_in per value: {len(x_space)}")
print(f"Output folder       : {sweep_dir.resolve()}")
print("=====================================================================")
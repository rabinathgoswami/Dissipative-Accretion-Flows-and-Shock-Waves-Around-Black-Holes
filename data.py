
import numpy as np
from pathlib import Path

from parameters import p
from functions import master_critical_point_solver


# ==========================================
# DATA GENERATION FOR lambda_in = 1.2
# ==========================================

lambda_in = 1.2

# 1000 equally spaced points, including 1 and 500
x_space = np.linspace(2.0, 1000.0, 5000)

# Output directory
output_dir = Path("data") / "lambda1o2"
output_dir.mkdir(parents=True, exist_ok=True)


# Store results for the three classifications
nodal_data = []
saddle_data = []
spiral_data = []

# Track unsuccessful calculations
failed_points = 0
zero_root_points = 0


for x_in in x_space:

    try:
        (
            M_xc,
            v_xc,
            a_xc,
            N,
            D,
            dvdx_roots,
            eps,
            discriminant
        ) = master_critical_point_solver(p, x_in, lambda_in)

        root1, root2 = dvdx_roots

        # Each row contains:
        # x_in, lambda_in, M_xc, v_xc, a_xc, N, D,
        # dvdx_root_1, dvdx_root_2, eps, discriminant

        row = [
            x_in,
            lambda_in,
            M_xc,
            v_xc,
            a_xc,
            N,
            D,
            root1,
            root2,
            eps,
            discriminant
        ]

        root_tol = 1e-8


        # 1. Real roots with the same sign: Nodal
        if (
            not np.isnan(root1)
            and not np.isnan(root2)
            and abs(root1) > root_tol
            and abs(root2) > root_tol
            and root1 * root2 > 0
        ):
            nodal_data.append(row)

        # 2. Real roots with opposite signs: Saddle
        elif (
            not np.isnan(root1)
            and not np.isnan(root2)
            and abs(root1) > root_tol
            and abs(root2) > root_tol
            and root1 * root2 < 0
        ):
            saddle_data.append(row)

        # 3. Everything else: Spiral
        else:
            spiral_data.append(row)
            
    except (
        ValueError,
        ZeroDivisionError,
        OverflowError,
        FloatingPointError
    ):
        failed_points += 1



# ==========================================
# SAVE DATA TO .DAT FILES
# ==========================================

header = (
    "x_in lambda_in M_xc v_xc a_xc N D "
    "dvdx_root_1 dvdx_root_2 eps discriminant"
)


def save_data(filename, data):
    filepath = output_dir / filename

    # Ensure even an empty dataset produces a valid empty file
    # with its column header.
    np.savetxt(
        filepath,
        np.asarray(data, dtype=float).reshape(-1, 11),
        header=header,
        fmt="%.12e"
    )

    print(f"{filename}: {len(data)} rows saved")


save_data("nodal1o2.dat", nodal_data)
save_data("saddle1o2.dat", saddle_data)
save_data("spiral1o2.dat", spiral_data)


# ==========================================
# SUMMARY
# ==========================================

print("\n========== DATA GENERATION SUMMARY ==========")
print(f"lambda_in       : {lambda_in}")
print(f"Total x_in      : {len(x_space)}")
print(f"Nodal points    : {len(nodal_data)}")
print(f"Saddle points   : {len(saddle_data)}")
print(f"Spiral points   : {len(spiral_data)}")
print(f"Zero-root cases : {zero_root_points}")
print(f"Failed points   : {failed_points}")
print(f"Output folder   : {output_dir.resolve()}")
print("=============================================")

import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# ==========================================
# SETTINGS
# ==========================================

data_dir = Path("data") / "lam_given"
out_dir = Path("output") / "given_lambda_outputs"

SAVE_PNG = True
SAVE_PDF = True
SHOW_PLOTS = False        # True = also open the plot windows

# Column indices in the .dat files:
# x_in lambda_in M_xc v_xc a_xc N D dvdx_root_1 dvdx_root_2 eps discriminant
COL_X, COL_EPS = 0, 9

# Colour = critical-point type. Listed in drawing order (last is drawn on top).
CLASS_STYLE = {
    "saddle": dict(color="black", label="Saddle"),
    "spiral": dict(color="blue",  label="Spiral"),
    "nodal":  dict(color="red",   label="Nodal"),
}

# Marker shape = lambda (cycles if you have more lambdas than markers)
LAMBDA_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

POINT_SIZE = 6
POINT_ALPHA = 0.6

YLIM = (-0.08, 0.05)                  # fixed vertical axis limits for all plots

# Legend positions inside the axes (anchor = lower right corner of each legend)
TYPE_LEGEND_ANCHOR = (0.99, 0.01)     # critical-point type legend (bottom)
LAMBDA_LEGEND_ANCHOR = (0.99, 0.23)   # lambda legend, stacked just above it


# ==========================================
# HELPERS
# ==========================================

def tag_to_float(tag):
    """'1o65' -> 1.65, '0o001' -> 0.001"""
    return float(tag.replace("o", "."))


def load_dat(path):
    """Return (x, eps) arrays; empty arrays if the file is missing or has no rows."""
    if not path.exists():
        return np.array([]), np.array([])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # numpy warns on header-only files
        arr = np.loadtxt(path, ndmin=2)
    if arr.size == 0:
        return np.array([]), np.array([])
    return arr[:, COL_X], arr[:, COL_EPS]


# ==========================================
# MAIN
# ==========================================

if not data_dir.exists():
    raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()}")

out_dir.mkdir(parents=True, exist_ok=True)

case_pattern = re.compile(r"alpha(\d+o\d+)_beta(\d+o\d+)$")
lam_pattern = re.compile(r"lambda(\d+o\d+)$")

case_folders = sorted(d for d in data_dir.iterdir() if d.is_dir() and case_pattern.match(d.name))
if not case_folders:
    raise FileNotFoundError(f"No alpha..._beta... folders found in {data_dir.resolve()}")

for case_dir in case_folders:
    m = case_pattern.match(case_dir.name)
    alpha_pi, beta = tag_to_float(m.group(1)), tag_to_float(m.group(2))

    # lambda folders, sorted numerically
    lam_dirs = [d for d in case_dir.iterdir() if d.is_dir() and lam_pattern.match(d.name)]
    lam_dirs.sort(key=lambda d: tag_to_float(lam_pattern.match(d.name).group(1)))
    if not lam_dirs:
        print(f"{case_dir.name}: no lambda folders, skipped")
        continue

    fig, ax = plt.subplots(figsize=(10, 6.3))

    lam_values = []
    for i, lam_dir in enumerate(lam_dirs):
        lam_tag = lam_pattern.match(lam_dir.name).group(1)
        lam_values.append(tag_to_float(lam_tag))
        marker = LAMBDA_MARKERS[i % len(LAMBDA_MARKERS)]

        for cls, style in CLASS_STYLE.items():          # saddle, spiral, then nodal on top
            x, eps = load_dat(lam_dir / f"{cls}{lam_tag}.dat")
            if x.size == 0:
                continue
            ax.scatter(
                np.log10(x), eps,
                color=style["color"], marker=marker,
                s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0,
            )

    # Legend 1: critical-point type (colour)
    type_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               color=s["color"], label=s["label"])
        for s in CLASS_STYLE.values()
    ]
    leg_type = ax.legend(handles=type_handles, title="Critical-point type",
                         loc="lower right", bbox_to_anchor=TYPE_LEGEND_ANCHOR,
                         framealpha=0.9)
    ax.add_artist(leg_type)

    # Legend 2: lambda (marker shape)
    lam_handles = [
        Line2D([0], [0], marker=LAMBDA_MARKERS[i % len(LAMBDA_MARKERS)], linestyle="None",
               markersize=8, color="dimgray", label=rf"$\lambda_{{in}} = {lam:g}$")
        for i, lam in enumerate(lam_values)
    ]
    ax.legend(handles=lam_handles, title=r"$\lambda_{in}$",
              loc="lower right", bbox_to_anchor=LAMBDA_LEGEND_ANCHOR,
              framealpha=0.9)

    ax.set_xlabel(r"$\log_{10}(x_{\rm in})$", fontsize=14)
    ax.set_ylim(*YLIM)
    ax.set_ylabel(r"$\varepsilon_{\rm critical}$", fontsize=14)
    ax.set_title(
        rf"Critical Energy vs $\log_{{10}}(x_{{\rm in}})$ for Different $\lambda_{{\rm in}}$ "
        rf"with $\alpha_\pi = {alpha_pi:g},\ \beta = {beta:g}$",
        fontsize=14,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()

    base = f"eps_vs_log10_xin_{case_dir.name}"
    if SAVE_PNG:
        fig.savefig(out_dir / f"{base}.png", dpi=300, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(out_dir / f"{base}.pdf", bbox_inches="tight")
    print(f"saved: {out_dir / base}.(png/pdf)")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

print(f"\nAll plots saved in: {out_dir.resolve()}")
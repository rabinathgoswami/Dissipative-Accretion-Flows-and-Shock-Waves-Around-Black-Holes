import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# ==========================================
# SETTINGS
# ==========================================

data_dir = Path("data") / "eps_given"
out_dir = Path("output") / "given_eps_outputs"

SAVE_PNG = True
SAVE_PDF = True
SHOW_PLOTS = False        # True = also open the plot windows

# Column indices in the .dat files:
# x_in eps lambda_in M_xc v_xc a_xc dvdx_root_1 dvdx_root_2
COL_X, COL_LAMBDA = 0, 2

# Colour = critical-point type. Listed in drawing order (last is drawn on top).
CLASS_STYLE = {
    "saddle": dict(color="black", label="Saddle"),
    "spiral": dict(color="blue",  label="Spiral"),
    "nodal":  dict(color="red",   label="Nodal"),
}

# Marker shape = eps (cycles if you have more eps values than markers)
EPS_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

POINT_SIZE = 6
POINT_ALPHA = 0.6

YLIM = None                           # e.g. (1.0, 2.5) to fix the vertical axis; None = automatic

# Legend positions inside the axes (axes-fraction coordinates)
TYPE_LEGEND_LOC = "lower left"
TYPE_LEGEND_ANCHOR = (0.01, 0.01)     # critical-point type legend (bottom left)
EPS_LEGEND_LOC = "lower left"
EPS_LEGEND_ANCHOR = (0.17, 0.01)      # eps legend, just to its right


# ==========================================
# HELPERS
# ==========================================

def tag_to_float(tag):
    """'1o65' -> 1.65, '0o001' -> 0.001, 'm0o04' -> -0.04"""
    sign = -1.0 if tag.startswith("m") else 1.0
    return sign * float(tag.lstrip("m").replace("o", "."))


def load_dat(path):
    """Return (x, lambda) arrays; empty arrays if the file is missing or has no rows."""
    if not path.exists():
        return np.array([]), np.array([])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # numpy warns on header-only files
        arr = np.loadtxt(path, ndmin=2)
    if arr.size == 0:
        return np.array([]), np.array([])
    return arr[:, COL_X], arr[:, COL_LAMBDA]


# ==========================================
# MAIN
# ==========================================

if not data_dir.exists():
    raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()}")

out_dir.mkdir(parents=True, exist_ok=True)

case_pattern = re.compile(r"alpha(\d+o\d+)_beta(\d+o\d+)$")
eps_pattern = re.compile(r"eps(m?\d+o\d+)$")

case_folders = sorted(d for d in data_dir.iterdir() if d.is_dir() and case_pattern.match(d.name))
if not case_folders:
    raise FileNotFoundError(f"No alpha..._beta... folders found in {data_dir.resolve()}")

for case_dir in case_folders:
    m = case_pattern.match(case_dir.name)
    alpha_pi, beta = tag_to_float(m.group(1)), tag_to_float(m.group(2))

    # eps folders, sorted numerically (most negative first)
    eps_dirs = [d for d in case_dir.iterdir() if d.is_dir() and eps_pattern.match(d.name)]
    eps_dirs.sort(key=lambda d: tag_to_float(eps_pattern.match(d.name).group(1)))
    if not eps_dirs:
        print(f"{case_dir.name}: no eps folders, skipped")
        continue

    fig, ax = plt.subplots(figsize=(10, 6.3))

    eps_values = []
    for i, eps_dir in enumerate(eps_dirs):
        eps_tag = eps_pattern.match(eps_dir.name).group(1)
        eps_values.append(tag_to_float(eps_tag))
        marker = EPS_MARKERS[i % len(EPS_MARKERS)]

        for cls, style in CLASS_STYLE.items():          # saddle, spiral, then nodal on top
            x, lam = load_dat(eps_dir / f"{cls}{eps_tag}.dat")
            if x.size == 0:
                continue
            ax.scatter(
                np.log10(x), lam,
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
                         loc=TYPE_LEGEND_LOC, bbox_to_anchor=TYPE_LEGEND_ANCHOR,
                         framealpha=0.9)
    ax.add_artist(leg_type)

    # Legend 2: eps (marker shape)
    eps_handles = [
        Line2D([0], [0], marker=EPS_MARKERS[i % len(EPS_MARKERS)], linestyle="None",
               markersize=8, color="dimgray", label=rf"$\varepsilon = {e:g}$")
        for i, e in enumerate(eps_values)
    ]
    ax.legend(handles=eps_handles, title=r"$\varepsilon$",
              loc=EPS_LEGEND_LOC, bbox_to_anchor=EPS_LEGEND_ANCHOR,
              framealpha=0.9)

    if YLIM is not None:
        ax.set_ylim(*YLIM)
    ax.set_xlabel(r"$\log_{10}(x_{\rm in})$", fontsize=14)
    ax.set_ylabel(r"$\lambda_{\rm in}$", fontsize=14)
    ax.set_title(
        rf"$\lambda_{{\rm in}}$ vs $\log_{{10}}(x_{{\rm in}})$ for Different $\varepsilon$ "
        rf"with $\alpha_\pi = {alpha_pi:g},\ \beta = {beta:g}$",
        fontsize=14,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()

    base = f"lambda_vs_log10_xin_{case_dir.name}"
    if SAVE_PNG:
        fig.savefig(out_dir / f"{base}.png", dpi=300, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(out_dir / f"{base}.pdf", bbox_inches="tight")
    print(f"saved: {out_dir / base}.(png/pdf)")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

print(f"\nAll plots saved in: {out_dir.resolve()}")
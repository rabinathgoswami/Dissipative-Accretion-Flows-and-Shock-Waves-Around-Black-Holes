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

POINT_MARKER = "o"                    # same marker for every lambda
POINT_SIZE = 6
POINT_ALPHA = 0.6

YLIM = (-0.08, 0.05)                  # fixed vertical axis limits for all plots

# Curve labels: each lambda value is placed automatically in free space next to its
# own curve. A position is only accepted if the label does not touch any data point,
# another label, or the legend.
LABEL_FONTSIZE = 10
LABEL_TEXT = "{:g}"                   # text of each label; the legend explains that these are lambda_in values
LABEL_CLEARANCE_PTS = 3               # minimum empty space around each label (in points)
LABEL_SEARCH_RADIUS_PTS = 110         # how far from the curve's minimum to look for free space

# Legend position inside the axes (anchor = lower right corner)
TYPE_LEGEND_LOC = "lower right"
TYPE_LEGEND_ANCHOR = (0.99, 0.01)


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


def rect_to_points_distance(centres, hw, hh, pts, chunk=400):
    """Distance from each rectangle (centre, half-width hw, half-height hh) to the nearest
    point in `pts`. Returns an array with one value per rectangle (0 = a point is inside)."""
    out = np.empty(len(centres))
    if len(pts) == 0:
        out[:] = np.inf
        return out
    for s in range(0, len(centres), chunk):
        c = centres[s:s + chunk]
        dx = np.maximum(np.abs(pts[None, :, 0] - c[:, None, 0]) - hw, 0.0)
        dy = np.maximum(np.abs(pts[None, :, 1] - c[:, None, 1]) - hh, 0.0)
        out[s:s + chunk] = np.sqrt(dx * dx + dy * dy).min(axis=1)
    return out


def place_curve_labels(fig, ax, curves, legend, make_text):
    """Place one label per curve in free space, close to (and clearly nearest to) its own curve."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    clearance = LABEL_CLEARANCE_PTS * px_per_pt
    radius = LABEL_SEARCH_RADIUS_PTS * px_per_pt
    step = max(2.0, 2.0 * px_per_pt)

    ax_bb = ax.get_window_extent(renderer)
    curve_px = [ax.transData.transform(np.column_stack([xs, ys])) for _, xs, ys in curves]

    # rectangles that labels must not overlap: legend first, then labels already placed
    blocked = []
    if legend is not None:
        bb = legend.get_window_extent(renderer)
        blocked.append((bb.x0, bb.y0, bb.x1, bb.y1))

    for k, (value, xs, ys) in enumerate(curves):
        text = make_text(value)

        # measure the exact size of this label
        probe = ax.text(0, 0, text, fontsize=LABEL_FONTSIZE, ha="center", va="center")
        bb = probe.get_window_extent(renderer)
        probe.remove()
        hw, hh = bb.width / 2.0, bb.height / 2.0

        own = curve_px[k]
        others = np.vstack([p for j, p in enumerate(curve_px) if j != k]) if len(curve_px) > 1 else np.empty((0, 2))
        anchor = own[np.argmin(ys)]                       # minimum of this curve (data space min eps)

        # keep only points near the search area (speeds things up)
        lo, hi = anchor - radius - 2 * max(hw, hh), anchor + radius + 2 * max(hw, hh)
        near = lambda P: P[(P[:, 0] > lo[0]) & (P[:, 0] < hi[0]) & (P[:, 1] > lo[1]) & (P[:, 1] < hi[1])]
        own_n, oth_n = near(own), near(others)

        # candidate centres on a grid around the anchor, kept inside the axes
        gx = np.arange(anchor[0] - radius, anchor[0] + radius + 1, step)
        gy = np.arange(anchor[1] - radius, anchor[1] + radius + 1, step)
        C = np.array(np.meshgrid(gx, gy)).reshape(2, -1).T
        inside = ((C[:, 0] - hw > ax_bb.x0 + clearance) & (C[:, 0] + hw < ax_bb.x1 - clearance) &
                  (C[:, 1] - hh > ax_bb.y0 + clearance) & (C[:, 1] + hh < ax_bb.y1 - clearance))
        C = C[inside]

        # reject positions overlapping the legend or earlier labels
        for (x0, y0, x1, y1) in blocked:
            clash = ((C[:, 0] + hw + clearance > x0) & (C[:, 0] - hw - clearance < x1) &
                     (C[:, 1] + hh + clearance > y0) & (C[:, 1] - hh - clearance < y1))
            C = C[~clash]

        d_own = rect_to_points_distance(C, hw, hh, own_n)
        d_oth = rect_to_points_distance(C, hw, hh, oth_n)
        free = (d_own > clearance) & (d_oth > clearance)
        valid = free & (d_own < d_oth)               # label must be nearer to its own curve than to any other

        if valid.any():
            Cf, do, dt = C[valid], d_own[valid], d_oth[valid]
            # prefer: close to own curve, clearly farther from other curves, near the minimum
            score = (do - 0.3 * np.minimum(dt, 5 * clearance)
                     + 0.02 * np.hypot(*(Cf - anchor).T)
                     + 0.5 * np.maximum(0.0, Cf[:, 1] - anchor[1]))      # mild preference for below the minimum
            centre = Cf[np.argmin(score)]
        else:
            print(f"  warning: no free spot found for label '{text}'; try a smaller LABEL_FONTSIZE")
            centre = anchor + np.array([0.0, -(hh + clearance + 4 * px_per_pt)])

        x_d, y_d = ax.transData.inverted().transform(centre)
        ax.text(x_d, y_d, text, fontsize=LABEL_FONTSIZE, ha="center", va="center", zorder=5)
        blocked.append((centre[0] - hw, centre[1] - hh, centre[0] + hw, centre[1] + hh))


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

    curves = []                                  # (lambda value, all x, all eps) per lambda
    for lam_dir in lam_dirs:
        lam_tag = lam_pattern.match(lam_dir.name).group(1)
        all_x, all_eps = [], []

        for cls, style in CLASS_STYLE.items():          # saddle, spiral, then nodal on top
            x, eps = load_dat(lam_dir / f"{cls}{lam_tag}.dat")
            if x.size == 0:
                continue
            ax.scatter(
                np.log10(x), eps,
                color=style["color"], marker=POINT_MARKER,
                s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0,
            )
            all_x.append(np.log10(x))
            all_eps.append(eps)

        if all_x:
            curves.append((tag_to_float(lam_tag), np.concatenate(all_x), np.concatenate(all_eps)))

    ax.set_ylim(*YLIM)

    # Legend: critical-point type (colour)
    type_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               color=s["color"], label=s["label"])
        for s in CLASS_STYLE.values()
    ]
    type_handles.append(Line2D([], [], linestyle="None", marker="",
                               label=r"Curve labels: $\lambda_{in}$"))
    legend = ax.legend(handles=type_handles, title="Critical-point type",
                       loc=TYPE_LEGEND_LOC, bbox_to_anchor=TYPE_LEGEND_ANCHOR,
                       framealpha=0.9)

    ax.set_xlabel(r"$\log_{10}(x_{\rm in})$", fontsize=14)
    ax.set_ylabel(r"$\varepsilon_{\rm critical}$", fontsize=14)
    ax.set_title(
        rf"Critical Energy vs $\log_{{10}}(x_{{\rm in}})$ for Different $\lambda_{{\rm in}}$ "
        rf"with $\alpha_\pi = {alpha_pi:g},\ \beta = {beta:g}$",
        fontsize=14,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ax.set_xlim(ax.get_xlim())                   # freeze the view before placing labels

    # Place each lambda label in free space next to its own curve
    place_curve_labels(fig, ax, curves, legend,
                       make_text=lambda v: LABEL_TEXT.format(v))

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
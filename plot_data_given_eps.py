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

POINT_MARKER = "o"                    # same marker for every eps
POINT_SIZE = 6
POINT_ALPHA = 0.6

YLIM = None                           # e.g. (0.8, 2.6) to fix the vertical axis; None = automatic

# Keplerian angular momentum curve:  lambda_K(x) = x^(3/2) / (sqrt(2) (x - 1))
SHOW_KEPLERIAN = True
KEPLER_COLOR = "green"
KEPLER_LINEWIDTH = 2.0
KEPLER_LINESTYLE = "--"               # "--" dashed, ":" dotted, "-." dash-dot, "-" solid
KEPLER_ALPHA = 0.6                    # 1.0 = fully opaque, lower = lighter
KEPLER_ZORDER = 4                     # drawn above the data points so it stands out

# Curve labels (the eps value written next to the end of each curve). Each label is placed
# automatically in free space: it never touches a data point, the Keplerian curve, another
# label or the legend. If it has to sit away from the curve end, a thin leader line is drawn.
LABEL_FONTSIZE = 11
LABEL_CLEARANCE_PTS = 4               # minimum empty space around each label (in points)
LABEL_SEARCH_RADIUS_PTS = 120         # how far from the curve end to look for free space
LABEL_LEADER_MIN_PTS = 16             # draw a leader line if the label is farther than this from the end
AXES_MARGIN = 0.10                    # extra room around the data so end labels stay inside

# Legend position inside the axes (axes-fraction coordinates)
TYPE_LEGEND_LOC = "lower left"
TYPE_LEGEND_ANCHOR = (0.01, 0.01)


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


def keplerian_lambda(x):
    """Keplerian angular momentum: lambda_K(x) = x^(3/2) / (sqrt(2) (x - 1))."""
    return x**1.5 / (np.sqrt(2.0) * (x - 1.0))


def rect_to_points_distance(centres, hw, hh, pts, chunk=400):
    """Distance from each rectangle (centre, half-width hw, half-height hh) to the nearest
    point in `pts`. Returns one value per rectangle (0 = a point lies inside it)."""
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


def place_end_labels(fig, ax, curves, legend, extra_points_px, make_text):
    """Write one label near the high-x end of each curve, in free space."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    clearance = LABEL_CLEARANCE_PTS * px_per_pt
    radius = LABEL_SEARCH_RADIUS_PTS * px_per_pt
    step = max(2.0, 2.0 * px_per_pt)

    ax_bb = ax.get_window_extent(renderer)
    curve_px = [ax.transData.transform(np.column_stack([xs, ys])) for _, xs, ys in curves]
    obstacles = np.vstack(curve_px + ([extra_points_px] if len(extra_points_px) else []))

    blocked = []                                   # rectangles: legend, then labels already placed
    if legend is not None:
        bb = legend.get_window_extent(renderer)
        blocked.append((bb.x0, bb.y0, bb.x1, bb.y1))

    for k, (value, xs, ys) in enumerate(curves):
        text = make_text(value)

        probe = ax.text(0, 0, text, fontsize=LABEL_FONTSIZE, ha="center", va="center")
        bb = probe.get_window_extent(renderer)
        probe.remove()
        hw, hh = bb.width / 2.0, bb.height / 2.0

        # end of the curve (largest x) and its direction of travel
        order = np.argsort(xs)
        pts = curve_px[k][order]
        end = pts[-1]
        back = pts[max(0, len(pts) - 11)]
        tangent = end - back
        tangent = tangent / np.hypot(*tangent) if np.hypot(*tangent) > 0 else np.array([1.0, 0.0])

        lo, hi = end - radius - 2 * max(hw, hh), end + radius + 2 * max(hw, hh)
        near = obstacles[(obstacles[:, 0] > lo[0]) & (obstacles[:, 0] < hi[0]) &
                         (obstacles[:, 1] > lo[1]) & (obstacles[:, 1] < hi[1])]

        gx = np.arange(end[0] - radius, end[0] + radius + 1, step)
        gy = np.arange(end[1] - radius, end[1] + radius + 1, step)
        C = np.array(np.meshgrid(gx, gy)).reshape(2, -1).T
        inside = ((C[:, 0] - hw > ax_bb.x0 + clearance) & (C[:, 0] + hw < ax_bb.x1 - clearance) &
                  (C[:, 1] - hh > ax_bb.y0 + clearance) & (C[:, 1] + hh < ax_bb.y1 - clearance))
        C = C[inside]
        for (x0, y0, x1, y1) in blocked:
            clash = ((C[:, 0] + hw + clearance > x0) & (C[:, 0] - hw - clearance < x1) &
                     (C[:, 1] + hh + clearance > y0) & (C[:, 1] - hh - clearance < y1))
            C = C[~clash]

        d_data = rect_to_points_distance(C, hw, hh, near)
        free = d_data > clearance
        d_end = rect_to_points_distance(C, hw, hh, end[None, :])

        if free.any():
            Cf, de = C[free], d_end[free]
            v = Cf - end
            cos_t = (v @ tangent) / np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)
            score = de - 6.0 * px_per_pt * cos_t          # close to the end, preferably straight ahead of it
            centre = Cf[np.argmin(score)]
            dist_end = de[np.argmin(score)]
        else:
            print(f"  warning: no free spot found for label '{text}'; try a smaller LABEL_FONTSIZE")
            centre = end + tangent * (hw + 20 * px_per_pt)
            dist_end = 0.0

        x_d, y_d = ax.transData.inverted().transform(centre)
        if dist_end > LABEL_LEADER_MIN_PTS * px_per_pt:
            end_d = ax.transData.inverted().transform(end)
            ax.annotate(text, xy=end_d, xytext=(x_d, y_d), textcoords="data",
                        ha="center", va="center", fontsize=LABEL_FONTSIZE, zorder=5,
                        arrowprops=dict(arrowstyle="-", color="0.35", lw=0.8,
                                        shrinkA=2, shrinkB=3))
        else:
            ax.text(x_d, y_d, text, fontsize=LABEL_FONTSIZE, ha="center", va="center", zorder=5)
        blocked.append((centre[0] - hw, centre[1] - hh, centre[0] + hw, centre[1] + hh))


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
    ax.margins(x=AXES_MARGIN, y=AXES_MARGIN)

    curves = []                                  # (eps value, all x, all lambda) per eps
    for eps_dir in eps_dirs:
        eps_tag = eps_pattern.match(eps_dir.name).group(1)
        all_x, all_lam = [], []

        for cls, style in CLASS_STYLE.items():          # saddle, spiral, then nodal on top
            x, lam = load_dat(eps_dir / f"{cls}{eps_tag}.dat")
            if x.size == 0:
                continue
            ax.scatter(
                np.log10(x), lam,
                color=style["color"], marker=POINT_MARKER,
                s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0,
            )
            all_x.append(np.log10(x))
            all_lam.append(lam)

        if all_x:
            curves.append((tag_to_float(eps_tag), np.concatenate(all_x), np.concatenate(all_lam)))

    if YLIM is not None:
        ax.set_ylim(*YLIM)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()    # view set by the data (+ margins) and YLIM

    # Keplerian curve over the same x range as the data. The limits are restored afterwards so the
    # part of the curve that rises far above the data does not stretch the vertical axis.
    kepler_xy = None
    if SHOW_KEPLERIAN and curves:
        log_x_all = np.concatenate([xs for _, xs, _ in curves])
        x_k = np.logspace(log_x_all.min(), log_x_all.max(), 1500)
        kepler_xy = np.column_stack([np.log10(x_k), keplerian_lambda(x_k)])
        ax.plot(kepler_xy[:, 0], kepler_xy[:, 1],
                color=KEPLER_COLOR, lw=KEPLER_LINEWIDTH, ls=KEPLER_LINESTYLE,
                alpha=KEPLER_ALPHA, zorder=KEPLER_ZORDER)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Legend: critical-point type (colour)
    type_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               color=s["color"], label=s["label"])
        for s in CLASS_STYLE.values()
    ]
    if SHOW_KEPLERIAN:
        type_handles.append(Line2D([0], [0], color=KEPLER_COLOR, lw=KEPLER_LINEWIDTH,
                                   ls=KEPLER_LINESTYLE, alpha=KEPLER_ALPHA,
                                   label=r"Keplerian $\lambda_K(x)$"))
    legend = ax.legend(handles=type_handles, title="Critical-point type",
              loc=TYPE_LEGEND_LOC, bbox_to_anchor=TYPE_LEGEND_ANCHOR,
              framealpha=0.9)

    ax.set_xlabel(r"$\log_{10}(x_{\rm in})$", fontsize=14)
    ax.set_ylabel(r"$\lambda_{\rm in}$", fontsize=14)
    ax.set_title(
        rf"$\lambda_{{\rm in}}$ vs $\log_{{10}}(x_{{\rm in}})$ for Different $\varepsilon$ "
        rf"with $\alpha_\pi = {alpha_pi:g},\ \beta = {beta:g}$",
        fontsize=14,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())   # freeze the view before placing labels

    # Write each eps value near the end of its own curve, in free space
    kepler_px = ax.transData.transform(kepler_xy) if kepler_xy is not None else np.empty((0, 2))
    place_end_labels(fig, ax, curves, legend, kepler_px,
                     make_text=lambda v: rf"$\varepsilon = {v:g}$")

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
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# ==========================================
# SETTINGS
# ==========================================

data_dir = Path("data") / "param_given"
out_dir = Path("output") / "param_given_outputs"

SAVE_PNG = True
SAVE_PDF = True
SHOW_PLOTS = False        # True = also open the plot windows

# "log10" -> horizontal axis is log10(x_in);  "linear" -> horizontal axis is x_in
X_MODE = "log10"

# Column indices in the .dat files:
# x_in lambda_in M_xc v_xc a_xc N D dvdx_root_1 dvdx_root_2 eps discriminant
COL_X, COL_EPS = 0, 9

# Colour = critical-point type. Listed in drawing order (last is drawn on top).
CLASS_STYLE = {
    "saddle": dict(color="black", label="Saddle"),
    "spiral": dict(color="blue",  label="Spiral"),
    "nodal":  dict(color="red",   label="Nodal"),
}

POINT_MARKER = "o"                    # same marker for every curve
POINT_SIZE = 6
POINT_ALPHA = 0.6

# Vertical axis limits, SHARED by both panels so they can be compared directly.
# None = automatic: from just below the lowest data point of EITHER sweep up to Y_AUTO_MAX
# (the steep rise at small x_in is cut off, as in the other plots).
YLIM = None
Y_AUTO_MAX = 0.05

FIGSIZE = (16, 6.3)                   # both panels have the same size

# Curve labels: each value of the swept parameter is placed automatically in free space next
# to its own curve (never touching a data point, another label or the legend). Curves that lie
# on top of each other share ONE label such as "0, 0.001".
LABEL_FONTSIZE = 10
LABEL_MIN_FONTSIZE = 7.5              # in crowded spots a label may shrink down to this size
LABEL_CLEARANCE_PTS = 3               # preferred empty space around each label (in points)
LABEL_SEARCH_RADIUS_PTS = 110         # how far from the curve's minimum to look for free space
COINCIDE_TOL_PTS = 3.0                # curves closer than this (in points) are treated as one curve

# Legend position: "best" lets matplotlib pick the corner with the least data under it
TYPE_LEGEND_LOC = "best"


# ==========================================
# SWEEP DEFINITIONS
# ==========================================

# For each sweep: how its folders are named, and how to write it on the plot
SWEEPS = {
    "alpha_pi": dict(short="alpha", fixed_short="beta",
                     sym=r"\alpha_\pi", fixed_sym=r"\beta"),
    "beta":     dict(short="beta",  fixed_short="alpha",
                     sym=r"\beta", fixed_sym=r"\alpha_\pi"),
}


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


def x_transform(x):
    return np.log10(x) if X_MODE == "log10" else x


def x_label():
    return r"$\log_{10}(x_{\rm in})$" if X_MODE == "log10" else r"$x_{\rm in}$"


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


def _one_way_gap(P, Q):
    """90th-percentile distance from points of P to the nearest point of Q (pixels)."""
    if len(P) == 0 or len(Q) == 0:
        return np.inf
    P = P[::max(1, len(P) // 200)]
    d = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return np.percentile(d, 90)


def group_coincident(curve_px, tol_px):
    """Group curves that lie on top of each other. Returns a list of lists of curve indices."""
    groups, used = [], set()
    for i in range(len(curve_px)):
        if i in used:
            continue
        g = [i]
        used.add(i)
        for j in range(i + 1, len(curve_px)):
            if j in used:
                continue
            if _one_way_gap(curve_px[i], curve_px[j]) < tol_px and _one_way_gap(curve_px[j], curve_px[i]) < tol_px:
                g.append(j)
                used.add(j)
        groups.append(g)
    return groups


def place_curve_labels(fig, ax, curves, legend):
    """Place one label per (group of coincident) curve(s) in free space next to the curve.

    Order of attempts for every label: (1) next to its own curve, full size; (2) the same with a
    slightly smaller font / tighter clearance; (3) if the curves are too crowded, in the nearest
    free spot with a thin leader line (drawn BEHIND the data points) to the curve."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    radius = LABEL_SEARCH_RADIUS_PTS * px_per_pt
    step = max(2.0, 2.0 * px_per_pt)

    ax_bb = ax.get_window_extent(renderer)
    curve_px = [ax.transData.transform(np.column_stack([xs, ys])) for _, xs, ys in curves]
    groups = group_coincident(curve_px, COINCIDE_TOL_PTS * px_per_pt)

    blocked = []                                     # rectangles: legend, then labels already placed
    if legend is not None:
        bb = legend.get_window_extent(renderer)
        blocked.append((bb.x0, bb.y0, bb.x1, bb.y1))

    # (font size, clearance in points) tried in this order
    f0, f1 = LABEL_FONTSIZE, max(LABEL_MIN_FONTSIZE, LABEL_FONTSIZE - 2)
    attempts = [(f0, LABEL_CLEARANCE_PTS), (max(f1, f0 - 1), 2.5), (f1, 2.0), (f1, 1.2)]

    def measure(text, fontsize):
        probe = ax.text(0, 0, text, fontsize=fontsize, ha="center", va="center")
        bb = probe.get_window_extent(renderer)
        probe.remove()
        return bb.width / 2.0, bb.height / 2.0

    for g in groups:
        text = ", ".join(f"{curves[i][0]:g}" for i in g)

        own = np.vstack([curve_px[i] for i in g])
        rest = [curve_px[j] for j in range(len(curves)) if j not in g]
        others = np.vstack(rest) if rest else np.empty((0, 2))
        first = g[0]
        anchor = curve_px[first][np.argmin(curves[first][2])]     # lowest point (min eps) of the curve

        centre, fontsize, last = None, LABEL_FONTSIZE, None
        for fontsize, clr_pt in attempts:
            clearance = clr_pt * px_per_pt
            hw, hh = measure(text, fontsize)

            lo, hi = anchor - radius - 2 * max(hw, hh), anchor + radius + 2 * max(hw, hh)
            near = lambda P: P[(P[:, 0] > lo[0]) & (P[:, 0] < hi[0]) & (P[:, 1] > lo[1]) & (P[:, 1] < hi[1])]
            own_n, oth_n = near(own), near(others)

            gx = np.arange(anchor[0] - radius, anchor[0] + radius + 1, step)
            gy = np.arange(anchor[1] - radius, anchor[1] + radius + 1, step)
            C = np.array(np.meshgrid(gx, gy)).reshape(2, -1).T
            inside = ((C[:, 0] - hw > ax_bb.x0 + clearance) & (C[:, 0] + hw < ax_bb.x1 - clearance) &
                      (C[:, 1] - hh > ax_bb.y0 + clearance) & (C[:, 1] + hh < ax_bb.y1 - clearance))
            C = C[inside]
            for (x0, y0, x1, y1) in blocked:
                clash = ((C[:, 0] + hw + clearance > x0) & (C[:, 0] - hw - clearance < x1) &
                         (C[:, 1] + hh + clearance > y0) & (C[:, 1] - hh - clearance < y1))
                C = C[~clash]

            d_own = rect_to_points_distance(C, hw, hh, own_n)
            d_oth = rect_to_points_distance(C, hw, hh, oth_n)
            free = (d_own > clearance) & (d_oth > clearance)
            valid = free & (d_own < d_oth)           # label nearer to its own curve than to any other
            last = (hw, hh, C, d_own, free, fontsize)

            if valid.any():
                Cf, do, dt = C[valid], d_own[valid], d_oth[valid]
                score = (do - 0.3 * np.minimum(dt, 5 * clearance)
                         + 0.02 * np.hypot(*(Cf - anchor).T)
                         + 0.5 * np.maximum(0.0, Cf[:, 1] - anchor[1]))   # mild preference for below the minimum
                centre = Cf[np.argmin(score)]
                break

        if centre is not None:                       # placed next to its own curve
            x_d, y_d = ax.transData.inverted().transform(centre)
            ax.text(x_d, y_d, text, fontsize=fontsize, ha="center", va="center", zorder=5)
            blocked.append((centre[0] - hw, centre[1] - hh, centre[0] + hw, centre[1] + hh))
            continue

        # crowded: nearest free spot (preferably below) + a leader line behind the data
        hw, hh, C, d_own, free, fontsize = last
        if free.any():
            Cf, do = C[free], d_own[free]
            centre = Cf[np.argmin(do + 0.5 * np.maximum(0.0, Cf[:, 1] - anchor[1]))]
            d = anchor - centre
            t = min(hw / abs(d[0]) if d[0] else np.inf, hh / abs(d[1]) if d[1] else np.inf, 1.0)
            start = centre + d * t                       # where the leader leaves the label box
            (sx, sy), (ex, ey) = ax.transData.inverted().transform([start, anchor])
            ax.plot([sx, ex], [sy, ey], color="0.35", lw=0.8, zorder=0.9, solid_capstyle="butt")
        else:
            print(f"  warning: no free spot found for label '{text}'; try a smaller LABEL_FONTSIZE")
            centre = anchor + np.array([0.0, -(hh + 4 * px_per_pt)])

        x_d, y_d = ax.transData.inverted().transform(centre)
        ax.text(x_d, y_d, text, fontsize=fontsize, ha="center", va="center", zorder=5)
        blocked.append((centre[0] - hw, centre[1] - hh, centre[0] + hw, centre[1] + hh))


# ==========================================
# MAIN
# ==========================================

def load_sweep(sweep_dir, sw):
    """Read one sweep folder. Returns a list of (swept value, x array, eps array), sorted by value."""
    value_pattern = re.compile(rf"{sw['short']}(\d+o\d+)$")
    value_dirs = [d for d in sweep_dir.iterdir() if d.is_dir() and value_pattern.match(d.name)]
    value_dirs.sort(key=lambda d: tag_to_float(value_pattern.match(d.name).group(1)))

    curves = []
    for value_dir in value_dirs:
        v_tag = value_pattern.match(value_dir.name).group(1)
        parts = {}
        for cls in CLASS_STYLE:
            x, eps = load_dat(value_dir / f"{cls}{v_tag}.dat")
            if x.size:
                parts[cls] = (x_transform(x), eps)
        if parts:
            xs = np.concatenate([p[0] for p in parts.values()])
            es = np.concatenate([p[1] for p in parts.values()])
            curves.append((tag_to_float(v_tag), xs, es, parts))
    return curves


def draw_panel(ax, curves, sw, fixed_value, lambda_in, show_ylabel):
    """Scatter the curves of one sweep on `ax`; returns the legend."""
    for _, _, _, parts in curves:
        for cls, style in CLASS_STYLE.items():          # saddle, spiral, then nodal on top
            if cls in parts:
                x, eps = parts[cls]
                ax.scatter(x, eps, color=style["color"], marker=POINT_MARKER,
                           s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               color=s["color"], label=s["label"])
        for s in CLASS_STYLE.values()
    ]
    handles.append(Line2D([], [], linestyle="None", marker="", label=rf"Curve labels: ${sw['sym']}$"))
    legend = ax.legend(handles=handles, title="Critical-point type",
                       loc=TYPE_LEGEND_LOC, framealpha=0.9)

    ax.set_xlabel(x_label(), fontsize=14)
    if show_ylabel:
        ax.set_ylabel(r"$\varepsilon_{\rm critical}$", fontsize=14)
    ax.set_title(
        rf"Varying ${sw['sym']}$:  ${sw['fixed_sym']} = {fixed_value:g}$,  $\lambda_{{\rm in}} = {lambda_in:g}$",
        fontsize=13,
    )
    ax.grid(alpha=0.3)
    return legend


if not data_dir.exists():
    raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()}")

out_dir.mkdir(parents=True, exist_ok=True)

alpha_sw, beta_sw = SWEEPS["alpha_pi"], SWEEPS["beta"]

# alpha sweeps:  alpha_sweep_beta<b>_lambda<l>      beta sweeps:  beta_sweep_alpha<a>_lambda<l>
alpha_pat = re.compile(r"alpha_sweep_beta(\d+o\d+)_lambda(\d+o\d+)$")
beta_pat = re.compile(r"beta_sweep_alpha(\d+o\d+)_lambda(\d+o\d+)$")

alpha_folders = sorted(d for d in data_dir.iterdir() if d.is_dir() and alpha_pat.match(d.name))
beta_folders = sorted(d for d in data_dir.iterdir() if d.is_dir() and beta_pat.match(d.name))

if not alpha_folders or not beta_folders:
    raise FileNotFoundError(
        f"Need BOTH sweeps in {data_dir.resolve()}: found {len(alpha_folders)} alpha-sweep and "
        f"{len(beta_folders)} beta-sweep folder(s). Run the data script once with "
        f"SWEEP_PARAM = 'alpha_pi' and once with 'beta' (using the same lambda_in)."
    )

# pair alpha sweeps and beta sweeps that were run at the same lambda
pairs = [(af, bf) for af in alpha_folders for bf in beta_folders
         if alpha_pat.match(af.name).group(2) == beta_pat.match(bf.name).group(2)]
if not pairs:
    raise FileNotFoundError("The alpha sweeps and beta sweeps were run at different lambda_in values; "
                            "re-run one of them so that both use the same lambda_in.")

for alpha_dir, beta_dir in pairs:
    beta_fixed = tag_to_float(alpha_pat.match(alpha_dir.name).group(1))     # beta held fixed in the alpha sweep
    alpha_fixed = tag_to_float(beta_pat.match(beta_dir.name).group(1))      # alpha held fixed in the beta sweep
    lam_tag = alpha_pat.match(alpha_dir.name).group(2)
    lambda_in = tag_to_float(lam_tag)

    curves_a = load_sweep(alpha_dir, alpha_sw)
    curves_b = load_sweep(beta_dir, beta_sw)
    if not curves_a or not curves_b:
        print(f"{alpha_dir.name} / {beta_dir.name}: one of the sweeps has no data, skipped")
        continue

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True, sharey=True)

    leg_a = draw_panel(ax_a, curves_a, alpha_sw, beta_fixed, lambda_in, show_ylabel=True)
    leg_b = draw_panel(ax_b, curves_b, beta_sw, alpha_fixed, lambda_in, show_ylabel=False)

    # one common vertical range for both panels
    if YLIM is not None:
        ylim = YLIM
    else:
        y_min = min(float(c[2].min()) for c in curves_a + curves_b)
        ylim = (y_min - 0.15 * (Y_AUTO_MAX - y_min), Y_AUTO_MAX)
    ax_a.set_ylim(*ylim)                          # shared, so this sets both panels

    fig.suptitle(
        rf"Critical Energy vs {x_label()}:  ${alpha_sw['sym']}$ sweep (left) vs ${beta_sw['sym']}$ sweep (right),"
        rf"  $\lambda_{{\rm in}} = {lambda_in:g}$",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ax_a.set_xlim(ax_a.get_xlim())                # freeze the shared view before placing labels
    ax_a.set_ylim(*ylim)

    # labels, one panel at a time (tuples reduced to what the placement code needs)
    place_curve_labels(fig, ax_a, [(v, xs, es) for v, xs, es, _ in curves_a], leg_a)
    place_curve_labels(fig, ax_b, [(v, xs, es) for v, xs, es, _ in curves_b], leg_b)

    x_name = "log10_xin" if X_MODE == "log10" else "xin"
    base = (f"sweep_comparison_{x_name}_lambda{lam_tag}_"
            f"{alpha_dir.name.split('_lambda')[0]}_{beta_dir.name.split('_lambda')[0]}")
    if SAVE_PNG:
        fig.savefig(out_dir / f"{base}.png", dpi=300, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(out_dir / f"{base}.pdf", bbox_inches="tight")
    print(f"saved: {out_dir / base}.(png/pdf)")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

print(f"\nAll plots saved in: {out_dir.resolve()}")
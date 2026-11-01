import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# ==========================================
# SETTINGS
# ==========================================

SWEEP_PARAM = "alpha_pi"          # "alpha_pi" or "beta"  <- which sweep to plot (same switch as in the data script)

data_dir = Path("data") / "param_eps_given"
out_dir = Path("output") / "param_eps_given_outputs"

SAVE_PNG = True
SAVE_PDF = True
SHOW_PLOTS = False        # True = also open the plot windows

# "log10" -> horizontal axis is log10(x_in);  "linear" -> horizontal axis is x_in
X_MODE = "log10"

# Column indices in the .dat files:
# x_in eps lambda_in M_xc v_xc a_xc dvdx_root_1 dvdx_root_2
COL_X, COL_LAMBDA = 0, 2

# Colour = critical-point type. Listed in drawing order (last is drawn on top).
CLASS_STYLE = {
    "saddle": dict(color="black", label="Saddle"),
    "spiral": dict(color="blue",  label="Spiral"),
    "nodal":  dict(color="red",   label="Nodal"),
}

POINT_MARKER = "o"                    # same marker for every curve
POINT_SIZE = 6
POINT_ALPHA = 0.6

# Vertical axis limits. None = automatic: the range of the data plus a small margin.
# (The Keplerian curve never stretches the axis.)
YLIM = None
Y_MARGIN = 0.06                       # fraction of the data range added above and below
AXES_MARGIN_X = 0.08                  # extra room left/right so labels at the curve ends stay inside

# Keplerian angular momentum curve:  lambda_K(x) = x^(3/2) / (sqrt(2) (x - 1))
SHOW_KEPLERIAN = True
KEPLER_COLOR = "green"
KEPLER_LINEWIDTH = 2.0
KEPLER_LINESTYLE = "--"               # "--" dashed, ":" dotted, "-." dash-dot, "-" solid
KEPLER_ALPHA = 0.6                    # 1.0 = fully opaque, lower = lighter
KEPLER_ZORDER = 4                     # above the data points

# Curve labels: each value of the swept parameter is placed automatically in free space next
# to its own curve (never touching a data point, the Keplerian curve, another label or the
# legend). Curves that lie on top of each other (e.g. alpha_pi = 0 and 0.001) share ONE label
# such as "0, 0.001".
LABEL_ANCHOR = "end"                  # "end": near the high-x end of each curve;  "min": near its lowest point
LABEL_FONTSIZE = 10
LABEL_MIN_FONTSIZE = 7.5              # in crowded spots a label may shrink down to this size
LABEL_CLEARANCE_PTS = 3               # preferred empty space around each label (in points)
LABEL_SEARCH_RADIUS_PTS = 110         # how far from the anchor to look for free space
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
    """'1o65' -> 1.65, '0o001' -> 0.001, 'm0o02' -> -0.02"""
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


def x_transform(x):
    return np.log10(x) if X_MODE == "log10" else x


def x_label():
    return r"$\log_{10}(x_{\rm in})$" if X_MODE == "log10" else r"$x_{\rm in}$"


def keplerian_lambda(x):
    """Keplerian angular momentum: lambda_K(x) = x^(3/2) / (sqrt(2) (x - 1))."""
    return x**1.5 / (np.sqrt(2.0) * (x - 1.0))


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


def curve_anchor(px, xs, ys):
    """Point (in pixels) where a curve's label should sit next to, and the direction the curve
    is heading there (None when anchoring at the lowest point)."""
    if LABEL_ANCHOR == "min":
        return px[np.argmin(ys)], None
    order = np.argsort(xs)
    pts = px[order]
    end = pts[-1]
    t = end - pts[max(0, len(pts) - 11)]
    n = np.hypot(*t)
    return end, (t / n if n > 0 else np.array([1.0, 0.0]))


def place_curve_labels(fig, ax, curves, legend, extra_points_px):
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
        if len(extra_points_px):
            rest.append(extra_points_px)             # the Keplerian curve counts as "another curve"
        others = np.vstack(rest) if rest else np.empty((0, 2))
        first = g[0]
        anchor, tangent = curve_anchor(curve_px[first], curves[first][1], curves[first][2])

        def direction_bonus(Cf):                     # prefer spots straight ahead of the curve's end
            v = Cf - anchor
            cos_t = (v @ tangent) / np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)
            return -6.0 * px_per_pt * cos_t

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
                score = do - 0.3 * np.minimum(dt, 5 * clearance) + 0.02 * np.hypot(*(Cf - anchor).T)
                if tangent is not None:
                    score = score + direction_bonus(Cf)
                else:
                    score = score + 0.5 * np.maximum(0.0, Cf[:, 1] - anchor[1])   # mild preference for below
                centre = Cf[np.argmin(score)]
                break

        if centre is not None:                       # placed next to its own curve
            x_d, y_d = ax.transData.inverted().transform(centre)
            ax.text(x_d, y_d, text, fontsize=fontsize, ha="center", va="center", zorder=5)
            blocked.append((centre[0] - hw, centre[1] - hh, centre[0] + hw, centre[1] + hh))
            continue

        # crowded: nearest free spot + a leader line behind the data
        hw, hh, C, d_own, free, fontsize = last
        if free.any():
            Cf, do = C[free], d_own[free]
            centre = Cf[np.argmin(do)]
            d = anchor - centre
            t = min(hw / abs(d[0]) if d[0] else np.inf, hh / abs(d[1]) if d[1] else np.inf, 1.0)
            start = centre + d * t                       # where the leader leaves the label box
            (sx, sy), (ex, ey) = ax.transData.inverted().transform([start, anchor])
            ax.plot([sx, ex], [sy, ey], color="0.35", lw=0.8, zorder=0.9, solid_capstyle="butt")
        else:
            print(f"  warning: no free spot found for label '{text}'; try a smaller LABEL_FONTSIZE")
            centre = anchor + (tangent * (hw + 8 * px_per_pt) if tangent is not None
                               else np.array([0.0, -(hh + 4 * px_per_pt)]))

        x_d, y_d = ax.transData.inverted().transform(centre)
        ax.text(x_d, y_d, text, fontsize=fontsize, ha="center", va="center", zorder=5)
        blocked.append((centre[0] - hw, centre[1] - hh, centre[0] + hw, centre[1] + hh))


# ==========================================
# MAIN
# ==========================================

if SWEEP_PARAM not in SWEEPS:
    raise ValueError(f'SWEEP_PARAM must be "alpha_pi" or "beta", got {SWEEP_PARAM!r}')
sw = SWEEPS[SWEEP_PARAM]

if not data_dir.exists():
    raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()}")

out_dir.mkdir(parents=True, exist_ok=True)

# sweep folders, e.g. alpha_sweep_beta0o01_eps0o005  /  beta_sweep_alpha0o001_epsm0o02
sweep_pattern = re.compile(rf"{sw['short']}_sweep_{sw['fixed_short']}(\d+o\d+)_eps(m?\d+o\d+)$")
value_pattern = re.compile(rf"{sw['short']}(\d+o\d+)$")

sweep_folders = sorted(d for d in data_dir.iterdir() if d.is_dir() and sweep_pattern.match(d.name))
if not sweep_folders:
    raise FileNotFoundError(
        f"No '{sw['short']}_sweep_{sw['fixed_short']}..._eps...' folders found in {data_dir.resolve()}.\n"
        f"Run the data script with SWEEP_PARAM = {SWEEP_PARAM!r} first."
    )

for sweep_dir in sweep_folders:
    m = sweep_pattern.match(sweep_dir.name)
    fixed_value, eps_in = tag_to_float(m.group(1)), tag_to_float(m.group(2))

    # one folder per value of the swept parameter, sorted numerically
    value_dirs = [d for d in sweep_dir.iterdir() if d.is_dir() and value_pattern.match(d.name)]
    value_dirs.sort(key=lambda d: tag_to_float(value_pattern.match(d.name).group(1)))
    if not value_dirs:
        print(f"{sweep_dir.name}: no value folders, skipped")
        continue

    fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.margins(x=AXES_MARGIN_X)

    curves = []                                  # (swept value, all x, all lambda) per value
    for value_dir in value_dirs:
        v_tag = value_pattern.match(value_dir.name).group(1)
        all_x, all_lam = [], []

        for cls, style in CLASS_STYLE.items():          # saddle, spiral, then nodal on top
            x, lam = load_dat(value_dir / f"{cls}{v_tag}.dat")
            if x.size == 0:
                continue
            ax.scatter(
                x_transform(x), lam,
                color=style["color"], marker=POINT_MARKER,
                s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0,
            )
            all_x.append(x_transform(x))
            all_lam.append(lam)

        if all_x:
            curves.append((tag_to_float(v_tag), np.concatenate(all_x), np.concatenate(all_lam)))

    if not curves:
        print(f"{sweep_dir.name}: no data points, skipped")
        plt.close(fig)
        continue

    # vertical limits from the data only
    if YLIM is not None:
        ylim = YLIM
    else:
        y_lo = min(float(ys.min()) for _, _, ys in curves)
        y_hi = max(float(ys.max()) for _, _, ys in curves)
        pad = Y_MARGIN * (y_hi - y_lo if y_hi > y_lo else 1.0)
        ylim = (y_lo - pad, y_hi + pad)
    ax.set_ylim(*ylim)
    xlim = ax.get_xlim()

    # Keplerian curve over the same x range as the data; the limits are restored afterwards so
    # the part of the curve far above the data does not stretch the axes
    kepler_xy = None
    if SHOW_KEPLERIAN:
        u_all = np.concatenate([xs for _, xs, _ in curves])
        if X_MODE == "log10":
            x_k = np.logspace(u_all.min(), u_all.max(), 1500)
            u_k = np.log10(x_k)
        else:
            x_k = np.linspace(u_all.min(), u_all.max(), 1500)
            u_k = x_k
        kepler_xy = np.column_stack([u_k, keplerian_lambda(x_k)])
        ax.plot(kepler_xy[:, 0], kepler_xy[:, 1], color=KEPLER_COLOR, lw=KEPLER_LINEWIDTH,
                ls=KEPLER_LINESTYLE, alpha=KEPLER_ALPHA, zorder=KEPLER_ZORDER)
        ax.set_xlim(xlim)
        ax.set_ylim(*ylim)

    # Legend: critical-point type (colour) + what the curve labels mean (+ Keplerian curve)
    type_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               color=s["color"], label=s["label"])
        for s in CLASS_STYLE.values()
    ]
    type_handles.append(Line2D([], [], linestyle="None", marker="",
                               label=rf"Curve labels: ${sw['sym']}$"))
    if SHOW_KEPLERIAN:
        type_handles.append(Line2D([0], [0], color=KEPLER_COLOR, lw=KEPLER_LINEWIDTH,
                                   ls=KEPLER_LINESTYLE, alpha=KEPLER_ALPHA,
                                   label=r"Keplerian $\lambda_K(x)$"))
    legend = ax.legend(handles=type_handles, title="Critical-point type",
                       loc=TYPE_LEGEND_LOC, framealpha=0.9)

    ax.set_xlabel(x_label(), fontsize=14)
    ax.set_ylabel(r"$\lambda_{\rm in}$", fontsize=14)
    ax.set_title(
        rf"$\lambda_{{\rm in}}$ vs {x_label()} for Different ${sw['sym']}$ "
        rf"with ${sw['fixed_sym']} = {fixed_value:g},\ \varepsilon = {eps_in:g}$",
        fontsize=14,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ax.set_xlim(ax.get_xlim())                   # freeze the view before placing labels
    ax.set_ylim(ax.get_ylim())

    kepler_px = ax.transData.transform(kepler_xy) if kepler_xy is not None else np.empty((0, 2))
    place_curve_labels(fig, ax, curves, legend, kepler_px)

    base = f"lambda_vs_{'log10_xin' if X_MODE == 'log10' else 'xin'}_{sweep_dir.name}"
    if SAVE_PNG:
        fig.savefig(out_dir / f"{base}.png", dpi=300, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(out_dir / f"{base}.pdf", bbox_inches="tight")
    print(f"saved: {out_dir / base}.(png/pdf)")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

print(f"\nAll plots saved in: {out_dir.resolve()}")
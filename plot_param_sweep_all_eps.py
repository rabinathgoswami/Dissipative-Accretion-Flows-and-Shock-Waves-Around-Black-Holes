from pathlib import Path
import re
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ==========================================
# SETTINGS
# ==========================================

data_dir = Path("data") / "param_eps_given"
out_dir = Path("output") / "param_eps_given_outputs"

SAVE_PNG = True
SAVE_PDF = True
SHOW_PLOTS = False

# "log10" -> horizontal axis is log10(x_in);  "linear" -> horizontal axis is x_in
X_MODE = "log10"

# Column indices in the .dat files
COL_X, COL_LAMBDA = 0, 2

CLASS_STYLE = {
    "saddle": dict(color="black", label="Saddle"),
    "spiral": dict(color="blue", label="Spiral"),
    "nodal": dict(color="red", label="Nodal"),
}

POINT_MARKER = "o"
POINT_SIZE = 1
POINT_ALPHA = 0.6

YLIM = None
Y_MARGIN = 0.06
AXES_MARGIN_X = 0.08

FIGSIZE = (16, 6.3)

SHOW_KEPLERIAN = True
KEPLER_COLOR = "green"
KEPLER_LINEWIDTH = 2.0
KEPLER_LINESTYLE = "--"
KEPLER_ALPHA = 0.6
KEPLER_ZORDER = 4

TYPE_LEGEND_LOC = "best"

# ==========================================
# MANUAL LABELS CONFIGURATION
# ==========================================
# Format: (x_pos, y_pos, "Text string")
# Define coordinates directly in the axis data units:

MANUAL_LABELS_LEFT = [
    (1.70, 2.01, r"$\alpha_\pi = 0$"),
    # (1.60, 1.85, r"$\alpha_\pi = 0.04,\, 0.06$"),
    (1.50, 1.88, r"$\alpha_\pi = 0.3$"),
]

MANUAL_LABELS_RIGHT = [
    (0.77, 1.75, r"$\beta = 3.0$"),
    # (1.65, 1.95, r"$\beta = 0.4,\, 0.6$"),
    (0.8, 1.58, r"$\beta = 0$"),
]

# ==========================================
# SWEEP DEFINITIONS & HELPERS
# ==========================================

SWEEPS = {
    "alpha_pi": dict(
        short="alpha",
        fixed_short="beta",
        sym=r"\alpha_\pi",
        fixed_sym=r"\beta",
    ),
    "beta": dict(
        short="beta",
        fixed_short="alpha",
        sym=r"\beta",
        fixed_sym=r"\alpha_\pi",
    ),
}


def tag_to_float(tag):
  """'1o65' -> 1.65, '0o001' -> 0.001, 'm0o02' -> -0.02"""
  sign = -1.0 if tag.startswith("m") else 1.0
  return sign * float(tag.lstrip("m").replace("o", "."))


def load_dat(path):
  if not path.exists():
    return np.array([]), np.array([])
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
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


def load_sweep(sweep_dir, sw):
  value_pattern = re.compile(rf"{sw['short']}(\d+o\d+)$")
  value_dirs = [
      d
      for d in sweep_dir.iterdir()
      if d.is_dir() and value_pattern.match(d.name)
  ]
  value_dirs.sort(
      key=lambda d: tag_to_float(value_pattern.match(d.name).group(1))
  )

  curves = []
  for value_dir in value_dirs:
    v_tag = value_pattern.match(value_dir.name).group(1)
    parts = {}
    for cls in CLASS_STYLE:
      x, lam = load_dat(value_dir / f"{cls}{v_tag}.dat")
      if x.size:
        parts[cls] = (x_transform(x), lam)
    if parts:
      xs = np.concatenate([p[0] for p in parts.values()])
      ls = np.concatenate([p[1] for p in parts.values()])
      curves.append((tag_to_float(v_tag), xs, ls, parts))
  return curves


def draw_points(ax, curves):
  for _, _, _, parts in curves:
    for cls, style in CLASS_STYLE.items():
      if cls in parts:
        x, lam = parts[cls]
        ax.scatter(
            x,
            lam,
            color=style["color"],
            marker=POINT_MARKER,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            linewidths=0,
        )


def make_legend(ax):
  handles = [
      Line2D(
          [0],
          [0],
          marker="o",
          linestyle="None",
          markersize=8,
          color=s["color"],
          label=s["label"],
      )
      for s in CLASS_STYLE.values()
  ]
  if SHOW_KEPLERIAN:
    handles.append(
        Line2D(
            [0],
            [0],
            color=KEPLER_COLOR,
            lw=KEPLER_LINEWIDTH,
            ls=KEPLER_LINESTYLE,
            alpha=KEPLER_ALPHA,
            label=r"Keplerian $\lambda_K(x)$",
        )
    )
  return ax.legend(
      handles=handles,
      title="Critical-point type",
      loc=TYPE_LEGEND_LOC,
      framealpha=0.9,
  )


# ==========================================
# MAIN EXECUTION
# ==========================================

if not data_dir.exists():
  raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()}")

out_dir.mkdir(parents=True, exist_ok=True)

alpha_sw, beta_sw = SWEEPS["alpha_pi"], SWEEPS["beta"]
alpha_pat = re.compile(r"alpha_sweep_beta(\d+o\d+)_eps(m?\d+o\d+)$")
beta_pat = re.compile(r"beta_sweep_alpha(\d+o\d+)_eps(m?\d+o\d+)$")

alpha_folders = sorted(
    d for d in data_dir.iterdir() if d.is_dir() and alpha_pat.match(d.name)
)
beta_folders = sorted(
    d for d in data_dir.iterdir() if d.is_dir() and beta_pat.match(d.name)
)

if not alpha_folders or not beta_folders:
  raise FileNotFoundError(
      f"Need BOTH sweeps in {data_dir.resolve()}: found {len(alpha_folders)}"
      f" alpha-sweep and {len(beta_folders)} beta-sweep folder(s)."
  )

pairs = [
    (af, bf)
    for af in alpha_folders
    for bf in beta_folders
    if alpha_pat.match(af.name).group(2) == beta_pat.match(bf.name).group(2)
]

if not pairs:
  raise FileNotFoundError(
      "The alpha sweeps and beta sweeps were run at different eps values."
  )

for alpha_dir, beta_dir in pairs:
  beta_fixed = tag_to_float(alpha_pat.match(alpha_dir.name).group(1))
  alpha_fixed = tag_to_float(beta_pat.match(beta_dir.name).group(1))
  eps_tag = alpha_pat.match(alpha_dir.name).group(2)
  eps_in = tag_to_float(eps_tag)

  curves_a = load_sweep(alpha_dir, alpha_sw)
  curves_b = load_sweep(beta_dir, beta_sw)
  if not curves_a or not curves_b:
    continue

  fig, (ax_a, ax_b) = plt.subplots(
      1, 2, figsize=FIGSIZE, sharex=True, sharey=True
  )
  ax_a.margins(x=AXES_MARGIN_X)

  draw_points(ax_a, curves_a)
  draw_points(ax_b, curves_b)

  # Determine vertical range from data
  if YLIM is not None:
    ylim = YLIM
  else:
    y_lo = min(float(c[2].min()) for c in curves_a + curves_b)
    y_hi = max(float(c[2].max()) for c in curves_a + curves_b)
    pad = Y_MARGIN * (y_hi - y_lo if y_hi > y_lo else 1.0)
    ylim = (y_lo - pad, y_hi + pad)
  ax_a.set_ylim(*ylim)
  xlim = ax_a.get_xlim()

  # Draw Keplerian curve
  if SHOW_KEPLERIAN:
    u_all = np.concatenate([c[1] for c in curves_a + curves_b])
    if X_MODE == "log10":
      x_k = np.logspace(u_all.min(), u_all.max(), 1500)
      u_k = np.log10(x_k)
    else:
      x_k = np.linspace(u_all.min(), u_all.max(), 1500)
      u_k = x_k
    kepler_y = keplerian_lambda(x_k)
    for ax in (ax_a, ax_b):
      ax.plot(
          u_k,
          kepler_y,
          color=KEPLER_COLOR,
          lw=KEPLER_LINEWIDTH,
          ls=KEPLER_LINESTYLE,
          alpha=KEPLER_ALPHA,
          zorder=KEPLER_ZORDER,
      )
    ax_a.set_xlim(xlim)
    ax_a.set_ylim(*ylim)

  make_legend(ax_a)
  make_legend(ax_b)

  for ax in (ax_a, ax_b):
    ax.set_xlabel(x_label(), fontsize=14)
    ax.grid(alpha=0.3)
  ax_a.set_ylabel(r"$\lambda_{\rm in}$", fontsize=14)
  ax_a.set_title(
      rf"Varying ${alpha_sw['sym']}$:  ${alpha_sw['fixed_sym']} ="
      rf" {beta_fixed:g}$,  $\varepsilon = {eps_in:g}$",
      fontsize=13,
  )
  ax_b.set_title(
      rf"Varying ${beta_sw['sym']}$:  ${beta_sw['fixed_sym']} ="
      rf" {alpha_fixed:g}$,  $\varepsilon = {eps_in:g}$",
      fontsize=13,
  )

  fig.suptitle(
      rf"$\lambda_{{\rm in}}$ vs {x_label()}:  ${alpha_sw['sym']}$ sweep (left)"
      rf" vs ${beta_sw['sym']}$ sweep (right),  $\varepsilon = {eps_in:g}$",
      fontsize=15,
  )

  # --- Manual Label Placement ---
  for x_pos, y_pos, text in MANUAL_LABELS_LEFT:
    ax_a.text(
        x_pos,
        y_pos,
        text,
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )

  for x_pos, y_pos, text in MANUAL_LABELS_RIGHT:
    ax_b.text(
        x_pos,
        y_pos,
        text,
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )

  fig.tight_layout(rect=[0, 0, 1, 0.95])

  x_name = "log10_xin" if X_MODE == "log10" else "xin"
  base = (
      f"lambda_sweep_comparison_{x_name}_eps{eps_tag}_"
      f"{alpha_dir.name.split('_eps')[0]}_{beta_dir.name.split('_eps')[0]}"
  )
  if SAVE_PNG:
    fig.savefig(out_dir / f"{base}.png", dpi=300, bbox_inches="tight")
  if SAVE_PDF:
    fig.savefig(out_dir / f"{base}.pdf", bbox_inches="tight")
  print(f"saved: {out_dir / base}.(png/pdf)")

  if SHOW_PLOTS:
    plt.show()
  plt.close(fig)

print(f"\nAll plots saved in: {out_dir.resolve()}")
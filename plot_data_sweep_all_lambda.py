from pathlib import Path
import re
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ==========================================
# SETTINGS
# ==========================================

data_dir = Path("data") / "param_given"
out_dir = Path("output") / "param_given_outputs"

SAVE_PNG = True
SAVE_PDF = True
SHOW_PLOTS = False

# "log10" -> horizontal axis is log10(x_in);  "linear" -> horizontal axis is x_in
X_MODE = "log10"

# Column indices in the .dat files
COL_X, COL_EPS = 0, 9

CLASS_STYLE = {
    "saddle": dict(color="black", label="Saddle"),
    "spiral": dict(color="blue", label="Spiral"),
    "nodal": dict(color="red", label="Nodal"),
}

POINT_MARKER = "o"
POINT_SIZE = 2
POINT_ALPHA = 0.6

YLIM = None
Y_AUTO_MAX = 0.05
FIGSIZE = (16, 6.3)
TYPE_LEGEND_LOC = "best"

# ==========================================
# MANUAL LABELS CONFIGURATION
# ==========================================
# Format: (x_pos, y_pos, "Text string")
# Edit/add coordinates directly in plot data space:

MANUAL_LABELS_LEFT = [
    (0.55, -0.015, r"$\alpha_\pi = 0$"),
    (0.50, -0.04, r"$\alpha_\pi = 0.3$"),
    # (0.55, -0.028, r"$\alpha_\pi = 0.08,\, 0.1$"),
]

MANUAL_LABELS_RIGHT = [
    (0.60, -0.001, r"$\beta = 3$"),
    # (0.65, -0.018, r"$\beta = 0.4,\, 0.6$"),
    (0.52, -0.024, r"$\beta = 0$"),
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
  return float(tag.replace("o", "."))


def load_dat(path):
  if not path.exists():
    return np.array([]), np.array([])
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    arr = np.loadtxt(path, ndmin=2)
  if arr.size == 0:
    return np.array([]), np.array([])
  return arr[:, COL_X], arr[:, COL_EPS]


def x_transform(x):
  return np.log10(x) if X_MODE == "log10" else x


def x_label():
  return r"$\log_{10}(x_{\rm in})$" if X_MODE == "log10" else r"$x_{\rm in}$"


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
      x, eps = load_dat(value_dir / f"{cls}{v_tag}.dat")
      if x.size:
        parts[cls] = (x_transform(x), eps)
    if parts:
      xs = np.concatenate([p[0] for p in parts.values()])
      es = np.concatenate([p[1] for p in parts.values()])
      curves.append((tag_to_float(v_tag), xs, es, parts))
  return curves


def draw_panel(ax, curves, sw, fixed_value, lambda_in, show_ylabel):
  for _, _, _, parts in curves:
    for cls, style in CLASS_STYLE.items():
      if cls in parts:
        x, eps = parts[cls]
        ax.scatter(
            x,
            eps,
            color=style["color"],
            marker=POINT_MARKER,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            linewidths=0,
        )

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
  legend = ax.legend(
      handles=handles,
      title="Critical-point type",
      loc=TYPE_LEGEND_LOC,
      framealpha=0.9,
  )

  ax.set_xlabel(x_label(), fontsize=14)
  if show_ylabel:
    ax.set_ylabel(r"$\varepsilon_{\rm critical}$", fontsize=14)
  ax.set_title(
      rf"Varying ${sw['sym']}$:  ${sw['fixed_sym']} = {fixed_value:g}$,"
      rf"  $\lambda_{{\rm in}} = {lambda_in:g}$",
      fontsize=13,
  )
  ax.grid(alpha=0.3)
  return legend


# ==========================================
# MAIN EXECUTION
# ==========================================

if not data_dir.exists():
  raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()}")

out_dir.mkdir(parents=True, exist_ok=True)

alpha_sw, beta_sw = SWEEPS["alpha_pi"], SWEEPS["beta"]
alpha_pat = re.compile(r"alpha_sweep_beta(\d+o\d+)_lambda(\d+o\d+)$")
beta_pat = re.compile(r"beta_sweep_alpha(\d+o\d+)_lambda(\d+o\d+)$")

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

for alpha_dir, beta_dir in pairs:
  beta_fixed = tag_to_float(alpha_pat.match(alpha_dir.name).group(1))
  alpha_fixed = tag_to_float(beta_pat.match(beta_dir.name).group(1))
  lam_tag = alpha_pat.match(alpha_dir.name).group(2)
  lambda_in = tag_to_float(lam_tag)

  curves_a = load_sweep(alpha_dir, alpha_sw)
  curves_b = load_sweep(beta_dir, beta_sw)
  if not curves_a or not curves_b:
    continue

  fig, (ax_a, ax_b) = plt.subplots(
      1, 2, figsize=FIGSIZE, sharex=True, sharey=True
  )

  draw_panel(ax_a, curves_a, alpha_sw, beta_fixed, lambda_in, show_ylabel=True)
  draw_panel(
      ax_b, curves_b, beta_sw, alpha_fixed, lambda_in, show_ylabel=False
  )

  if YLIM is not None:
    ylim = YLIM
  else:
    y_min = min(float(c[2].min()) for c in curves_a + curves_b)
    ylim = (y_min - 0.15 * (Y_AUTO_MAX - y_min), Y_AUTO_MAX)
  ax_a.set_ylim(*ylim)

  fig.suptitle(
      rf"Critical Energy vs {x_label()}:  ${alpha_sw['sym']}$ sweep (left) vs"
      rf" ${beta_sw['sym']}$ sweep (right),  $\lambda_{{\rm in}} ="
      rf" {lambda_in:g}$",
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
      f"sweep_comparison_{x_name}_lambda{lam_tag}_"
      f"{alpha_dir.name.split('_lambda')[0]}_{beta_dir.name.split('_lambda')[0]}"
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
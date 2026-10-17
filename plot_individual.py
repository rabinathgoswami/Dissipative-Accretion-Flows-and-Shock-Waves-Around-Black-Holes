import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D


# ==========================================
# PATHS
# ==========================================

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"
output_dir = project_dir / "output"
individual_dir = output_dir / "individual"

individual_dir.mkdir(parents=True, exist_ok=True)


# ==========================================
# SETTINGS
# ==========================================

lambda_folders = {
    1.6: "lambda1o6",
    1.65: "lambda1o65",
    1.7: "lambda1o7",
    1.75: "lambda1o75",
    1.8: "lambda1o8",
}

lambda_markers = {
    1.6: "o",
    1.65: "s",
    1.7: "^",
    1.75: "D",
    1.8: "v",
}

# Colours represent critical-point classification
classification_files = {
    "Saddle": ("saddle", "black"),
    "Nodal": ("nodal", "red"),
    "Spiral": ("spiral", "blue"),
}

# Set to True if you want the same y-axis range on every plot
# (makes the individual plots directly comparable)
SHARED_YLIM = False


# ==========================================
# LOAD DATA (once per lambda)
# ==========================================

def load_lambda_data(folder_name):
    """Return {classification: (x_plot, eps_plot)} for one lambda folder."""
    folder_path = data_dir / folder_name
    result = {}

    for classification, (file_prefix, _) in classification_files.items():
        filepath = folder_path / f"{file_prefix}{folder_name[6:]}.dat"

        if not filepath.exists():
            print(f"File not found, skipping: {filepath}")
            continue

        try:
            data = np.loadtxt(filepath, comments="#", ndmin=2)
        except ValueError:
            print(f"No usable numerical data in: {filepath}")
            continue

        if data.size == 0 or data.shape[1] < 11:
            print(f"Empty or invalid data file: {filepath}")
            continue

        x_in = data[:, 0]   # Column 0: x_in
        eps = data[:, 9]    # Column 9: eps

        valid = np.isfinite(x_in) & (x_in > 0) & np.isfinite(eps)

        result[classification] = (np.log10(x_in[valid]), eps[valid])

    return result


all_data = {lam: load_lambda_data(f) for lam, f in lambda_folders.items()}

# Optional shared y-limits
ylim = None
if SHARED_YLIM:
    all_eps = [
        e for d in all_data.values() for (_, e) in d.values() if e.size
    ]
    if all_eps:
        lo = min(e.min() for e in all_eps)
        hi = max(e.max() for e in all_eps)
        pad = 0.05 * (hi - lo)
        ylim = (lo - pad, hi + pad)


# ==========================================
# ONE FIGURE PER LAMBDA
# ==========================================

for lam, folder_name in lambda_folders.items():

    fig, ax = plt.subplots(figsize=(9, 6))

    for classification, (_, colour) in classification_files.items():
        if classification not in all_data[lam]:
            continue

        x_plot, eps_plot = all_data[lam][classification]

        ax.scatter(
            x_plot,
            eps_plot,
            color=colour,
            marker=lambda_markers[lam],
            s=5,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
        )

    # Legend: critical-point classification (colours)
    handles = [
        Line2D(
            [], [],
            color=colour,
            marker=lambda_markers[lam],
            linestyle="None",
            markersize=6,
            label=classification,
        )
        for classification, (_, colour) in classification_files.items()
    ]
    ax.legend(handles=handles, title="Critical-point type", loc="best")

    ax.set_xlabel(r"$\log_{10}(x_{\mathrm{in}})$", fontsize=13)
    ax.set_ylabel(r"$\epsilon_{\mathrm{critical}}$", fontsize=13)
    ax.set_title(
        rf"Critical Energy vs $\log_{{10}}(x_{{\mathrm{{in}}}})$ "
        rf"for $\lambda_{{\mathrm{{in}}}}={lam}$",
        fontsize=14,
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    png_path = individual_dir / f"eps_vs_log_xin_{folder_name}.png"
    pdf_path = individual_dir / f"eps_vs_log_xin_{folder_name}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

print("\nAll individual plots saved in:", individual_dir)
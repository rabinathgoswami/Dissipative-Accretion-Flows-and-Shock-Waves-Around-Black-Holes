
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

output_dir.mkdir(parents=True, exist_ok=True)


# ==========================================
# LAMBDA VALUES AND MARKERS
# ==========================================

lambda_folders = {
    1.2: "lambda1o2",
    1.4: "lambda1o4",
    1.6: "lambda1o6",
    1.8: "lambda1o8",
}

lambda_markers = {
    1.2: "o",
    1.4: "s",
    1.6: "^",
    1.8: "D",
}

# Colours represent critical-point classification
classification_files = {
    "Saddle": ("saddle", "black"),
    "Nodal": ("nodal", "red"),
    "Spiral": ("spiral", "blue"),
}


# ==========================================
# FIGURE
# ==========================================

fig, ax = plt.subplots(figsize=(11, 7))


# ==========================================
# LOAD DATA AND PLOT
# ==========================================

for lam, folder_name in lambda_folders.items():

    folder_path = data_dir / folder_name

    for classification, (file_prefix, colour) in (
        classification_files.items()
    ):

        filename = f"{file_prefix}{folder_name[6:]}.dat"
        filepath = folder_path / filename

        if not filepath.exists():
            print(f"File not found, skipping: {filepath}")
            continue

        # Load numerical data, ignoring the comment header
        try:
            data = np.loadtxt(filepath, comments="#", ndmin=2)
        except ValueError:
            print(f"No usable numerical data in: {filepath}")
            continue

        if data.size == 0 or data.shape[1] < 11:
            print(f"Empty or invalid data file: {filepath}")
            continue

        # Column 0: x_in
        # Column 7: eps
        x_in = data[:, 0]
        eps = data[:, 9]

        # Remove invalid values before plotting
        valid = (
            np.isfinite(x_in)
            & (x_in > 0)
            & np.isfinite(eps)
        )

        x_plot = np.log10(x_in[valid])
        eps_plot = eps[valid]

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



# ==========================================
# LEGENDS
# ==========================================

# Legend 1: Critical-point classification (colours)
classification_handles = [
    Line2D(
        [], [],
        color=colour,
        marker="o",
        linestyle="None",
        markersize=6,
        label=classification,
    )
    for classification, (_, colour)
    in classification_files.items()
]

legend1 = ax.legend(
    handles=classification_handles,
    title="Critical-point type",
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
)

# Legend 2: Lambda values (marker shapes)
lambda_handles = [
    Line2D(
        [], [],
        color="dimgray",
        marker=marker,
        linestyle="None",
        markersize=6,
        label=rf"$\lambda_{{in}}={lam}$",
    )
    for lam, marker in lambda_markers.items()
]

ax.add_artist(legend1)

ax.legend(
    handles=lambda_handles,
    title=r"$\lambda_{in}$",
    loc="upper left",
    bbox_to_anchor=(1.02, 0.58),
    borderaxespad=0,
)


# Reserve space on the right for both legends
fig.subplots_adjust(right=0.75)


# ==========================================
# LABELS AND FORMATTING
# ==========================================

ax.set_xlabel(r"$\ln(x_{\mathrm{in}})$", fontsize=13)
ax.set_ylabel(r"$\epsilon_{\mathrm{critical}}$", fontsize=13)

ax.set_title(
    r"Critical Energy vs $\ln(x_{\mathrm{in}})$ "
    r"for Different $\lambda_{\mathrm{in}}$",
    fontsize=14,
)

ax.grid(True, alpha=0.25)
fig.tight_layout()


# ==========================================
# SAVE FIGURE
# ==========================================

png_path = output_dir / "eps_vs_ln_xin_all_lambdas.png"
pdf_path = output_dir / "eps_vs_ln_xin_all_lambdas.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print("\nPlot saved successfully:")
print(png_path)
print(pdf_path)

plt.show()
fig.tight_layout(rect=[0, 0, 0.75, 1])
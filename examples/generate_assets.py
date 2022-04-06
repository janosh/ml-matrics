# %%
from shutil import which
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from plotly.graph_objs._figure import Figure

from pymatviz.correlation import marchenko_pastur
from pymatviz.cumulative import cum_err, cum_res
from pymatviz.elements import (
    hist_elemental_prevalence,
    ptable_heatmap,
    ptable_heatmap_plotly,
    ptable_heatmap_ratio,
)
from pymatviz.histograms import residual_hist, spacegroup_hist, true_pred_hist
from pymatviz.parity import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)
from pymatviz.quantile import qq_gaussian
from pymatviz.ranking import err_decay
from pymatviz.relevance import precision_recall_curve, roc_curve
from pymatviz.struct_vis import plot_structure_2d
from pymatviz.sunburst import spacegroup_sunburst
from pymatviz.utils import ROOT


# %%
plt.rc("font", size=14)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("axes", titlesize=16, titleweight="bold")
plt.rcParams["figure.constrained_layout.use"] = True


df_ptable = pd.read_csv(f"{ROOT}/pymatviz/elements.csv").set_index("symbol")


# random classification data
np.random.seed(42)
y_binary = np.random.choice([0, 1], 100)
y_proba = np.clip(y_binary - 0.1 * np.random.normal(scale=5, size=100), 0.2, 0.9)


df_steels = load_dataset("matbench_steels")
df_expt_gap = load_dataset("matbench_expt_gap")


# na_filter=False so sodium amide (NaN) is not parsed as 'not a number'
df_roost_ens = pd.read_csv(f"{ROOT}/data/ex-ensemble-roost.csv", na_filter=False)

y_true = df_roost_ens.target
y_pred = df_roost_ens.filter(like="pred").mean(1)
y_var_epi = df_roost_ens.filter(like="pred").var(1)
y_var_ale = (df_roost_ens.filter(like="ale") ** 2).mean(1)
y_std = np.sqrt(y_var_ale + y_var_epi)


def save_mpl_fig(filename: str) -> None:
    """Save current Matplotlib figure as SVG to assets/ folder. Compresses SVG file with
    svgo CLI if available in PATH.

    Args:
        filename (str): Name of SVG file (w/o extension).
    """
    filepath = f"{ROOT}/assets/{filename}.svg"
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

    if (svgo := which("svgo")) is not None:
        call([svgo, "--multipass", filepath])


def save_compress_plotly(fig: Figure, filename: str) -> None:
    """Save Plotly figure as SVG and HTML to assets/ folder. Compresses SVG file with
    svgo CLI if available in PATH.

    Args:
        fig (Figure): Plotly Figure instance.
        filename (str): Name of SVG file (w/o extension).
    """
    filepath = f"{ROOT}/assets/{filename}.svg"
    fig.write_image(filepath)

    if (svgo := which("svgo")) is not None:
        call([svgo, "--multipass", filepath])


# %% Parity Plots
density_scatter(y_pred, y_true)
save_mpl_fig("density_scatter")


density_scatter_with_hist(y_pred, y_true)
save_mpl_fig("density_scatter_with_hist")


density_hexbin(y_pred, y_true)
save_mpl_fig("density_scatter_hex")


density_hexbin_with_hist(y_pred, y_true)
save_mpl_fig("density_scatter_hex_with_hist")


scatter_with_err_bar(y_pred, y_true, yerr=y_std)
save_mpl_fig("scatter_with_err_bar")


residual_vs_actual(y_true, y_pred)
save_mpl_fig("residual_vs_actual")


# %% Elemental Plots
ptable_heatmap(df_expt_gap.composition, log=True)
title = f"Matbench glass elemental prevalence for {len(df_expt_gap):,} compositions"
plt.suptitle(title, fontsize=20, fontweight="bold", y=0.96)
save_mpl_fig("ptable_heatmap")

ptable_heatmap(df_ptable.atomic_mass)
plt.suptitle("Atomic mass heatmap", fontsize=20, fontweight="bold", y=0.96)
save_mpl_fig("ptable_heatmap_atomic_mass")

ptable_heatmap(df_expt_gap.composition, heat_labels="percent")
title = "Matbench glass elemental prevalence in percent"
plt.suptitle(title, fontsize=20, fontweight="bold", y=0.96)
save_mpl_fig("ptable_heatmap_percent")

ptable_heatmap_ratio(df_expt_gap.composition, df_steels.composition, log=True)
title = "Elemental prevalence ratios from Matbench glass to steel"
plt.suptitle(title, fontsize=20, fontweight="bold", y=0.96)
save_mpl_fig("ptable_heatmap_ratio")

hist_elemental_prevalence(df_expt_gap.composition, keep_top=15, v_offset=1)
save_mpl_fig("hist_elemental_prevalence")


# %% Plotly interactive periodic table heatmap
fig = ptable_heatmap_plotly(
    df_ptable.atomic_mass,
    hover_cols=["atomic_mass", "atomic_number"],
    hover_data="density = " + df_ptable.density.astype(str) + " g/cm^3",
)
fig.update_layout(
    title=dict(text="<b>Atomic mass heatmap</b>", x=0.4, y=0.94, font_size=20)
)
fig.show()
save_compress_plotly(fig, "ptable_heatmap_plotly_more_hover_data")

fig = ptable_heatmap_plotly(df_expt_gap.composition, heat_labels="percent")
title = "Matbench Experimental Bandgap Elemental Prevalence"
fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.4, y=0.94, font_size=20))
fig.show()
save_compress_plotly(fig, "ptable_heatmap_plotly_percent_labels")


# %% Quantile/Calibration Plots
qq_gaussian(y_pred, y_true, y_std)
save_mpl_fig("normal_prob_plot")


qq_gaussian(y_pred, y_true, {"overconfident": y_std, "underconfident": 1.5 * y_std})
save_mpl_fig("normal_prob_plot_multiple")


# %% Cumulative Plots
cum_err(y_pred, y_true)
save_mpl_fig("cumulative_error")


cum_res(y_pred, y_true)
save_mpl_fig("cumulative_residual")


# %% Ranking Plots
err_decay(y_true, y_pred, y_std)
save_mpl_fig("err_decay")

eps = 0.2 * np.random.randn(*y_std.shape)

err_decay(y_true, y_pred, {"better": y_std, "worse": y_std + eps})
save_mpl_fig("err_decay_multiple")


# %% Relevance Plots
roc_curve(y_binary, y_proba)
save_mpl_fig("roc_curve")


precision_recall_curve(y_binary, y_proba)
save_mpl_fig("precision_recall_curve")


# %% Histogram Plots
residual_hist(y_true, y_pred)
save_mpl_fig("residual_hist")

true_pred_hist(y_true, y_pred, y_std)
save_mpl_fig("true_pred_hist")


# %%
df_phonons = load_dataset("matbench_phonons")

df_phonons[["spg_symbol", "spg_num"]] = [
    struct.get_space_group_info() for struct in df_phonons.structure
]

spacegroup_hist(df_phonons.spg_num)
save_mpl_fig("spg_num_hist")

spacegroup_hist(df_phonons.spg_symbol)
save_mpl_fig("spg_symbol_hist")


# %% Sunburst Plots
fig = spacegroup_sunburst(df_phonons.spg_num, show_values="percent")
save_compress_plotly(fig, "spg_num_sunburst")

fig = spacegroup_sunburst(df_phonons.spg_symbol, show_values="percent")
save_compress_plotly(fig, "spg_symbol_sunburst")


# %% Correlation Plots
# plot eigenvalue distribution of a pure-noise correlation matrix
# i.e. the correlation matrix contains no significant correlations
# beyond the spurious correlation that occurs randomly
n_rows, n_cols = 500, 1000
rand_wide_mat = np.random.normal(0, 1, size=(n_rows, n_cols))

corr_mat = np.corrcoef(rand_wide_mat)

marchenko_pastur(corr_mat, gamma=n_cols / n_rows)
save_mpl_fig("marchenko_pastur")

# plot eigenvalue distribution of a correlation matrix with significant
# (i.e. non-noise) eigenvalue
n_rows, n_cols = 50, 400
linear_matrix = np.arange(n_rows * n_cols).reshape(n_rows, n_cols) / n_cols

corr_mat = np.corrcoef(linear_matrix + rand_wide_mat[:n_rows, :n_cols])

marchenko_pastur(corr_mat, gamma=n_cols / n_rows)
save_mpl_fig("marchenko_pastur_significant_eval")

# plot eigenvalue distribution of a rank-deficient correlation matrix
n_rows, n_cols = 600, 500
rand_tall_mat = np.random.normal(0, 1, size=(n_rows, n_cols))

corr_mat_rank_deficient = np.corrcoef(rand_tall_mat)

marchenko_pastur(corr_mat_rank_deficient, gamma=n_cols / n_rows)
save_mpl_fig("marchenko_pastur_rank_deficient")


# %%
df_phonons = load_dataset("matbench_phonons")

fig, axs = plt.subplots(3, 4, figsize=(12, 12))

for struct, ax in zip(df_phonons.structure.head(12), axs.flat):
    ax = plot_structure_2d(struct, ax=ax)
    ax.set_title(struct.composition.reduced_formula, fontsize=14)

save_mpl_fig("mp-structures-2d")

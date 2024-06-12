# %%
import numpy as np
from matminer.datasets import load_dataset
from tqdm import tqdm

from pymatviz.enums import Key
from pymatviz.histograms import (
    elements_hist,
    plot_histogram,
    spacegroup_hist,
    true_pred_hist,
)
from pymatviz.io import save_and_compress_svg
from pymatviz.templates import set_plotly_template
from pymatviz.utils import VALID_BACKENDS


set_plotly_template("pymatviz_white")


# %%
df_phonons = load_dataset("matbench_phonons")
df_expt_gap = load_dataset("matbench_expt_gap")

df_phonons[["spg_symbol", Key.spacegroup]] = [
    struct.get_space_group_info()
    for struct in tqdm(df_phonons[Key.structure], desc="Getting spacegroups")
]

# Random regression data
rand_regression_size = 500
y_true = np.random.normal(5, 4, rand_regression_size)
y_pred = 1.2 * y_true - 2 * np.random.normal(0, 1, rand_regression_size)
y_std = (y_true - y_pred) * 10 * np.random.normal(0, 0.1, rand_regression_size)


# %% Histogram Plots
ax = true_pred_hist(y_true, y_pred, y_std)
save_and_compress_svg(ax, "true-pred-hist")

ax = elements_hist(df_expt_gap.composition, keep_top=15, v_offset=1)
save_and_compress_svg(ax, "hist-elemental-prevalence")


# %% Spacegroup histograms
for backend in VALID_BACKENDS:
    fig = spacegroup_hist(df_phonons[Key.spacegroup], backend=backend)
    save_and_compress_svg(fig, f"spg-num-hist-{backend}")

    fig = spacegroup_hist(df_phonons.spg_symbol, backend=backend)
    save_and_compress_svg(fig, f"spg-symbol-hist-{backend}")


# %% matbench experimental band gap distribution
fig = plot_histogram(df_expt_gap["gap expt"], log_y=True)

save_and_compress_svg(fig, "matbench-expt-gap-hist")

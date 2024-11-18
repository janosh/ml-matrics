# %%
import numpy as np

import pymatviz as pmv


# %% Random regression data
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
y_true = np_rng.normal(5, 4, rand_regression_size)
y_pred = 1.2 * y_true - 2 * np_rng.normal(0, 1, rand_regression_size)
y_std = (y_true - y_pred) * 10 * np_rng.normal(0, 0.1, rand_regression_size)


# %% Uncertainty Plots
ax = pmv.qq_gaussian(
    y_pred, y_true, y_std, identity_line={"line_kwargs": {"color": "red"}}
)
pmv.io.save_and_compress_svg(ax, "normal-prob-plot")


ax = pmv.qq_gaussian(
    y_pred, y_true, {"over-confident": y_std, "under-confident": 1.5 * y_std}
)
pmv.io.save_and_compress_svg(ax, "normal-prob-plot-multiple")

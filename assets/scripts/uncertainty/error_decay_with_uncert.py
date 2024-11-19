# %%
import numpy as np

import pymatviz as pmv
from pymatviz.test.data import get_regression_data


np_rng = np.random.default_rng(seed=0)
dummy_data = get_regression_data()


# %% Uncertainty Plots
ax = pmv.error_decay_with_uncert(dummy_data.y_true, dummy_data.y_pred, dummy_data.y_std)
pmv.io.save_and_compress_svg(ax, "error-decay-with-uncert")

eps = 0.2 * np_rng.standard_normal(*dummy_data.y_std.shape)

ax = pmv.error_decay_with_uncert(
    dummy_data.y_true,
    dummy_data.y_pred,
    {"better": dummy_data.y_std, "worse": dummy_data.y_std + eps},
)
pmv.io.save_and_compress_svg(ax, "error-decay-with-uncert-multiple")

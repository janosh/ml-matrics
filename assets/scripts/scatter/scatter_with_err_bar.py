# %%
from __future__ import annotations

import plotly.express as px
import plotly.io as pio

import pymatviz as pmv
from pymatviz.test.config import config_matplotlib
from pymatviz.test.data import get_regression_data


px.defaults.template = "pymatviz_white"
pio.templates.default = "pymatviz_white"

dummy_data = get_regression_data()

# Configure matplotlib
config_matplotlib()


# %% scatter with error bar
ax = pmv.scatter_with_err_bar(
    dummy_data.y_pred, dummy_data.y_true, yerr=dummy_data.y_std
)
pmv.io.save_and_compress_svg(ax, "scatter-with-err-bar")

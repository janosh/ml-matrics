# %%
from __future__ import annotations

import plotly.express as px
import plotly.io as pio

import pymatviz as pmv
from pymatviz.test.config import config_matplotlib


px.defaults.template = "pymatviz_white"
pio.templates.default = "pymatviz_white"

dummy_data = pmv.data.regression()
# Configure matplotlib
config_matplotlib()


# %% density scatter
ax = pmv.density_scatter(dummy_data.y_pred, dummy_data.y_true)
pmv.io.save_and_compress_svg(ax, "density-scatter")

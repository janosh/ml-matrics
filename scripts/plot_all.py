# %%
import matplotlib.pyplot as plt
import pandas as pd

from mlmatrics import (
    ROOT,
    cum_err,
    cum_err_cum_res,
    cum_res,
    density_scatter,
    density_scatter_hex,
    density_scatter_hex_with_hist,
    density_scatter_with_hist,
    err_decay,
    hist_elemental_prevalence,
    precision_recall_curve,
    ptable_elemental_prevalence,
    roc_curve,
    std_calibration,
)

# %%
y_binary, y_proba, y_clf = pd.read_csv(f"{ROOT}/data/rand_clf.csv").to_numpy().T
xs, y_pred, y_true = pd.read_csv(f"{ROOT}/data/rand_regr.csv").to_numpy().T


def savefig(filename: str) -> None:
    plt.savefig(f"{ROOT}/assets/{filename}.svg", bbox_inches="tight")
    plt.close()


# %%
density_scatter(xs, y_pred, xlabel="foo", ylabel="bar")
savefig("density_scatter")


density_scatter_with_hist(xs, y_pred)
savefig("density_scatter_with_hist")


density_scatter_hex(xs, y_pred)
savefig("density_scatter_hex")


density_scatter_hex_with_hist(xs, y_pred)
savefig("density_scatter_hex_with_hist")


# %%
df = pd.read_csv(f"{ROOT}/data/mp-n_elements<2.csv")


ptable_elemental_prevalence(df.formula)
savefig("ptable_elemental_prevalence")


hist_elemental_prevalence(df.formula, keep_top=10)
savefig("hist_elemental_prevalence")


# %%
std_calibration(xs, y_pred, xs)
savefig("std_calibration_single")


std_calibration(xs, y_pred, {"foo": xs, "bar": 0.1 * xs})
savefig("std_calibration_multiple")


# %%
cum_err(y_pred, y_true)
savefig("cumulative_error")


cum_res(y_pred, y_true)
savefig("cumulative_residual")


cum_err_cum_res(y_pred, y_true)
savefig("cumulative_error_cumulative_residual")


# %%
err_decay(y_true, y_pred, y_true - y_pred)
savefig("err_decay")


# %%
roc_curve(y_binary, y_proba)
savefig("roc_curve")


precision_recall_curve(y_binary, y_proba)
savefig("precision_recall_curve")

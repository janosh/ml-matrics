from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from pymatviz import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)
from pymatviz.utils import Array
from tests.conftest import df_x_y


@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("cmap", [None, "Greens"])
@pytest.mark.parametrize(
    "stats",
    [False, True, dict(prefix="test", loc="lower right", prop=dict(fontsize=10))],
)
@pytest.mark.parametrize("df, x, y", df_x_y)
def test_density_scatter(
    df: pd.DataFrame | None,
    x: Array | str,
    y: str | Array,
    log: bool,
    sort: bool,
    cmap: str | None,
    stats: bool | dict[str, Any],
) -> None:
    density_scatter(df=df, x=x, y=y, log=log, sort=sort, cmap=cmap, stats=stats)


@pytest.mark.parametrize("df, x, y", df_x_y)
def test_density_scatter_with_hist(
    df: pd.DataFrame | None, x: str | Array, y: str | Array
) -> None:
    density_scatter_with_hist(df=df, x=x, y=y)


@pytest.mark.parametrize("df, x, y", df_x_y)
def test_density_hexbin(
    df: pd.DataFrame | None, x: str | Array, y: str | Array
) -> None:
    density_hexbin(df=df, x=x, y=y)


@pytest.mark.parametrize("df, x, y", df_x_y)
def test_density_hexbin_with_hist(
    df: pd.DataFrame | None, x: str | Array, y: str | Array
) -> None:
    density_hexbin_with_hist(df=df, x=x, y=y)


@pytest.mark.parametrize("df, x, y", df_x_y)
def test_scatter_with_err_bar(
    df: pd.DataFrame | None, x: str | Array, y: str | Array
) -> None:
    if df is not None:
        err = abs(df[x] - df[y])
    else:
        err = abs(x - y)  # type: ignore[operator]
    scatter_with_err_bar(df=df, x=x, y=y, yerr=err)
    scatter_with_err_bar(df=df, x=x, y=y, xerr=err)


@pytest.mark.parametrize("df, x, y", df_x_y)
def test_residual_vs_actual(
    df: pd.DataFrame | None, x: str | Array, y: str | Array
) -> None:
    residual_vs_actual(df=df, y_true=x, y_pred=y)

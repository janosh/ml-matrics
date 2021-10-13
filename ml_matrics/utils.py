from os.path import abspath, dirname
from typing import TYPE_CHECKING, Any, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import r2_score


if TYPE_CHECKING:
    from numpy.typing import NDArray

    NumArray = NDArray[Union[np.float64, np.int_]]
else:
    NumArray = Sequence[Union[int, float]]
    NDArray = Sequence

ROOT: str = dirname(dirname(abspath(__file__)))


def with_hist(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    cell: GridSpec = None,
    bins: int = 100,
) -> Axes:
    """Call before creating a plot and use the returned `ax_main` for all
    subsequent plotting ops to create a grid of plots with the main plot in
    the lower left and narrow histograms along its x- and/or y-axes displayed
    above and near the right edge.

    Args:
        xs (array): x values.
        ys (array): y values.
        cell (GridSpec, optional): Cell of a plt GridSpec at which to add the
            grid of plots. Defaults to None.
        bins (int, optional): Resolution/bin count of the histograms. Defaults to 100.

    Returns:
        Axes: The axes to be used to for the main plot.
    """
    fig = plt.gcf()

    gs = (cell.subgridspec if cell else fig.add_gridspec)(
        2, 2, width_ratios=(6, 1), height_ratios=(1, 5), wspace=0, hspace=0
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # x_hist
    ax_histx.hist(xs, bins=bins, rwidth=0.8)
    ax_histx.axis("off")

    # y_hist
    ax_histy.hist(ys, bins=bins, rwidth=0.8, orientation="horizontal")
    ax_histy.axis("off")

    return ax_main


def softmax(arr: NDArray[np.float64], axis: int = -1) -> NDArray[np.float64]:
    """Compute the softmax of an array along an axis."""
    exp = np.exp(arr)
    return exp / exp.sum(axis=axis, keepdims=True)


def one_hot(targets: Sequence[int], n_classes: int = None) -> NDArray[np.int_]:
    """Get a one-hot encoded version of `targets` containing `n_classes`."""
    if n_classes is None:
        n_classes = np.max(targets) + 1
    return np.eye(n_classes)[targets]


def annotate_bar_heights(
    ax: Axes = None,
    voffset: int = 10,
    hoffset: int = 0,
    labels: Sequence[Union[str, int, float]] = None,
    fontsize: int = 14,
) -> None:
    """Annotate histograms with a label indicating the height/count of each bar.

    Args:
        ax (matplotlib.axes.Axes): The axes to annotate.
        voffset (int): Vertical offset between the labels and the bars.
        hoffset (int): Horizontal offset between the labels and the bars.
        labels (list[str]): Labels used for annotating bars. Falls back to the
            y-value of each bar if None.
        fontsize (int): Annotated text size in pts. Defaults to 14.
    """
    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = [int(patch.get_height()) for patch in ax.patches]

    for rect, label in zip(ax.patches, labels):

        y_pos = rect.get_height()
        x_pos = rect.get_x() + rect.get_width() / 2 + hoffset

        if ax.get_yscale() == "log":
            y_pos = y_pos + np.log(voffset)
        else:
            y_pos = y_pos + voffset

        # place label at end of the bar and center horizontally
        ax.annotate(label, (x_pos, y_pos), ha="center", fontsize=fontsize)
        # ensure enough vertical space to display label above highest bar
        ax.margins(y=0.1)


def add_mae_r2_box(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    ax: Axes = None,
    loc: str = "lower right",
    prec: int = 3,
    **kwargs: Any,
) -> None:
    """Provide a set of x and y values of equal length and an optional Axes object
    on which to print the values' mean absolute error and R^2 coefficient of
    determination.

    Args:
        xs (array, optional): x values.
        ys (array, optional): y values.
        ax (plt.Axes, optional): plt.Axes object. Defaults to None.
        loc (str, optional): Where on the plot to place the AnchoredText object.
            Defaults to "lower right".
        prec (int, optional): # of decimal places in printed metrics. Defaults to 3.
    """
    if ax is None:
        ax = plt.gca()

    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[mask], ys[mask]

    mae_str = f"$\\mathrm{{MAE}} = {np.abs(xs - ys).mean():.{prec}f}$\n"

    r2_str = f"$R^2 = {r2_score(xs, ys):.{prec}f}$"

    frameon: bool = kwargs.pop("frameon", False)
    text_box = AnchoredText(mae_str + r2_str, loc=loc, frameon=frameon, **kwargs)
    ax.add_artist(text_box)

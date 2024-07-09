"""Project (nest) a custom plot into a periodic table."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatter, LogLocator
from pymatgen.core import Element

from pymatviz.colors import ELEM_COLORS_JMOL, ELEM_COLORS_VESTA, ELEM_TYPE_COLORS
from pymatviz.enums import ElemColorMode, ElemColorScheme, Key
from pymatviz.ptable._process_data import (
    PTableData,
    SupportedDataType,
    SupportedValueType,
)
from pymatviz.utils import df_ptable


if TYPE_CHECKING:
    from typing import Any, Callable

    import pandas as pd
    from matplotlib.typing import ColorType


class PTableProjector:
    """Project (nest) a custom plot into a periodic table.

    Attributes:
        cmap (Colormap | None): Colormap.
        data (pd.DataFrame): Data for plotting.
        ptable_data (PTableData): Internal data container.
        norm (Normalize): Data min-max normalizer.
        hide_f_block (bool): Whether to hide f-block.
        elem_types (set[str]): Types of elements present.
        elem_type_colors (dict[str, str]): Element typed based colors.
        elem_colors (dict): Element based colors.

    Scopes mentioned in this plotter:
        plot: Refers to the global Figure.
        axis: Refers to the Axis where child plotter would plot.
        child: Refers to the child plotter, for example, ax.plot().
    """

    def __init__(
        self,
        *,
        data: SupportedDataType | PTableData,
        log: bool = False,
        colormap: str | Colormap = "viridis",
        tile_size: tuple[float, float] = (0.75, 0.75),
        plot_kwargs: dict[str, Any] | None = None,
        exclude_elements: Sequence[str] = (),
        on_empty: Literal["hide", "show"] = "hide",
        hide_f_block: bool | Literal["AUTO"] = "AUTO",
        elem_type_colors: dict[str, str] | None = None,
        elem_colors: ElemColorScheme | dict[str, ColorType] = ElemColorScheme.vesta,
    ) -> None:
        """Initialize a ptable projector.

        Default figsize is set to (0.75 * n_groups, 0.75 * n_periods),
        and axes would be turned off by default.

        Args:
            data (SupportedDataType | PTableData): The data to be visualized.
            log (bool): Whether to show colorbar in log scale.
            colormap (str | Colormap): The colormap to use. Defaults to "viridis".
            tile_size (tuple[float, float]): The relative tile height and width.
            plot_kwargs (dict): Additional keyword arguments to
                pass to the plt.subplots function call.
            exclude_elements (Sequence[str]): Elements to exclude.
            on_empty ("hide" | "show"): Hide or show tile if no data provided.
            hide_f_block (bool | "AUTO"): Hide f-block (Lanthanum and Actinium series).
                Defaults to "AUTO", meaning hide if no data present.
            elem_type_colors (dict | None): Element typed based colors.
            elem_colors (dict | ElemColors): Element-specific colors.
        """
        # Set colors
        self.cmap: Colormap = colormap
        self._elem_type_colors = ELEM_TYPE_COLORS | (elem_type_colors or {})
        self.elem_colors = elem_colors  # type: ignore[assignment]

        # Preprocess data
        self.log = log
        self.data: pd.DataFrame = data

        # Remove excluded element from internal data to avoid metadata pollution
        self.exclude_elements = exclude_elements
        self.ptable_data.drop_elements(exclude_elements)

        self.on_empty = on_empty
        self.hide_f_block = hide_f_block  # type: ignore[assignment]

        # Initialize periodic table canvas
        n_periods = df_ptable.row.max()
        n_groups = df_ptable.column.max()

        if self.hide_f_block:
            n_periods -= 3

        # Set figure size
        plot_kwargs = plot_kwargs or {}
        height, width = tile_size
        plot_kwargs.setdefault("figsize", (height * n_groups, width * n_periods))

        self.fig, self.axes = plt.subplots(n_periods, n_groups, **plot_kwargs)

        # Turn off all axes
        for ax in self.axes.flat:
            ax.axis("off")

    @property
    def cmap(self) -> Colormap | None:
        """The periodic table's Colormap."""
        return self._cmap

    @cmap.setter
    def cmap(self, colormap: str | Colormap | None) -> None:
        """Set the periodic table's Colormap."""
        self._cmap = None if colormap is None else plt.get_cmap(colormap)

    @property
    def data(self) -> pd.DataFrame:
        """The preprocessed data."""
        return self.ptable_data.data

    @data.setter
    def data(self, data: SupportedDataType) -> None:
        """Preprocess and set the data. Also set normalizer."""
        if not isinstance(data, PTableData):
            data = PTableData(data)

        self.ptable_data = data

        # Update norm when data is updated
        self.norm = (None, None)

    @property
    def norm(self) -> Normalize | LogNorm:
        """Data normalizer."""
        return self._norm

    @norm.setter
    def norm(self, value_range: tuple[float | None, float | None]) -> None:
        """Set normalizer by value range.

        Args:
            value_range (tuple[float | None, float | None]): The upper and
                lower bound of values, use None for auto detect from metadata.
        """
        value_range = (
            value_range[0] or self.ptable_data.data.attrs["vmin"],
            value_range[1] or self.ptable_data.data.attrs["vmax"],
        )
        self._norm: Normalize | LogNorm = (
            LogNorm(*value_range) if self.log else Normalize(*value_range)
        )

    @property
    def anomalies(self) -> dict[str, set[Literal["nan", "inf"]]] | Literal["NA"]:
        """Element symbol to anomalies ("nan/inf") mapping."""
        return self.ptable_data.anomalies

    @property
    def hide_f_block(self) -> bool:
        """Whether to hide f-block in plots."""
        return self._hide_f_block

    @hide_f_block.setter
    def hide_f_block(self, hide_f_block: bool | Literal["AUTO"]) -> None:
        """If hide_f_block is "AUTO", would detect if data is present."""
        if hide_f_block == "AUTO":
            f_block_elements_has_data = {
                atom_num
                for atom_num in [*range(57, 72), *range(89, 104)]  # rare earths
                if (elem := Element.from_Z(atom_num).symbol) in self.data.index
                and len(self.data.loc[elem, Key.heat_val]) > 0
            }

            self._hide_f_block = not bool(f_block_elements_has_data)

        else:
            self._hide_f_block = hide_f_block

    @property
    def elem_types(self) -> set[str]:
        """Set of element types present in data."""
        return set(df_ptable.loc[self.data.index, "type"])

    @property
    def elem_type_colors(self) -> dict[str, str]:
        """Element type based colors mapping.

        Example:
            dict(Nonmetal="green", Halogen="teal", Metal="lightblue")
        """
        return self._elem_type_colors or {}

    @elem_type_colors.setter
    def elem_type_colors(self, elem_type_colors: dict[str, str]) -> None:
        self._elem_type_colors |= elem_type_colors or {}

    @property
    def elem_colors(self) -> dict[str, ColorType]:
        """Element-based colors."""
        return self._elem_colors

    @elem_colors.setter
    def elem_colors(
        self,
        elem_colors: ElemColorScheme | dict[str, ColorType] = ElemColorScheme.vesta,
    ) -> None:
        """Set elem_colors.

        Args:
            elem_colors ("vesta" | "jmol" | dict[str, ColorType]): Use VESTA
                or Jmol color mapping, or a custom {"element": Color} mapping.
                Defaults to "vesta".
        """
        if elem_colors == "vesta":
            self._elem_colors = ELEM_COLORS_VESTA
        elif elem_colors == "jmol":
            self._elem_colors = ELEM_COLORS_JMOL
        elif isinstance(elem_colors, dict):
            self._elem_colors = elem_colors
        else:
            raise ValueError(
                f"elem_colors must be 'vesta', 'jmol', or a custom dict, "
                f"got {elem_colors=}"
            )

    def get_elem_type_color(
        self,
        elem_symbol: str,
        default: ColorType = "white",
    ) -> ColorType:
        """Get element type color by element symbol.

        Args:
            elem_symbol (str): The element symbol.
            default (str): Default color if not found in mapping.
        """
        elem_type = df_ptable.loc[elem_symbol].get("type", None)
        return self.elem_type_colors.get(elem_type, default)

    def add_child_plots(
        self,
        child_plotter: Callable[..., None],
        *,
        child_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        ax_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Add custom child plots to the periodic table grid.

        Args:
            child_plotter (Callable): The child plotter.
            child_kwargs (dict): Arguments to pass to the child plotter call.
            tick_kwargs (dict): Keyword arguments to pass to ax.tick_params().
            ax_kwargs (dict): Keyword arguments to pass to ax.set().
        """
        # Update kwargs
        child_kwargs = child_kwargs or {}
        ax_kwargs = ax_kwargs or {}
        tick_kwargs = {"axis": "both", "which": "major", "labelsize": 8} | (
            tick_kwargs or {}
        )

        for element in Element:
            # Hide f-block
            if self.hide_f_block and (element.is_lanthanoid or element.is_actinoid):
                continue

            # Get axis index by element symbol
            symbol: str = element.symbol
            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax: plt.Axes = self.axes[row - 1][column - 1]

            # Get and check tile data
            try:
                plot_data: np.ndarray | Sequence[float] = self.data.loc[
                    symbol, Key.heat_val
                ]
            except KeyError:  # skip element without data
                plot_data = None

            if (plot_data is None or len(plot_data) == 0) and self.on_empty == "hide":
                continue

            # Call child plotter
            child_plotter(ax, plot_data, tick_kwargs=tick_kwargs, **child_kwargs)

            # Pass axis kwargs
            if ax_kwargs:
                ax.set(**ax_kwargs)

    def add_elem_symbols(
        self,
        text: str | Callable[[Element], str] = lambda elem: elem.symbol,
        *,
        pos: tuple[float, float] = (0.5, 0.5),
        text_color: ColorType
        | dict[str, ColorType]
        | Literal[ElemColorMode.element_types] = "black",
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Add element symbols for each tile.

        Args:
            text (str | Callable): The text to add to the tile.
                If a callable, it should accept a pymatgen Element and return a
                string. If a string, it can contain a format
                specifier for an `elem` variable which will be replaced by the element.
            pos (tuple): The position of the text relative to the axes.
            text_color (ColorType | dict[str, ColorType]): The color of the text.
                Defaults to "black". Could take the following type:
                    - ColorType: The same color for all elements.
                    - dict[str, ColorType]: An element to color mapping.
                    - "element-types": Use color from self.elem_type_colors.
            kwargs (dict): Additional keyword arguments to pass to the `ax.text`.
        """
        # Update symbol kwargs
        kwargs = kwargs or {}
        kwargs.setdefault("fontsize", 12)

        # Add symbol for each element
        for element in Element:
            # Hide f-block
            if self.hide_f_block and (element.is_lanthanoid or element.is_actinoid):
                continue

            # Get axis index by element symbol
            symbol: str = element.symbol
            if symbol not in self.data.index and self.on_empty == "hide":
                continue

            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax: plt.Axes = self.axes[row - 1][column - 1]

            content = text(element) if callable(text) else text.format(elem=element)

            # Generate symbol text color
            if text_color == ElemColorMode.element_types:
                symbol_color = self.get_elem_type_color(symbol, "black")
            elif isinstance(text_color, dict):
                symbol_color = text_color.get(symbol, "black")
            else:
                symbol_color = text_color

            ax.text(
                *pos,
                content,
                color=symbol_color,
                ha="center",
                va="center",
                transform=ax.transAxes,
                **kwargs,
            )

    def add_colorbar(
        self,
        title: str,
        *,
        coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.02),
        cbar_range: tuple[float | None, float | None] = (None, None),
        cbar_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Add colorbar.

        Args:
            title (str): Title for the colorbar.
            coords (tuple): Coordinates of the colorbar (left, bottom, width, height).
                Defaults to (0.18, 0.8, 0.42, 0.02).
            cbar_range (tuple): Colorbar values display range, use None for auto
                detection for the corresponding boundary.
            cbar_kwargs (dict): Additional keyword arguments to pass to fig.colorbar().
            title_kwargs (dict): Additional keyword arguments for the colorbar title.
        """
        # Update colorbar kwargs
        cbar_kwargs = {"orientation": "horizontal"} | (cbar_kwargs or {})
        title_kwargs = {"fontsize": 12, "pad": 10, "label": title} | (
            title_kwargs or {}
        )

        # Format for log scale
        if self.log:
            cbar_kwargs |= {
                "norm": "log",
                "format": LogFormatter(10),
                "ticks": LogLocator(base=10),
            }

        # Update colorbar range
        self.norm = cbar_range

        # Check colormap
        if self.cmap is None:
            raise ValueError("Cannot add colorbar without a colormap.")

        # Add colorbar
        cbar_ax = self.fig.add_axes(coords)
        self.fig.colorbar(
            plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
            cax=cbar_ax,
            **cbar_kwargs,
        )

        # Set colorbar title
        cbar_ax.set_title(**title_kwargs)

    def add_elem_type_legend(
        self,
        kwargs: dict[str, Any] | None,
    ) -> None:
        """Add a legend to show the colors based on element types.

        Args:
            kwargs (dict | None): Keyword arguments passed to
                plt.legend() for customizing legend appearance.
        """
        kwargs = kwargs or {}

        font_size = 10

        # Get present elements
        legend_elements = [
            plt.Line2D(
                *([0], [0]),
                marker="s",
                color="w",
                label=elem_class,
                markerfacecolor=color,
                markersize=1.2 * font_size,
            )
            for elem_class, color in self.elem_type_colors.items()
            if elem_class in self.elem_types
        ]

        default_legend_kwargs = dict(
            loc="center left",
            bbox_to_anchor=(0, -42),
            ncol=6,
            frameon=False,
            fontsize=font_size,
            handlelength=1,  # more compact legend
        )
        kwargs = default_legend_kwargs | kwargs

        plt.legend(handles=legend_elements, **kwargs)

    def set_elem_background_color(self, alpha: float = 0.1) -> None:
        """Set element tile background color by element type.

        Args:
            alpha (float): Transparency.
        """
        for element in Element:
            # Hide f-block
            if self.hide_f_block and (element.is_lanthanoid or element.is_actinoid):
                continue

            # Get element axis
            symbol = element.symbol
            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax: plt.Axes = self.axes[row - 1][column - 1]

            # Set background color by element type
            rgb = mpl.colors.to_rgb(self.get_elem_type_color(symbol))
            ax.set_facecolor((*rgb, alpha))


class ChildPlotters:
    """Child plotters for PTableProjector to make different types
    (heatmap/line/scatter/histogram) for individual element tiles.
    """

    @staticmethod
    def rectangle(
        ax: plt.axes,
        data: SupportedValueType,
        norm: Normalize,
        cmap: Colormap,
        start_angle: float,
        tick_kwargs: dict[str, Any],  # noqa: ARG004
    ) -> None:
        """Rectangle heatmap plotter. Could be evenly split
        depending on the length of the data (could mix and match).

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for the child plotter.
            norm (Normalize): Normalizer for data-color mapping.
            cmap (Colormap): Colormap used for value mapping.
            start_angle (float): The starting angle for the splits in degrees,
                and the split proceeds counter-clockwise (0 refers to the x-axis).
            tick_kwargs: For compatibility with other plotters.
        """
        # Map values to colors
        if isinstance(data, (Sequence, np.ndarray)):
            colors = [cmap(norm(value)) for value in data]
        else:
            raise TypeError("Unsupported data type.")

        # Add the pie chart
        ax.pie(
            np.ones(len(colors)),
            colors=colors,
            startangle=start_angle,
            wedgeprops={"clip_on": True},
        )

        # Crop the central rectangle from the pie chart
        rect = Rectangle(xy=(-0.5, -0.5), width=1, height=1, fc="none", ec="none")
        ax.set_clip_path(rect)

        ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))

    @staticmethod
    def scatter(
        ax: plt.axes,
        data: SupportedValueType,
        norm: Normalize,
        cmap: Colormap,
        tick_kwargs: dict[str, Any],
        **child_kwargs: Any,
    ) -> None:
        """Scatter plotter.

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for the child plotter.
            norm (Normalize): Normalizer for data-color mapping.
            cmap (Colormap): Colormap.
            child_kwargs (dict): kwargs to pass to the child plotter call.
            tick_kwargs (dict): kwargs to pass to ax.tick_params().
        """
        # Add scatter
        if len(data) == 2:
            ax.scatter(x=data[0], y=data[1], **child_kwargs)
        elif len(data) == 3:
            ax.scatter(x=data[0], y=data[1], c=cmap(norm(data[2])), **child_kwargs)

        # Set tick labels
        ax.tick_params(**tick_kwargs)

        # Hide the right and top spines
        ax.axis("on")  # turned off by default
        ax.spines[["right", "top"]].set_visible(False)

    @staticmethod
    def line(
        ax: plt.axes,
        data: SupportedValueType,
        tick_kwargs: dict[str, Any],
        **child_kwargs: Any,
    ) -> None:
        """Line plotter.

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for the child plotter.
            child_kwargs (dict): kwargs to pass to the child plotter call.
            tick_kwargs (dict): kwargs to pass to ax.tick_params().
        """
        # Add line
        ax.plot(data[0], data[1], **child_kwargs)

        # Set tick labels
        ax.tick_params(**tick_kwargs)

        # Hide the right and top spines
        ax.axis("on")  # turned off by default
        ax.spines[["right", "top"]].set_visible(False)

    @staticmethod
    def histogram(
        ax: plt.axes,
        data: SupportedValueType,
        cmap: Colormap,
        cbar_axis: Literal["x", "y"],
        tick_kwargs: dict[str, Any],
        **child_kwargs: Any,
    ) -> None:
        """Histogram plotter.

        Taken from https://stackoverflow.com/questions/23061657/
        plot-histogram-with-colors-taken-from-colormap

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for the child plotter.
            cmap (Colormap): Colormap.
            cbar_axis ("x" | "y"): The axis colormap would be based on.
            child_kwargs: kwargs to pass to the child plotter call.
            tick_kwargs (dict): kwargs to pass to ax.tick_params().
        """
        # Preprocess x_range if only one boundary is given
        x_range = child_kwargs.pop("range", None)

        if x_range is None or x_range == [None, None]:
            x_range = [np.min(data), np.max(data)]

        elif x_range[0] is None:
            x_range = [np.min(data), x_range[1]]

        else:
            x_range = [x_range[0], np.max(data)]

        # Add histogram
        n, bins, patches = ax.hist(data, range=x_range, **child_kwargs)

        # Scale values according to axis
        if cbar_axis == "x":
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            cols = bin_centers - min(bin_centers)
            cols /= max(cols)

        else:
            cols = (n - n.min()) / (n.max() - n.min())

        # Apply colors
        for col, patch in zip(cols, patches):
            plt.setp(patch, "facecolor", cmap(col))

        # Set tick labels
        ax.tick_params(**tick_kwargs)
        ax.set(yticklabels=(), yticks=())
        # Set x-ticks to min/max only
        ax.set(xticks=[math.floor(x_range[0]), math.ceil(x_range[1])])

        # Hide the right, left and top spines
        ax.axis("on")
        ax.spines[["right", "top", "left"]].set_visible(False)

"""Resolve impact panel geometry and displayed year ticks for shared figures."""

import math
from collections.abc import Sequence

from matplotlib.ticker import MaxNLocator
import numpy as np

_DOUBLE_BASE_ROWS = 5
_DOUBLE_BASE_ROW_HEIGHT = 3.1
_DOUBLE_BASE_HEIGHT = _DOUBLE_BASE_ROWS * _DOUBLE_BASE_ROW_HEIGHT
_DOUBLE_FIG_WIDTH = 22.0
_SINGLE_FIG_WIDTH = 16.0
DOUBLE_COLUMN_TITLE_TOP = 0.965
TRANSITION_PANEL_TITLE_PAD = 14
MULTI_YEAR_SINGLE_PANEL_FIGURE_SIZE = (12.5, 5.4)
MULTI_YEAR_TWO_PANEL_FIGURE_SIZE = (15.5, 3.6)
MULTI_YEAR_COMPACT_PLOT_HEIGHT_IN = 4.5
SINGLE_IMPACT_SINGLE_YEAR_FIGURE_SIZE = (12.5, 3.6)
SINGLE_IMPACT_MULTI_YEAR_FIGURE_SIZE = (12.5, 2.7)
SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN = 2.25
MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN = SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN * 1.3
MULTI_IMPACT_PANEL_WIDTH_IN = 18.0
MULTI_IMPACT_PANEL_MIN_HEIGHT_IN = 7.2
MULTI_IMPACT_PANEL_COMPACT_MIN_HEIGHT_IN = SINGLE_IMPACT_SINGLE_YEAR_FIGURE_SIZE[1]
MULTI_IMPACT_PANEL_ROW_HEIGHT_IN = 2.45


def single_impact_figure_size(*, single_year: bool) -> tuple[float, float]:
    """Return the compact figure size for one impact or non LCIA based panel."""
    return (
        SINGLE_IMPACT_SINGLE_YEAR_FIGURE_SIZE
        if single_year
        else SINGLE_IMPACT_MULTI_YEAR_FIGURE_SIZE
    )


def multi_impact_panel_figure_size(*, nrows: int, compact: bool = False) -> tuple[float, float]:
    """Return the shared figure size for two column impact panel products."""
    row_count = max(1, int(nrows))
    row_height = (
        SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN if bool(compact) else MULTI_IMPACT_PANEL_ROW_HEIGHT_IN
    )
    min_height = (
        MULTI_IMPACT_PANEL_COMPACT_MIN_HEIGHT_IN
        if bool(compact)
        else MULTI_IMPACT_PANEL_MIN_HEIGHT_IN
    )
    return (
        MULTI_IMPACT_PANEL_WIDTH_IN,
        max(min_height, row_height * float(row_count)),
    )


def resolve_layout(*, impacts_count: int) -> dict[str, float | int | str]:
    """Resolve subplot layout by number of impact categories."""
    count = max(1, int(impacts_count))
    if count <= 5:
        ncols = 1
        nrows = count
        return {
            "layout": "single",
            "ncols": ncols,
            "nrows": nrows,
            "fig_width": _SINGLE_FIG_WIDTH,
            "fig_height": _DOUBLE_BASE_HEIGHT,
        }
    ncols = 2
    nrows = int(math.ceil(float(count) / 2.0))
    fig_height = _DOUBLE_BASE_HEIGHT
    if count > 10:
        fig_height = max(_DOUBLE_BASE_HEIGHT, _DOUBLE_BASE_ROW_HEIGHT * float(nrows))
    return {
        "layout": "double",
        "ncols": ncols,
        "nrows": nrows,
        "fig_width": _DOUBLE_FIG_WIDTH,
        "fig_height": fig_height,
    }


def bottom_panel_indices(*, panel_count: int, ncols: int) -> set[int]:
    """Return active panel indices that sit lowest in each plotted column."""
    bottoms: dict[int, int] = {}
    for index in range(max(0, int(panel_count))):
        bottoms[index % max(1, int(ncols))] = index
    return set(bottoms.values())


def show_panel_x_labels(
    *,
    panel_index: int,
    bottom_indices: set[int],
) -> bool:
    """Return whether one subplot in a multi-row panel should show x tick labels."""
    return bool(int(panel_index) in bottom_indices)


def hide_x_axis_tick_labels(axis) -> None:
    """Hide x tick labels and tick marks while preserving x scale ownership."""
    axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axis.set_xlabel("")


def hide_unused_axes(*, axes, used: int) -> None:
    """Hide unused axes in a rectangular subplot grid."""
    for index in range(max(0, int(used)), axes.size):
        axes[index // axes.shape[1], index % axes.shape[1]].axis("off")


def format_single_year_category_axis(
    axis,
    *,
    positions: Sequence[int | float],
    labels: Sequence[str],
) -> None:
    """Apply readable category labels for exact single year bars or violins."""
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=30, ha="right")


def resolve_polar_availability(n_impacts: int) -> bool:
    """Return True if polar/radial figures are appropriate for this impact count."""
    return n_impacts >= 2


def build_year_columns(*, years: list[int], step: int) -> np.ndarray:
    """Build displayed x axis tick years from data years and step."""
    unique_years = sorted({int(y) for y in years})
    if not unique_years:
        return np.array([], dtype=int)
    first_year = unique_years[0]
    cols = np.array(
        [y for y in unique_years if (int(y) - first_year) % int(step) == 0],
        dtype=int,
    )
    return cols


def format_integer_year_axis(
    axis,
    *,
    years: Sequence[int],
    rotation: float = 0.0,
    ha: str = "center",
) -> None:
    """Apply automatic integer year ticks to one x axis."""
    unique_years = sorted({int(year) for year in years})
    if not unique_years:
        return
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.tick_params(axis="x", labelrotation=rotation)
    for label in axis.get_xticklabels():
        label.set_ha(ha)

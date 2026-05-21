"""Shared uncertainty interval rendering primitives."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pyaesa.shared.figures.figure_footer import (
    align_lower_legend_top_to_layout,
    legend_display_rows,
    reserve_footer_space,
)

BAND_ALPHA_OUTER = 0.14
BAND_ALPHA_INNER = 0.24
SUMMARY_COLUMNS = ("mean", "median", "p25", "p75", "p5", "p95")


def trajectory_band_legend_handles(
    *,
    color: Any,
    pair_count: int | None = None,
    sampling_method: str | None = None,
    outer_alpha: float = BAND_ALPHA_OUTER,
    inner_alpha: float = BAND_ALPHA_INNER,
) -> list[Any]:
    """Return legend handles describing uncertainty interval geometry."""
    handles: list[Any] = [
        Line2D([0], [0], color=color, linewidth=2.2, label="Mean line"),
        Line2D([0], [0], color=color, linewidth=1.6, linestyle=":", label="Median line"),
        Patch(facecolor=color, alpha=float(inner_alpha), label="p25 to p75 band"),
        Patch(facecolor=color, alpha=float(outer_alpha), label="p5 to p95 band"),
    ]
    if pair_count is not None:
        pair_label = f"{int(pair_count)} AR6 CC model-scenario pairs"
        if sampling_method is not None:
            pair_label = f"{pair_label}; sampling method: {str(sampling_method).strip().lower()}"
        handles.append(
            Line2D(
                [0],
                [0],
                color="none",
                linewidth=0.0,
                label=pair_label,
            )
        )
    return handles


def render_trajectory_band_legend_below(
    fig,
    *,
    color: Any,
    prefix_handles: Sequence[Any] = (),
    pair_count: int | None = None,
    sampling_method: str | None = None,
    ncol: int | None = None,
    outer_alpha: float = BAND_ALPHA_OUTER,
    inner_alpha: float = BAND_ALPHA_INNER,
    title_rows: int = 0,
    handler_map: Mapping[type, object] | None = None,
    legend_kwargs: Mapping[str, object] | None = None,
    extra_height_in: float = 0.0,
    frameon: bool = True,
    title: str | None = None,
) -> None:
    """Render a below figure legend for trajectory interval geometry."""
    handles = [
        *list(prefix_handles),
        *trajectory_band_legend_handles(
            color=color,
            pair_count=pair_count,
            sampling_method=sampling_method,
            outer_alpha=outer_alpha,
            inner_alpha=inner_alpha,
        ),
    ]
    labels = [str(handle.get_label()).strip() for handle in handles]
    columns = max(1, len(labels) if ncol is None else int(ncol))
    title_row_count = 1 if title is not None and str(title).strip() else 0
    layout = reserve_footer_space(
        fig,
        rows=legend_display_rows(labels, ncol=columns),
        note_lines=0,
        title_rows=title_rows + title_row_count,
        extra_height_in=extra_height_in,
    )
    legend = fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, layout.anchor_y),
        ncol=columns,
        frameon=frameon,
        fontsize="small",
        title=title,
        title_fontsize="small",
        handler_map=handler_map,
        **dict(legend_kwargs or {}),
    )
    if title is not None and str(title).strip():
        legend.get_title().set_fontweight("bold")
    align_lower_legend_top_to_layout(fig, legend, layout=layout)


def render_trajectory_band(
    axis,
    *,
    years: np.ndarray,
    summaries: Mapping[str, object],
    color: Any,
    value_scale: float = 1.0,
    label: str = "_nolegend_",
    line_alpha: float = 0.82,
    mean_linewidth: float = 1.9,
    mean_linestyle: str = "-",
    median_linewidth: float = 1.4,
    outer_alpha: float = BAND_ALPHA_OUTER,
    inner_alpha: float = BAND_ALPHA_INNER,
    line_path_effects: Sequence[Any] | None = None,
) -> dict[str, np.ndarray]:
    """Render one uncertainty interval from precomputed summary columns."""
    x = np.asarray(years, dtype=np.int64)
    values = {
        column: _numeric_array(summaries[column]) * float(value_scale) for column in SUMMARY_COLUMNS
    }
    axis.fill_between(
        x,
        values["p5"],
        values["p95"],
        color=color,
        alpha=float(outer_alpha),
        linewidth=0,
        zorder=1,
    )
    axis.fill_between(
        x,
        values["p25"],
        values["p75"],
        color=color,
        alpha=float(inner_alpha),
        linewidth=0,
        zorder=2,
    )
    mean_line = axis.plot(
        x,
        values["mean"],
        color=color,
        linewidth=float(mean_linewidth),
        linestyle=mean_linestyle,
        alpha=float(line_alpha),
        label=label,
        zorder=4,
    )[0]
    if line_path_effects is not None:
        mean_line.set_path_effects(list(line_path_effects))
    axis.plot(
        x,
        values["median"],
        color=color,
        linewidth=float(median_linewidth),
        linestyle=":",
        alpha=0.92,
        label="_nolegend_",
        zorder=5,
    )
    return values


def _numeric_array(values: object) -> np.ndarray:
    series = pd.Series(pd.to_numeric(values, errors="raise"))
    return series.to_numpy(dtype=np.float64)

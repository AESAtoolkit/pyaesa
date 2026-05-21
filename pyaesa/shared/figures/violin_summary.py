"""Shared legend contracts for uncertainty violin summary figures."""

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from pyaesa.shared.figures.figure_footer import (
    align_lower_legend_top_to_layout,
    legend_display_rows,
    reserve_footer_space,
)

VIOLIN_ALPHA = 0.32
VIOLIN_WIDTH = 0.78
VIOLIN_MEDIAN_MARKER_SIZE = 9.5
VIOLIN_MEDIAN_LINEWIDTH = 2.2
VIOLIN_COMPACT_TICK_HALF_WIDTH = 0.045
VIOLIN_COMPACT_TICK_LINEWIDTH = 0.9
VIOLIN_COMPACT_DOTTED_LINESTYLE = (0, (0.35, 0.45))
VIOLIN_LEGEND_HANDLE_LENGTH = 4.0
VIOLIN_LEGEND_HANDLE_HEIGHT = 3.8
VIOLIN_LEGEND_BORDER_PAD = 0.65
VIOLIN_LEGEND_LABEL_SPACING = 0.95
VIOLIN_FULL_LEGEND_VERTICAL_SHIFT = -0.94
VIOLIN_MEAN_MEDIAN_LEGEND_VERTICAL_SHIFT = -0.80
VIOLIN_LEGEND_FOOTER_EXTRA_HEIGHT_IN = 0.0
VIOLIN_SUMMARY_LEGEND_GROUP_TITLE = "Violin summary"
ViolinSummaryMode = Literal["full", "mean_median"]
VIOLIN_FULL_LEGEND_LABEL = (
    "Width is proportional to the distribution of Monte Carlo runs.\n"
    "distribution: min to max\nwhisker: p5 to p95; bar: p25 to p75\n"
    "tick: median; circle: mean"
)
VIOLIN_MEAN_MEDIAN_LEGEND_LABEL = (
    "Width is proportional to the distribution of Monte Carlo runs.\n"
    "distribution: min to max\nsolid tick: median; dotted tick: mean"
)
VIOLIN_LEGEND_LABEL = VIOLIN_FULL_LEGEND_LABEL


class ViolinSummaryLegendHandle(Line2D):
    """Proxy artist for the composite violin summary legend glyph."""

    def __init__(self, *, color: Any = "#4d4d4d", summary: ViolinSummaryMode = "full") -> None:
        label = VIOLIN_FULL_LEGEND_LABEL if summary == "full" else VIOLIN_MEAN_MEDIAN_LEGEND_LABEL
        super().__init__([], [], color=color, label=label)
        self.summary: ViolinSummaryMode = summary


class ViolinSummaryLegendHandler(HandlerBase):
    """Render one violin body with embedded summary markers in legend space."""

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ) -> list[Any]:
        del legend, fontsize
        handle = cast(ViolinSummaryLegendHandle, orig_handle)
        color = handle.get_color()
        x_center = float(xdescent) + 0.5 * float(width)
        y_shift = (
            VIOLIN_FULL_LEGEND_VERTICAL_SHIFT
            if handle.summary == "full"
            else VIOLIN_MEAN_MEDIAN_LEGEND_VERTICAL_SHIFT
        ) * float(height)
        y_bottom = float(ydescent) + y_shift + 0.03 * float(height)
        y_top = float(ydescent) + y_shift + 0.97 * float(height)
        y_values = np.linspace(y_bottom, y_top, 28)
        y_center = 0.5 * (y_bottom + y_top)
        y_sigma = 0.28 * max(y_top - y_bottom, 1e-12)
        half_widths = 0.07 * float(width) + 0.28 * float(width) * np.exp(
            -(((y_values - y_center) / y_sigma) ** 2)
        )
        left = np.column_stack([x_center - half_widths, y_values])
        right = np.column_stack([x_center + half_widths, y_values])[::-1]
        vertices = np.vstack([left, right, left[:1]])
        codes = [Path.MOVETO, *([Path.LINETO] * (len(vertices) - 2)), Path.CLOSEPOLY]
        body = PathPatch(
            Path(vertices, codes),
            facecolor=color,
            edgecolor=color,
            alpha=VIOLIN_ALPHA,
            linewidth=1.0,
            transform=trans,
        )
        y_p5 = float(ydescent) + y_shift + 0.10 * float(height)
        y_q1 = float(ydescent) + y_shift + 0.36 * float(height)
        y_median = float(ydescent) + y_shift + 0.50 * float(height)
        y_mean = float(ydescent) + y_shift + 0.61 * float(height)
        y_q3 = float(ydescent) + y_shift + 0.70 * float(height)
        y_p95 = float(ydescent) + y_shift + 0.90 * float(height)
        artists: list[Any] = [body]
        if handle.summary == "full":
            whisker = Line2D([x_center, x_center], [y_p5, y_p95], color=color, linewidth=1.7)
            interval = Line2D([x_center, x_center], [y_q1, y_q3], color=color, linewidth=5.0)
            artists.extend([whisker, interval])
        tick_half_width = 0.22 * float(width)
        median = Line2D(
            [x_center - tick_half_width, x_center + tick_half_width],
            [y_median, y_median],
            color=color,
            linewidth=VIOLIN_MEDIAN_LINEWIDTH
            if handle.summary == "full"
            else VIOLIN_COMPACT_TICK_LINEWIDTH,
            linestyle="-",
        )
        artists.append(median)
        if handle.summary == "full":
            mean_fill = Line2D(
                [x_center],
                [y_mean],
                color="white",
                marker="o",
                markersize=6.2,
                markeredgewidth=0.0,
                linestyle="",
            )
            mean_ring = Line2D(
                [x_center],
                [y_mean],
                color=color,
                marker="o",
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.4,
                markersize=5.0,
                linestyle="",
            )
            artists.extend([mean_fill, mean_ring])
        else:
            mean = Line2D(
                [x_center - tick_half_width, x_center + tick_half_width],
                [y_mean, y_mean],
                color=color,
                linewidth=VIOLIN_COMPACT_TICK_LINEWIDTH,
                linestyle=VIOLIN_COMPACT_DOTTED_LINESTYLE,
            )
            artists.append(mean)
        for artist in artists:
            artist.set_transform(trans)
        return artists


def violin_summary_legend_handles(*, summary: ViolinSummaryMode = "full") -> list[Any]:
    """Return legend handles describing violin distribution summaries."""
    return [ViolinSummaryLegendHandle(summary=summary)]


def violin_summary_legend_handler_map() -> dict[type, HandlerBase]:
    """Return the Matplotlib handler map for violin summary legend handles."""
    return {ViolinSummaryLegendHandle: ViolinSummaryLegendHandler()}


def violin_summary_legend_kwargs() -> dict[str, float]:
    """Return legend sizing kwargs that keep the composite violin glyph readable."""
    return {
        "handlelength": VIOLIN_LEGEND_HANDLE_LENGTH,
        "handleheight": VIOLIN_LEGEND_HANDLE_HEIGHT,
        "borderpad": VIOLIN_LEGEND_BORDER_PAD,
        "labelspacing": VIOLIN_LEGEND_LABEL_SPACING,
    }


def violin_summary_footer_extra_height() -> float:
    """Return additional footer height needed by the composite violin legend glyph."""
    return VIOLIN_LEGEND_FOOTER_EXTRA_HEIGHT_IN


def render_violin_summary_legend_below(
    fig,
    *,
    summary: ViolinSummaryMode = "full",
    extra_entries: Sequence[Any] = (),
    extra_height_in: float = 0.0,
    frameon: bool = True,
    title: str | None = None,
) -> None:
    """Render the shared below figure violin summary legend."""
    handles = [*list(extra_entries), *violin_summary_legend_handles(summary=summary)]
    labels = [str(handle.get_label()).strip() for handle in handles]
    ncol = min(3, len(labels))
    title_rows = 1 if title is not None and str(title).strip() else 0
    layout = reserve_footer_space(
        fig,
        rows=legend_display_rows(labels, ncol=ncol),
        note_lines=0,
        title_rows=title_rows,
        extra_height_in=violin_summary_footer_extra_height() + float(extra_height_in),
    )
    legend = fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, layout.anchor_y),
        ncol=ncol,
        frameon=frameon,
        fontsize="small",
        title=title,
        title_fontsize="small",
        handler_map=violin_summary_legend_handler_map(),
        **violin_summary_legend_kwargs(),
    )
    if title is not None and str(title).strip():
        legend.get_title().set_fontweight("bold")
    align_lower_legend_top_to_layout(fig, legend, layout=layout)


def render_violin_summaries(
    axis,
    *,
    values: list[np.ndarray],
    positions: np.ndarray,
    colors: list[Any],
    value_scale: float = 1.0,
    width: float = VIOLIN_WIDTH,
    summary: ViolinSummaryMode = "full",
    alpha: float = VIOLIN_ALPHA,
) -> list[dict[str, float]]:
    """Render violin distributions and preconfigured summary markers."""
    scaled = [_finite_array(value) * float(value_scale) for value in values]
    if not scaled:
        return []
    bodies = axis.violinplot(
        scaled,
        positions=positions,
        showextrema=False,
        widths=float(width),
    )
    stats_by_position: list[dict[str, float]] = []
    for index, (numeric, color) in enumerate(zip(scaled, colors, strict=True)):
        body = bodies["bodies"][index]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(float(alpha))
        stats = violin_summary_stats(numeric)
        position = float(positions[index])
        if summary == "full":
            axis.vlines(position, stats["p5"], stats["p95"], color=color, linewidth=1.3)
            axis.vlines(position, stats["p25"], stats["p75"], color=color, linewidth=4.0)
        if summary == "full":
            axis.scatter(
                [position],
                [stats["mean"]],
                color="white",
                s=46,
                marker="o",
                linewidths=0.0,
                label="_nolegend_",
                zorder=4.5,
            )
            axis.scatter(
                [position],
                [stats["mean"]],
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                s=30,
                marker="o",
                label="_nolegend_",
                zorder=5,
            )
            axis.plot(
                [position],
                [stats["median"]],
                color=color,
                marker="_",
                markersize=VIOLIN_MEDIAN_MARKER_SIZE,
                markeredgewidth=VIOLIN_MEDIAN_LINEWIDTH,
                linestyle="",
                zorder=6,
            )
        else:
            left = position - VIOLIN_COMPACT_TICK_HALF_WIDTH
            right = position + VIOLIN_COMPACT_TICK_HALF_WIDTH
            axis.plot(
                [left, right],
                [stats["median"], stats["median"]],
                color=color,
                linewidth=VIOLIN_COMPACT_TICK_LINEWIDTH,
                linestyle="-",
                zorder=6,
            )
            axis.plot(
                [left, right],
                [stats["mean"], stats["mean"]],
                color=color,
                linewidth=VIOLIN_COMPACT_TICK_LINEWIDTH,
                linestyle=VIOLIN_COMPACT_DOTTED_LINESTYLE,
                zorder=6,
            )
        stats_by_position.append(stats)
    return stats_by_position


def violin_summary_stats(values: np.ndarray) -> dict[str, float]:
    """Return visual summary statistics for one violin distribution."""
    numeric = _finite_array(values)
    return {
        "mean": float(np.nanmean(numeric)),
        "p5": float(np.nanpercentile(numeric, 5)),
        "p25": float(np.nanpercentile(numeric, 25)),
        "median": float(np.nanpercentile(numeric, 50)),
        "p75": float(np.nanpercentile(numeric, 75)),
        "p95": float(np.nanpercentile(numeric, 95)),
    }


def _finite_array(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    return numeric[np.isfinite(numeric)]

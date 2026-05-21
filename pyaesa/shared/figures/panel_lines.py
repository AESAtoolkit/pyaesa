"""Render panelized time series lines from prepared deterministic panel payloads."""

from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from pyaesa.shared.figures.layout import DOUBLE_COLUMN_TITLE_TOP, resolve_layout
from pyaesa.shared.figures.multi_year_transitions import (
    TransitionMarker,
    render_transition_markers,
)
from pyaesa.shared.figures.nonnegative_axis import require_nonnegative_figure_ylim
from pyaesa.shared.figures.deterministic_legends import (
    bind_deterministic_legend_group,
    render_grouped_deterministic_legend_below,
)
from pyaesa.shared.figures.save import save_figure
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.title_contract import (
    build_resolved_figure_title,
)
from pyaesa.shared.figures.titles import render_figure_title, title_layout_top
from pyaesa.shared.tabular.scalars import sanitize_token

plt.switch_backend("Agg")

PanelSeriesItem: TypeAlias = tuple[str, list[int], list[float], str]
PanelSeries: TypeAlias = list[tuple[str, list[PanelSeriesItem]]]
PanelMarkers: TypeAlias = dict[str, list[TransitionMarker]]


def render_panel_series(
    *,
    panel_series: PanelSeries,
    output_path: Path,
    title_parts: dict[str, str | None],
    ylabel: str,
    dpi: int,
    output_format: str,
    y_percent: bool = False,
    panel_markers: PanelMarkers | None = None,
    split_panels: bool = False,
    overlay_panels: bool = False,
    footer_note: str | None = None,
    force_zero_ymin: bool = False,
) -> list[Path]:
    """Render one panel per row figure from prepared time series payloads."""
    if not panel_series:
        return []
    panel_count = len(panel_series)
    if overlay_panels:
        fig, axis = plt.subplots(figsize=(11.8, 6.8))
        merged_markers: dict[tuple[int, str, str], TransitionMarker] = {}
        for panel_label, series_items in panel_series:
            for series_label, years, values, legend_group in series_items:
                label = series_label or None
                line = axis.plot(years, values, linewidth=1.6, alpha=0.9, label=label)[0]
                bind_deterministic_legend_group(line, legend_group)
            for marker in [] if panel_markers is None else panel_markers.get(panel_label, []):
                merged_markers[(int(marker.year), str(marker.label), str(marker.color))] = marker
        render_transition_markers(axis, markers=list(merged_markers.values()))
        axis.set_xlabel("")
        axis.set_ylabel(format_scientific_figure_text(ylabel))
        if y_percent:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        if force_zero_ymin:
            axis.set_ylim(
                *require_nonnegative_figure_ylim(
                    values=np.asarray(
                        [
                            value
                            for _panel, items in panel_series
                            for _label, _years, values, _group in items
                            for value in values
                        ],
                        dtype=float,
                    ),
                    context=build_resolved_figure_title(
                        title_parts=title_parts,
                        panel_count=panel_count,
                    ),
                )
            )
        axis.grid(alpha=0.25)
        legend_labels = [
            label for _panel, items in panel_series for label, _y, _v, _g in items if label
        ]
        if legend_labels:
            render_grouped_deterministic_legend_below(axis, legend_note=footer_note)
        overlay_panel_title = str(panel_series[0][0]).strip() if panel_count == 1 else None
        figure_title = build_resolved_figure_title(
            title_parts=title_parts,
            panel_title=overlay_panel_title,
            panel_count=panel_count,
        )
        render_figure_title(fig, figure_title)
        fig.subplots_adjust(top=title_layout_top(fig, figure_title, default_top=0.94))
        return save_figure(fig, output_path, dpi=dpi, output_format=output_format)
    if split_panels:
        paths: list[Path] = []
        for panel_label, series_items in panel_series:
            fig, axis = plt.subplots(figsize=(9.4, 6.2))
            panel_text = str(panel_label).strip()
            for series_label, years, values, legend_group in series_items:
                line = axis.plot(
                    years,
                    values,
                    linewidth=1.6,
                    alpha=0.9,
                    label=series_label or None,
                )[0]
                bind_deterministic_legend_group(line, legend_group)
            markers = [] if panel_markers is None else panel_markers.get(panel_label, [])
            render_transition_markers(axis, markers=markers)
            axis.set_xlabel("")
            axis.set_ylabel(format_scientific_figure_text(ylabel))
            if y_percent:
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            if force_zero_ymin:
                axis.set_ylim(
                    *require_nonnegative_figure_ylim(
                        values=np.asarray(
                            [
                                value
                                for _label, _years, values, _group in series_items
                                for value in values
                            ],
                            dtype=float,
                        ),
                        context=build_resolved_figure_title(
                            title_parts=title_parts,
                            panel_title=panel_text,
                            panel_count=1,
                        ),
                    )
                )
            axis.grid(alpha=0.25)
            legend_labels = [label for label, _years, _values, _group in series_items if label]
            if legend_labels:
                render_grouped_deterministic_legend_below(axis, legend_note=footer_note)
            figure_title = build_resolved_figure_title(
                title_parts=title_parts,
                panel_title=panel_text,
                panel_count=1,
            )
            render_figure_title(fig, figure_title)
            fig.subplots_adjust(top=title_layout_top(fig, figure_title, default_top=0.94))
            paths.extend(
                save_figure(
                    fig,
                    output_path
                    if not panel_text
                    else output_path.parent / f"{output_path.name}__{sanitize_token(panel_label)}",
                    dpi=dpi,
                    output_format=output_format,
                )
            )
        return paths
    layout = resolve_layout(impacts_count=len(panel_series))
    fig, axes = plt.subplots(
        int(layout["nrows"]),
        int(layout["ncols"]),
        figsize=(float(layout["fig_width"]), float(layout["fig_height"])),
        squeeze=False,
    )
    flat_axes = list(axes.flatten())
    for axis in flat_axes[len(panel_series) :]:
        axis.set_visible(False)
    for axis, (panel_label, series_items) in zip(flat_axes, panel_series, strict=False):
        for series_label, years, values, legend_group in series_items:
            line = axis.plot(
                years,
                values,
                linewidth=1.6,
                alpha=0.9,
                label=series_label or None,
            )[0]
            bind_deterministic_legend_group(line, legend_group)
        markers = [] if panel_markers is None else panel_markers.get(panel_label, [])
        render_transition_markers(axis, markers=markers)
        axis.set_xlabel("")
        axis.set_ylabel(format_scientific_figure_text(ylabel))
        if y_percent:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        if force_zero_ymin:
            axis.set_ylim(
                *require_nonnegative_figure_ylim(
                    values=np.asarray(
                        [
                            value
                            for _label, _years, values, _group in series_items
                            for value in values
                        ],
                        dtype=float,
                    ),
                    context=build_resolved_figure_title(
                        title_parts=title_parts,
                        panel_count=panel_count,
                    ),
                )
            )
        axis.grid(alpha=0.25)
        panel_title = str(panel_label).strip()
        if panel_title:
            axis.set_title(format_scientific_figure_text(panel_title), loc="left")
    if any(
        label for _panel, items in panel_series for label, _years, _values, _group in items if label
    ):
        render_grouped_deterministic_legend_below(flat_axes[0], legend_note=footer_note)
    figure_panel_title = str(panel_series[0][0]).strip() if panel_count == 1 else None
    figure_title = build_resolved_figure_title(
        title_parts=title_parts,
        panel_title=figure_panel_title,
        panel_count=panel_count,
    )
    render_figure_title(fig, figure_title)
    fig.subplots_adjust(
        top=title_layout_top(
            fig,
            figure_title,
            default_top=DOUBLE_COLUMN_TITLE_TOP,
            panel_title_pad=5 if panel_count > 1 else 0,
        )
    )
    return save_figure(
        fig,
        output_path,
        dpi=dpi,
        output_format=output_format,
    )

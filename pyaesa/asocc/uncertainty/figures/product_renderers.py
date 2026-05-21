"""Rendering policy for aSoCC uncertainty figure products."""

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from pyaesa.asocc.figures.product_renderers import (
    apply_asocc_y_axis_policy,
)
from pyaesa.asocc.figures.product_renderers import plot_scope as plot_deterministic_scope
from pyaesa.asocc.uncertainty.figures.scope_planner import (
    SUMMARY_STAT_COLUMNS,
    VALUE_ARRAY_COLUMN,
    visible_values,
)
from pyaesa.asocc.uncertainty.figures.transition_planner import transition_year
from pyaesa.shared.figures.deterministic_legends import (
    bind_deterministic_legend_group,
    render_grouped_deterministic_legend_below,
)
from pyaesa.shared.figures.deterministic_legends_methods import legend_group_from_row
from pyaesa.shared.figures.figure_footer import set_footer_min_plot_height
from pyaesa.shared.figures.lcia_metadata import (
    ordered_impact_panels,
    resolve_frame_impact_title,
)
from pyaesa.shared.figures.colors import (
    DEFAULT_SINGLE_SERIES_COLOR,
    MULTI_METHOD_LINE_ALPHA,
    distinct_colors,
    single_or_distinct_colors,
)
from pyaesa.shared.figures.layout import (
    DOUBLE_COLUMN_TITLE_TOP,
    MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN,
    SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN,
    TRANSITION_PANEL_TITLE_PAD,
    bottom_panel_indices,
    format_single_year_category_axis,
    format_integer_year_axis,
    multi_impact_panel_figure_size,
    single_impact_figure_size,
    show_panel_x_labels,
)
from pyaesa.shared.figures.multi_year_transitions import (
    TransitionMarker,
    render_transition_markers,
)
from pyaesa.shared.figures.paths import output_file_path
from pyaesa.shared.figures.trajectory_bands import (
    SUMMARY_COLUMNS,
    render_trajectory_band,
    render_trajectory_band_legend_below,
)
from pyaesa.shared.figures.titles import render_figure_title, title_layout_top
from pyaesa.shared.figures.value_order import (
    order_labels_by_average_score,
    row_average_score,
)
from pyaesa.shared.figures.violin_summary import (
    ViolinSummaryMode,
    VIOLIN_SUMMARY_LEGEND_GROUP_TITLE,
    render_violin_summaries,
    render_violin_summary_legend_below,
    violin_summary_legend_handler_map,
    violin_summary_legend_handles,
    violin_summary_legend_kwargs,
    violin_summary_footer_extra_height,
)
from pyaesa.shared.figures.asocc_transition_policy import (
    ASOCC_TRANSITION_SERIES_EXCLUDED_COLUMNS,
)

_LINE_ALPHA = 0.82
_PANEL_TITLE_PAD = 5
_TWO_COLUMN_PANEL_HSPACE = 0.32
_TWO_COLUMN_TRANSITION_HSPACE = 0.42
MEAN_LINE_NOTE = "The lines represent Monte Carlo runs mean values."


def plot_band_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_impact_in_label: bool,
    include_method_in_label: bool = True,
) -> list[Path]:
    """Render one multi-year uncertainty interval scope."""
    plot_frame = frame.copy()
    include_impact_in_legend = _include_impact_in_legend(
        plot_frame,
        include_impact_in_label=include_impact_in_label,
    )
    if len(_ordered_impacts(plot_frame)) > 1:
        return _plot_impact_panel_band_scope(
            frame=plot_frame,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
        )
    fig, axis = plt.subplots(figsize=single_impact_figure_size(single_year=False))
    set_footer_min_plot_height(
        fig,
        height_in=_single_panel_min_height(
            frame=plot_frame,
            include_method_in_label=include_method_in_label,
        ),
    )
    visible_years, transitions = _render_band_payload(
        axis=axis,
        frame=plot_frame,
        group_legend=group_legend,
        include_impact_in_label=include_impact_in_legend,
        include_method_in_label=include_method_in_label,
    )
    _format_year_axis(axis, visible_years)
    _format_axes(axis)
    axis.set_title(title, fontweight="bold", pad=26 if transitions else 6)
    axis.grid(alpha=0.25)
    render_transition_markers(axis, markers=transitions)
    _render_trajectory_band_legend_box(fig=fig, axis=axis)
    return _save(fig=fig, output_stem=output_stem, output_format=output_format, dpi=dpi)


def _plot_impact_panel_band_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_method_in_label: bool,
) -> list[Path]:
    """Render one LCIA method as two columns of impact specific band panels."""
    impacts = _ordered_impacts(frame)
    ncols = 2
    nrows = (len(impacts) + 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=multi_impact_panel_figure_size(nrows=nrows),
        squeeze=False,
        sharey=True,
    )
    common_top = _visible_band_top(frame)
    first_axis = axes[0, 0]
    has_transitions = False
    for index, impact in enumerate(impacts):
        row = index // ncols
        column = index % ncols
        axis = axes[row, column]
        panel_frame = frame.loc[frame["impact"].astype(str).eq(str(impact))].copy()
        visible_years, transitions = _render_band_payload(
            axis=axis,
            frame=panel_frame,
            group_legend=group_legend,
            include_impact_in_label=False,
            include_method_in_label=include_method_in_label,
        )
        _format_year_axis(axis, visible_years)
        _format_axes(axis, data_top=common_top)
        if transitions:
            has_transitions = True
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(
            _impact_panel_title(panel_frame, impact=str(impact)),
            loc="left",
            pad=TRANSITION_PANEL_TITLE_PAD if transitions else _PANEL_TITLE_PAD,
        )
        axis.grid(alpha=0.25)
        render_transition_markers(axis, markers=transitions)
    for index in range(len(impacts), nrows * ncols):
        row = index // ncols
        column = index % ncols
        axes[row, column].axis("off")
    render_figure_title(fig, title)
    fig.subplots_adjust(
        hspace=_TWO_COLUMN_TRANSITION_HSPACE if has_transitions else _TWO_COLUMN_PANEL_HSPACE,
        wspace=0.16,
        top=title_layout_top(
            fig,
            title,
            default_top=DOUBLE_COLUMN_TITLE_TOP,
            panel_title_pad=TRANSITION_PANEL_TITLE_PAD if has_transitions else _PANEL_TITLE_PAD,
        ),
    )
    _render_trajectory_band_legend_box(fig=fig, axis=first_axis)
    return _save(fig=fig, output_stem=output_stem, output_format=output_format, dpi=dpi)


def _render_band_payload(
    *,
    axis,
    frame: pd.DataFrame,
    group_legend: bool,
    include_impact_in_label: bool,
    include_method_in_label: bool,
) -> tuple[list[int], list[TransitionMarker]]:
    line_alpha = MULTI_METHOD_LINE_ALPHA if group_legend else _LINE_ALPHA
    colors = _series_color_map(
        frame,
        include_impact_in_label=include_impact_in_label,
        include_method_in_label=include_method_in_label,
    )
    visible_years: list[int] = []
    transitions: dict[int, TransitionMarker] = {}
    for label, group in _series_groups(
        frame,
        include_impact_in_label=include_impact_in_label,
        include_method_in_label=include_method_in_label,
    ):
        ordered = group.sort_values("year", kind="stable")
        years = pd.Series(pd.to_numeric(ordered["year"], errors="raise"), copy=False).astype(int)
        visible_years.extend(int(year) for year in years.tolist())
        color = colors[label]
        year_values = years.to_numpy(dtype=int)
        render_trajectory_band(
            axis,
            years=year_values,
            summaries={column: ordered[column] for column in SUMMARY_COLUMNS},
            color=color,
            value_scale=100.0,
            label="_nolegend_",
            line_alpha=line_alpha,
        )
        marker = transition_year(ordered)
        if marker is not None:
            transitions[int(marker)] = TransitionMarker(
                year=int(marker),
                label="retrospective/prospective transition",
                color="#7d7d7d",
            )
    return visible_years, list(transitions.values())


def plot_mean_line_scope(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_impact_in_label: bool,
    include_method_in_label: bool = True,
) -> list[Path]:
    """Render a method comparison scope with deterministic aSoCC visuals."""
    return plot_deterministic_scope(
        frame=frame,
        requested_years=requested_years,
        output_stem=output_stem,
        title=title,
        dpi=dpi,
        output_format=output_format,
        group_legend=group_legend,
        include_impact_in_label=include_impact_in_label,
        include_method_in_label=include_method_in_label,
        figure_note=MEAN_LINE_NOTE,
    )


def plot_violin_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_impact_in_label: bool,
    include_method_in_label: bool = True,
) -> list[Path]:
    """Render one single year uncertainty violin scope."""
    plot_frame = frame.copy()
    if group_legend and len(_ordered_impacts(plot_frame)) > 1:
        return _plot_impact_panel_violin_scope(
            frame=plot_frame,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            include_method_in_label=include_method_in_label,
        )
    axis_impact_label = _include_impact_in_legend(
        plot_frame,
        include_impact_in_label=include_impact_in_label,
    )
    rows = _violin_entries(
        frame,
        include_impact_in_label=axis_impact_label,
        include_method_in_label=include_method_in_label,
    )
    fig, axis = plt.subplots(figsize=single_impact_figure_size(single_year=True))
    set_footer_min_plot_height(fig, height_in=SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN)
    _render_violin_payload(
        axis=axis,
        rows=rows,
        colors=_single_axis_violin_colors(
            row_count=len(rows),
            impact_axis_labels=axis_impact_label,
            group_legend=group_legend,
        ),
        group_legend=group_legend,
        visible_labels=set(),
    )
    _format_axes(axis)
    axis.set_xlabel("")
    axis.set_title(title, fontweight="bold", pad=6)
    axis.grid(alpha=0.25, axis="y")
    if group_legend:
        _attach_violin_legend_handles(axis)
        render_grouped_deterministic_legend_below(
            axis,
            handler_map=violin_summary_legend_handler_map(),
            legend_kwargs=violin_summary_legend_kwargs(),
            legend_kwargs_group_title=VIOLIN_SUMMARY_LEGEND_GROUP_TITLE,
            legend_extra_height_in=violin_summary_footer_extra_height(),
            hidden_group_titles={VIOLIN_SUMMARY_LEGEND_GROUP_TITLE},
        )
    else:
        render_violin_summary_legend_below(fig)
    return _save(fig=fig, output_stem=output_stem, output_format=output_format, dpi=dpi)


def _plot_impact_panel_violin_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    include_method_in_label: bool,
) -> list[Path]:
    """Render one LCIA method as impact specific single year violin panels."""
    impacts = _ordered_impacts(frame)
    ncols = 2
    nrows = (len(impacts) + 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=multi_impact_panel_figure_size(nrows=nrows),
        squeeze=False,
        sharey=True,
    )
    common_top = _visible_violin_top(frame)
    label_order = _violin_label_order(
        frame,
        include_impact_in_label=False,
        include_method_in_label=include_method_in_label,
    )
    color_map = _violin_color_map(label_order)
    first_axis = axes[0, 0]
    bottom_label_indices = bottom_panel_indices(panel_count=len(impacts), ncols=ncols)
    visible_labels: set[str] = set()
    for index, impact in enumerate(impacts):
        row = index // ncols
        column = index % ncols
        axis = axes[row, column]
        panel_frame = frame.loc[frame["impact"].astype(str).eq(str(impact))].copy()
        rows = _violin_entries(
            panel_frame,
            include_impact_in_label=False,
            include_method_in_label=include_method_in_label,
            label_order=label_order,
        )
        _render_violin_payload(
            axis=axis,
            rows=rows,
            colors=[color_map[label] for label, _values, _row in rows],
            group_legend=True,
            visible_labels=visible_labels,
            show_x_labels=show_panel_x_labels(
                panel_index=index,
                bottom_indices=bottom_label_indices,
            ),
            summary="mean_median",
        )
        _format_axes(axis, data_top=common_top)
        axis.set_xlabel("")
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(
            _impact_panel_title(panel_frame, impact=str(impact)),
            loc="left",
            pad=_PANEL_TITLE_PAD,
        )
        axis.grid(alpha=0.25, axis="y")
    for index in range(len(impacts), nrows * ncols):
        row = index // ncols
        column = index % ncols
        axes[row, column].axis("off")
    render_figure_title(fig, title)
    fig.subplots_adjust(
        hspace=_TWO_COLUMN_PANEL_HSPACE,
        wspace=0.16,
        top=title_layout_top(
            fig,
            title,
            default_top=DOUBLE_COLUMN_TITLE_TOP,
            panel_title_pad=_PANEL_TITLE_PAD,
        ),
    )
    _attach_violin_legend_handles(first_axis, summary="mean_median")
    render_grouped_deterministic_legend_below(
        first_axis,
        handler_map=violin_summary_legend_handler_map(),
        legend_kwargs=violin_summary_legend_kwargs(),
        legend_kwargs_group_title=VIOLIN_SUMMARY_LEGEND_GROUP_TITLE,
        legend_extra_height_in=violin_summary_footer_extra_height(),
        hidden_group_titles={VIOLIN_SUMMARY_LEGEND_GROUP_TITLE},
    )
    return _save(fig=fig, output_stem=output_stem, output_format=output_format, dpi=dpi)


def _render_violin_payload(
    *,
    axis,
    rows: list[tuple[str, np.ndarray, pd.Series]],
    colors: list[str],
    group_legend: bool,
    visible_labels: set[str],
    show_x_labels: bool = True,
    summary: ViolinSummaryMode = "full",
) -> None:
    """Render violin bodies and summary markers for one axis."""
    values = [entry[1] * 100.0 for entry in rows]
    positions = np.arange(1, len(rows) + 1, dtype=float)
    render_violin_summaries(
        axis,
        values=values,
        positions=positions,
        colors=colors,
        summary=summary,
    )
    for index, (label, _numeric, row) in enumerate(rows):
        color = colors[index]
        visible_label = label if group_legend and label not in visible_labels else "_nolegend_"
        handle = Line2D(
            [],
            [],
            color=color,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.2,
            linestyle="",
            markersize=5.0,
            label=visible_label,
        )
        if visible_label != "_nolegend_":
            bind_deterministic_legend_group(handle, legend_group_from_row(row))
            axis.add_line(handle)
            visible_labels.add(label)
    labels = [entry[0] for entry in rows]
    if show_x_labels and any(str(label).strip() for label in labels):
        format_single_year_category_axis(
            axis,
            positions=positions.tolist(),
            labels=labels,
        )
        return
    axis.set_xticks(positions.tolist())
    axis.set_xticklabels([])
    axis.tick_params(axis="x", length=0)


def _violin_color_map(labels: list[str]) -> dict[str, str]:
    return {label: color for label, color in zip(labels, distinct_colors(len(labels)), strict=True)}


def _single_axis_violin_colors(
    *,
    row_count: int,
    impact_axis_labels: bool,
    group_legend: bool,
) -> list[str]:
    if impact_axis_labels and not group_legend:
        return [DEFAULT_SINGLE_SERIES_COLOR] * int(row_count)
    return distinct_colors(int(row_count))


def _violin_label_order(
    frame: pd.DataFrame,
    *,
    include_impact_in_label: bool,
    include_method_in_label: bool,
) -> list[str]:
    scores: dict[str, list[float]] = {}
    for _index, row in frame.iterrows():
        series = pd.Series(row, copy=False)
        label = _row_label(
            series,
            include_impact_in_label=include_impact_in_label,
            include_method_in_label=include_method_in_label,
        )
        scores.setdefault(label, []).append(_row_average_score(series))
    return order_labels_by_average_score(scores)


def _attach_violin_legend_handles(axis, *, summary: ViolinSummaryMode = "full") -> None:
    for handle in _violin_legend_handles(summary=summary):
        bind_deterministic_legend_group(handle, VIOLIN_SUMMARY_LEGEND_GROUP_TITLE)
        axis.add_line(handle)


def _violin_legend_handles(*, summary: ViolinSummaryMode = "full") -> list[Any]:
    return violin_summary_legend_handles(summary=summary)


def _series_groups(
    frame: pd.DataFrame,
    *,
    include_impact_in_label: bool,
    include_method_in_label: bool,
):
    excluded = {
        *SUMMARY_STAT_COLUMNS,
        "public_row_id",
        VALUE_ARRAY_COLUMN,
        *ASOCC_TRANSITION_SERIES_EXCLUDED_COLUMNS,
    }
    group_columns = [
        column
        for column in frame.columns
        if column not in {*excluded, "year"} and not str(column).startswith("__figure")
    ]
    for _key, group in frame.groupby(group_columns, dropna=False, sort=True):
        first = pd.Series(group.iloc[0], copy=False)
        yield (
            _row_label(
                first,
                include_impact_in_label=include_impact_in_label,
                include_method_in_label=include_method_in_label,
            ),
            group,
        )


def _series_color_map(
    frame: pd.DataFrame,
    *,
    include_impact_in_label: bool,
    include_method_in_label: bool,
) -> dict[str, str]:
    labels = list(
        dict.fromkeys(
            label
            for label, _group in _series_groups(
                frame,
                include_impact_in_label=include_impact_in_label,
                include_method_in_label=include_method_in_label,
            )
        )
    )
    return single_or_distinct_colors(labels)


def _single_panel_min_height(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
) -> float:
    labels = list(
        dict.fromkeys(
            label
            for label, _group in _series_groups(
                frame,
                include_impact_in_label=False,
                include_method_in_label=include_method_in_label,
            )
        )
    )
    return (
        MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN
        if include_method_in_label and len(labels) > 1
        else SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN
    )


def _violin_entries(
    frame: pd.DataFrame,
    *,
    include_impact_in_label: bool,
    include_method_in_label: bool,
    label_order: list[str] | None = None,
) -> list[tuple[str, np.ndarray, pd.Series]]:
    entries: list[tuple[str, np.ndarray, pd.Series]] = []
    for _index, row in frame.iterrows():
        values = np.asarray(row[VALUE_ARRAY_COLUMN], dtype=np.float64)
        values = values[~np.isnan(values)]
        series = pd.Series(row, copy=False)
        label = _row_label(
            series,
            include_impact_in_label=include_impact_in_label,
            include_method_in_label=include_method_in_label,
        )
        entries.append((label, values, series))
    return _sort_violin_entries(entries, label_order=label_order)


def _sort_violin_entries(
    entries: list[tuple[str, np.ndarray, pd.Series]],
    *,
    label_order: list[str] | None,
) -> list[tuple[str, np.ndarray, pd.Series]]:
    if label_order is None:
        return sorted(entries, key=lambda item: (-float(np.nanmean(item[1])), item[0]))
    ranks = {label: index for index, label in enumerate(label_order)}
    return sorted(entries, key=lambda item: (ranks.get(item[0], len(ranks)), item[0]))


def _row_label(
    row: pd.Series,
    *,
    include_impact_in_label: bool,
    include_method_in_label: bool,
) -> str:
    parts = [str(row.get("__method", "aSoCC")).strip()] if include_method_in_label else []
    impact = row.get("impact")
    if (
        include_impact_in_label
        and impact is not None
        and not pd.isna(impact)
        and str(impact).strip()
    ):
        parts.append(str(impact).strip())
    return " | ".join(part for part in parts if part)


def _include_impact_in_legend(frame: pd.DataFrame, *, include_impact_in_label: bool) -> bool:
    """Return whether impact adds information to legend entries for this scope."""
    if not include_impact_in_label:
        return False
    return len(visible_values(frame, "impact")) > 1


def _render_trajectory_band_legend_box(*, fig, axis) -> None:
    render_trajectory_band_legend_below(
        fig,
        color=_trajectory_band_legend_color(axis),
        ncol=4,
    )


def _trajectory_band_legend_color(axis) -> Any:
    return axis.lines[0].get_color()


def _ordered_impacts(frame: pd.DataFrame) -> list[str]:
    impacts = visible_values(frame, "impact")
    if len(impacts) <= 1:
        return impacts
    return ordered_impact_panels(
        lcia_method=visible_values(frame, "lcia_method")[0], impacts=impacts
    )


def _row_average_score(row: pd.Series) -> float:
    return cast(
        float,
        row_average_score(
            row,
            value_array_column=VALUE_ARRAY_COLUMN,
            scalar_columns=("mean", "value"),
        ),
    )


def _impact_panel_title(frame: pd.DataFrame, *, impact: str) -> str:
    title = resolve_frame_impact_title(frame)
    return str(title).strip() if title is not None else str(impact).strip()


def _visible_band_top(frame: pd.DataFrame) -> float | None:
    values = np.concatenate(
        [
            pd.Series(pd.to_numeric(frame[column], errors="raise")).to_numpy(dtype=float)
            for column in ("mean", "median", "p25", "p75", "p5", "p95")
        ]
    )
    return float(np.max(values[np.isfinite(values)])) * 100.0


def _visible_violin_top(frame: pd.DataFrame) -> float | None:
    values = np.concatenate(
        [np.asarray(values, dtype=np.float64) for values in frame[VALUE_ARRAY_COLUMN].tolist()]
    )
    return float(np.max(values[np.isfinite(values)])) * 100.0


def _format_year_axis(axis, visible_years: list[int]) -> None:
    years = sorted({int(year) for year in visible_years})
    format_integer_year_axis(axis, years=years)


def _format_axes(axis, *, data_top: float | None = None) -> None:
    axis.set_ylabel("aSoCC (%)")
    apply_asocc_y_axis_policy(axis, data_top=data_top)


def _save(*, fig, output_stem: Path, output_format: str, dpi: int) -> list[Path]:
    path = output_file_path(base_path=output_stem, output_format=output_format)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight", format=output_format)
    plt.close(fig)
    return [path]

"""Rendering policy for aCC uncertainty interval and mean line products."""

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from pyaesa.acc.deterministic.figures.product_renderers import (
    plot_scope as plot_deterministic_scope,
    prepare_plot_rows,
)
from pyaesa.acc.figures.common import (
    BUDGET_VALUES_COLUMN,
    MEAN_LINE_NOTE,
    MAX_CC_BOUND,
    MIN_CC_BOUND,
    PAIR_COUNT_COLUMN,
    VALUE_ARRAY_COLUMN,
    apply_acc_axis_policy,
    cc_bound_layer_key,
    cumulative_budget_unit_label,
    format_year_axis,
    has_static_min_max_bounds,
    impact_panel_title,
    is_dynamic_scope,
    ordered_impacts,
    panel_unit_label,
    save_figure,
)
from pyaesa.shared.figures.colors import (
    DEFAULT_SINGLE_SERIES_COLOR,
    MULTI_METHOD_LINE_ALPHA,
    single_or_distinct_colors,
)
from pyaesa.shared.figures.deterministic_legends import (
    bind_deterministic_legend_group,
    render_grouped_deterministic_legend_below,
)
from pyaesa.shared.figures.deterministic_legends_methods import legend_group_from_row
from pyaesa.shared.figures.asocc_transition_policy import (
    ASOCC_TRANSITION_SERIES_EXCLUDED_COLUMNS,
    asocc_transition_year,
)
from pyaesa.shared.figures.layout import (
    DOUBLE_COLUMN_TITLE_TOP,
    MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN,
    MULTI_YEAR_TWO_PANEL_FIGURE_SIZE,
    SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN,
    TRANSITION_PANEL_TITLE_PAD,
    bottom_panel_indices,
    hide_unused_axes,
    multi_impact_panel_figure_size,
    single_impact_figure_size,
    show_panel_x_labels,
)
from pyaesa.shared.figures.figure_footer import (
    render_two_panel_legends_below,
    set_footer_min_plot_height,
)
from pyaesa.shared.figures.multi_year_transitions import (
    TransitionMarker,
    render_transition_markers,
)
from pyaesa.shared.figures.trajectory_bands import (
    SUMMARY_COLUMNS,
    BAND_ALPHA_INNER,
    BAND_ALPHA_OUTER,
    render_trajectory_band,
    render_trajectory_band_legend_below,
    trajectory_band_legend_handles,
)
from pyaesa.shared.figures.dynamic_ar6 import (
    MODEL_SCENARIO_SAMPLING_METHOD_COLUMN,
    model_scenario_sampling_method,
)
from pyaesa.shared.figures.titles import render_figure_title, title_layout_top
from pyaesa.shared.figures.value_order import (
    finite_average,
    order_labels_by_average_within_group_rank,
)
from pyaesa.shared.figures.violin_summary import (
    render_violin_summaries,
    violin_summary_footer_extra_height,
    violin_summary_legend_handler_map,
    violin_summary_legend_handles,
    violin_summary_legend_kwargs,
)

_DEFAULT_COLOR = DEFAULT_SINGLE_SERIES_COLOR
_NEUTRAL_BAND_LEGEND_COLOR = "#555555"
_STATIC_BOUND_COLORS = {
    MIN_CC_BOUND: "#54A24B",
    MAX_CC_BOUND: "#E68613",
}
_LINE_ALPHA = 0.82
_PANEL_TITLE_PAD = 5
_TWO_COLUMN_PANEL_HSPACE = 0.32
_TWO_COLUMN_TRANSITION_HSPACE = 0.42
_STATIC_BOUND_BAND_ALPHA_OUTER = 0.10
_STATIC_BOUND_BAND_ALPHA_INNER = 0.32
_DYNAMIC_TITLE_CLEARANCE_PAD_EXTRA = 10


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
    """Render one multi year aCC uncertainty interval scope."""
    del group_legend, include_impact_in_label
    impacts = ordered_impacts(frame)
    if len(impacts) > 1:
        return _plot_impact_panel_band_scope(
            frame=frame,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            include_method_in_label=include_method_in_label,
        )
    if is_dynamic_scope(frame) and BUDGET_VALUES_COLUMN in frame.columns:
        return _plot_dynamic_band_scope(
            frame=frame,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            include_method_in_label=include_method_in_label,
        )
    fig, axis = plt.subplots(figsize=single_impact_figure_size(single_year=False))
    set_footer_min_plot_height(
        fig,
        height_in=_single_panel_min_height(
            frame=frame,
            include_method_in_label=include_method_in_label,
        ),
    )
    values, years, markers = _render_band_axis(
        axis=axis,
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    _format_band_axis(axis=axis, frame=frame, values=values, years=years, show_x_labels=True)
    axis.set_title(title, fontweight="bold", pad=26 if markers else 6)
    render_transition_markers(axis, markers=markers)
    render_trajectory_band_legend_below(
        fig,
        color=_trajectory_legend_color(axis=axis, frame=frame),
        prefix_handles=_static_bound_band_handles(frame),
        pair_count=_dynamic_pair_count(frame),
        sampling_method=(
            model_scenario_sampling_method(frame) if PAIR_COUNT_COLUMN in frame.columns else None
        ),
        ncol=_trajectory_legend_columns(frame),
    )
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


def _plot_dynamic_band_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    include_method_in_label: bool,
) -> list[Path]:
    fig, (axis, budget_axis) = plt.subplots(
        ncols=2,
        figsize=MULTI_YEAR_TWO_PANEL_FIGURE_SIZE,
        gridspec_kw={"width_ratios": [3.0, 1.22], "wspace": 0.28},
    )
    set_footer_min_plot_height(
        fig,
        height_in=_dynamic_two_panel_min_height(
            frame=frame,
            include_method_in_label=include_method_in_label,
        ),
    )
    render_figure_title(fig, title)
    values, years, markers = _render_band_axis(
        axis=axis,
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    _format_band_axis(axis=axis, frame=frame, values=values, years=years, show_x_labels=True)
    title_pad = TRANSITION_PANEL_TITLE_PAD if markers else 6
    fig.subplots_adjust(
        top=title_layout_top(
            fig,
            title,
            default_top=0.89,
            panel_title_pad=title_pad + _DYNAMIC_TITLE_CLEARANCE_PAD_EXTRA,
        )
    )
    axis.set_title("Pathways", fontweight="bold", pad=title_pad)
    render_transition_markers(axis, markers=markers)
    _render_dynamic_budget_axis(
        axis=budget_axis,
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    budget_axis.set_title("Cumulative budget", fontweight="bold", pad=title_pad)
    pathway_handles = trajectory_band_legend_handles(
        color=_trajectory_legend_color(axis=axis, frame=frame),
        pair_count=_dynamic_pair_count(frame),
        sampling_method=model_scenario_sampling_method(frame),
    )
    render_two_panel_legends_below(
        fig,
        left_axis=axis,
        right_axis=budget_axis,
        left_handles=pathway_handles,
        right_handles=violin_summary_legend_handles(),
        left_ncol=len(pathway_handles),
        right_ncol=1,
        right_handler_map=violin_summary_legend_handler_map(),
        right_legend_kwargs=violin_summary_legend_kwargs(),
        extra_height_in=violin_summary_footer_extra_height(),
    )
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


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
    """Render a method comparison scope with deterministic aCC visuals."""
    del include_impact_in_label
    if is_dynamic_scope(frame) and BUDGET_VALUES_COLUMN in frame.columns:
        return _plot_dynamic_mean_scope(
            frame=frame,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
        )
    plot_frame = frame.copy()
    plot_frame["value"] = pd.to_numeric(plot_frame["mean"], errors="raise")
    return plot_deterministic_scope(
        frame=prepare_plot_rows(plot_frame),
        requested_years=requested_years,
        output_stem=output_stem,
        title=title,
        dpi=dpi,
        output_format=output_format,
        group_legend=group_legend,
        include_method_in_label=include_method_in_label,
        figure_note=MEAN_LINE_NOTE,
    )


def _plot_dynamic_mean_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_method_in_label: bool,
) -> list[Path]:
    fig, (axis, budget_axis) = plt.subplots(
        ncols=2,
        figsize=MULTI_YEAR_TWO_PANEL_FIGURE_SIZE,
        gridspec_kw={
            "width_ratios": [1.0, 1.0] if group_legend else [3.0, 1.22],
            "wspace": 0.28,
        },
    )
    set_footer_min_plot_height(
        fig,
        height_in=_dynamic_two_panel_min_height(
            frame=frame,
            include_method_in_label=include_method_in_label,
        ),
    )
    render_figure_title(fig, title)
    values, years, markers = _render_mean_line_axis(
        axis=axis,
        frame=frame,
        group_legend=group_legend,
        include_method_in_label=include_method_in_label,
    )
    _format_band_axis(axis=axis, frame=frame, values=values, years=years, show_x_labels=True)
    title_pad = TRANSITION_PANEL_TITLE_PAD if markers else 6
    fig.subplots_adjust(
        top=title_layout_top(
            fig,
            title,
            default_top=0.89,
            panel_title_pad=title_pad + _DYNAMIC_TITLE_CLEARANCE_PAD_EXTRA,
        )
    )
    axis.set_title("Pathways", fontweight="bold", pad=title_pad)
    render_transition_markers(axis, markers=markers)
    _render_dynamic_budget_axis(
        axis=budget_axis,
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    budget_axis.set_title("Cumulative budget", fontweight="bold", pad=title_pad)
    _render_dynamic_mean_legend(axis=axis, budget_axis=budget_axis, frame=frame)
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


def _plot_impact_panel_band_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    include_method_in_label: bool,
) -> list[Path]:
    impacts = ordered_impacts(frame)
    label_order = _series_label_order(
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    color_map = _band_color_map(label_order)
    ncols = 2
    nrows = (len(impacts) + 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=multi_impact_panel_figure_size(nrows=nrows),
        squeeze=False,
    )
    bottom_indices = bottom_panel_indices(panel_count=len(impacts), ncols=ncols)
    has_transitions = False
    for index, impact in enumerate(impacts):
        axis = axes[index // ncols, index % ncols]
        panel = frame.loc[frame["impact"].astype(str).eq(str(impact))].copy()
        values, years, markers = _render_band_axis(
            axis=axis,
            frame=panel,
            include_method_in_label=include_method_in_label,
            label_order=label_order,
            color_map=color_map,
        )
        _format_band_axis(
            axis=axis,
            frame=panel,
            values=values,
            years=years,
            show_x_labels=show_panel_x_labels(
                panel_index=index,
                bottom_indices=bottom_indices,
            ),
        )
        if markers:
            has_transitions = True
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(
            impact_panel_title(panel, impact=str(impact)),
            loc="left",
            pad=TRANSITION_PANEL_TITLE_PAD if markers else _PANEL_TITLE_PAD,
        )
        render_transition_markers(axis, markers=markers)
    hide_unused_axes(axes=axes, used=len(impacts))
    render_figure_title(fig, title)
    top = title_layout_top(
        fig,
        title,
        default_top=DOUBLE_COLUMN_TITLE_TOP,
        panel_title_pad=TRANSITION_PANEL_TITLE_PAD if has_transitions else _PANEL_TITLE_PAD,
    )
    fig.subplots_adjust(
        hspace=_TWO_COLUMN_TRANSITION_HSPACE if has_transitions else _TWO_COLUMN_PANEL_HSPACE,
        wspace=0.16,
        top=top,
    )
    render_trajectory_band_legend_below(
        fig,
        color=_trajectory_legend_color(axis=None, frame=frame),
        prefix_handles=_static_bound_band_handles(frame),
        pair_count=_dynamic_pair_count(frame),
        sampling_method=(
            model_scenario_sampling_method(frame) if PAIR_COUNT_COLUMN in frame.columns else None
        ),
        ncol=_trajectory_legend_columns(frame),
    )
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


def _render_band_axis(
    *,
    axis,
    frame: pd.DataFrame,
    include_method_in_label: bool,
    label_order: list[str] | None = None,
    color_map: dict[str, str] | None = None,
):
    labels = label_order or _series_labels(
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    colors = color_map if color_map is not None else _band_color_map(labels)
    years: list[int] = []
    values: list[np.ndarray] = []
    markers: dict[int, TransitionMarker] = {}
    for label, group in _ordered_series_groups(
        frame=frame,
        include_method_in_label=include_method_in_label,
        label_order=label_order,
    ):
        ordered = group.sort_values("year", kind="stable")
        bound = _group_bound(ordered)
        combined_bounds = has_static_min_max_bounds(frame)
        color = _static_bound_color(bound) if combined_bounds else colors[label]
        year_values = pd.Series(pd.to_numeric(ordered["year"], errors="raise")).astype(int)
        run_values = render_trajectory_band(
            axis,
            years=year_values.to_numpy(dtype=int),
            summaries={column: ordered[column] for column in SUMMARY_COLUMNS},
            color=color,
            line_alpha=MULTI_METHOD_LINE_ALPHA if include_method_in_label else _LINE_ALPHA,
            outer_alpha=_STATIC_BOUND_BAND_ALPHA_OUTER if combined_bounds else BAND_ALPHA_OUTER,
            inner_alpha=_STATIC_BOUND_BAND_ALPHA_INNER if combined_bounds else BAND_ALPHA_INNER,
        )
        years.extend(int(year) for year in year_values.tolist())
        values.extend(run_values[column] for column in SUMMARY_COLUMNS)
        for marker in filter(lambda value: value is not None, [asocc_transition_year(ordered)]):
            marker_year = int(cast(int, marker))
            markers[marker_year] = TransitionMarker(
                year=marker_year,
                label="retrospective/prospective transition",
                color="#7d7d7d",
            )
    return np.concatenate(values) if values else np.empty(0), years, list(markers.values())


def _render_mean_line_axis(
    *,
    axis,
    frame: pd.DataFrame,
    group_legend: bool,
    include_method_in_label: bool,
) -> tuple[np.ndarray, list[int], list[TransitionMarker]]:
    labels = _series_labels(frame=frame, include_method_in_label=include_method_in_label)
    colors = _band_color_map(labels)
    years: list[int] = []
    values: list[float] = []
    markers: dict[int, TransitionMarker] = {}
    visible: set[str] = set()
    for label, group in _ordered_series_groups(
        frame=frame,
        include_method_in_label=include_method_in_label,
        label_order=None,
    ):
        ordered = group.sort_values("year", kind="stable")
        row = pd.Series(ordered.iloc[0], copy=False)
        year_values = pd.Series(pd.to_numeric(ordered["year"], errors="raise")).astype(int)
        mean = pd.Series(pd.to_numeric(ordered["mean"], errors="raise")).astype(float)
        visible_label = label if label and label not in visible else "_nolegend_"
        line = axis.plot(
            year_values.to_numpy(dtype=int),
            mean.to_numpy(dtype=float),
            linewidth=1.7,
            color=colors[label],
            alpha=MULTI_METHOD_LINE_ALPHA if include_method_in_label else _LINE_ALPHA,
            label=visible_label,
        )[0]
        bind_deterministic_legend_group(line, legend_group_from_row(row))
        visible.add(label)
        years.extend(int(year) for year in year_values.tolist())
        values.extend(float(value) for value in mean.tolist())
        for marker in filter(lambda value: value is not None, [asocc_transition_year(ordered)]):
            marker_year = int(cast(int, marker))
            markers[marker_year] = TransitionMarker(
                year=marker_year,
                label="retrospective/prospective transition",
                color="#7d7d7d",
            )
    return np.asarray(values, dtype=float), years, list(markers.values())


def _format_band_axis(
    *,
    axis,
    frame: pd.DataFrame,
    values: np.ndarray,
    years: list[int],
    show_x_labels: bool,
) -> None:
    year_values = sorted({int(year) for year in years})
    axis.set_xlim(min(year_values) - 0.5, max(year_values) + 0.5)
    format_year_axis(axis, years=year_values, show_labels=show_x_labels)
    apply_acc_axis_policy(axis, values=values, context="aCC uncertainty interval figure")
    axis.set_ylabel(panel_unit_label(frame))
    axis.grid(alpha=0.25)


def _render_dynamic_budget_axis(
    *,
    axis,
    frame: pd.DataFrame,
    include_method_in_label: bool,
) -> np.ndarray:
    entries = _dynamic_budget_entries(
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    labels = list(dict.fromkeys(label for label, _values, _color in entries))
    positions = np.asarray(
        [labels.index(label) for label, _values, _color in entries],
        dtype=float,
    )
    colors = [color for _label, _values, color in entries]
    values = [np.asarray(payload, dtype=np.float64) for _label, payload, _color in entries]
    render_violin_summaries(
        axis,
        values=values,
        positions=positions,
        colors=colors,
        width=0.34,
    )
    axis.set_xlim(-0.5, max(0.5, float(len(labels)) - 0.5))
    ticks = np.arange(len(labels), dtype=float)
    if include_method_in_label and any(str(label).strip() for label in labels):
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels, rotation=45, ha="right")
    else:
        axis.set_xticks(ticks)
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    numeric = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    apply_acc_axis_policy(axis, values=numeric, context="aCC uncertainty dynamic budget panel")
    axis.set_ylabel(cumulative_budget_unit_label(frame))
    axis.set_xlabel("")
    axis.grid(alpha=0.25, axis="y")
    return numeric


def _render_dynamic_mean_legend(*, axis: Any, budget_axis: Any, frame: pd.DataFrame) -> None:
    violin_handle = violin_summary_legend_handles()[0]
    bind_deterministic_legend_group(violin_handle, "Uncertainty")
    budget_axis.add_line(violin_handle)
    pair_count = int(pd.Series(pd.to_numeric(frame[PAIR_COUNT_COLUMN], errors="raise")).max())
    pair_label = f"{pair_count} AR6 CC model-scenario pairs"
    sampling_method = model_scenario_sampling_method(frame)
    if sampling_method is not None:
        pair_label = f"{pair_label}; sampling method: {sampling_method}"
    render_grouped_deterministic_legend_below(
        axis,
        legend_note=f"{MEAN_LINE_NOTE}\n{pair_label}.",
        handler_map=violin_summary_legend_handler_map(),
        legend_kwargs=violin_summary_legend_kwargs(),
        legend_kwargs_group_title="Uncertainty",
    )


def _band_color_map(labels: list[str]) -> dict[str, str]:
    return single_or_distinct_colors(labels)


def _dynamic_two_panel_min_height(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
) -> float:
    labels = _series_labels(frame=frame, include_method_in_label=include_method_in_label)
    return (
        MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN
        if include_method_in_label and len(labels) > 1
        else SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN
    )


def _single_panel_min_height(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
) -> float:
    labels = _series_labels(frame=frame, include_method_in_label=include_method_in_label)
    return (
        MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN
        if include_method_in_label and len(labels) > 1
        else SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN
    )


def _dynamic_budget_entries(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
) -> list[tuple[str, np.ndarray, str]]:
    labels = _series_labels(frame=frame, include_method_in_label=include_method_in_label)
    colors = _band_color_map(labels)
    entries: list[tuple[str, np.ndarray, str]] = []
    for label, group in _ordered_series_groups(
        frame=frame,
        include_method_in_label=include_method_in_label,
        label_order=None,
    ):
        row = pd.Series(group.iloc[0], copy=False)
        values = np.asarray(row[BUDGET_VALUES_COLUMN], dtype=np.float64)
        color = colors[label]
        entries.append((label if include_method_in_label else "Study period", values, color))
    return entries


def _dynamic_pair_count(frame: pd.DataFrame) -> int | None:
    if PAIR_COUNT_COLUMN not in frame.columns:
        return None
    values = pd.Series(pd.to_numeric(frame[PAIR_COUNT_COLUMN], errors="raise")).astype(int)
    return int(values.max())


def _series_groups(*, frame: pd.DataFrame, include_method_in_label: bool):
    excluded = {
        *SUMMARY_COLUMNS,
        "std",
        "min",
        "max",
        "public_row_id",
        VALUE_ARRAY_COLUMN,
        BUDGET_VALUES_COLUMN,
        PAIR_COUNT_COLUMN,
        MODEL_SCENARIO_SAMPLING_METHOD_COLUMN,
        *ASOCC_TRANSITION_SERIES_EXCLUDED_COLUMNS,
    }
    group_columns = [
        column
        for column in frame.columns
        if column not in {*excluded, "year"} and not str(column).startswith("__figure")
    ]
    for _key, group in frame.groupby(group_columns, dropna=False, sort=True):
        row = pd.Series(group.iloc[0], copy=False)
        yield _row_label(row, include_method_in_label=include_method_in_label), group


def _series_labels(*, frame: pd.DataFrame, include_method_in_label: bool) -> list[str]:
    return list(
        dict.fromkeys(
            label
            for label, _group in _series_groups(
                frame=frame,
                include_method_in_label=include_method_in_label,
            )
        )
    )


def _ordered_series_groups(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
    label_order: list[str] | None,
) -> list[tuple[str, pd.DataFrame]]:
    groups = list(_series_groups(frame=frame, include_method_in_label=include_method_in_label))
    combined_bounds = has_static_min_max_bounds(frame)
    if label_order is None:
        if combined_bounds:
            label_ranks = {
                label: index
                for index, label in enumerate(dict.fromkeys(label for label, _ in groups))
            }
            return sorted(
                groups,
                key=lambda item: (
                    label_ranks[item[0]],
                    cc_bound_layer_key(_group_bound(item[1])),
                ),
            )
        return groups
    ranks = {label: index for index, label in enumerate(label_order)}
    return sorted(
        groups,
        key=lambda item: (
            ranks.get(item[0], len(ranks)),
            item[0],
            cc_bound_layer_key(_group_bound(item[1])) if combined_bounds else 0,
        ),
    )


def _series_label_order(*, frame: pd.DataFrame, include_method_in_label: bool) -> list[str]:
    values: list[tuple[str, str, float | None]] = []
    for label, group in _series_groups(
        frame=frame,
        include_method_in_label=include_method_in_label,
    ):
        score = _series_average_score(group)
        impact = str(pd.Series(group.iloc[0], copy=False).get("impact", "")).strip()
        values.append((impact, label, score))
    return order_labels_by_average_within_group_rank(values)


def _series_average_score(group: pd.DataFrame) -> float | None:
    values = pd.Series(pd.to_numeric(group["mean"], errors="raise")).to_numpy(dtype=float)
    return finite_average(values.tolist())


def _row_label(row: pd.Series, *, include_method_in_label: bool) -> str:
    return str(row.get("__method", "")).strip() if include_method_in_label else "aCC"


def _trajectory_legend_color(*, axis, frame: pd.DataFrame) -> Any:
    if has_static_min_max_bounds(frame):
        return _NEUTRAL_BAND_LEGEND_COLOR
    if axis is None:
        return _DEFAULT_COLOR
    return axis.lines[0].get_color() if axis.lines else _DEFAULT_COLOR


def _group_bound(group: pd.DataFrame) -> str:
    return str(pd.Series(group.iloc[0], copy=False).get("cc_bound", "")).strip()


def _trajectory_legend_columns(frame: pd.DataFrame) -> int | None:
    if has_static_min_max_bounds(frame):
        return 3
    return None


def _static_bound_color(bound: str) -> str:
    return _STATIC_BOUND_COLORS.get(bound, _DEFAULT_COLOR)


def _static_bound_band_handles(frame: pd.DataFrame) -> list[Patch]:
    if not has_static_min_max_bounds(frame):
        return []
    return [
        Patch(
            facecolor=_static_bound_color(MIN_CC_BOUND),
            alpha=_STATIC_BOUND_BAND_ALPHA_INNER,
            label="min CC",
        ),
        Patch(
            facecolor=_static_bound_color(MAX_CC_BOUND),
            alpha=_STATIC_BOUND_BAND_ALPHA_INNER,
            label="max CC",
        ),
    ]

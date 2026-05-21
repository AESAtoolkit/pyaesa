"""aCC uncertainty violin figure rendering."""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from pyaesa.acc.figures.common import (
    MAX_CC_BOUND,
    MIN_CC_BOUND,
    VALUE_ARRAY_COLUMN,
    apply_acc_axis_policy,
    cc_bound_layer_key,
    cc_bound_order_key,
    has_static_min_max_bounds,
    impact_panel_title,
    ordered_impacts,
    panel_unit_label,
    save_figure,
    visible_values,
)
from pyaesa.shared.figures.colors import single_or_distinct_colors
from pyaesa.shared.figures.deterministic_legends import (
    bind_deterministic_legend_group,
    render_grouped_deterministic_legend_below,
)
from pyaesa.shared.figures.deterministic_legends_methods import legend_group_from_row
from pyaesa.shared.figures.layout import (
    DOUBLE_COLUMN_TITLE_TOP,
    SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN,
    bottom_panel_indices,
    format_single_year_category_axis,
    hide_unused_axes,
    multi_impact_panel_figure_size,
    single_impact_figure_size,
    show_panel_x_labels,
)
from pyaesa.shared.figures.figure_footer import set_footer_min_plot_height
from pyaesa.shared.figures.titles import render_figure_title, title_layout_top
from pyaesa.shared.figures.value_order import order_labels_by_average_within_group_rank
from pyaesa.shared.figures.violin_summary import (
    VIOLIN_SUMMARY_LEGEND_GROUP_TITLE,
    ViolinSummaryMode,
    render_violin_summaries,
    render_violin_summary_legend_below,
    violin_summary_footer_extra_height,
    violin_summary_legend_handler_map,
    violin_summary_legend_handles,
    violin_summary_legend_kwargs,
)

_TWO_COLUMN_PANEL_HSPACE = 0.32
_STATIC_BOUND_OFFSET = 0.0
_STATIC_BOUND_VIOLIN_WIDTH = 0.58
_STATIC_BOUND_LEGEND_TITLE = "Static CC bounds"
_STATIC_BOUND_COLORS = {
    MIN_CC_BOUND: "#54A24B",
    MAX_CC_BOUND: "#E68613",
}


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
    """Render one exact single year aCC uncertainty violin scope."""
    del include_impact_in_label
    impacts = ordered_impacts(frame)
    if len(impacts) > 1:
        return _plot_impact_panel_violin_scope(
            frame=frame,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
        )
    rows = _violin_entries(frame, include_method_in_label=include_method_in_label)
    fig, axis = plt.subplots(figsize=single_impact_figure_size(single_year=True))
    set_footer_min_plot_height(fig, height_in=SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN)
    values = _render_violin_payload(
        axis=axis,
        rows=rows,
        colors=_row_colors(rows, use_bound_colors=not group_legend),
        group_legend=group_legend,
        visible_labels=set(),
        show_x_labels=_show_violin_labels(rows),
        summary="full",
    )
    _format_violin_axis(axis=axis, frame=frame, values=values)
    axis.set_title(title, fontweight="bold", pad=6)
    if group_legend:
        _attach_static_bound_legend_handles(axis, rows=rows, use_bound_colors=False)
        _attach_violin_legend_handles(axis, summary="full")
        render_grouped_deterministic_legend_below(
            axis,
            handler_map=violin_summary_legend_handler_map(),
            legend_kwargs=violin_summary_legend_kwargs(),
            legend_kwargs_group_title=VIOLIN_SUMMARY_LEGEND_GROUP_TITLE,
            legend_extra_height_in=violin_summary_footer_extra_height(),
            hidden_group_titles={VIOLIN_SUMMARY_LEGEND_GROUP_TITLE},
        )
    else:
        render_violin_summary_legend_below(
            fig,
            summary="full",
            extra_entries=_static_bound_legend_handles(
                rows=rows,
                use_bound_colors=True,
            ),
        )
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


def _plot_impact_panel_violin_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_method_in_label: bool,
) -> list[Path]:
    impacts = ordered_impacts(frame)
    ncols = 2
    nrows = (len(impacts) + 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=multi_impact_panel_figure_size(nrows=nrows, compact=not group_legend),
        squeeze=False,
    )
    bottom_indices = bottom_panel_indices(panel_count=len(impacts), ncols=ncols)
    label_order = _violin_label_order(frame, include_method_in_label=include_method_in_label)
    color_map = single_or_distinct_colors(label_order)
    visible_labels: set[str] = set()
    summary: ViolinSummaryMode = "mean_median" if group_legend else "full"
    for index, impact in enumerate(impacts):
        axis = axes[index // ncols, index % ncols]
        panel = frame.loc[frame["impact"].astype(str).eq(str(impact))].copy()
        rows = _violin_entries(
            panel,
            include_method_in_label=include_method_in_label,
            label_order=label_order,
        )
        values = _render_violin_payload(
            axis=axis,
            rows=rows,
            colors=[
                _static_bound_color(
                    color_map[label],
                    _row_bound(row),
                    use_bound_colors=not group_legend,
                )
                for label, _values, row in rows
            ],
            group_legend=group_legend,
            visible_labels=visible_labels,
            show_x_labels=show_panel_x_labels(
                panel_index=index,
                bottom_indices=bottom_indices,
            )
            and _show_violin_labels(rows),
            summary=summary,
        )
        _format_violin_axis(axis=axis, frame=panel, values=values)
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(impact_panel_title(panel, impact=str(impact)), loc="left", pad=5)
    hide_unused_axes(axes=axes, used=len(impacts))
    render_figure_title(fig, title)
    fig.subplots_adjust(
        hspace=_TWO_COLUMN_PANEL_HSPACE,
        wspace=0.16,
        top=title_layout_top(fig, title, default_top=DOUBLE_COLUMN_TITLE_TOP, panel_title_pad=5),
    )
    if group_legend:
        _attach_static_bound_legend_handles(
            axes[0, 0],
            rows=_violin_entries(frame, include_method_in_label=include_method_in_label),
            use_bound_colors=False,
        )
        _attach_violin_legend_handles(axes[0, 0], summary=summary)
        render_grouped_deterministic_legend_below(
            axes[0, 0],
            handler_map=violin_summary_legend_handler_map(),
            legend_kwargs=violin_summary_legend_kwargs(),
            legend_kwargs_group_title=VIOLIN_SUMMARY_LEGEND_GROUP_TITLE,
            legend_extra_height_in=violin_summary_footer_extra_height(),
            hidden_group_titles={VIOLIN_SUMMARY_LEGEND_GROUP_TITLE},
        )
    else:
        render_violin_summary_legend_below(
            fig,
            summary=summary,
            extra_entries=_static_bound_legend_handles(
                rows=_violin_entries(frame, include_method_in_label=include_method_in_label),
                use_bound_colors=True,
            ),
        )
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


def _render_violin_payload(
    *,
    axis,
    rows: list[tuple[str, np.ndarray, pd.Series]],
    colors: list[str],
    group_legend: bool,
    visible_labels: set[str],
    show_x_labels: bool,
    summary: ViolinSummaryMode,
) -> np.ndarray:
    if _rows_have_static_min_max_bounds(rows):
        return _render_static_min_max_violin_payload(
            axis=axis,
            rows=rows,
            colors=colors,
            group_legend=group_legend,
            visible_labels=visible_labels,
            show_x_labels=show_x_labels,
            summary=summary,
        )
    values = [entry[1] for entry in rows]
    positions = np.arange(1, len(rows) + 1, dtype=float)
    render_violin_summaries(
        axis,
        values=values,
        positions=positions,
        colors=colors,
        summary=summary,
    )
    for index, (label, _numeric, row) in enumerate(rows):
        visible_label = label if group_legend and label not in visible_labels else "_nolegend_"
        if visible_label == "_nolegend_":
            continue
        handle = Line2D(
            [],
            [],
            color=colors[index],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=colors[index],
            linestyle="",
            markersize=5.0,
            label=visible_label,
        )
        bind_deterministic_legend_group(handle, legend_group_from_row(row))
        axis.add_line(handle)
        visible_labels.add(label)
    labels = [label for label, _values, _row in rows]
    if show_x_labels:
        format_single_year_category_axis(axis, positions=positions.tolist(), labels=labels)
    else:
        axis.set_xticks(positions.tolist())
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    axis.set_xlim(float(positions.min()) - 0.7, float(positions.max()) + 0.7)
    return np.concatenate([np.asarray(value, dtype=np.float64) for value in values])


def _render_static_min_max_violin_payload(
    *,
    axis,
    rows: list[tuple[str, np.ndarray, pd.Series]],
    colors: list[str],
    group_legend: bool,
    visible_labels: set[str],
    show_x_labels: bool,
    summary: ViolinSummaryMode,
) -> np.ndarray:
    labels = list(dict.fromkeys(label for label, _values, _row in rows))
    label_positions = {label: float(index + 1) for index, label in enumerate(labels)}
    label_rank = {label: index for index, label in enumerate(labels)}
    render_items = sorted(
        zip(rows, colors, strict=True),
        key=lambda item: (
            label_rank[item[0][0]],
            cc_bound_layer_key(_row_bound(item[0][2])),
        ),
    )
    render_rows = [row for row, _color in render_items]
    render_colors = [color for _row, color in render_items]
    values = [entry[1] for entry in render_rows]
    positions = np.asarray(
        [
            label_positions[label] + _bound_offset(_row_bound(row))
            for label, _values, row in render_rows
        ],
        dtype=float,
    )
    render_violin_summaries(
        axis,
        values=values,
        positions=positions,
        colors=render_colors,
        width=_STATIC_BOUND_VIOLIN_WIDTH,
        summary=summary,
    )
    for index, (label, _numeric, row) in enumerate(render_rows):
        visible_label = label if group_legend and label not in visible_labels else "_nolegend_"
        if visible_label == "_nolegend_":
            continue
        handle = Line2D(
            [],
            [],
            color=_base_color(render_colors[index], _row_bound(row)),
            marker="o",
            markerfacecolor="white",
            markeredgecolor=_base_color(render_colors[index], _row_bound(row)),
            linestyle="",
            markersize=5.0,
            label=visible_label,
        )
        bind_deterministic_legend_group(handle, legend_group_from_row(row))
        axis.add_line(handle)
        visible_labels.add(label)
    if show_x_labels and any(str(label).strip() for label in labels):
        format_single_year_category_axis(
            axis,
            positions=[label_positions[label] for label in labels],
            labels=labels,
        )
    else:
        axis.set_xticks([label_positions[label] for label in labels])
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    axis.set_xlim(float(positions.min()) - 0.55, float(positions.max()) + 0.55)
    return np.concatenate([np.asarray(value, dtype=np.float64) for value in values])


def _format_violin_axis(*, axis, frame: pd.DataFrame, values: np.ndarray) -> None:
    apply_acc_axis_policy(axis, values=values, context="aCC uncertainty violin figure")
    axis.set_xlabel("")
    axis.set_ylabel(panel_unit_label(frame))
    axis.grid(alpha=0.25, axis="y")


def _violin_entries(
    frame: pd.DataFrame,
    *,
    include_method_in_label: bool,
    label_order: list[str] | None = None,
) -> list[tuple[str, np.ndarray, pd.Series]]:
    entries = []
    for _index, row in frame.iterrows():
        values = np.asarray(row[VALUE_ARRAY_COLUMN], dtype=np.float64)
        values = values[np.isfinite(values)]
        series = pd.Series(row, copy=False)
        label = _row_label(series, include_method_in_label=include_method_in_label)
        entries.append((label, values, series))
    if label_order is None:
        return sorted(
            entries,
            key=lambda item: (
                -float(np.mean(item[1])),
                item[0],
                cc_bound_order_key(_row_bound(item[2])),
            ),
        )
    ranks = {label: index for index, label in enumerate(label_order)}
    return sorted(
        entries,
        key=lambda item: (
            ranks.get(item[0], len(ranks)),
            item[0],
            cc_bound_order_key(_row_bound(item[2])),
        ),
    )


def _violin_label_order(frame: pd.DataFrame, *, include_method_in_label: bool) -> list[str]:
    rank_values: list[tuple[str, str, float]] = []
    for label, run_values, row in _violin_entries(
        frame,
        include_method_in_label=include_method_in_label,
    ):
        impact = str(row.get("impact", "")).strip()
        rank_values.append((impact, label, float(np.mean(run_values))))
    return order_labels_by_average_within_group_rank(rank_values)


def _row_label(row: pd.Series, *, include_method_in_label: bool) -> str:
    if include_method_in_label:
        return str(row.get("__method", "")).strip()
    impacts = visible_values(pd.DataFrame([row.to_dict()]), "impact")
    return impacts[0] if len(impacts) > 1 else ""


def _show_violin_labels(rows: list[tuple[str, np.ndarray, pd.Series]]) -> bool:
    return any(str(label).strip() for label, _values, _row in rows)


def _attach_violin_legend_handles(axis, *, summary: ViolinSummaryMode) -> None:
    for handle in violin_summary_legend_handles(summary=summary):
        bind_deterministic_legend_group(handle, VIOLIN_SUMMARY_LEGEND_GROUP_TITLE)
        axis.add_line(handle)


def _rows_have_static_min_max_bounds(rows: list[tuple[str, np.ndarray, pd.Series]]) -> bool:
    frame = pd.DataFrame([row.to_dict() for _label, _values, row in rows])
    return has_static_min_max_bounds(frame)


def _row_colors(
    rows: list[tuple[str, np.ndarray, pd.Series]],
    *,
    use_bound_colors: bool,
) -> list[str]:
    labels = list(dict.fromkeys(label for label, _values, _row in rows))
    color_map = single_or_distinct_colors(labels)
    return [
        _static_bound_color(
            color_map[label],
            _row_bound(row),
            use_bound_colors=use_bound_colors,
        )
        for label, _values, row in rows
    ]


def _row_bound(row: pd.Series) -> str:
    return str(row.get("cc_bound", "")).strip()


def _bound_offset(bound: str) -> float:
    return -_STATIC_BOUND_OFFSET if bound == MIN_CC_BOUND else _STATIC_BOUND_OFFSET


def _static_bound_color(color: str, bound: str, *, use_bound_colors: bool) -> str:
    if use_bound_colors:
        return _STATIC_BOUND_COLORS[bound]
    target, fraction = {
        MIN_CC_BOUND: ("#ffffff", 0.38),
        MAX_CC_BOUND: ("#000000", 0.22),
    }[bound]
    return _mix_color(color, target, fraction)


def _base_color(color: str, bound: str) -> str:
    target, fraction = {
        MIN_CC_BOUND: ("#000000", 0.18),
        MAX_CC_BOUND: ("#ffffff", 0.28),
    }[bound]
    return _mix_color(color, target, fraction)


def _mix_color(color: str, target: str, fraction: float) -> str:
    base = np.asarray(mcolors.to_rgb(color), dtype=np.float64)
    other = np.asarray(mcolors.to_rgb(target), dtype=np.float64)
    mixed = base * (1.0 - float(fraction)) + other * float(fraction)
    return str(mcolors.to_hex((float(mixed[0]), float(mixed[1]), float(mixed[2]))))


def _static_bound_legend_handles(
    *,
    rows: list[tuple[str, np.ndarray, pd.Series]],
    use_bound_colors: bool,
) -> list[Line2D]:
    if not _rows_have_static_min_max_bounds(rows):
        return []
    return [
        Line2D(
            [0],
            [0],
            color=_static_bound_color(
                "#4f83c2",
                MIN_CC_BOUND,
                use_bound_colors=use_bound_colors,
            ),
            marker="s",
            linestyle="",
            markersize=7.0,
            label="min CC",
        ),
        Line2D(
            [0],
            [0],
            color=_static_bound_color(
                "#4f83c2",
                MAX_CC_BOUND,
                use_bound_colors=use_bound_colors,
            ),
            marker="s",
            linestyle="",
            markersize=7.0,
            label="max CC",
        ),
    ]


def _attach_static_bound_legend_handles(
    axis,
    *,
    rows: list[tuple[str, np.ndarray, pd.Series]],
    use_bound_colors: bool,
) -> None:
    for handle in _static_bound_legend_handles(
        rows=rows,
        use_bound_colors=use_bound_colors,
    ):
        bind_deterministic_legend_group(handle, _STATIC_BOUND_LEGEND_TITLE)
        axis.add_line(handle)

"""ASR uncertainty violin figure rendering."""

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from pyaesa.asr.figures.axis import ASR_LOG_SCALE, ASRScaleMode
from pyaesa.asr.figures.common import (
    VALUE_ARRAY_COLUMN,
    apply_scaled_asr_axis_policy,
    asr_axis_limits,
    impact_panel_title,
    ordered_impacts,
    save_figure,
    visible_values,
)
from pyaesa.asr.figures.frequency import (
    FNT_FRACTION_COLUMN,
    fnt_box_legend_entry,
    render_fnt_box_groups,
    render_fnt_boxes,
)
from pyaesa.asr.figures.risk_guides import (
    ASR_RISK_LEGEND_GROUP_TITLE,
    ASR_RISK_BACKGROUND_VIOLIN_ALPHA,
    asr_risk_scale_footer_extra_height,
    render_asr_risk_scale_footer,
)
from pyaesa.shared.figures.colors import single_or_distinct_colors
from pyaesa.shared.figures.deterministic_legends import (
    bind_deterministic_legend_group,
    render_grouped_deterministic_legend_below,
)
from pyaesa.shared.figures.deterministic_legends_methods import legend_group_from_row
from pyaesa.shared.figures.figure_footer import set_footer_min_plot_height
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
_TWO_COLUMN_FREQUENCY_PANEL_HSPACE = 0.60
_TWO_COLUMN_FREQUENCY_PANEL_TOP = DOUBLE_COLUMN_TITLE_TOP
_TWO_COLUMN_PANEL_TOP = DOUBLE_COLUMN_TITLE_TOP
_FREQUENCY_TITLE_PAD = 18


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
    scale_mode: ASRScaleMode = ASR_LOG_SCALE,
) -> list[Path]:
    """Render one exact single year ASR uncertainty violin scope."""
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
            scale_mode=scale_mode,
        )
    rows = _violin_entries(frame, include_method_in_label=include_method_in_label)
    fig, axis = plt.subplots(figsize=single_impact_figure_size(single_year=True))
    set_footer_min_plot_height(fig, height_in=SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN)
    values = _render_violin_payload(
        axis=axis,
        rows=rows,
        colors=_colors_for_labels([label for label, _values, _row in rows]),
        group_legend=group_legend,
        visible_labels=set(),
        show_x_labels=_show_violin_labels(rows),
        summary="full",
    )
    _format_violin_axis(
        axis=axis,
        frame=frame,
        values=values,
        grouped_legend=group_legend,
        scale_mode=scale_mode,
    )
    _annotate_frequencies(axis=axis, rows=rows)
    axis.set_title(
        title,
        fontweight="bold",
        pad=_FREQUENCY_TITLE_PAD if _has_frequency_rows(rows) else 24,
    )
    risk_extra = asr_risk_scale_footer_extra_height()
    if group_legend:
        _attach_violin_legend_handles(axis, frame=frame, summary="full")
        render_grouped_deterministic_legend_below(
            axis,
            handler_map=violin_summary_legend_handler_map(),
            legend_kwargs=violin_summary_legend_kwargs(),
            legend_kwargs_group_title=VIOLIN_SUMMARY_LEGEND_GROUP_TITLE,
            legend_extra_height_in=violin_summary_footer_extra_height(),
            extra_height_in=risk_extra,
            hidden_group_titles={VIOLIN_SUMMARY_LEGEND_GROUP_TITLE},
        )
    else:
        render_violin_summary_legend_below(
            fig,
            summary="full",
            extra_entries=[_fnt_legend_handle(frame)],
            extra_height_in=risk_extra,
            frameon=False,
            title="Uncertainty",
        )
    render_asr_risk_scale_footer(fig, frame=frame)
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
    scale_mode: ASRScaleMode,
) -> list[Path]:
    impacts = ordered_impacts(frame)
    ncols = 2
    nrows = (len(impacts) + 1) // ncols
    has_frequency = FNT_FRACTION_COLUMN in frame.columns
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
    frequency_groups = []
    summary: ViolinSummaryMode = "mean_median" if group_legend else "full"
    common_limits = _common_violin_limits(frame, scale_mode=scale_mode)
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
            colors=[color_map[label] for label, _values, _row in rows],
            group_legend=group_legend,
            visible_labels=visible_labels,
            show_x_labels=show_panel_x_labels(
                panel_index=index,
                bottom_indices=bottom_indices,
            )
            and _show_violin_labels(rows),
            summary=summary,
        )
        _format_violin_axis(
            axis=axis,
            frame=panel,
            values=values,
            grouped_legend=group_legend,
            limits=common_limits,
            scale_mode=scale_mode,
        )
        frequency_groups.append((axis, _frequency_entries(rows=rows)))
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(
            impact_panel_title(panel, impact=str(impact)),
            loc="left",
            pad=_FREQUENCY_TITLE_PAD if _has_frequency_rows(rows) else 24,
        )
    render_fnt_box_groups(frequency_groups)
    hide_unused_axes(axes=axes, used=len(impacts))
    render_figure_title(fig, title)
    hspace = _TWO_COLUMN_FREQUENCY_PANEL_HSPACE if has_frequency else _TWO_COLUMN_PANEL_HSPACE
    default_top = _TWO_COLUMN_FREQUENCY_PANEL_TOP if has_frequency else _TWO_COLUMN_PANEL_TOP
    panel_title_pad = _FREQUENCY_TITLE_PAD if has_frequency else 24
    top = title_layout_top(
        fig,
        title,
        default_top=default_top,
        panel_title_pad=panel_title_pad,
    )
    fig.subplots_adjust(hspace=hspace, wspace=0.16, top=top)
    risk_extra = asr_risk_scale_footer_extra_height()
    if group_legend:
        _attach_violin_legend_handles(axes[0, 0], frame=frame, summary=summary)
        render_grouped_deterministic_legend_below(
            axes[0, 0],
            handler_map=violin_summary_legend_handler_map(),
            legend_kwargs=violin_summary_legend_kwargs(),
            legend_kwargs_group_title=VIOLIN_SUMMARY_LEGEND_GROUP_TITLE,
            legend_extra_height_in=violin_summary_footer_extra_height(),
            extra_height_in=risk_extra,
            hidden_group_titles={VIOLIN_SUMMARY_LEGEND_GROUP_TITLE},
        )
    else:
        render_violin_summary_legend_below(
            fig,
            summary=summary,
            extra_entries=[_fnt_legend_handle(frame)],
            extra_height_in=risk_extra,
            frameon=False,
            title="Uncertainty",
        )
    render_asr_risk_scale_footer(fig, frame=frame)
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
    values = [entry[1] for entry in rows]
    positions = np.arange(1, len(rows) + 1, dtype=float)
    render_violin_summaries(
        axis,
        values=values,
        positions=positions,
        colors=colors,
        summary=summary,
        alpha=ASR_RISK_BACKGROUND_VIOLIN_ALPHA,
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


def _colors_for_labels(labels: list[str]) -> list[str]:
    color_map = single_or_distinct_colors(labels)
    return [color_map[label] for label in labels]


def _format_violin_axis(
    *,
    axis,
    frame: pd.DataFrame,
    values: np.ndarray,
    grouped_legend: bool,
    limits: tuple[float, float] | None = None,
    scale_mode: ASRScaleMode = ASR_LOG_SCALE,
) -> None:
    apply_scaled_asr_axis_policy(
        axis,
        values=values,
        frame=frame,
        scale_mode=scale_mode,
        grouped_legend=grouped_legend,
        limits=limits,
    )
    axis.grid(alpha=0.25, axis="y")


def _common_violin_limits(frame: pd.DataFrame, *, scale_mode: ASRScaleMode) -> tuple[float, float]:
    values = [
        np.asarray(payload, dtype=np.float64) for payload in frame[VALUE_ARRAY_COLUMN].tolist()
    ]
    numeric_values = np.concatenate(values)
    return asr_axis_limits(values=numeric_values, frame=frame, scale_mode=scale_mode)


def _violin_entries(
    frame: pd.DataFrame,
    *,
    include_method_in_label: bool,
    label_order: list[str] | None = None,
) -> list[tuple[str, np.ndarray, pd.Series]]:
    entries = []
    for _index, row in frame.iterrows():
        values = np.asarray(row[VALUE_ARRAY_COLUMN], dtype=np.float64)
        series = pd.Series(row, copy=False)
        label = _row_label(series, include_method_in_label=include_method_in_label)
        entries.append((label, values, series))
    if label_order is None:
        return sorted(entries, key=lambda item: (-float(np.mean(item[1])), item[0]))
    ranks = {label: index for index, label in enumerate(label_order)}
    return sorted(entries, key=lambda item: (ranks.get(item[0], len(ranks)), item[0]))


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


def _attach_violin_legend_handles(
    axis,
    *,
    frame: pd.DataFrame,
    summary: ViolinSummaryMode,
) -> None:
    for handle in violin_summary_legend_handles(summary=summary):
        bind_deterministic_legend_group(handle, VIOLIN_SUMMARY_LEGEND_GROUP_TITLE)
        axis.add_line(handle)
    fnt_handle, _label = _fnt_legend_entry(frame)
    bind_deterministic_legend_group(fnt_handle, ASR_RISK_LEGEND_GROUP_TITLE)
    axis.add_line(fnt_handle)


def _fnt_legend_entry(frame: pd.DataFrame) -> tuple[Line2D, str]:
    cc_source = visible_values(frame, "lcia_method")[0]
    handle, label = fnt_box_legend_entry(cc_source=cc_source)
    return handle, label


def _fnt_legend_handle(frame: pd.DataFrame) -> Line2D:
    handle, _label = _fnt_legend_entry(frame)
    return handle


def _annotate_frequencies(
    *,
    axis,
    rows: list[tuple[str, np.ndarray, pd.Series]],
) -> None:
    render_fnt_boxes(axis, entries=_frequency_entries(rows=rows))


def _frequency_entries(
    *,
    rows: list[tuple[str, np.ndarray, pd.Series]],
) -> list[tuple[float, float]]:
    entries = [
        (float(index), float(cast(float | int | str, row.to_dict()[FNT_FRACTION_COLUMN])))
        for index, (_label, _values, row) in enumerate(rows, start=1)
    ]
    return entries


def _has_frequency_rows(rows: list[tuple[str, np.ndarray, pd.Series]]) -> bool:
    return bool(rows) and FNT_FRACTION_COLUMN in rows[0][2].index

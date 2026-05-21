"""Rendering policy for deterministic aCC figures."""

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyaesa.acc.figures.common import (
    DYNAMIC_SCOPE_COLUMNS,
    MAX_CC_BOUND,
    MIN_CC_BOUND,
    acc_scope_stem,
    apply_acc_axis_policy,
    cumulative_budget_unit_label,
    format_year_axis,
    has_static_min_max_bounds,
    impact_panel_title,
    is_dynamic_scope,
    ordered_impacts,
    panel_unit_label,
    requested_single_year,
    save_figure,
    scope_slices,
    scope_title,
    static_asocc_ssp_slices,
    visible_values,
)
from pyaesa.shared.figures.asocc_transition_policy import asocc_transition_year
from pyaesa.shared.figures.deterministic_variant_display import (
    base_variant_groups,
    has_complete_variant_roles,
    variant_note,
    variant_role_row,
    variant_styles,
)
from pyaesa.shared.figures.deterministic_variant_method_note import (
    write_variant_compression_method_note,
)
from pyaesa.shared.figures.deterministic_variant_compressor import (
    IDENTITY_COLUMNS,
    MAX_ROLE,
    MIN_ROLE,
    ROLE_COLUMN,
    VALUE_COLUMN,
    VARIANT_COLUMNS,
    YEAR_COLUMN,
    compress_variants,
)
from pyaesa.shared.figures.colors import (
    MULTI_METHOD_LINE_ALPHA,
    distinct_colors,
)
from pyaesa.shared.figures.deterministic_legends import (
    bind_deterministic_legend_group,
    render_grouped_deterministic_legend_below,
)
from pyaesa.shared.figures.deterministic_legends_methods import legend_group_from_row
from pyaesa.shared.figures.figure_footer import (
    render_below_figure_legend,
    set_footer_min_plot_height,
)
from pyaesa.shared.figures.jobs import PlannedFigureJob, render_figure_jobs
from pyaesa.shared.figures.layout import (
    DOUBLE_COLUMN_TITLE_TOP,
    MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN,
    MULTI_YEAR_TWO_PANEL_FIGURE_SIZE,
    SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN,
    TRANSITION_PANEL_TITLE_PAD,
    bottom_panel_indices,
    format_single_year_category_axis,
    multi_impact_panel_figure_size,
    single_impact_figure_size,
    show_panel_x_labels,
)
from pyaesa.shared.figures.lcia_scopes import lcia_impact_slices
from pyaesa.shared.figures.multi_year_transitions import (
    TransitionMarker,
    render_transition_markers,
)
from pyaesa.shared.figures.selector_slices import selector_slices
from pyaesa.shared.figures.titles import (
    FIGURE_TITLE_DETAIL_CLEARANCE_IN,
    FIGURE_TITLE_DETAIL_OFFSET_IN,
    render_dynamic_ar6_title,
    title_layout_top,
)
from pyaesa.shared.figures.value_order import order_labels_by_average_within_group_rank
from pyaesa.shared.runtime.reporting.status import StatusSink

_STYLE_COLUMN = "__variant_style"
_COLOR_COLUMN = "__series_color"
_VARIANT_SCORE_COLUMN = "__variant_score_value"
_LABEL_COLUMN = "__series_label"
_LINE_ALPHA = 0.82
_PANEL_TITLE_PAD = 5
_TWO_COLUMN_PANEL_HSPACE = 0.32
_TWO_COLUMN_TRANSITION_HSPACE = 0.42
_DYNAMIC_TITLE_CLEARANCE_PAD_EXTRA = 10
_DYNAMIC_MULTI_METHOD_TITLE_CLEARANCE_PAD_EXTRA = 0
_DYNAMIC_MULTI_METHOD_TITLE_TOP = 0.965
_DYNAMIC_MULTI_METHOD_TITLE_DETAIL_OFFSET_IN = 0.34
_DYNAMIC_MULTI_METHOD_TITLE_DETAIL_CLEARANCE_IN = 0.18


def render_products(
    *,
    rows: pd.DataFrame,
    figures_root: Path,
    requested_years: list[int],
    cc_type: str,
    dpi: int,
    output_format: str,
    per_method: bool = True,
    multi_method: bool = True,
    status: StatusSink | None = None,
) -> list[Path]:
    """Render deterministic aCC per method and multi-method products."""
    jobs = _plan_jobs(
        rows=rows,
        figures_root=figures_root,
        requested_years=requested_years,
        cc_type=cc_type,
        dpi=dpi,
        output_format=output_format,
        per_method=per_method,
        multi_method=multi_method,
    )
    paths = render_figure_jobs(source="deterministic_acc", jobs=jobs, status=status)
    write_variant_compression_method_note(figures_root=figures_root, rows=rows)
    return paths


def _plan_jobs(
    *,
    rows: pd.DataFrame,
    figures_root: Path,
    requested_years: list[int],
    cc_type: str,
    dpi: int,
    output_format: str,
    per_method: bool,
    multi_method: bool,
) -> list[PlannedFigureJob]:
    jobs: list[PlannedFigureJob] = []
    studied_year = requested_single_year(requested_years)
    if multi_method:
        for scope in _multi_method_branch_slices(rows=rows, cc_type=cc_type):
            for selector_token, selector_title, selector_scope in selector_slices(scope):
                prepared_scope = _prepare_rows(selector_scope)
                if _has_multiple_methods(prepared_scope):
                    split_impacts = (
                        studied_year is None and len(ordered_impacts(prepared_scope)) > 1
                    )
                    product_scopes = (
                        lcia_impact_slices(prepared_scope) if split_impacts else [prepared_scope]
                    )
                    for product_scope in product_scopes:
                        jobs.append(
                            _job(
                                kind="multi_method",
                                label="multi_method",
                                frame=product_scope,
                                figures_root=figures_root / "multi_method",
                                requested_years=requested_years,
                                studied_year=studied_year,
                                dpi=dpi,
                                output_format=output_format,
                                group_legend=True,
                                include_method_in_label=True,
                                include_impact=split_impacts,
                                selector_token=selector_token,
                                selector_title=selector_title,
                            )
                        )
    if per_method:
        for scope in _per_method_branch_slices(rows=rows, cc_type=cc_type):
            for selector_token, selector_title, selector_scope in selector_slices(scope):
                prepared_scope = _prepare_rows(selector_scope)
                for method in visible_values(prepared_scope, "__method"):
                    method_rows = prepared_scope.loc[
                        prepared_scope["__method"].astype(str).eq(method)
                    ]
                    jobs.append(
                        _job(
                            kind="per_method",
                            label=method,
                            frame=method_rows.copy(),
                            figures_root=figures_root / "per_method",
                            requested_years=requested_years,
                            studied_year=studied_year,
                            dpi=dpi,
                            output_format=output_format,
                            group_legend=False,
                            include_method_in_label=False,
                            selector_token=selector_token,
                            selector_title=selector_title,
                        )
                    )
    return jobs


def _multi_method_branch_slices(*, rows: pd.DataFrame, cc_type: str) -> list[pd.DataFrame]:
    common = ("lcia_method",)
    if str(cc_type) == "static":
        scoped = []
        for branch_scope in scope_slices(rows, common):
            scoped.extend(static_asocc_ssp_slices(branch_scope))
        return scoped
    return scope_slices(rows, (*common, *DYNAMIC_SCOPE_COLUMNS))


def _per_method_branch_slices(*, rows: pd.DataFrame, cc_type: str) -> list[pd.DataFrame]:
    common = ("lcia_method",)
    if str(cc_type) == "static":
        scoped = []
        for branch_scope in scope_slices(rows, common):
            for method in visible_values(branch_scope, "__method"):
                method_scope = branch_scope.loc[
                    branch_scope["__method"].astype(str).eq(method)
                ].copy()
                scoped.extend(static_asocc_ssp_slices(method_scope))
        return scoped
    scoped = []
    for branch_scope in scope_slices(rows, (*common, *DYNAMIC_SCOPE_COLUMNS)):
        for method in visible_values(branch_scope, "__method"):
            scoped.append(branch_scope.loc[branch_scope["__method"].astype(str).eq(method)].copy())
    return scoped


def _prepare_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return prepare_plot_rows(rows)


def prepare_plot_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic aCC rows after variant compression and style assignment."""
    scored_rows, score_column = _with_acc_variant_scores(rows)
    compressed = compress_variants(scored_rows, score_column=score_column)
    if _VARIANT_SCORE_COLUMN in compressed.columns:
        compressed = compressed.drop(columns=[_VARIANT_SCORE_COLUMN])
    out = _static_min_max_display_rows(compressed)
    out[_STYLE_COLUMN] = (
        _static_min_max_styles(out) if has_static_min_max_bounds(out) else variant_styles(out)
    )
    out[_LABEL_COLUMN] = out["__method"].astype(str).str.strip()
    label_values = [str(value) for value in out[_LABEL_COLUMN].tolist()]
    label_count = len(dict.fromkeys(label_values))
    color_map = {
        label: color
        for label, color in zip(
            dict.fromkeys(label_values),
            distinct_colors(label_count),
            strict=True,
        )
    }
    out[_COLOR_COLUMN] = [color_map[label] for label in label_values]
    return out


def _with_acc_variant_scores(rows: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return aCC rows with cumulative dynamic scores for variant compression."""
    if not is_dynamic_scope(rows):
        return rows, VALUE_COLUMN
    variant_columns = [column for column in VARIANT_COLUMNS if column in rows.columns]
    if not variant_columns:
        return rows, VALUE_COLUMN
    out = rows.copy()
    group_columns = [
        column
        for column in (*IDENTITY_COLUMNS, *DYNAMIC_SCOPE_COLUMNS, *variant_columns)
        if column in out.columns and column != YEAR_COLUMN
    ]
    out[_VARIANT_SCORE_COLUMN] = out.groupby(group_columns, dropna=False, sort=False)[
        VALUE_COLUMN
    ].transform("sum")
    return out, _VARIANT_SCORE_COLUMN


def _job(
    *,
    kind: str,
    label: str,
    frame: pd.DataFrame,
    figures_root: Path,
    requested_years: list[int],
    studied_year: int | None,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_method_in_label: bool,
    include_impact: bool = False,
    selector_token: str = "all",
    selector_title: str | None = None,
) -> PlannedFigureJob:
    output_base = figures_root / acc_scope_stem(
        label,
        frame,
        include_impact=include_impact,
        selector_token=selector_token,
        studied_year=studied_year,
    )
    title = scope_title(
        "aCC",
        None if label == "multi_method" else label,
        frame,
        include_impact=include_impact or len(ordered_impacts(frame)) == 1,
        selector_title=selector_title,
        studied_year=studied_year,
    )
    return PlannedFigureJob(
        kind=kind,
        label=output_base.name,
        render=lambda: plot_scope(
            frame=frame,
            requested_years=requested_years,
            output_stem=output_base,
            title=title,
            dpi=dpi,
            output_format=output_format,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
        ),
    )


def plot_scope(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_method_in_label: bool,
    figure_note: str | None = None,
) -> list[Path]:
    """Render one deterministic aCC figure scope."""
    impacts = ordered_impacts(frame)
    single_year = requested_single_year(requested_years) is not None
    if len(impacts) > 1:
        return _plot_impact_panels(
            frame=frame,
            requested_years=requested_years,
            impacts=impacts,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
            figure_note=figure_note,
        )
    if not single_year and is_dynamic_scope(frame):
        return _plot_dynamic_scope(
            frame=frame,
            requested_years=requested_years,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
            figure_note=figure_note,
        )
    fig, axis = plt.subplots(figsize=single_impact_figure_size(single_year=single_year))
    set_footer_min_plot_height(
        fig,
        height_in=(
            MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN
            if not single_year and group_legend
            else SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN
        ),
    )
    markers = _render_axis(
        axis=axis,
        frame=frame,
        requested_years=requested_years,
        group_legend=group_legend,
        show_x_labels=True,
        include_method_in_label=include_method_in_label,
    )
    axis.set_title(title, fontweight="bold", pad=26 if markers else 6)
    _render_footer(
        fig=fig,
        axis=axis,
        frame=frame,
        group_legend=group_legend,
        note=figure_note,
        single_year=single_year,
    )
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


def _plot_dynamic_scope(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_method_in_label: bool,
    figure_note: str | None,
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
        height_in=(
            MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN if group_legend else SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN
        ),
    )
    markers = _render_axis(
        axis=axis,
        frame=frame,
        requested_years=requested_years,
        group_legend=group_legend,
        show_x_labels=True,
        include_method_in_label=include_method_in_label,
    )
    title_pad = TRANSITION_PANEL_TITLE_PAD if markers else 6
    title_clearance_pad = title_pad + _DYNAMIC_TITLE_CLEARANCE_PAD_EXTRA
    title_top = 0.89
    title_detail_offset = FIGURE_TITLE_DETAIL_OFFSET_IN
    title_detail_clearance = FIGURE_TITLE_DETAIL_CLEARANCE_IN
    if group_legend:
        title_clearance_pad = title_pad + _DYNAMIC_MULTI_METHOD_TITLE_CLEARANCE_PAD_EXTRA
        title_top = _DYNAMIC_MULTI_METHOD_TITLE_TOP
        title_detail_offset = _DYNAMIC_MULTI_METHOD_TITLE_DETAIL_OFFSET_IN
        title_detail_clearance = _DYNAMIC_MULTI_METHOD_TITLE_DETAIL_CLEARANCE_IN
    fig.subplots_adjust(
        top=title_layout_top(
            fig,
            title,
            default_top=title_top,
            panel_title_pad=title_clearance_pad,
            detail_offset_in=title_detail_offset,
            detail_clearance_in=title_detail_clearance,
        )
    )
    axis.set_title("Pathways", fontweight="bold", pad=title_pad)
    _render_dynamic_budget_axis(
        axis=budget_axis,
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    budget_axis.set_title("Cumulative budget", fontweight="bold", pad=title_pad)
    _render_footer(
        fig=fig,
        axis=axis,
        budget_axis=budget_axis,
        frame=frame,
        group_legend=group_legend,
        note=figure_note,
        single_year=False,
    )
    fig.subplots_adjust(
        top=title_layout_top(
            fig,
            title,
            default_top=title_top,
            panel_title_pad=title_clearance_pad,
            detail_offset_in=title_detail_offset,
            detail_clearance_in=title_detail_clearance,
        )
    )
    render_dynamic_ar6_title(fig, title, detail_offset_in=title_detail_offset)
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


def _plot_impact_panels(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    impacts: list[str],
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_method_in_label: bool,
    figure_note: str | None,
) -> list[Path]:
    single_year = requested_single_year(requested_years) is not None
    single_year_entries = (
        _single_year_entries(frame, include_method_in_label=include_method_in_label)
        if single_year
        else None
    )
    bar_order = (
        _single_year_label_order_from_entries(single_year_entries)
        if single_year_entries is not None
        else None
    )
    ncols = 2
    nrows = (len(impacts) + 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=multi_impact_panel_figure_size(
            nrows=nrows,
            compact=single_year and not group_legend,
        ),
        squeeze=False,
    )
    bottom_indices = bottom_panel_indices(panel_count=len(impacts), ncols=ncols)
    first_axis = axes[0, 0]
    has_transitions = False
    for index, impact in enumerate(impacts):
        axis = axes[index // ncols, index % ncols]
        panel = frame.loc[frame["impact"].astype(str).eq(str(impact))].copy()
        panel_entries = (
            [
                entry
                for entry in single_year_entries
                if str(entry[1].get("impact", "")) == str(impact)
            ]
            if single_year_entries is not None
            else None
        )
        markers = _render_axis(
            axis=axis,
            frame=panel,
            requested_years=requested_years,
            group_legend=group_legend,
            show_x_labels=show_panel_x_labels(
                panel_index=index,
                bottom_indices=bottom_indices,
            ),
            include_method_in_label=include_method_in_label,
            bar_order=bar_order,
            single_year_entries=panel_entries,
        )
        if markers:
            has_transitions = True
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(
            impact_panel_title(panel, impact=impact),
            loc="left",
            pad=TRANSITION_PANEL_TITLE_PAD if markers else _PANEL_TITLE_PAD,
        )
    for index in range(len(impacts), nrows * ncols):
        axes[index // ncols, index % ncols].axis("off")
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
    _render_footer(
        fig=fig,
        axis=first_axis,
        frame=frame,
        group_legend=group_legend,
        note=figure_note,
        single_year=single_year,
    )
    top = title_layout_top(
        fig,
        title,
        default_top=DOUBLE_COLUMN_TITLE_TOP,
        panel_title_pad=TRANSITION_PANEL_TITLE_PAD if has_transitions else _PANEL_TITLE_PAD,
    )
    fig.subplots_adjust(top=top)
    render_dynamic_ar6_title(fig, title)
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


def _render_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    requested_years: list[int],
    group_legend: bool,
    show_x_labels: bool,
    include_method_in_label: bool,
    bar_order: list[str] | None = None,
    single_year_entries: list[tuple[str, pd.Series, float, float]] | None = None,
) -> list[TransitionMarker]:
    single_year = requested_single_year(requested_years) is not None
    if single_year:
        values = _render_single_year_bars(
            axis=axis,
            frame=frame,
            group_legend=group_legend,
            show_x_labels=show_x_labels,
            include_method_in_label=include_method_in_label,
            bar_order=bar_order,
            single_year_entries=single_year_entries,
        )
        markers: list[TransitionMarker] = []
    else:
        values, markers = _render_multi_year_lines(
            axis=axis,
            frame=frame,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
        )
        years = sorted({int(year) for year in frame[YEAR_COLUMN].tolist()})
        format_year_axis(axis, years=years, show_labels=show_x_labels)
        render_transition_markers(axis, markers=markers)
    apply_acc_axis_policy(axis, values=values, context="deterministic aCC figure")
    axis.set_ylabel(panel_unit_label(frame))
    axis.grid(alpha=0.25, axis="y" if single_year else "both")
    return markers


def _render_dynamic_budget_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    include_method_in_label: bool,
) -> np.ndarray:
    entries = _dynamic_budget_entries(
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    labels = list(dict.fromkeys(label for label, _row, _value, _style in entries))
    positions = {label: index for index, label in enumerate(labels)}
    values: list[float] = []
    for label in labels:
        label_entries = [entry for entry in entries if entry[0] == label]
        position = float(positions[label])
        color = _row_color(label_entries[0][1], group_legend=False)
        label_values = [float(value) for _label, _row, value, _style in label_entries]
        low = min(label_values)
        high = max(label_values)
        axis.bar(
            [position],
            [low],
            width=0.58,
            color=color,
            alpha=_LINE_ALPHA,
            zorder=3,
        )
        if high > low:
            axis.vlines(
                position,
                low,
                high,
                colors=color,
                linestyles=":",
                linewidth=1.8,
                zorder=4,
            )
            axis.hlines(
                high,
                position - 0.29,
                position + 0.29,
                colors=color,
                linestyles=":",
                linewidth=1.8,
                zorder=4,
            )
        values.extend(float(value) for value in label_values)
    x_positions = np.arange(len(labels), dtype=float)
    axis.set_xlim(-0.5, max(0.5, float(len(labels)) - 0.5))
    if include_method_in_label and any(str(label).strip() for label in labels):
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, rotation=45, ha="right")
    else:
        axis.set_xticks(x_positions)
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    numeric = np.asarray(values, dtype=float)
    apply_acc_axis_policy(axis, values=numeric, context="deterministic dynamic aCC budget panel")
    axis.set_ylabel(cumulative_budget_unit_label(frame))
    axis.set_xlabel("")
    axis.grid(alpha=0.25, axis="y")
    return numeric


def _dynamic_budget_entries(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
) -> list[tuple[str, pd.Series, float, str]]:
    entries: list[tuple[str, pd.Series, float, str]] = []
    for group in _line_groups(frame):
        ordered = group.sort_values(YEAR_COLUMN, kind="stable")
        row = pd.Series(ordered.iloc[0], copy=False)
        label = _method_label(row) if include_method_in_label else "Study period"
        values = pd.Series(pd.to_numeric(ordered[VALUE_COLUMN], errors="raise"), copy=False)
        entries.append((label, row, float(values.sum()), str(row[_STYLE_COLUMN])))
    return sorted(entries, key=lambda item: (item[0], 0 if item[3] != "dotted" else 1))


def _row_color(row: pd.Series, *, group_legend: bool) -> str:
    del group_legend
    return str(row[_COLOR_COLUMN])


def _render_single_year_bars(
    *,
    axis: Any,
    frame: pd.DataFrame,
    group_legend: bool,
    show_x_labels: bool,
    include_method_in_label: bool,
    bar_order: list[str] | None = None,
    single_year_entries: list[tuple[str, pd.Series, float, float]] | None = None,
) -> np.ndarray:
    entries = (
        single_year_entries
        if single_year_entries is not None
        else _single_year_entries(frame, include_method_in_label=include_method_in_label)
    )
    if bar_order is not None:
        entries = _order_single_year_entries(entries, bar_order=bar_order)
    values = []
    seen_labels: set[str] = set()
    for index, (label, row, low, high) in enumerate(entries):
        color = _row_color(row, group_legend=group_legend)
        visible_label = label if label and label not in seen_labels else "_nolegend_"
        if label:
            seen_labels.add(label)
        handle = axis.bar(
            [index],
            [low],
            width=0.72,
            color=color,
            alpha=_LINE_ALPHA,
            label=visible_label,
            zorder=2,
        )
        if group_legend:
            bind_deterministic_legend_group(handle, legend_group_from_row(row))
            bind_deterministic_legend_group(handle[0], legend_group_from_row(row))
        if high > low:
            axis.vlines(
                index,
                low,
                high,
                colors=color,
                linestyles=":",
                linewidth=1.9,
                zorder=3,
            )
            axis.hlines(
                high,
                index - 0.36,
                index + 0.36,
                colors=color,
                linestyles=":",
                linewidth=1.9,
            )
        values.extend([low, high])
    positions = list(range(len(entries)))
    labels = [label for label, _row, _low, _high in entries]
    if show_x_labels and any(labels):
        format_single_year_category_axis(axis, positions=positions, labels=labels)
    else:
        axis.set_xticks(positions)
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    axis.set_xlim(-0.65, len(entries) - 0.35)
    return np.asarray(values, dtype=float)


def _render_multi_year_lines(
    *,
    axis: Any,
    frame: pd.DataFrame,
    group_legend: bool,
    include_method_in_label: bool,
) -> tuple[np.ndarray, list[TransitionMarker]]:
    values: list[float] = []
    markers: dict[int, TransitionMarker] = {}
    seen: set[str] = set()
    for group in _line_groups(frame):
        ordered = group.sort_values(YEAR_COLUMN, kind="stable")
        row = pd.Series(ordered.iloc[0], copy=False)
        years = pd.Series(pd.to_numeric(ordered[YEAR_COLUMN], errors="raise")).astype(int)
        numeric = pd.Series(pd.to_numeric(ordered[VALUE_COLUMN], errors="raise")).astype(float)
        label = _row_label(row, include_method_in_label=include_method_in_label)
        visible_label = label if label and label not in seen else "_nolegend_"
        seen.add(label)
        line = axis.plot(
            years.to_numpy(dtype=int),
            numeric.to_numpy(dtype=float),
            linewidth=1.7,
            linestyle=":" if str(row[_STYLE_COLUMN]) == "dotted" else "-",
            color=_row_color(row, group_legend=group_legend),
            alpha=MULTI_METHOD_LINE_ALPHA if group_legend else _LINE_ALPHA,
            label=visible_label,
            zorder=2,
        )[0]
        if group_legend and visible_label != "_nolegend_":
            bind_deterministic_legend_group(line, legend_group_from_row(row))
        marker = asocc_transition_year(ordered)
        if marker is not None:
            markers[int(marker)] = TransitionMarker(
                year=int(marker),
                label="retrospective/prospective transition",
                color="#7d7d7d",
            )
        values.extend(numeric.tolist())
    return np.asarray(values, dtype=float), list(markers.values())


def _single_year_entries(
    frame: pd.DataFrame,
    *,
    include_method_in_label: bool,
) -> list[tuple[str, pd.Series, float, float]]:
    entries = []
    for group in _base_groups(frame):
        row = pd.Series(group.iloc[0], copy=False)
        low_row, high_row = _single_year_low_high_rows(group)
        label = _row_label(row, include_method_in_label=include_method_in_label)
        low_value = float(cast(float | int | str, low_row.to_dict()[VALUE_COLUMN]))
        high_value = float(cast(float | int | str, high_row.to_dict()[VALUE_COLUMN]))
        entries.append((label, row, low_value, high_value))
    return sorted(entries, key=lambda item: (-item[2], item[0]))


def _single_year_label_order_from_entries(
    entries: list[tuple[str, pd.Series, float, float]],
) -> list[str]:
    values: list[tuple[str, str, float]] = []
    for label, row, low_value, _high_value in entries:
        if not str(label).strip():
            continue
        impact = str(row.get("impact", "")).strip()
        values.append((impact, label, float(low_value)))
    return order_labels_by_average_within_group_rank(values)


def _order_single_year_entries(
    entries: list[tuple[str, pd.Series, float, float]],
    *,
    bar_order: list[str],
) -> list[tuple[str, pd.Series, float, float]]:
    ranks = {label: index for index, label in enumerate(bar_order)}
    return sorted(entries, key=lambda item: (ranks.get(item[0], len(ranks)), item[0]))


def _line_groups(frame: pd.DataFrame) -> list[pd.DataFrame]:
    columns = [
        column
        for column in [*IDENTITY_COLUMNS, "cc_bound", *VARIANT_COLUMNS, _STYLE_COLUMN]
        if column in frame.columns
    ]
    return [group.copy() for _key, group in frame.groupby(columns, dropna=False, sort=True)]


def _base_groups(frame: pd.DataFrame) -> list[pd.DataFrame]:
    return [group.copy() for group in base_variant_groups(frame)]


def _row_label(row: pd.Series, *, include_method_in_label: bool = True) -> str:
    return _method_label(row) if include_method_in_label else ""


def _method_label(row: pd.Series) -> str:
    return str(row.get("__method", "")).strip()


def _render_footer(
    *,
    fig: Any,
    axis: Any,
    frame: pd.DataFrame,
    group_legend: bool,
    note: str | None,
    single_year: bool,
    budget_axis: Any | None = None,
) -> None:
    footer = _combined_note(_variant_footer(frame, single_year=single_year), note)
    if group_legend:
        del budget_axis
        render_grouped_deterministic_legend_below(axis, legend_note=footer)
    else:
        render_below_figure_legend(fig, legend_note=footer, max_columns=3)


def _combined_note(first: str | None, second: str | None) -> str | None:
    parts = [str(note).strip() for note in (first, second) if str(note or "").strip()]
    return "\n".join(parts) if parts else None


def _has_multiple_methods(frame: pd.DataFrame) -> bool:
    return len(visible_values(frame, "__method")) > 1


def _static_min_max_display_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows used to run one combined min plus max static CC figure."""
    if not has_static_min_max_bounds(frame):
        return frame.copy()
    if ROLE_COLUMN not in frame.columns:
        return frame.copy()
    roles = pd.Series(frame[ROLE_COLUMN], copy=False).astype(str).str.strip()
    bounds = pd.Series(frame["cc_bound"], copy=False).astype(str).str.strip()
    no_role = roles.eq("") | roles.eq("nan") | frame[ROLE_COLUMN].isna()
    mask = (
        no_role
        | (bounds.eq(MIN_CC_BOUND) & roles.eq(MIN_ROLE))
        | (bounds.eq(MAX_CC_BOUND) & roles.eq(MAX_ROLE))
    )
    return frame.loc[mask].copy()


def _static_min_max_styles(frame: pd.DataFrame) -> list[str]:
    bounds = pd.Series(frame["cc_bound"], copy=False).astype(str).str.strip()
    return ["dotted" if bound == MAX_CC_BOUND else "solid" for bound in bounds.tolist()]


def _single_year_low_high_rows(group: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if has_static_min_max_bounds(group):
        min_rows = _bound_rows(group, bound=MIN_CC_BOUND)
        max_rows = _bound_rows(group, bound=MAX_CC_BOUND)
        low_row = _retained_bound_row(min_rows)
        high_row = _retained_bound_row(max_rows)
        return low_row, high_row
    has_variant_roles = has_complete_variant_roles(group)
    row = pd.Series(group.iloc[0], copy=False)
    low_row = variant_role_row(group, role=MIN_ROLE) if has_variant_roles else row
    high_row = variant_role_row(group, role=MAX_ROLE) if has_variant_roles else low_row
    return low_row, high_row


def _bound_rows(group: pd.DataFrame, *, bound: str) -> pd.DataFrame:
    bounds = pd.Series(group["cc_bound"], copy=False).astype(str).str.strip()
    return group.loc[bounds.eq(bound)]


def _retained_bound_row(group: pd.DataFrame) -> pd.Series:
    return pd.Series(group.iloc[0], copy=False)


def _variant_footer(frame: pd.DataFrame, *, single_year: bool) -> str | None:
    geometry = _static_min_max_geometry(frame, single_year=single_year)
    note = variant_note(frame, single_year=single_year, geometry_override=geometry)
    if note is not None:
        return note
    return _static_min_max_geometry(frame, single_year=single_year)


def _static_min_max_geometry(frame: pd.DataFrame, *, single_year: bool) -> str | None:
    if not has_static_min_max_bounds(frame):
        return None
    has_variants = ROLE_COLUMN in frame.columns and {
        str(value).strip()
        for value in frame[ROLE_COLUMN].tolist()
        if str(value).strip() and str(value).strip() != "nan"
    } == {MIN_ROLE, MAX_ROLE}
    if not has_variants:
        return (
            "Solid bar = min CC; dotted whisker and cap = max CC."
            if single_year
            else "Plain = min CC; dotted = max CC."
        )
    if single_year:
        return (
            "Solid bar = min CC lower retained combination; "
            "dotted whisker and cap = max CC upper retained combination."
        )
    return "Plain = min CC lower retained combination; dotted = max CC upper retained combination."

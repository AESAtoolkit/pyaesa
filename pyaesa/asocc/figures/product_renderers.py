"""Deterministic aSoCC figure product orchestration."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from pyaesa.shared.figures.colors import (
    MULTI_METHOD_LINE_ALPHA,
    single_or_distinct_colors,
)
from pyaesa.shared.figures.jobs import render_figure_jobs
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
from pyaesa.shared.figures.deterministic_legends import (
    bind_deterministic_legend_group,
    render_grouped_deterministic_legend_below,
)
from pyaesa.shared.figures.deterministic_legends_methods import legend_group_from_row
from pyaesa.shared.figures.figure_footer import (
    render_below_figure_legend,
    set_footer_min_plot_height,
)
from pyaesa.shared.figures.layout import (
    DOUBLE_COLUMN_TITLE_TOP,
    MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN,
    SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN,
    bottom_panel_indices,
    format_integer_year_axis,
    format_single_year_category_axis,
    multi_impact_panel_figure_size,
    single_impact_figure_size,
    show_panel_x_labels,
)
from pyaesa.shared.figures.lcia_metadata import (
    ordered_impact_panels,
    resolve_frame_impact_title,
)
from pyaesa.shared.figures.multi_year_transitions import (
    TransitionMarker,
    render_transition_markers,
)
from pyaesa.shared.figures.paths import output_file_path
from pyaesa.shared.figures.scientific_ticks import scientific_percent_tick_formatter
from pyaesa.shared.figures.titles import render_figure_title, title_layout_top
from pyaesa.shared.figures.value_order import order_labels_by_average_score
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.tabular.scalars import is_display_missing

from .multi_method_renderer import plan_multi_method_jobs
from .per_method_renderer import plan_per_method_jobs, series_transition, visible_values
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

_STYLE_COLUMN = "__variant_style"
_LEGEND_LABEL_COLUMN = "__variant_legend_label"
_AXIS_LABEL_COLUMN = "__axis_category_label"
_SHOW_LEGEND_COLUMN = "__show_variant_legend"
_COLOR_COLUMN = "__series_color"
_LINE_ALPHA = 0.78
_LEGEND_MAX_COLUMNS = 3
_ASOCC_Y_AXIS_TOP_PADDING_FRACTION = 0.08
_PANEL_TITLE_PAD = 5
_TWO_COLUMN_PANEL_HSPACE = 0.32


@dataclass(frozen=True)
class SingleYearBar:
    """Prepared single year category bar with compressed min and max values."""

    row: pd.Series
    axis_label: str
    label: str
    visible_label: str
    color: str
    min_value: float
    max_value: float


def render_products(
    *,
    rows: pd.DataFrame,
    figures_root: Path,
    requested_years: list[int],
    dpi: int,
    output_format: str,
    status_source: str = "deterministic_asocc",
    per_method: bool = True,
    multi_method: bool = True,
    status: StatusSink | None = None,
) -> list[Path]:
    """Render deterministic aSoCC multi-method products before per method products."""
    jobs = [
        *(
            plan_multi_method_jobs(
                rows=rows,
                figures_root=figures_root,
                requested_years=requested_years,
                dpi=dpi,
                output_format=output_format,
                plotter=plot_scope,
                row_preparer=prepare_plot_rows,
            )
            if multi_method
            else []
        ),
        *(
            plan_per_method_jobs(
                rows=rows,
                figures_root=figures_root,
                requested_years=requested_years,
                dpi=dpi,
                output_format=output_format,
                plotter=plot_scope,
                row_preparer=prepare_plot_rows,
            )
            if per_method
            else []
        ),
    ]
    paths = render_figure_jobs(source=status_source, jobs=jobs, status=status)
    write_variant_compression_method_note(figures_root=figures_root, rows=rows)
    return paths


def prepare_plot_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic aSoCC rows after compression and style assignment."""
    compressed = compress_variants(rows)
    return compressed.assign(**{_STYLE_COLUMN: variant_styles(compressed)})


def plot_scope(
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
    figure_note: str | None = None,
) -> list[Path]:
    """Render one deterministic aSoCC figure scope."""
    include_impact_in_legend = _include_impact_in_legend(
        frame,
        include_impact_in_label=include_impact_in_label,
    )
    single_year = len({int(year) for year in requested_years}) == 1
    axis_impact_label = include_impact_in_label and len(visible_values(frame, "impact")) > 1
    if single_year and axis_impact_label and not group_legend:
        include_impact_in_legend = False
    frame = with_variant_legend_labels(
        frame,
        include_impact_in_label=axis_impact_label if single_year else include_impact_in_label,
        include_impact_in_legend=include_impact_in_legend,
        include_method_in_label=include_method_in_label,
    )
    frame = with_series_colors(
        frame,
        include_impact_in_label=include_impact_in_legend,
        include_method_in_label=include_method_in_label,
    )
    if single_year and group_legend and len(_ordered_impacts(frame)) > 1:
        return _plot_single_year_impact_panel_scope(
            frame=frame,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
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
    visible_years: list[int] = []
    transition_markers: dict[int, TransitionMarker] = {}
    if single_year:
        render_single_year_bars(axis, frame=frame, group_legend=group_legend)
    else:
        for group in plot_groups(frame):
            ordered = group.sort_values(YEAR_COLUMN, kind="stable")
            years = cast(
                pd.Series,
                pd.to_numeric(pd.Series(ordered[YEAR_COLUMN], copy=False), errors="raise"),
            ).astype(int)
            values = cast(
                pd.Series,
                pd.to_numeric(pd.Series(ordered[VALUE_COLUMN], copy=False), errors="raise"),
            ).astype(float)
            year_values = years.to_numpy(dtype=int)
            plotted_values = values.to_numpy(dtype=float) * 100.0
            visible_years.extend(int(year) for year in year_values.tolist())
            first_row = pd.Series(ordered.iloc[0], copy=False)
            label = str(first_row[_LEGEND_LABEL_COLUMN])
            show_in_legend = bool(first_row[_SHOW_LEGEND_COLUMN])
            visible_label = label if show_in_legend else "_nolegend_"
            color = str(first_row[_COLOR_COLUMN])
            style = str(first_row.get(_STYLE_COLUMN, "solid"))
            marker = series_transition(group)
            line = plot_line(
                axis,
                year_values=year_values.tolist(),
                plotted_values=plotted_values.tolist(),
                label=visible_label,
                style=style,
                color=color,
                alpha=MULTI_METHOD_LINE_ALPHA if group_legend else _LINE_ALPHA,
            )
            if group_legend:
                bind_deterministic_legend_group(line, legend_group_from_row(first_row))
            if marker is not None:
                transition_markers[int(marker)] = TransitionMarker(
                    year=int(marker),
                    label="retrospective/prospective transition",
                    color="#7d7d7d",
                )
    if not single_year:
        _format_year_axis(axis, visible_years=visible_years)
    axis.set_title(title, fontweight="bold", pad=26 if transition_markers else 6)
    format_scope_axes(axis, single_year=single_year)
    axis.grid(alpha=0.25, axis="x" if single_year else "both")
    render_transition_markers(axis, markers=list(transition_markers.values()))
    note = _combined_note(
        variant_note(
            frame,
            single_year=single_year,
        ),
        figure_note,
    )
    if group_legend:
        render_grouped_deterministic_legend_below(axis, legend_note=note)
    elif not single_year:
        render_below_figure_legend(fig, legend_note=note, max_columns=_LEGEND_MAX_COLUMNS)
    else:
        render_below_figure_legend(fig, legend_note=note, max_columns=1)
    output_path = output_file_path(base_path=output_stem, output_format=output_format)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight", format=output_format)
    plt.close(fig)
    return [output_path]


def _plot_single_year_impact_panel_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    include_method_in_label: bool,
    figure_note: str | None,
) -> list[Path]:
    impacts = _ordered_impacts(frame)
    ncols = 2
    nrows = math.ceil(len(impacts) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=multi_impact_panel_figure_size(nrows=nrows),
        squeeze=False,
        sharey=True,
    )
    common_top = _visible_single_year_top(frame)
    bar_order = _single_year_bar_label_order(frame)
    bottom_label_indices = bottom_panel_indices(panel_count=len(impacts), ncols=ncols)
    first_axis = axes[0, 0]
    for index, impact in enumerate(impacts):
        row = index // ncols
        column = index % ncols
        axis = axes[row, column]
        panel = frame.loc[frame["impact"].astype(str).eq(str(impact))].copy()
        render_single_year_bars(
            axis,
            frame=panel,
            group_legend=True,
            show_x_labels=show_panel_x_labels(
                panel_index=index,
                bottom_indices=bottom_label_indices,
            ),
            bar_order=bar_order,
        )
        format_scope_axes(axis, single_year=True, data_top=common_top)
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(
            _impact_panel_title(panel, impact=str(impact)),
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
    note = _combined_note(
        variant_note(
            frame,
            single_year=True,
        ),
        figure_note,
    )
    render_grouped_deterministic_legend_below(first_axis, legend_note=note)
    output_path = output_file_path(base_path=output_stem, output_format=output_format)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight", format=output_format)
    plt.close(fig)
    return [output_path]


def render_single_year_bars(
    axis,
    *,
    frame: pd.DataFrame,
    group_legend: bool,
    show_x_labels: bool = True,
    bar_order: list[str] | None = None,
) -> None:
    """Render single year categories as min bars with max caps."""
    bars = single_year_bars(frame)
    if bar_order is not None:
        bars = _order_single_year_bars(bars, bar_order=bar_order)
    width = 0.72
    for index, entry in enumerate(bars):
        bar_container = axis.bar(
            [index],
            [entry.min_value],
            width=width,
            color=entry.color,
            alpha=_LINE_ALPHA,
            label=entry.visible_label,
            zorder=2,
        )
        if group_legend:
            group = legend_group_from_row(entry.row)
            bind_deterministic_legend_group(bar_container, group)
            bind_deterministic_legend_group(bar_container[0], group)
        if entry.max_value > entry.min_value:
            axis.vlines(
                index,
                entry.min_value,
                entry.max_value,
                colors=entry.color,
                linestyles=":",
                linewidth=1.9,
                alpha=0.95,
                zorder=3,
            )
            axis.hlines(
                entry.max_value,
                index - width / 2,
                index + width / 2,
                colors=entry.color,
                linestyles=":",
                linewidth=1.9,
                alpha=0.95,
                zorder=3,
            )
    positions = list(range(len(bars)))
    labels = [bar.axis_label for bar in bars]
    if show_x_labels and any(str(label).strip() for label in labels):
        format_single_year_category_axis(
            axis,
            positions=positions,
            labels=labels,
        )
    else:
        axis.set_xticks(positions)
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    axis.set_xlim(-0.65, len(bars) - 0.35)


def single_year_bars(frame: pd.DataFrame) -> list[SingleYearBar]:
    """Return single year category bars sorted by retained min value."""
    entries = [_single_year_bar(group) for group in _base_groups(frame)]
    sorted_entries = sorted(entries, key=lambda entry: (-entry.min_value, entry.axis_label))
    visible_labels: set[str] = set()
    out: list[SingleYearBar] = []
    for entry in sorted_entries:
        visible_label = entry.label
        if visible_label in visible_labels:
            visible_label = "_nolegend_"
        else:
            visible_labels.add(visible_label)
        out.append(
            SingleYearBar(
                row=entry.row,
                axis_label=entry.axis_label,
                label=entry.label,
                visible_label=visible_label,
                color=entry.color,
                min_value=entry.min_value,
                max_value=entry.max_value,
            )
        )
    return out


def _single_year_bar_label_order(frame: pd.DataFrame) -> list[str]:
    scores: dict[str, list[float]] = {}
    for bar in single_year_bars(frame):
        scores.setdefault(bar.label, []).append(float(bar.min_value))
    return order_labels_by_average_score(scores)


def _order_single_year_bars(
    bars: list[SingleYearBar],
    *,
    bar_order: list[str],
) -> list[SingleYearBar]:
    ranks = {label: index for index, label in enumerate(bar_order)}
    return sorted(bars, key=lambda bar: (ranks.get(bar.label, len(ranks)), bar.axis_label))


def _single_year_bar(group: pd.DataFrame) -> SingleYearBar:
    legend_row = _legend_row(group)
    has_variant_roles = has_complete_variant_roles(group)
    min_row = variant_role_row(group, role=MIN_ROLE) if has_variant_roles else legend_row
    max_row = variant_role_row(group, role=MAX_ROLE) if has_variant_roles else min_row
    min_value = _row_percent_value(min_row)
    max_value = _row_percent_value(max_row)
    return SingleYearBar(
        row=legend_row,
        axis_label=str(legend_row[_AXIS_LABEL_COLUMN]),
        label=str(legend_row[_LEGEND_LABEL_COLUMN]),
        visible_label=str(legend_row[_LEGEND_LABEL_COLUMN]),
        color=str(legend_row[_COLOR_COLUMN]),
        min_value=min_value,
        max_value=max_value,
    )


def _legend_row(group: pd.DataFrame) -> pd.Series:
    visible = group.loc[group[_SHOW_LEGEND_COLUMN].astype(bool)]
    if visible.empty:
        return pd.Series(group.iloc[0], copy=False)
    return pd.Series(visible.iloc[0], copy=False)


def _row_percent_value(row: pd.Series) -> float:
    value = cast(float | int | str, row.to_dict()[VALUE_COLUMN])
    return float(value) * 100.0


def plot_line(
    axis,
    *,
    year_values: list[int],
    plotted_values: list[float],
    label: str,
    style: str,
    color: str,
    alpha: float = _LINE_ALPHA,
):
    """Plot one trajectory with its compressed variant style."""
    common = {
        "linewidth": 1.7,
        "color": color,
        "alpha": float(alpha),
        "zorder": 2,
    }
    return axis.plot(
        year_values,
        plotted_values,
        linestyle=":" if style == "dotted" else "-",
        label=label,
        **common,
    )[0]


def plot_groups(frame: pd.DataFrame):
    """Yield visible line groups for one figure scope."""
    columns = [
        column
        for column in [*IDENTITY_COLUMNS, *VARIANT_COLUMNS, _STYLE_COLUMN]
        if column in frame.columns
    ]
    for key, group in frame.groupby(columns, dropna=False, sort=True):
        _ = key
        yield group


def with_variant_legend_labels(
    frame: pd.DataFrame,
    *,
    include_impact_in_label: bool,
    include_impact_in_legend: bool | None = None,
    include_method_in_label: bool = True,
) -> pd.DataFrame:
    """Attach one legend label per plain retained variant family."""
    work = frame.copy()
    legend_impact = (
        include_impact_in_label
        if include_impact_in_legend is None
        else bool(include_impact_in_legend)
    )
    work[_LEGEND_LABEL_COLUMN] = "value"
    work[_AXIS_LABEL_COLUMN] = "value"
    work[_SHOW_LEGEND_COLUMN] = True
    base_columns = [column for column in IDENTITY_COLUMNS if column in work.columns]
    for _key, group in work.groupby(base_columns, dropna=False):
        first_row = pd.Series(group.iloc[0], copy=False)
        axis_label = base_series_label(
            first_row,
            include_impact_in_label=include_impact_in_label,
            include_method_in_label=include_method_in_label,
        )
        legend_label = base_series_label(
            first_row,
            include_impact_in_label=legend_impact,
            include_method_in_label=include_method_in_label,
        )
        work.loc[group.index, _AXIS_LABEL_COLUMN] = axis_label
        has_variant_roles = has_complete_variant_roles(group)
        label = legend_label
        work.loc[group.index, _LEGEND_LABEL_COLUMN] = label
        if has_variant_roles:
            show_mask = group[ROLE_COLUMN].astype(str).ne(MAX_ROLE) & bool(str(label).strip())
        else:
            show_mask = group[_STYLE_COLUMN].astype(str).ne("dotted") & bool(str(label).strip())
        work.loc[group.index, _SHOW_LEGEND_COLUMN] = show_mask
    return work


def with_series_colors(
    frame: pd.DataFrame,
    *,
    include_impact_in_label: bool,
    include_method_in_label: bool = True,
) -> pd.DataFrame:
    """Attach deterministic nonrepeating colors to visible base series."""
    work = frame.copy()
    groups = _base_groups(work)
    keys = [
        base_series_label(
            pd.Series(group.iloc[0], copy=False),
            include_impact_in_label=include_impact_in_label,
            include_method_in_label=include_method_in_label,
        )
        for group in groups
    ]
    color_map = single_or_distinct_colors(keys)
    for group in groups:
        first_row = pd.Series(group.iloc[0], copy=False)
        key = base_series_label(
            first_row,
            include_impact_in_label=include_impact_in_label,
            include_method_in_label=include_method_in_label,
        )
        work.loc[group.index, _COLOR_COLUMN] = color_map[key]
    return work


def base_series_label(
    row: pd.Series,
    *,
    include_impact_in_label: bool,
    include_method_in_label: bool = True,
) -> str:
    """Return the compact legend label without variant values."""
    parts = [str(row["__method"]).strip()] if include_method_in_label else []
    if include_impact_in_label:
        impact = _visible_value(row.get("impact"))
        if impact is not None:
            parts.append(impact)
    return " | ".join(parts)


def _include_impact_in_legend(frame: pd.DataFrame, *, include_impact_in_label: bool) -> bool:
    """Return whether impact adds information to legend entries for this scope."""
    if not include_impact_in_label:
        return False
    return len(visible_values(frame, "impact")) > 1


def _visible_value(value: object) -> str | None:
    if is_display_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _combined_note(first: str | None, second: str | None) -> str | None:
    parts = [str(note).strip() for note in (first, second) if str(note or "").strip()]
    if not parts:
        return None
    return "\n".join(parts)


def _base_groups(frame: pd.DataFrame) -> list[pd.DataFrame]:
    return base_variant_groups(frame)


def _ordered_impacts(frame: pd.DataFrame) -> list[str]:
    impacts = visible_values(frame, "impact")
    if len(impacts) <= 1:
        return impacts
    return ordered_impact_panels(
        lcia_method=visible_values(frame, "lcia_method")[0], impacts=impacts
    )


def _visible_single_year_top(frame: pd.DataFrame) -> float | None:
    bars = single_year_bars(frame)
    return max((bar.max_value for bar in bars), default=None)


def _impact_panel_title(frame: pd.DataFrame, *, impact: str) -> str:
    title = resolve_frame_impact_title(frame)
    return str(title).strip() if title is not None else str(impact).strip()


def _format_year_axis(axis, *, visible_years: list[int]) -> None:
    years = sorted({int(year) for year in visible_years})
    format_integer_year_axis(axis, years=years)


def format_scope_axes(axis, *, single_year: bool, data_top: float | None = None) -> None:
    """Apply axis labels and value formatting for one aSoCC product type."""
    del single_year
    axis.set_xlabel("")
    axis.set_ylabel("aSoCC (%)")
    apply_asocc_y_axis_policy(axis, data_top=data_top)


def apply_asocc_y_axis_policy(axis, *, data_top: float | None = None) -> None:
    """Start aSoCC axes at zero and keep headroom above visible data."""
    current_top = float(axis.get_ylim()[1])
    resolved_data_top = float(axis.dataLim.ymax) if data_top is None else float(data_top)
    if math.isfinite(resolved_data_top) and resolved_data_top > 0.0:
        padded_top = resolved_data_top * (1.0 + _ASOCC_Y_AXIS_TOP_PADDING_FRACTION)
    else:
        padded_top = current_top
    axis.set_ylim(bottom=0.0, top=max(current_top, padded_top))
    axis.yaxis.set_major_formatter(FuncFormatter(scientific_percent_tick_formatter))

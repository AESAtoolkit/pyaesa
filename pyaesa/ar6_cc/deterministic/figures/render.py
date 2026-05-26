"""Deterministic figure renderers for dynamic carrying capacity outputs."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from pyaesa.ar6_cc.deterministic.request.contracts import (
    CC_FLOW_NEGATIVE,
    CC_FLOW_NET,
    CC_FLOW_POSITIVE,
)
from pyaesa.ar6_cc.deterministic.figures.period_panels import (
    ar6_period_panel_title_pad,
    combine_study_and_post_tables,
    render_deterministic_legends_below,
    render_study_transition,
    figure_year_columns,
    plot_budget_panel,
)
from pyaesa.ar6_cc.shared.runtime.figure_style import ar6_category_color, ar6_cc_flow_color
from pyaesa.ar6_cc.shared.runtime.figure_titles import ar6_cc_title
from pyaesa.shared.figures.layout import format_integer_year_axis
from pyaesa.shared.figures.multi_year_transitions import (
    normalized_requested_years,
)
from pyaesa.shared.figures.nonnegative_axis import apply_zero_floor_if_nonnegative
from pyaesa.shared.figures.save import save_figure
from pyaesa.shared.figures.scientific_ticks import scientific_tick_formatter
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.titles import render_figure_title, title_layout_top
from pyaesa.shared.runtime.reporting.figure_progress import render_with_progress
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.selectors.scenarios import ssp_partition_token

_DETERMINISTIC_DENSE_LINE_WIDTH = 1.1
_DETERMINISTIC_SPARSE_LINE_WIDTH = 1.6
_DETERMINISTIC_LEGEND_LINE_WIDTH = 2.6
_FLOW_LEGEND_LINE_WIDTH = 2.0
_SPARSE_PAIR_COUNT_LIMIT = 10


def _category_pair_count(frame: pd.DataFrame) -> int:
    """Return the number of unique model-scenario pairs in one category slice."""
    return int(
        len(
            frame.loc[:, ["cc_model", "cc_scenario"]].astype(str).drop_duplicates(ignore_index=True)
        )
    )


def _render_ssp_figure(
    *,
    variable_name: str,
    ssp_scenario: str,
    scoped_frame: pd.DataFrame,
    study_years: list[int],
    post_years: list[int],
    output_dir: Path,
    dpi: int,
    output_format: str,
) -> list[Path]:
    categories = sorted({str(value) for value in scoped_frame["cc_category"].astype(str)})
    flow_colors = _ar6_flow_colors()
    category_colors = _ar6_category_colors(categories)
    fig, (axis, budget_axis) = plt.subplots(
        ncols=2,
        figsize=(15.5, 7.2),
        gridspec_kw={"width_ratios": [3.0, 1.22], "wspace": 0.28},
    )
    figure_title = ar6_cc_title(
        variable_name=variable_name,
        ssp_scenario=ssp_scenario,
        categories=categories,
    )
    render_figure_title(fig, figure_title)
    panel_title_pad = ar6_period_panel_title_pad(post_years)
    fig.subplots_adjust(
        top=title_layout_top(
            fig,
            figure_title,
            default_top=0.94,
            panel_title_pad=panel_title_pad,
        )
    )
    legend_handles, budget_handles, visible_negative_flow = render_deterministic_ar6_cc_row(
        axis=axis,
        budget_axis=budget_axis,
        scoped_frame=scoped_frame,
        study_years=study_years,
        post_years=post_years,
        category_colors=category_colors,
        flow_colors=flow_colors,
        pathway_title="Pathways",
        budget_title="Cumulative budgets",
        show_x_labels=True,
        show_study_label=True,
        title_pad=panel_title_pad,
    )
    if visible_negative_flow:
        legend_handles.extend(_flow_style_legend_handles())
    ncol = min(3, len(legend_handles)) if visible_negative_flow else max(1, len(legend_handles))
    render_deterministic_legends_below(
        fig,
        pathway_axis=axis,
        budget_axis=budget_axis,
        pathway_handles=legend_handles,
        budget_handles=budget_handles,
        pathway_ncol=ncol,
        budget_ncol=1,
    )
    return save_figure(
        fig,
        output_dir / ssp_partition_token(ssp_scenario),
        dpi=dpi,
        output_format=output_format,
    )


def render_deterministic_ar6_cc_row(
    *,
    axis: Any,
    budget_axis: Any,
    scoped_frame: pd.DataFrame,
    study_years: list[int],
    post_years: list[int],
    category_colors: dict[str, str],
    flow_colors: dict[str, str] | None = None,
    pathway_title: str,
    budget_title: str,
    show_x_labels: bool,
    show_study_label: bool,
    title_pad: int | None = None,
) -> tuple[list[Any], list[Any], bool]:
    """Render deterministic AR6 CC pathways and cumulative budget on supplied axes."""
    years = [*study_years, *post_years]
    legend_handles: list[Any] = []
    minimum_value: float | None = None
    visible_negative_flow = False
    categories = sorted({str(value) for value in scoped_frame["cc_category"].astype(str)})
    for category in categories:
        category_frame = scoped_frame.loc[
            scoped_frame["cc_category"].astype(str) == category
        ].copy()
        grouped = category_frame.groupby(
            ["cc_flow", "cc_model", "cc_scenario"],
            dropna=False,
            sort=True,
        )
        category_color = category_colors[category]
        pair_count = _category_pair_count(category_frame)
        line_width = (
            _DETERMINISTIC_DENSE_LINE_WIDTH
            if pair_count > _SPARSE_PAIR_COUNT_LIMIT
            else _DETERMINISTIC_SPARSE_LINE_WIDTH
        )
        line_alpha = 0.35 if pair_count > _SPARSE_PAIR_COUNT_LIMIT else 1.0
        for group_key, group in grouped:
            cc_flow = str(group_key[0])
            row = group.iloc[0]
            plot_years = years
            values = row.loc[years].to_numpy(dtype=float).tolist()
            if cc_flow == CC_FLOW_NEGATIVE and not any(value < 0.0 for value in values):
                continue
            if cc_flow == CC_FLOW_NEGATIVE:
                visible_negative_flow = True
                first_negative_index = next(
                    index for index, value in enumerate(values) if value < 0.0
                )
                plot_years = years[first_negative_index:]
                values = values[first_negative_index:]
            axis.plot(
                plot_years,
                values,
                alpha=line_alpha,
                linewidth=line_width,
                color=category_color,
                linestyle=":" if cc_flow == CC_FLOW_NEGATIVE else "-",
            )
            local_min = min(values)
            minimum_value = local_min if minimum_value is None else min(minimum_value, local_min)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=category_color,
                linewidth=_DETERMINISTIC_LEGEND_LINE_WIDTH,
                label=f"Category {category} (n={pair_count})",
            )
        )
    panel_title_pad = (
        ar6_period_panel_title_pad(post_years) if title_pad is None else int(title_pad)
    )
    axis.set_title(pathway_title, fontweight="bold", pad=panel_title_pad)
    axis.set_xlabel("")
    axis.set_ylabel(format_scientific_figure_text(str(scoped_frame["impact_unit"].iloc[0])))
    axis.yaxis.set_major_formatter(FuncFormatter(scientific_tick_formatter))
    axis.set_xlim(float(min(years)) - 0.5, float(max(years)) + 0.5)
    format_integer_year_axis(axis, years=years)
    if not show_x_labels:
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    axis.grid(alpha=0.25)
    apply_zero_floor_if_nonnegative(axis=axis, minimum_value=minimum_value)
    if post_years:
        render_study_transition(
            axis,
            study_years=study_years,
            post_years=post_years,
            show_study_label=show_study_label,
        )
    budget_handles = plot_budget_panel(
        axis=budget_axis,
        frame=scoped_frame,
        study_years=study_years,
        post_years=post_years,
        category_colors=category_colors,
        flow_colors=None,
        negative_sequestration_style="dotted",
        title=budget_title,
        title_pad=panel_title_pad,
    )
    return legend_handles, budget_handles, visible_negative_flow


def _flow_style_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=_FLOW_LEGEND_LINE_WIDTH,
            linestyle="-",
            label="Solid lines: gross emissions",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=_FLOW_LEGEND_LINE_WIDTH,
            linestyle=":",
            label="Dotted lines: negative sequestration",
        ),
    ]


def _ar6_flow_colors() -> dict[str, str]:
    return {
        CC_FLOW_NET: ar6_cc_flow_color(CC_FLOW_NET),
        CC_FLOW_POSITIVE: ar6_cc_flow_color(CC_FLOW_POSITIVE),
        CC_FLOW_NEGATIVE: ar6_cc_flow_color(CC_FLOW_NEGATIVE),
    }


def _ar6_category_colors(categories: list[str]) -> dict[str, str]:
    """Return stable deterministic AR6 CC colors by climate category."""
    return {category: ar6_category_color(category=category) for category in categories}


def render_cc_pathway_figures(
    *,
    cc_table: pd.DataFrame,
    post_study_cc_table: pd.DataFrame | None = None,
    variable_name: str,
    output_dir: Path,
    dpi: int,
    output_format: str,
    requested_years: list[int] | None = None,
    status: StatusSink | None = None,
) -> list[Path]:
    """Render deterministic dynamic CC pathway figures one file per SSP."""
    if cc_table.empty:
        return []
    figure_frame = combine_study_and_post_tables(
        cc_table=cc_table,
        post_study_cc_table=post_study_cc_table,
    )
    year_column_map = figure_year_columns(figure_frame)
    year_columns = sorted(set(year_column_map.values()))
    years = sorted(int(column) for column in year_columns)
    study_years = (
        normalized_requested_years(requested_years) if requested_years is not None else years
    )
    post_years = [year for year in years if year > max(study_years)]
    ssp_values = sorted({str(value) for value in figure_frame["ssp_scenario"].astype(str)})
    return render_with_progress(
        source="deterministic_ar6_cc",
        items=ssp_values,
        describe=lambda item: str(item),
        render=lambda ssp_scenario: _render_ssp_figure(
            variable_name=variable_name,
            ssp_scenario=str(ssp_scenario),
            scoped_frame=figure_frame.loc[
                figure_frame["ssp_scenario"].astype(str) == str(ssp_scenario)
            ],
            study_years=study_years,
            post_years=post_years,
            output_dir=output_dir,
            dpi=dpi,
            output_format=output_format,
        ),
        total=len(ssp_values),
        status=status,
    )

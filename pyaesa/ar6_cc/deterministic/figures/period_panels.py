"""Study and post study figure panels for deterministic AR6 CC."""

from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from pyaesa.ar6_cc.deterministic.request.contracts import CC_FLOW_NEGATIVE
from pyaesa.shared.figures.figure_footer import (
    legend_display_rows,
    render_two_panel_legends_below,
    reserve_footer_space,
)
from pyaesa.shared.figures.multi_year_transitions import transition_boundary_x
from pyaesa.shared.figures.nonnegative_axis import apply_zero_floor_if_nonnegative
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.scientific_ticks import scientific_tick_formatter
from pyaesa.shared.figures.violin_summary import (
    render_violin_summaries,
    violin_summary_footer_extra_height,
    violin_summary_legend_handler_map,
    violin_summary_legend_handles,
    violin_summary_legend_kwargs,
)

CC_FIGURE_ID_COLUMNS = [
    "cc_model",
    "cc_scenario",
    "cc_category",
    "ssp_scenario",
    "cc_flow",
    "cc_variable",
    "impact_unit",
]

_MIN_VIOLIN_MODEL_SCENARIO_COUNT = 2
_POST_STUDY_SHADE_COLOR = "#7d7d7d"
_POST_STUDY_SHADE_ALPHA = 0.28
_TRANSITION_COLOR = "#7d7d7d"
_BUDGET_VIOLIN_WIDTH = 0.34
_BUDGET_HISTOGRAM_WIDTH = 0.34
_BUDGET_HISTOGRAM_ALPHA = 0.72
_BUDGET_NEGATIVE_HATCH = ".."
_PERIOD_LABEL_Y_OFFSET_PT = 4.0
_PERIOD_PANEL_TITLE_PAD = 32
_NO_PERIOD_PANEL_TITLE_PAD = 8

BudgetNegativeStyle = Literal["dotted", "plain"]


def ar6_period_panel_title_pad(post_years: Sequence[int]) -> int:
    """Return panel title padding matched to visible study/post labels."""
    return _PERIOD_PANEL_TITLE_PAD if list(post_years) else _NO_PERIOD_PANEL_TITLE_PAD


def figure_year_columns(frame: pd.DataFrame) -> dict[str | int, int]:
    """Return deterministic year columns mapped to integer labels for plotting."""
    return {
        column: int(column)
        for column in frame.columns
        if str(column).isdigit() and 1900 < int(column) < 2200
    }


def combine_study_and_post_tables(
    *,
    cc_table: pd.DataFrame,
    post_study_cc_table: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return one wide plotting frame spanning study and post study years."""
    study = _normalize_year_columns(cc_table)
    if post_study_cc_table is None or post_study_cc_table.empty:
        return study
    post = _normalize_year_columns(post_study_cc_table)
    post_years = sorted(figure_year_columns(post).values())
    post_columns = [*CC_FIGURE_ID_COLUMNS, *post_years]
    return study.merge(
        post.loc[:, post_columns],
        on=CC_FIGURE_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )


def render_study_transition(
    axis,
    *,
    study_years: list[int],
    post_years: list[int],
    show_study_label: bool = True,
) -> None:
    """Render the study to post study period divider on one pathway axis."""
    boundary = transition_boundary_x(int(post_years[0]))
    _render_transition(axis, boundary=boundary)
    _render_period_labels(
        axis,
        study_x=_period_midpoint(study_years),
        post_x=_period_midpoint(post_years),
        study_years=study_years,
        post_years=post_years,
        show_study_label=show_study_label,
    )


def plot_budget_panel(
    *,
    axis,
    frame: pd.DataFrame,
    study_years: list[int],
    post_years: list[int],
    category_colors: dict[str, str],
    negative_sequestration_style: BudgetNegativeStyle,
    flow_colors: dict[str, str] | None = None,
    title: str = "Cumulative budgets",
    title_pad: int | None = None,
) -> list[Any]:
    """Render the deterministic cumulative budget panel."""
    entries = _budget_entries(frame=frame, study_years=study_years, post_years=post_years)
    positions_by_period = _period_positions(study_years=study_years, post_years=post_years)
    any_violin = False
    any_histogram = False
    minimum_value: float | None = None
    for category, flow, period_segment, values in entries:
        position = positions_by_period[period_segment]
        color = (flow_colors or {}).get(str(flow), category_colors[str(category)])
        numeric = values[np.isfinite(values)]
        local_minimum = float(np.min(numeric))
        minimum_value = (
            local_minimum if minimum_value is None else min(minimum_value, local_minimum)
        )
        if len(numeric) >= _MIN_VIOLIN_MODEL_SCENARIO_COUNT:
            any_violin = True
            render_violin_summaries(
                axis,
                values=[numeric],
                positions=np.asarray([position], dtype=float),
                colors=[color],
                width=_BUDGET_VIOLIN_WIDTH,
            )
            continue
        any_histogram = True
        _render_budget_histogram(
            axis=axis,
            position=float(position),
            values=numeric,
            color=color,
            negative_sequestration=flow == CC_FLOW_NEGATIVE,
            negative_sequestration_style=negative_sequestration_style,
        )
    axis.set_xlim(
        float(min(positions_by_period.values())) - 0.5,
        float(max(positions_by_period.values())) + 0.5,
    )
    _hide_x_axis(axis)
    if post_years:
        _render_transition(
            axis,
            boundary=0.5,
            shade_zorder=6,
            line_zorder=7,
        )
        _render_period_labels(
            axis,
            study_x=positions_by_period["study"],
            post_x=positions_by_period["post_study_period"],
            study_years=study_years,
            post_years=post_years,
            show_study_label=True,
        )
    panel_title_pad = (
        ar6_period_panel_title_pad(post_years) if title_pad is None else int(title_pad)
    )
    axis.set_title(title, fontweight="bold", pad=panel_title_pad)
    axis.set_xlabel("")
    axis.set_ylabel(
        format_scientific_figure_text(_budget_unit_label(str(frame["impact_unit"].iloc[0])))
    )
    axis.yaxis.set_major_formatter(FuncFormatter(scientific_tick_formatter))
    axis.grid(alpha=0.25, axis="y")
    apply_zero_floor_if_nonnegative(axis=axis, minimum_value=minimum_value)
    handles: list[Any] = []
    if any_violin:
        handles.extend(violin_summary_legend_handles())
    if any_histogram and any_violin:
        handles.append(
            Patch(
                facecolor="#777777",
                edgecolor="#444444",
                alpha=_BUDGET_HISTOGRAM_ALPHA,
                label="Histogram: model-scenario budget distribution",
            )
        )
    return handles


def _render_budget_histogram(
    *,
    axis,
    position: float,
    values: np.ndarray,
    color: str,
    negative_sequestration: bool,
    negative_sequestration_style: BudgetNegativeStyle,
) -> None:
    numeric = values.astype(np.float64, copy=False)
    center = float(numeric[0])
    dotted_negative = negative_sequestration and negative_sequestration_style == "dotted"
    facecolor = "none" if dotted_negative else color
    edgecolor = color if negative_sequestration else "white"
    alpha = 1.0 if dotted_negative else _BUDGET_HISTOGRAM_ALPHA
    axis.bar(
        [position],
        [center],
        width=_BUDGET_HISTOGRAM_WIDTH,
        bottom=0.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=0.9 if negative_sequestration else 0.45,
        linestyle=":" if dotted_negative else "-",
        hatch=_BUDGET_NEGATIVE_HATCH if dotted_negative else None,
        zorder=4,
    )


def render_deterministic_legends_below(
    fig,
    *,
    pathway_axis,
    budget_axis,
    pathway_handles: list[Any],
    budget_handles: list[Any],
    pathway_ncol: int,
    budget_ncol: int,
) -> None:
    """Render separate pathway and budget legends under their panels."""
    has_violin = any(
        handle.__class__.__name__ == "ViolinSummaryLegendHandle" for handle in budget_handles
    )
    if not budget_handles:
        _render_centered_pathway_legend(
            fig,
            pathway_handles=pathway_handles,
            pathway_ncol=pathway_ncol,
        )
        return
    render_two_panel_legends_below(
        fig,
        left_axis=pathway_axis,
        right_axis=budget_axis,
        left_handles=pathway_handles,
        right_handles=budget_handles,
        left_ncol=pathway_ncol,
        right_ncol=budget_ncol,
        left_title="AR6 category | AR6 CC model-scenario pairs",
        right_handler_map=violin_summary_legend_handler_map() if has_violin else None,
        right_legend_kwargs=violin_summary_legend_kwargs() if has_violin else None,
        extra_height_in=violin_summary_footer_extra_height() if has_violin else 0.0,
        title_rows=1,
    )


def _render_centered_pathway_legend(
    fig,
    *,
    pathway_handles: list[Any],
    pathway_ncol: int,
) -> None:
    labels = [
        format_scientific_figure_text(str(handle.get_label()).strip()) for handle in pathway_handles
    ]
    layout = reserve_footer_space(
        fig,
        rows=legend_display_rows(labels, ncol=pathway_ncol),
        note_lines=0,
        title_rows=1,
    )
    fig.canvas.draw()
    fig.legend(
        handles=pathway_handles,
        labels=labels,
        title=format_scientific_figure_text("AR6 category | AR6 CC model-scenario pairs"),
        loc="upper center",
        bbox_to_anchor=(0.5, layout.top_y),
        ncol=int(pathway_ncol),
        frameon=True,
        fontsize="small",
    )


def _normalize_year_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [
        int(column) if isinstance(column, str) and column.isdigit() else column
        for column in renamed.columns
    ]
    return renamed


def _budget_entries(
    *,
    frame: pd.DataFrame,
    study_years: list[int],
    post_years: list[int],
) -> list[tuple[str, str, str, np.ndarray]]:
    segments: list[tuple[str, list[int]]] = [("study", study_years)]
    if post_years:
        segments.append(("post_study_period", post_years))
    entries: list[tuple[str, str, str, np.ndarray]] = []
    for key, group in frame.groupby(
        ["cc_category", "cc_flow"],
        dropna=False,
        sort=True,
    ):
        category, _flow = cast(tuple[object, object], key)
        category_text = str(category)
        flow_text = str(_flow)
        for segment_label, segment_years in segments:
            values = (
                group.loc[:, segment_years]
                .apply(pd.to_numeric, errors="raise")
                .sum(axis=1)
                .to_numpy(dtype=float)
            )
            entries.append((category_text, flow_text, segment_label, values))
    return entries


def _budget_unit_label(unit: str) -> str:
    return str(unit).replace("/yr", "").replace(" yr^-1", "").strip()


def _period_positions(*, study_years: list[int], post_years: list[int]) -> dict[str, float]:
    del study_years
    positions = {"study": 0.0}
    if post_years:
        positions["post_study_period"] = 1.0
    return positions


def _period_midpoint(years: list[int]) -> float:
    return 0.5 * (float(years[0]) + float(years[-1]))


def _render_transition(
    axis,
    *,
    boundary: float,
    shade_zorder: int = 0,
    line_zorder: int = 1,
) -> None:
    left, right = axis.get_xlim()
    axis.axvspan(
        float(boundary),
        float(right) + max(1e-9, abs(float(right) - float(left)) * 0.02),
        facecolor=_POST_STUDY_SHADE_COLOR,
        alpha=_POST_STUDY_SHADE_ALPHA,
        linewidth=0,
        zorder=shade_zorder,
        clip_on=True,
    )
    axis.axvline(
        float(boundary),
        color=_TRANSITION_COLOR,
        linestyle=":",
        linewidth=1.2,
        alpha=0.9,
        zorder=line_zorder,
    )
    axis.set_xlim(left, right)


def _render_period_labels(
    axis,
    *,
    study_x: float,
    post_x: float,
    study_years: list[int],
    post_years: list[int],
    show_study_label: bool,
) -> None:
    transform = axis.get_xaxis_transform() + transforms.ScaledTranslation(
        0.0,
        _PERIOD_LABEL_Y_OFFSET_PT / 72.0,
        axis.figure.dpi_scale_trans,
    )
    if show_study_label:
        axis.text(
            float(study_x),
            1.0,
            f"study period\n({study_years[0]}-{study_years[-1]})",
            color=_TRANSITION_COLOR,
            ha="center",
            va="bottom",
            fontsize=8,
            transform=transform,
            clip_on=False,
            zorder=8,
        )
    axis.text(
        float(post_x),
        1.0,
        f"post study period\n({post_years[0]}-{post_years[-1]})",
        color=_TRANSITION_COLOR,
        ha="center",
        va="bottom",
        fontsize=8,
        transform=transform,
        clip_on=False,
        zorder=8,
    )


def _hide_x_axis(axis) -> None:
    axis.set_xticks([])
    axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

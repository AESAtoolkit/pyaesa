"""Rendering policy for AR6 CC uncertainty figure products."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from pyaesa.ar6_cc.deterministic.request.contracts import (
    CC_FLOW_NEGATIVE,
    CC_FLOW_NET,
    CC_FLOW_POSITIVE,
)
from pyaesa.ar6_cc.uncertainty.figures.period_panels import (
    ar6_period_panel_title_pad,
    render_study_transition,
    render_uncertainty_legends_below,
    plot_uncertainty_budget_panel,
)
from pyaesa.ar6_cc.shared.runtime.figure_style import ar6_category_color, ar6_cc_flow_color
from pyaesa.ar6_cc.shared.runtime.figure_titles import ar6_cc_title
from pyaesa.shared.figures.layout import format_integer_year_axis
from pyaesa.shared.figures.nonnegative_axis import apply_zero_floor_if_nonnegative
from pyaesa.shared.figures.save import save_figure
from pyaesa.shared.figures.scientific_ticks import scientific_tick_formatter
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.trajectory_bands import (
    SUMMARY_COLUMNS,
    render_trajectory_band,
    trajectory_band_legend_handles,
)
from pyaesa.shared.figures.titles import render_figure_title, title_layout_top

_BAND_ALPHA_OUTER = 0.10
_BAND_ALPHA_INNER = 0.32
_NEUTRAL_BAND_LEGEND_COLOR = "#555555"
FLOW_COLORS = {
    CC_FLOW_NET: ar6_cc_flow_color(CC_FLOW_NET),
    CC_FLOW_POSITIVE: ar6_cc_flow_color(CC_FLOW_POSITIVE),
    CC_FLOW_NEGATIVE: ar6_cc_flow_color(CC_FLOW_NEGATIVE),
}
_FLOW_ORDER = {
    CC_FLOW_NET: 0,
    CC_FLOW_POSITIVE: 0,
    CC_FLOW_NEGATIVE: 1,
}


def plot_trajectory_band_scope(
    *,
    frame: pd.DataFrame,
    budget_frame: pd.DataFrame,
    output_stem: Path,
    title_categories: list[str],
    variable_name: str,
    ssp_scenario: str,
    pair_count: int,
    sampling_method: str,
    study_years: list[int],
    dpi: int,
    output_format: str,
) -> list[Path]:
    """Render one AR6 CC multi-year uncertainty scope."""
    scoped = frame.copy()
    scoped["year"] = _numeric_series(frame=scoped, column="year").astype(int)
    scoped = scoped.sort_values("year", kind="stable").reset_index(drop=True)
    fig, (axis, budget_axis) = plt.subplots(
        ncols=2,
        figsize=(15.5, 7.2),
        gridspec_kw={"width_ratios": [3.0, 1.22], "wspace": 0.28},
    )
    figure_title = ar6_cc_title(
        variable_name=variable_name,
        ssp_scenario=ssp_scenario,
        categories=title_categories,
    )
    render_figure_title(fig, figure_title)
    visible_years = sorted({int(year) for year in scoped["year"].tolist()})
    post_years = [year for year in visible_years if year > max(study_years)]
    panel_title_pad = ar6_period_panel_title_pad(post_years)
    fig.subplots_adjust(
        top=title_layout_top(
            fig,
            figure_title,
            default_top=0.94,
            panel_title_pad=panel_title_pad,
        )
    )
    pathway_handles, budget_handles, legend_color, use_flow_colors = render_uncertainty_ar6_cc_row(
        axis=axis,
        budget_axis=budget_axis,
        scoped_frame=scoped,
        budget_frame=budget_frame,
        flow_colors=FLOW_COLORS,
        study_years=study_years,
        pathway_title="Pathways",
        budget_title="Cumulative budgets",
        show_x_labels=True,
        show_study_label=True,
        title_pad=panel_title_pad,
    )
    labels = [str(handle.get_label()) for handle in pathway_handles]
    labels.extend(
        str(handle.get_label())
        for handle in trajectory_band_legend_handles(
            color=legend_color,
            pair_count=pair_count,
            sampling_method=sampling_method,
            outer_alpha=_BAND_ALPHA_OUTER,
            inner_alpha=_BAND_ALPHA_INNER,
        )
    )
    ncol = _legend_column_count(
        labels=labels,
        has_category_handles=any(
            isinstance(handle, Line2D) and str(handle.get_label()).startswith("Category ")
            for handle in pathway_handles
        ),
        has_flow_colors=use_flow_colors,
    )
    render_uncertainty_legends_below(
        fig,
        pathway_axis=axis,
        budget_axis=budget_axis,
        trajectory_color=legend_color,
        pathway_handles=pathway_handles,
        budget_handles=budget_handles,
        pair_count=pair_count,
        sampling_method=sampling_method,
        pathway_ncol=ncol,
        budget_ncol=1,
        outer_alpha=_BAND_ALPHA_OUTER,
        inner_alpha=_BAND_ALPHA_INNER,
    )
    return save_figure(
        fig,
        output_stem,
        dpi=dpi,
        output_format=output_format,
    )


def render_uncertainty_ar6_cc_row(
    *,
    axis: Any,
    budget_axis: Any,
    scoped_frame: pd.DataFrame,
    budget_frame: pd.DataFrame,
    flow_colors: dict[str, str],
    study_years: list[int],
    pathway_title: str,
    budget_title: str,
    show_x_labels: bool,
    show_study_label: bool,
    title_pad: int | None = None,
) -> tuple[list[Any], list[Any], str, bool]:
    """Render uncertainty AR6 CC pathways and cumulative budget on supplied axes."""
    category_handles: list[Any] = []
    minimum_p5: float | None = None
    visible_years: list[int] = []
    category_groups = _category_groups(scoped_frame)
    category_colors = _category_colors(frame=scoped_frame, groups=category_groups)
    use_flow_colors = _uses_flow_colors(scoped_frame)
    legend_color = (
        _NEUTRAL_BAND_LEGEND_COLOR if use_flow_colors else next(iter(category_colors.values()))
    )
    for category, group in category_groups:
        for flow, flow_group in _flow_groups(group):
            color = _flow_color(flow) if use_flow_colors else category_colors[category]
            years = np.asarray(
                pd.Series(flow_group.loc[:, "year"], copy=False).to_numpy(),
                dtype=np.int64,
            )
            values = render_trajectory_band(
                axis,
                years=years,
                summaries={column: flow_group[column] for column in SUMMARY_COLUMNS},
                color=color,
                line_alpha=1.0,
                mean_linewidth=2.2,
                median_linewidth=1.6,
                outer_alpha=_BAND_ALPHA_OUTER,
                inner_alpha=_BAND_ALPHA_INNER,
            )
            visible_years.extend(int(year) for year in years.tolist())
            local_minimum = float(np.nanmin(values["p5"]))
            minimum_p5 = local_minimum if minimum_p5 is None else min(minimum_p5, local_minimum)
        if len(category_groups) > 1 and not use_flow_colors:
            color = category_colors[category]
            category_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2.2,
                    label=f"Category {category}",
                )
            )
    post_years = [
        year for year in sorted({int(value) for value in visible_years}) if year > max(study_years)
    ]
    panel_title_pad = (
        ar6_period_panel_title_pad(post_years) if title_pad is None else int(title_pad)
    )
    axis.set_title(pathway_title, fontweight="bold", pad=panel_title_pad)
    axis.set_xlabel("")
    axis.set_ylabel(format_scientific_figure_text(str(scoped_frame["impact_unit"].iloc[0])))
    axis.yaxis.set_major_formatter(FuncFormatter(scientific_tick_formatter))
    axis.grid(alpha=0.25)
    axis.set_xlim(float(min(visible_years)) - 0.5, float(max(visible_years)) + 0.5)
    _format_year_axis(axis, np.asarray(visible_years, dtype=np.int64))
    if not show_x_labels:
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    apply_zero_floor_if_nonnegative(axis=axis, minimum_value=minimum_p5)
    if post_years:
        render_study_transition(
            axis,
            study_years=study_years,
            post_years=post_years,
            show_study_label=show_study_label,
        )
    budget_handles = plot_uncertainty_budget_panel(
        axis=budget_axis,
        frame=budget_frame,
        flow_colors=flow_colors,
        study_years=study_years,
        post_years=post_years,
        title=budget_title,
        title_pad=panel_title_pad,
    )
    pathway_handles = [
        *category_handles,
        *(_flow_legend_handles() if use_flow_colors else []),
    ]
    return pathway_handles, budget_handles, legend_color, use_flow_colors


def _category_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Return category sorted plotting groups for one figure."""
    if "cc_category" not in frame.columns:
        return [("AR6 CC", frame)]
    groups: list[tuple[str, pd.DataFrame]] = []
    for category, group in frame.groupby("cc_category", dropna=False, sort=True):
        groups.append((str(category), group.copy()))
    return groups


def _category_colors(
    *,
    frame: pd.DataFrame,
    groups: list[tuple[str, pd.DataFrame]],
) -> dict[str, str]:
    """Return stable AR6 category colors for category resolved figures."""
    if "cc_category" not in frame.columns:
        return {category: FLOW_COLORS[CC_FLOW_NET] for category, _group in groups}
    return {category: ar6_category_color(category=category) for category, _group in groups}


def _flow_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Return flow sorted plotting groups inside one category scope."""
    groups: list[tuple[str, pd.DataFrame]] = []
    for flow, group in frame.groupby("cc_flow", dropna=False, sort=True):
        groups.append((str(flow), group.sort_values("year", kind="stable").copy()))
    return sorted(groups, key=lambda item: (_FLOW_ORDER.get(item[0], 99), item[0]))


def _uses_flow_colors(frame: pd.DataFrame) -> bool:
    """Return whether flow color coding is required for this figure."""
    return CC_FLOW_NEGATIVE in set(frame["cc_flow"].astype(str))


def _flow_color(flow: str) -> str:
    """Return the AR6 CC uncertainty band color for one flow."""
    return FLOW_COLORS[str(flow)]


def _flow_legend_handles() -> list[Patch]:
    """Return flow color legend handles for gross mode uncertainty figures."""
    return [
        Patch(
            facecolor=FLOW_COLORS[CC_FLOW_POSITIVE],
            alpha=_BAND_ALPHA_INNER,
            label="Gross emissions",
        ),
        Patch(
            facecolor=FLOW_COLORS[CC_FLOW_NEGATIVE],
            alpha=_BAND_ALPHA_INNER,
            label="Negative sequestration",
        ),
    ]


def _legend_column_count(
    *,
    labels: list[str],
    has_category_handles: bool,
    has_flow_colors: bool,
) -> int:
    """Return footer legend columns for AR6 CC uncertainty figures."""
    if has_flow_colors:
        return min(3, len(labels))
    if has_category_handles:
        return min(4, len(labels))
    return len(labels)


def _format_year_axis(axis, years: np.ndarray) -> None:
    unique_years = sorted({int(year) for year in years.tolist()})
    if not unique_years:
        return
    format_integer_year_axis(axis, years=unique_years)


def _numeric_series(*, frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one numeric frame column as a pandas Series."""
    return pd.Series(
        pd.to_numeric(pd.Series(frame.loc[:, column], copy=False), errors="raise"),
        index=frame.index,
    )

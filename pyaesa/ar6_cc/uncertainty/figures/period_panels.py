"""Study and post study figure panels for AR6 CC uncertainty."""

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from pyaesa.shared.figures.figure_footer import render_two_panel_legends_below
from pyaesa.shared.figures.multi_year_transitions import transition_boundary_x
from pyaesa.shared.figures.nonnegative_axis import apply_zero_floor_if_nonnegative
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.scientific_ticks import scientific_tick_formatter
from pyaesa.shared.figures.trajectory_bands import trajectory_band_legend_handles
from pyaesa.shared.figures.violin_summary import (
    render_violin_summaries,
    violin_summary_footer_extra_height,
    violin_summary_legend_handler_map,
    violin_summary_legend_handles,
    violin_summary_legend_kwargs,
)

_POST_STUDY_SHADE_COLOR = "#7d7d7d"
_POST_STUDY_SHADE_ALPHA = 0.28
_TRANSITION_COLOR = "#7d7d7d"
_BUDGET_VIOLIN_WIDTH = 0.34
_PERIOD_LABEL_Y_OFFSET_PT = 4.0
_PERIOD_PANEL_TITLE_PAD = 32
_NO_PERIOD_PANEL_TITLE_PAD = 8


def ar6_period_panel_title_pad(post_years: Sequence[int]) -> int:
    """Return panel title padding matched to visible study/post labels."""
    return _PERIOD_PANEL_TITLE_PAD if list(post_years) else _NO_PERIOD_PANEL_TITLE_PAD


def render_study_transition(
    axis,
    *,
    study_years: Sequence[int],
    post_years: Sequence[int],
    show_study_label: bool = True,
) -> None:
    """Render the study to post study period divider on one uncertainty axis."""
    post_values = sorted(int(year) for year in post_years)
    study_values = sorted(int(year) for year in study_years)
    boundary = transition_boundary_x(post_values[0])
    _render_transition(axis, boundary=boundary)
    _render_period_labels(
        axis,
        study_x=_period_midpoint(study_values),
        post_x=_period_midpoint(post_values),
        study_years=study_values,
        post_years=post_values,
        show_study_label=show_study_label,
    )


def plot_uncertainty_budget_panel(
    *,
    axis,
    frame: pd.DataFrame,
    flow_colors: Mapping[str, str],
    study_years: Sequence[int],
    post_years: Sequence[int],
    title: str = "Cumulative budgets",
    title_pad: int | None = None,
) -> list[object]:
    """Render exact cumulative budget run distributions for one figure scope."""
    rows = _budget_entries(frame)
    positions_by_period = _period_positions(
        study_years=study_years,
        post_years=post_years,
    )
    positions = np.asarray(
        [positions_by_period[str(row["period_segment"])] for row in rows],
        dtype=float,
    )
    colors = [flow_colors[str(row["cc_flow"])] for row in rows]
    values = [np.asarray(row["values"], dtype=np.float64) for row in rows]
    finite_values = np.concatenate([value[np.isfinite(value)] for value in values])
    render_violin_summaries(
        axis,
        values=values,
        positions=positions,
        colors=colors,
        width=_BUDGET_VIOLIN_WIDTH,
    )
    axis.set_xlim(
        float(min(positions_by_period.values())) - 0.5,
        float(max(positions_by_period.values())) + 0.5,
    )
    _hide_x_axis(axis)
    post_values = sorted(int(year) for year in post_years)
    if post_values:
        _render_transition(axis, boundary=0.5)
        _render_period_labels(
            axis,
            study_x=positions_by_period["study_period"],
            post_x=positions_by_period["post_study_period"],
            study_years=sorted(int(year) for year in study_years),
            post_years=post_values,
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
    apply_zero_floor_if_nonnegative(axis=axis, minimum_value=float(np.min(finite_values)))
    return violin_summary_legend_handles()


def render_uncertainty_legends_below(
    fig,
    *,
    pathway_axis: Any,
    budget_axis: Any,
    trajectory_color: Any,
    pathway_handles: Sequence[Any],
    budget_handles: Sequence[Any],
    pair_count: int,
    sampling_method: str,
    pathway_ncol: int,
    budget_ncol: int,
    outer_alpha: float,
    inner_alpha: float,
) -> None:
    """Render separate pathway and budget legends under their panels."""
    pathway_entries = [
        *list(pathway_handles),
        *trajectory_band_legend_handles(
            color=trajectory_color,
            pair_count=pair_count,
            sampling_method=sampling_method,
            outer_alpha=outer_alpha,
            inner_alpha=inner_alpha,
        ),
    ]
    budget_entries = list(budget_handles)
    render_two_panel_legends_below(
        fig,
        left_axis=pathway_axis,
        right_axis=budget_axis,
        left_handles=pathway_entries,
        right_handles=budget_entries,
        left_ncol=pathway_ncol,
        right_ncol=budget_ncol,
        right_handler_map=violin_summary_legend_handler_map(),
        right_legend_kwargs=violin_summary_legend_kwargs(),
        extra_height_in=violin_summary_footer_extra_height(),
    )


def _budget_entries(frame: pd.DataFrame) -> list[dict[str, object]]:
    has_category = "cc_category" in frame.columns
    sort_columns = [
        *([] if not has_category else ["cc_category"]),
        "period_segment",
        "cc_flow",
    ]
    rows: list[dict[str, object]] = []
    for _key, group in frame.sort_values(sort_columns, kind="stable").groupby(
        sort_columns,
        dropna=False,
        sort=False,
    ):
        row = pd.Series(group.iloc[0], copy=False)
        rows.append(
            {
                "cc_flow": str(row["cc_flow"]),
                "period_segment": str(row["period_segment"]),
                "values": row["__budget_values"],
            }
        )
    return rows


def _budget_unit_label(unit: str) -> str:
    return str(unit).replace("/yr", "").replace(" yr^-1", "").strip()


def _period_positions(
    *,
    study_years: Sequence[int],
    post_years: Sequence[int],
) -> dict[str, float]:
    del study_years
    positions = {"study_period": 0.0}
    if post_years:
        positions["post_study_period"] = 1.0
    return positions


def _period_midpoint(years: Sequence[int]) -> float:
    return 0.5 * (float(years[0]) + float(years[-1]))


def _render_transition(axis, *, boundary: float) -> None:
    left, right = axis.get_xlim()
    axis.axvspan(
        float(boundary),
        float(right) + max(1e-9, abs(float(right) - float(left)) * 0.02),
        facecolor=_POST_STUDY_SHADE_COLOR,
        alpha=_POST_STUDY_SHADE_ALPHA,
        linewidth=0,
        zorder=0,
        clip_on=True,
    )
    axis.axvline(
        float(boundary),
        color=_TRANSITION_COLOR,
        linestyle=":",
        linewidth=1.2,
        alpha=0.9,
        zorder=1,
    )
    axis.set_xlim(left, right)


def _render_period_labels(
    axis,
    *,
    study_x: float,
    post_x: float,
    study_years: Sequence[int],
    post_years: Sequence[int],
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
    )


def _hide_x_axis(axis) -> None:
    axis.set_xticks([])
    axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

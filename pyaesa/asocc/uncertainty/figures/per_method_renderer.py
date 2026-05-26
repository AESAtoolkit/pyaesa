"""Per method aSoCC uncertainty figure job planning."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd

from pyaesa.asocc.uncertainty.figures.scope_planner import (
    SUMMARY_STAT_COLUMNS,
    VALUE_ARRAY_COLUMN,
    FigureContext,
    scoped_stem,
    scope_title,
    single_requested_year,
    visible_values,
)
from pyaesa.shared.figures.jobs import PlannedFigureJob
from pyaesa.shared.figures.scenario_scopes import (
    preplanned_scenario_scope_slices,
    repeat_invariant_rows_into_scenarios,
)
from pyaesa.shared.figures.selector_slices import selector_slices
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.tabular.scalars import is_display_missing


def plan_per_method_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    plotter: Callable[..., list[Path]],
    kind: str,
) -> Iterator[PlannedFigureJob]:
    """Plan one uncertainty figure per method and final scope."""
    for method in _methods(rows):
        method_rows = rows.loc[rows["__method"].astype(str).eq(method)].copy()
        for scoped_rows in figure_ssp_slices(method_rows, context=context):
            for lcia_rows in _per_method_lcia_slices(scoped_rows, context=context):
                for selector_token, selector_title, selector_rows in selector_slices(lcia_rows):
                    output_base = (
                        context.figures_root
                        / "per_method"
                        / scoped_stem(
                            method,
                            selector_rows,
                            include_impact=False,
                            selector_token=selector_token,
                            studied_year=single_requested_year(context),
                        )
                    )
                    title = scope_title(
                        method,
                        selector_rows,
                        selector_title=selector_title,
                        studied_year=single_requested_year(context),
                    )
                    yield PlannedFigureJob(
                        kind=kind,
                        label=output_base.name,
                        render=_planned_plot(
                            plotter=plotter,
                            frame=selector_rows,
                            output_stem=output_base,
                            title=title,
                            context=context,
                            group_legend=False,
                            include_impact_in_label=True,
                            include_method_in_label=False,
                        ),
                    )


def figure_ssp_slices(frame: pd.DataFrame, *, context: FigureContext) -> Iterator[pd.DataFrame]:
    """Yield final SSP slices from a prepared figure frame."""
    if "__figure_ssp_scope" in frame.columns:
        return preplanned_scenario_scope_slices(
            frame,
            scenario_column=ASOCC_SSP_SCENARIO_COLUMN,
            scope_column="__figure_ssp_scope",
            identity_excluded_columns=_ssp_scope_identity_exclusions(),
        )
    return repeat_invariant_rows_into_scenarios(
        frame,
        scenario_column=ASOCC_SSP_SCENARIO_COLUMN,
        scope_column="__figure_ssp_scope",
        requested_scenarios=context.requested_ssps,
        identity_excluded_columns=_ssp_scope_identity_exclusions(),
    )


def _ssp_scope_identity_exclusions() -> set[str]:
    return {*SUMMARY_STAT_COLUMNS, VALUE_ARRAY_COLUMN}


def lcia_slices(frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield per method LCIA method slices."""
    values = visible_values(frame, "lcia_method")
    if not values:
        yield frame.copy()
        return
    for value in values:
        yield frame.loc[frame["lcia_method"].astype(str).eq(value)].copy()


def _per_method_lcia_slices(
    frame: pd.DataFrame, *, context: FigureContext
) -> Iterator[pd.DataFrame]:
    del context
    return lcia_slices(frame)


def _methods(rows: pd.DataFrame) -> list[str]:
    return sorted(
        {
            str(value).strip()
            for value in rows["__method"].tolist()
            if not is_display_missing(value) and str(value).strip()
        }
    )


def _planned_plot(
    *,
    plotter: Callable[..., list[Path]],
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    context: FigureContext,
    group_legend: bool,
    include_impact_in_label: bool,
    include_method_in_label: bool = True,
) -> Callable[[], list[Path]]:
    def _render() -> list[Path]:
        return plotter(
            frame=frame,
            output_stem=output_stem,
            title=title,
            dpi=context.figure_dpi,
            output_format=context.figure_output_format,
            group_legend=group_legend,
            include_impact_in_label=include_impact_in_label,
            include_method_in_label=include_method_in_label,
        )

    return _render

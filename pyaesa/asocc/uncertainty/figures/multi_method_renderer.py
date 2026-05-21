"""Multi-method aSoCC uncertainty figure job planning."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from pyaesa.asocc.figures.multi_method_renderer import (
    expand_generic_lcia_rows,
    lcia_impact_slices,
)
from pyaesa.asocc.figures.multi_method_renderer import (
    plan_multi_method_jobs as plan_deterministic_multi_method_jobs,
)
from pyaesa.asocc.figures.product_renderers import prepare_plot_rows
from pyaesa.asocc.uncertainty.figures.per_method_renderer import (
    _planned_plot,
    figure_ssp_slices,
    lcia_slices,
)
from pyaesa.asocc.uncertainty.figures.row_reader import deterministic_mean_rows
from pyaesa.asocc.uncertainty.figures.scope_planner import (
    FigureContext,
    scoped_stem,
    scope_title,
    single_requested_year,
    visible_values,
)
from pyaesa.shared.figures.jobs import PlannedFigureJob
from pyaesa.shared.figures.selector_slices import selector_slices


def plan_multi_method_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    plotter: Callable[..., list[Path]],
    kind: str,
) -> list[PlannedFigureJob]:
    """Plan aSoCC uncertainty method comparison products."""
    jobs: list[PlannedFigureJob] = []
    studied_year = single_requested_year(context)
    for scoped_rows in figure_ssp_slices(expand_generic_lcia_rows(rows), context=context):
        for lcia_rows in lcia_slices(scoped_rows):
            for selector_token, selector_title, selector_rows in selector_slices(lcia_rows):
                product_scopes = (
                    [selector_rows]
                    if studied_year is not None
                    else lcia_impact_slices(selector_rows)
                )
                include_impact = (
                    studied_year is None and len(visible_values(selector_rows, "impact")) > 1
                )
                for impact_rows in product_scopes:
                    output_base = (
                        context.figures_root
                        / "multi_method"
                        / scoped_stem(
                            "multi_method",
                            impact_rows,
                            include_impact=include_impact,
                            selector_token=selector_token,
                            studied_year=studied_year,
                        )
                    )
                    title = scope_title(
                        None,
                        impact_rows,
                        selector_title=selector_title,
                        studied_year=studied_year,
                    )
                    jobs.append(
                        PlannedFigureJob(
                            kind=kind,
                            label=output_base.name,
                            render=_planned_plot(
                                plotter=plotter,
                                frame=impact_rows,
                                output_stem=output_base,
                                title=title,
                                context=context,
                                group_legend=True,
                                include_impact_in_label=False,
                            ),
                        ),
                    )
    return jobs


def plan_multi_method_mean_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    plotter: Callable[..., list[Path]],
) -> list[PlannedFigureJob]:
    """Plan multi-method mean products through deterministic aSoCC visuals."""
    scoped = [
        deterministic_mean_rows(rows=scoped_rows)
        for scoped_rows in figure_ssp_slices(rows, context=context)
    ]
    mean_rows = pd.concat(
        scoped,
        ignore_index=True,
    )
    return plan_deterministic_multi_method_jobs(
        rows=mean_rows,
        figures_root=context.figures_root,
        requested_years=list(context.requested_years),
        dpi=context.figure_dpi,
        output_format=context.figure_output_format,
        plotter=plotter,
        row_preparer=prepare_plot_rows,
    )


def plan_inter_method_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    plotter: Callable[..., list[Path]],
    kind: str,
) -> list[PlannedFigureJob]:
    """Plan method invariant inter-method uncertainty products."""
    jobs: list[PlannedFigureJob] = []
    studied_year = single_requested_year(context)
    for scoped_rows in figure_ssp_slices(rows, context=context):
        for impact_rows in _inter_method_lcia_slices(scoped_rows, context=context):
            for selector_token, selector_title, selector_rows in selector_slices(impact_rows):
                output_base = (
                    context.figures_root
                    / "inter_method"
                    / scoped_stem(
                        "inter_method",
                        selector_rows,
                        include_impact=False,
                        selector_token=selector_token,
                        studied_year=studied_year,
                    )
                )
                title = scope_title(
                    "inter-method",
                    selector_rows,
                    selector_title=selector_title,
                    studied_year=studied_year,
                )
                jobs.append(
                    PlannedFigureJob(
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
                    ),
                )
    return jobs


def _inter_method_lcia_slices(frame: pd.DataFrame, *, context: FigureContext) -> list[pd.DataFrame]:
    del context
    return lcia_slices(frame)

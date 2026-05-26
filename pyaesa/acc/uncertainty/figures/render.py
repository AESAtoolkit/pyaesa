"""aCC uncertainty figure orchestration."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd

from pyaesa.acc.figures.common import (
    DYNAMIC_CC_TYPE,
    acc_scope_stem,
    ordered_impacts,
    scope_slices,
    scope_title,
    static_asocc_ssp_slices,
    visible_values,
)
from pyaesa.acc.uncertainty.figures.axis_validator import validate_inactive_axes_for_figures
from pyaesa.acc.uncertainty.figures.metadata import clear_uncertainty_figure_scope
from pyaesa.acc.uncertainty.figures.product_renderers import (
    plot_band_scope,
    plot_mean_line_scope,
)
from pyaesa.acc.uncertainty.figures.violin_renderers import plot_violin_scope
from pyaesa.acc.uncertainty.figures.row_reader import (
    attach_dynamic_budget_values,
    collapsed_value_rows,
    prepared_identity_rows,
    prepared_summary_rows,
    read_figure_tables,
    value_rows_from_runs,
)
from pyaesa.acc.uncertainty.evaluation.summary import (
    ACC_SUMMARY_SCOPE_COLUMN,
    ACC_SUMMARY_SCOPE_INTER_METHOD,
    ACC_SUMMARY_SCOPE_PER_METHOD,
)
from pyaesa.acc.uncertainty.figures.scope_planner import (
    FigureContext,
    build_figure_context,
    single_requested_year,
)
from pyaesa.acc.uncertainty.runtime.models import ACCUncertaintyRunPaths
from pyaesa.acc.uncertainty.sources.source_keys import ASOCC_INTER_METHOD_SOURCE
from pyaesa.shared.figures.jobs import PlannedFigureJob, render_figure_jobs
from pyaesa.shared.figures.lcia_scopes import lcia_impact_slices
from pyaesa.shared.figures.request_validation import (
    validate_consecutive_multi_year_figure_request,
)
from pyaesa.shared.figures.selector_slices import selector_slices
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest

Plotter = Callable[..., list[Path]]


def render_acc_uncertainty_figures(
    *,
    manifest: UncertaintyManifest,
    paths: ACCUncertaintyRunPaths,
    figure_options: dict | None,
    figure_format: dict | None,
    status: StatusSink | None = None,
) -> list[Path]:
    """Render requested aCC uncertainty figures from public run artifacts."""
    context = build_figure_context(
        manifest=manifest,
        paths=paths,
        figure_options=figure_options,
        figure_format=figure_format,
    )
    validate_consecutive_multi_year_figure_request(
        requested_years=list(context.requested_years),
        family_label="aCC uncertainty",
    )
    if not (
        context.per_method
        or context.multi_method
        or (context.inter_method and ASOCC_INTER_METHOD_SOURCE in set(context.active_sources))
    ):
        clear_uncertainty_figure_scope(paths=paths)
        return []
    tables = read_figure_tables(
        context=context,
        include_summary=single_requested_year(context) is None,
    )
    validate_inactive_axes_for_figures(identity=tables.identity, context=context)
    clear_uncertainty_figure_scope(paths=paths)
    return render_figure_jobs(
        source="uncertainty_acc",
        jobs=lambda: _uncertainty_jobs(
            context=context,
            identity=tables.identity,
            summary=tables.summary,
        ),
        status=status,
    )


def _uncertainty_jobs(
    *,
    context: FigureContext,
    identity: pd.DataFrame,
    summary: pd.DataFrame,
) -> Iterator[PlannedFigureJob]:
    if single_requested_year(context) is None:
        return _multi_year_jobs(context=context, identity=identity, summary=summary)
    return _single_year_jobs(context=context, identity=identity)


def _single_year_jobs(
    *, context: FigureContext, identity: pd.DataFrame
) -> Iterator[PlannedFigureJob]:
    identity_rows = prepared_identity_rows(context=context, identity=identity)
    value_rows = value_rows_from_runs(context=context, identity_rows=identity_rows)
    method_rows = collapsed_value_rows(
        rows=value_rows,
        context=context,
        include_method_axis=True,
    )
    if context.inter_method and ASOCC_INTER_METHOD_SOURCE in set(context.active_sources):
        yield from _plan_inter_method_jobs(
            rows=_collapsed_inter_method_rows(rows=value_rows, context=context),
            context=context,
            plotter=plot_violin_scope,
            kind="single_year",
        )
    if context.multi_method:
        yield from _plan_multi_method_jobs(
            rows=method_rows,
            context=context,
            plotter=plot_violin_scope,
            kind="single_year",
        )
    if context.per_method:
        yield from _plan_per_method_jobs(
            rows=method_rows,
            context=context,
            plotter=plot_violin_scope,
            kind="single_year",
        )


def _multi_year_jobs(
    *,
    context: FigureContext,
    identity: pd.DataFrame,
    summary: pd.DataFrame,
) -> Iterator[PlannedFigureJob]:
    active = set(context.active_sources)
    if ASOCC_INTER_METHOD_SOURCE in active:
        summary_rows = prepared_summary_rows(context=context, summary=summary)
        method_rows = _summary_scope_rows(rows=summary_rows, scope=ACC_SUMMARY_SCOPE_PER_METHOD)
        inter_rows = _summary_scope_rows(rows=summary_rows, scope=ACC_SUMMARY_SCOPE_INTER_METHOD)
        if _is_dynamic_rows(method_rows):
            identity_rows = prepared_identity_rows(context=context, identity=identity)
            value_rows = value_rows_from_runs(context=context, identity_rows=identity_rows)
            method_rows = attach_dynamic_budget_values(
                summary_rows=method_rows,
                value_rows=value_rows,
                context=context,
                include_method_axis=True,
            )
            inter_rows = attach_dynamic_budget_values(
                summary_rows=inter_rows,
                value_rows=value_rows,
                context=context,
                include_method_axis=False,
            )
    else:
        method_rows = prepared_summary_rows(context=context, summary=summary)
        if _is_dynamic_rows(method_rows):
            identity_rows = prepared_identity_rows(context=context, identity=identity)
            value_rows = value_rows_from_runs(context=context, identity_rows=identity_rows)
            method_rows = attach_dynamic_budget_values(
                summary_rows=method_rows,
                value_rows=value_rows,
                context=context,
                include_method_axis=True,
            )
        inter_rows = method_rows.iloc[0:0].copy()
    if context.inter_method:
        yield from _plan_inter_method_jobs(
            rows=inter_rows,
            context=context,
            plotter=plot_band_scope,
            kind="multi_year",
        )
    if context.multi_method:
        yield from _plan_multi_method_jobs(
            rows=method_rows,
            context=context,
            plotter=plot_mean_line_scope,
            kind="multi_year",
            mean_line=True,
        )
    if context.per_method:
        yield from _plan_per_method_jobs(
            rows=method_rows,
            context=context,
            plotter=plot_band_scope,
            kind="multi_year",
        )


def _plan_per_method_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    plotter: Plotter,
    kind: str,
) -> Iterator[PlannedFigureJob]:
    for method in visible_values(rows, "__method"):
        method_rows = rows.loc[rows["__method"].astype(str).eq(str(method))].copy()
        yield from _plan_scope_jobs(
            rows=method_rows,
            context=context,
            role="per_method",
            label=str(method),
            title_label=str(method),
            plotter=plotter,
            kind=kind,
            group_legend=False,
            include_method_in_label=False,
        )


def _plan_multi_method_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    plotter: Plotter,
    kind: str,
    mean_line: bool = False,
) -> Iterator[PlannedFigureJob]:
    if len(visible_values(rows, "__method")) <= 1:
        return
    yield from _plan_scope_jobs(
        rows=rows,
        context=context,
        role="multi_method",
        label="multi_method",
        title_label=None,
        plotter=plotter,
        kind=kind,
        group_legend=True,
        include_method_in_label=True,
        split_multi_year_impacts=kind == "multi_year",
        mean_line=mean_line,
    )


def _plan_inter_method_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    plotter: Plotter,
    kind: str,
) -> Iterator[PlannedFigureJob]:
    if rows.empty:
        return
    yield from _plan_scope_jobs(
        rows=rows,
        context=context,
        role="inter_method",
        label="inter_method",
        title_label="inter-method",
        plotter=plotter,
        kind=kind,
        group_legend=False,
        include_method_in_label=False,
    )


def _plan_scope_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    role: str,
    label: str,
    title_label: str | None,
    plotter: Plotter,
    kind: str,
    group_legend: bool,
    include_method_in_label: bool,
    split_multi_year_impacts: bool = False,
    mean_line: bool = False,
) -> Iterator[PlannedFigureJob]:
    studied_year = single_requested_year(context)
    for scope in _figure_scopes(rows=rows, context=context):
        for selector_token, selector_title, selector_scope in selector_slices(scope):
            split_impacts = (
                split_multi_year_impacts
                and studied_year is None
                and len(ordered_impacts(selector_scope)) > 1
            )
            product_scopes = (
                lcia_impact_slices(selector_scope) if split_impacts else [selector_scope]
            )
            for product_scope in product_scopes:
                yield _scope_job(
                    scope=product_scope,
                    context=context,
                    role=role,
                    label=label,
                    title_label=title_label,
                    plotter=plotter,
                    kind=kind,
                    group_legend=group_legend,
                    include_method_in_label=include_method_in_label,
                    mean_line=mean_line,
                    studied_year=studied_year,
                    include_impact=split_impacts,
                    selector_token=selector_token,
                    selector_title=selector_title,
                )


def _scope_job(
    *,
    scope: pd.DataFrame,
    context: FigureContext,
    role: str,
    label: str,
    title_label: str | None,
    plotter: Plotter,
    kind: str,
    group_legend: bool,
    include_method_in_label: bool,
    mean_line: bool,
    studied_year: int | None,
    include_impact: bool,
    selector_token: str,
    selector_title: str,
) -> PlannedFigureJob:
    """Return one planned aCC uncertainty figure for a final figure scope."""
    output_base = (
        context.figures_root
        / role
        / acc_scope_stem(
            label,
            scope,
            include_impact=include_impact,
            selector_token=selector_token,
            studied_year=studied_year,
        )
    )
    title = scope_title(
        "aCC uncertainty",
        title_label,
        scope,
        include_impact=include_impact or len(ordered_impacts(scope)) == 1,
        selector_title=selector_title,
        studied_year=studied_year,
    )
    return PlannedFigureJob(
        kind=kind,
        label=output_base.name,
        render=_planned_plot(
            plotter=plotter,
            frame=scope,
            output_stem=output_base,
            title=title,
            context=context,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
            mean_line=mean_line,
        ),
    )


def _planned_plot(
    *,
    plotter: Plotter,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    context: FigureContext,
    group_legend: bool,
    include_method_in_label: bool,
    mean_line: bool,
):
    def _render() -> list[Path]:
        if mean_line:
            return plotter(
                frame=frame,
                requested_years=list(context.requested_years),
                output_stem=output_stem,
                title=title,
                dpi=context.figure_dpi,
                output_format=context.figure_output_format,
                group_legend=group_legend,
                include_impact_in_label=False,
                include_method_in_label=include_method_in_label,
            )
        return plotter(
            frame=frame,
            output_stem=output_stem,
            title=title,
            dpi=context.figure_dpi,
            output_format=context.figure_output_format,
            group_legend=group_legend,
            include_impact_in_label=True,
            include_method_in_label=include_method_in_label,
        )

    return _render


def _figure_scopes(*, rows: pd.DataFrame, context: FigureContext) -> Iterator[pd.DataFrame]:
    """Yield one aCC uncertainty figure scope at a time."""
    for cc_type, cc_rows in rows.groupby("cc_type", dropna=False, sort=True):
        if str(cc_type) == "static":
            for ssp_rows in static_asocc_ssp_slices(
                cc_rows,
                requested_ssps=context.requested_asocc_ssps,
            ):
                yield from scope_slices(ssp_rows, ("lcia_method",))
            continue
        dynamic_columns = [
            column
            for column in (
                "lcia_method",
                AR6_CC_SSP_SCENARIO_COLUMN,
                "cc_category",
                "cc_model",
                "cc_scenario",
            )
            if column in cc_rows.columns
        ]
        yield from scope_slices(cc_rows, tuple(dynamic_columns))


def _collapsed_inter_method_rows(*, rows: pd.DataFrame, context: FigureContext) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for cc_type, cc_rows in rows.groupby("cc_type", dropna=False, sort=True):
        if str(cc_type) == "static":
            scoped_frames = static_asocc_ssp_slices(
                cc_rows,
                requested_ssps=context.requested_asocc_ssps,
            )
        else:
            scoped_frames = [cc_rows.copy()]
        for scoped in scoped_frames:
            parts.extend(
                collapsed_value_rows(
                    rows=selector_rows,
                    context=context,
                    include_method_axis=False,
                )
                for _selector_token, _selector_title, selector_rows in selector_slices(scoped)
            )
    return pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0].copy()


def _is_dynamic_rows(rows: pd.DataFrame) -> bool:
    return "cc_type" in rows.columns and visible_values(rows, "cc_type") == [DYNAMIC_CC_TYPE]


def _summary_scope_rows(*, rows: pd.DataFrame, scope: str) -> pd.DataFrame:
    scoped = rows.loc[rows[ACC_SUMMARY_SCOPE_COLUMN].astype(str).eq(scope)].copy()
    if scope == ACC_SUMMARY_SCOPE_INTER_METHOD and "__method" in scoped.columns:
        scoped["__method"] = ""
    return scoped.drop(columns=[ACC_SUMMARY_SCOPE_COLUMN])

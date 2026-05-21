"""aSoCC uncertainty figure orchestration."""

from pathlib import Path

import pandas as pd

from pyaesa.asocc.figures.multi_method_renderer import has_multiple_methods
from pyaesa.asocc.figures.product_renderers import render_products as render_deterministic_products
from pyaesa.asocc.uncertainty.figures.metadata import clear_uncertainty_figure_scope
from pyaesa.asocc.uncertainty.figures.multi_method_renderer import (
    plan_inter_method_jobs,
    plan_multi_method_mean_jobs,
    plan_multi_method_jobs,
)
from pyaesa.asocc.uncertainty.figures.per_method_renderer import (
    figure_ssp_slices,
    plan_per_method_jobs,
)
from pyaesa.asocc.uncertainty.figures.product_renderers import (
    plot_band_scope,
    plot_mean_line_scope,
    plot_violin_scope,
)
from pyaesa.asocc.uncertainty.figures.row_reader import (
    collapsed_violin_rows,
    deterministic_rows_from_summary,
    prepared_identity_rows,
    prepared_summary_rows,
    read_figure_tables,
    violin_rows_from_compact_runs,
    violin_rows_from_sparse_runs,
    drop_empty_value_rows,
)
from pyaesa.asocc.uncertainty.engine.evaluation.summary_identity import (
    ASOCC_SUMMARY_SCOPE_COLUMN,
    ASOCC_SUMMARY_SCOPE_INTER_METHOD,
    ASOCC_SUMMARY_SCOPE_PER_METHOD,
)
from pyaesa.asocc.uncertainty.figures.scope_planner import (
    FigureContext,
    build_figure_context,
    single_requested_year,
)
from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.asocc.uncertainty.sources.names import INTER_METHOD_SOURCE
from pyaesa.shared.figures.jobs import PlannedFigureJob, render_figure_jobs
from pyaesa.shared.figures.request_validation import (
    validate_consecutive_multi_year_figure_request,
)
from pyaesa.shared.figures.selector_slices import selector_slices
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def render_asocc_uncertainty_figures(
    *,
    manifest: UncertaintyManifest,
    paths: AsoccUncertaintyRunPaths,
    figure_options: dict | None,
    figure_format: dict | None,
    status: StatusSink | None = None,
) -> list[Path]:
    """Render requested aSoCC uncertainty figures from public run artifacts."""
    context = build_figure_context(
        manifest=manifest,
        paths=paths,
        figure_options=figure_options,
        figure_format=figure_format,
    )
    validate_consecutive_multi_year_figure_request(
        requested_years=list(context.requested_years),
        family_label="aSoCC uncertainty",
    )
    if not (
        context.per_method
        or context.multi_method
        or (context.inter_method and INTER_METHOD_SOURCE in set(context.active_sources))
    ):
        clear_uncertainty_figure_scope(paths=paths)
        return []
    single_year = single_requested_year(context) is not None
    tables = read_figure_tables(
        context=context,
        include_summary=(not context.active_sources or not single_year),
    )
    clear_uncertainty_figure_scope(paths=paths)
    if not context.active_sources:
        deterministic_rows = _planned_ssp_rows(
            rows=deterministic_rows_from_summary(
                context=context,
                summary=tables.summary,
            ),
            context=context,
        )
        return render_deterministic_products(
            rows=deterministic_rows,
            figures_root=context.figures_root,
            requested_years=list(context.requested_years),
            dpi=context.figure_dpi,
            output_format=context.figure_output_format,
            status_source="uncertainty_asocc",
            per_method=context.per_method,
            multi_method=context.multi_method,
            status=status,
        )
    jobs = _uncertainty_jobs(context=context, identity=tables.identity, summary=tables.summary)
    return render_figure_jobs(source="uncertainty_asocc", jobs=jobs, status=status)


def _uncertainty_jobs(
    *,
    context: FigureContext,
    identity,
    summary,
) -> list[PlannedFigureJob]:
    if single_requested_year(context) is None:
        return _multi_year_jobs(context=context, identity=identity, summary=summary)
    return _single_year_jobs(context=context, identity=identity, summary=summary)


def _single_year_jobs(
    *,
    context: FigureContext,
    identity,
    summary,
) -> list[PlannedFigureJob]:
    del summary
    identity_rows = prepared_identity_rows(context=context, identity=identity)
    if INTER_METHOD_SOURCE in set(context.active_sources):
        method_rows = drop_empty_value_rows(
            rows=violin_rows_from_sparse_runs(context=context, identity_rows=identity_rows)
        )
        inter_rows = _collapsed_inter_method_value_rows(rows=method_rows, context=context)
        return [
            *(
                plan_inter_method_jobs(
                    rows=inter_rows,
                    context=context,
                    plotter=plot_violin_scope,
                    kind="inter_method",
                )
                if context.inter_method
                else []
            ),
            *(
                plan_multi_method_jobs(
                    rows=method_rows,
                    context=context,
                    plotter=plot_violin_scope,
                    kind="multi_method",
                )
                if context.multi_method
                else []
            ),
            *(
                plan_per_method_jobs(
                    rows=method_rows,
                    context=context,
                    plotter=plot_violin_scope,
                    kind="per_method",
                )
                if context.per_method
                else []
            ),
        ]
    rows = drop_empty_value_rows(
        rows=violin_rows_from_compact_runs(context=context, identity_rows=identity_rows)
    )
    multi_method_jobs = (
        plan_multi_method_jobs(
            rows=rows,
            context=context,
            plotter=plot_violin_scope,
            kind="multi_method",
        )
        if context.multi_method and has_multiple_methods(rows)
        else []
    )
    return [
        *multi_method_jobs,
        *(
            plan_per_method_jobs(
                rows=rows,
                context=context,
                plotter=plot_violin_scope,
                kind="per_method",
            )
            if context.per_method
            else []
        ),
    ]


def _multi_year_jobs(
    *,
    context: FigureContext,
    identity,
    summary,
) -> list[PlannedFigureJob]:
    if INTER_METHOD_SOURCE in set(context.active_sources):
        rows = prepared_summary_rows(context=context, summary=summary)
        method_rows = _summary_scope_rows(rows=rows, scope=ASOCC_SUMMARY_SCOPE_PER_METHOD)
        inter_rows = _summary_scope_rows(rows=rows, scope=ASOCC_SUMMARY_SCOPE_INTER_METHOD)
        return [
            *(
                plan_inter_method_jobs(
                    rows=inter_rows,
                    context=context,
                    plotter=plot_band_scope,
                    kind="inter_method",
                )
                if context.inter_method
                else []
            ),
            *(
                plan_multi_method_mean_jobs(
                    rows=method_rows,
                    context=context,
                    plotter=plot_mean_line_scope,
                )
                if context.multi_method
                else []
            ),
            *(
                plan_per_method_jobs(
                    rows=method_rows,
                    context=context,
                    plotter=plot_band_scope,
                    kind="per_method",
                )
                if context.per_method
                else []
            ),
        ]
    rows = prepared_summary_rows(context=context, summary=summary)
    return [
        *(
            plan_multi_method_mean_jobs(
                rows=rows,
                context=context,
                plotter=plot_mean_line_scope,
            )
            if context.multi_method
            else []
        ),
        *(
            plan_per_method_jobs(
                rows=rows,
                context=context,
                plotter=plot_band_scope,
                kind="per_method",
            )
            if context.per_method
            else []
        ),
    ]


def _collapsed_inter_method_value_rows(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
) -> pd.DataFrame:
    parts = [
        collapsed_violin_rows(rows=selector_rows)
        for scoped_rows in figure_ssp_slices(rows, context=context)
        for _selector_token, _selector_title, selector_rows in selector_slices(scoped_rows)
    ]
    return pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0].copy()


def _planned_ssp_rows(*, rows: pd.DataFrame, context: FigureContext) -> pd.DataFrame:
    parts = figure_ssp_slices(rows, context=context)
    return pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0].copy()


def _summary_scope_rows(*, rows: pd.DataFrame, scope: str) -> pd.DataFrame:
    scoped = rows.loc[rows[ASOCC_SUMMARY_SCOPE_COLUMN].astype(str).eq(scope)].copy()
    return scoped.drop(columns=[ASOCC_SUMMARY_SCOPE_COLUMN])

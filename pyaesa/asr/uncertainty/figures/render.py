"""ASR uncertainty figure orchestration."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyaesa.acc.uncertainty.sources.source_keys import ASOCC_INTER_METHOD_SOURCE
from pyaesa.asr.figures.common import (
    DYNAMIC_SCOPE_COLUMNS,
    VALUE_ARRAY_COLUMN,
    asr_scale_mode_for_values,
    asr_scope_stem,
    asr_scope_title,
    ordered_impacts,
    scope_slices,
    static_asocc_ssp_slices,
    visible_values,
)
from pyaesa.asr.figures.axis import ASR_NORMAL_SCALE, ASRScaleMode
from pyaesa.asr.figures.frequency import CUMULATIVE_FNT_FRACTION_COLUMN, FNT_FRACTION_COLUMN
from pyaesa.asr.figures.dynamic_global_ar6 import (
    UncertaintyGlobalAR6Source,
    uncertainty_global_ar6_source,
)
from pyaesa.asr.uncertainty.figures.metadata import clear_uncertainty_figure_scope
from pyaesa.asr.uncertainty.figures.component_data import (
    ComponentDiagnosticRows,
    load_component_diagnostic_rows,
)
from pyaesa.asr.uncertainty.figures.product_renderers import (
    plot_band_scope,
    plot_mean_line_scope,
)
from pyaesa.asr.uncertainty.figures.polar_renderers import plot_polar_scope
from pyaesa.asr.uncertainty.figures.row_reader import (
    attach_dynamic_pair_counts,
    collapsed_value_rows,
    cumulative_value_rows_from_runs,
    prepared_cumulative_frequency_rows,
    prepared_cumulative_identity_rows,
    prepared_frequency_rows,
    prepared_identity_rows,
    prepared_summary_rows,
    read_figure_tables,
    SUMMARY_STAT_COLUMNS,
    value_rows_from_runs,
)
from pyaesa.asr.uncertainty.evaluation.summary import (
    ASR_CUMULATIVE_FREQUENCY_VALUE_COLUMN,
    ASR_FREQUENCY_VALUE_COLUMN,
    ASR_SUMMARY_METRIC_COLUMN,
    ASR_SUMMARY_SCOPE_COLUMN,
    ASR_SUMMARY_SCOPE_INTER_METHOD,
    ASR_SUMMARY_SCOPE_PER_METHOD,
)
from pyaesa.asr.uncertainty.figures.scope_planner import (
    FigureContext,
    build_figure_context,
    single_requested_year,
)
from pyaesa.asr.uncertainty.figures.violin_renderers import plot_violin_scope
from pyaesa.asr.uncertainty.runtime.models import ASRUncertaintyRunPaths
from pyaesa.shared.figures.jobs import PlannedFigureJob, render_figure_jobs
from pyaesa.shared.figures.lcia_scopes import lcia_impact_slices
from pyaesa.shared.figures.request_validation import (
    validate_consecutive_multi_year_figure_request,
)
from pyaesa.shared.figures.selector_slices import selector_slices
from pyaesa.shared.figures.dynamic_ar6 import DYNAMIC_AR6_CC_TYPE
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest

Plotter = Callable[..., list[Path]]


def render_asr_uncertainty_figures(
    *,
    manifest: UncertaintyManifest,
    paths: ASRUncertaintyRunPaths,
    figure_options: dict | None,
    figure_format: dict | None,
    status: StatusSink | None = None,
) -> list[Path]:
    """Render requested ASR uncertainty figures from public run artifacts."""
    context = build_figure_context(
        manifest=manifest,
        paths=paths,
        figure_options=figure_options,
        figure_format=figure_format,
    )
    validate_consecutive_multi_year_figure_request(
        requested_years=list(context.requested_years),
        family_label="ASR uncertainty",
    )
    if not (context.per_method or context.multi_method or _inter_method_active(context)):
        clear_uncertainty_figure_scope(paths=paths)
        return []
    tables = read_figure_tables(
        context=context,
        include_cumulative=single_requested_year(context) is None,
    )
    clear_uncertainty_figure_scope(paths=paths)
    jobs = _uncertainty_jobs(
        context=context,
        identity=tables.identity,
        summary=tables.summary,
        cumulative_identity=tables.cumulative_identity,
        cumulative_summary=tables.cumulative_summary,
    )
    return render_figure_jobs(source="uncertainty_asr", jobs=jobs, status=status)


def _uncertainty_jobs(
    *,
    context: FigureContext,
    identity: pd.DataFrame,
    summary: pd.DataFrame,
    cumulative_identity: pd.DataFrame,
    cumulative_summary: pd.DataFrame,
) -> list[PlannedFigureJob]:
    if single_requested_year(context) is None:
        return _multi_year_jobs(
            context=context,
            identity=identity,
            summary=summary,
            cumulative_identity=cumulative_identity,
            cumulative_summary=cumulative_summary,
        )
    return _single_year_jobs(context=context, identity=identity, summary=summary)


def _single_year_jobs(
    *, context: FigureContext, identity: pd.DataFrame, summary: pd.DataFrame
) -> list[PlannedFigureJob]:
    identity_rows = prepared_identity_rows(context=context, identity=identity)
    value_rows = value_rows_from_runs(context=context, identity_rows=identity_rows)
    summary_rows = prepared_summary_rows(context=context, summary=summary)
    frequency_rows = prepared_frequency_rows(context=context, summary=summary)
    method_summary_rows = _summary_scope_rows(summary_rows, ASR_SUMMARY_SCOPE_PER_METHOD)
    method_frequency_rows = _summary_scope_rows(frequency_rows, ASR_SUMMARY_SCOPE_PER_METHOD)
    method_rows = collapsed_value_rows(
        rows=value_rows,
        context=context,
        include_method_axis=True,
    )
    method_rows = _with_polar_summary_rows(
        rows=method_rows,
        summary_rows=method_summary_rows,
        frequency_rows=method_frequency_rows,
    )
    scale_modes = _scale_modes_by_lcia(value_rows)
    jobs = [
        *(
            _plan_multi_method_jobs(
                rows=method_rows,
                context=context,
                plotter=plot_violin_scope,
                kind="single_year",
                scale_modes=scale_modes,
            )
            if context.multi_method
            else []
        ),
        *(
            _plan_per_method_jobs(
                rows=method_rows,
                context=context,
                plotter=plot_violin_scope,
                kind="single_year",
                polar_for_multi_impact=bool(context.polar_years),
                scale_modes=scale_modes,
            )
            if context.per_method
            else []
        ),
    ]
    if _inter_method_active(context):
        inter_summary_rows = _summary_scope_rows(summary_rows, ASR_SUMMARY_SCOPE_INTER_METHOD)
        inter_frequency_rows = _summary_scope_rows(
            frequency_rows,
            ASR_SUMMARY_SCOPE_INTER_METHOD,
        )
        inter_rows = _collapsed_inter_method_value_rows(rows=value_rows, context=context)
        inter_rows = _with_polar_summary_rows(
            rows=inter_rows,
            summary_rows=inter_summary_rows,
            frequency_rows=inter_frequency_rows,
        )
        jobs = [
            *_plan_inter_method_jobs(
                rows=inter_rows,
                context=context,
                plotter=plot_violin_scope,
                kind="single_year",
                polar_for_multi_impact=bool(context.polar_years),
                scale_modes=scale_modes,
            ),
            *jobs,
        ]
    return jobs


def _multi_year_jobs(
    *,
    context: FigureContext,
    identity: pd.DataFrame,
    summary: pd.DataFrame,
    cumulative_identity: pd.DataFrame,
    cumulative_summary: pd.DataFrame,
) -> list[PlannedFigureJob]:
    dynamic_rows = _raw_identity_is_dynamic(identity)
    component_rows = load_component_diagnostic_rows(context=context) if dynamic_rows else None
    global_ar6_source = (
        uncertainty_global_ar6_source(manifest=context.manifest) if dynamic_rows else None
    )
    summary_rows = prepared_summary_rows(context=context, summary=summary)
    frequency_rows = prepared_frequency_rows(context=context, summary=summary)
    method_summary_rows = _summary_scope_rows(summary_rows, ASR_SUMMARY_SCOPE_PER_METHOD)
    method_frequency_rows = _summary_scope_rows(frequency_rows, ASR_SUMMARY_SCOPE_PER_METHOD)
    if dynamic_rows:
        cumulative_method_rows, cumulative_inter_rows = _cumulative_rows(
            context=context,
            cumulative_identity=cumulative_identity,
            cumulative_summary=cumulative_summary,
        )
    else:
        cumulative_method_rows = summary_rows.iloc[0:0].copy()
        cumulative_inter_rows = cumulative_method_rows.copy()
    if _inter_method_active(context):
        inter_summary_rows = _summary_scope_rows(summary_rows, ASR_SUMMARY_SCOPE_INTER_METHOD)
        inter_frequency_rows = _summary_scope_rows(
            frequency_rows,
            ASR_SUMMARY_SCOPE_INTER_METHOD,
        )
        identity_rows = prepared_identity_rows(context=context, identity=identity)
        method_rows = _with_frequency_summary(method_summary_rows, method_frequency_rows)
        if _is_dynamic_rows(method_rows):
            method_rows = attach_dynamic_pair_counts(
                summary_rows=method_rows,
                identity_rows=identity_rows,
                context=context,
                include_method_axis=True,
            )
        if dynamic_rows:
            method_rows = _with_cumulative_values(
                method_rows,
                cumulative_method_rows,
                context=context,
            )
        inter_rows = _with_frequency_summary(inter_summary_rows, inter_frequency_rows)
        if _is_dynamic_rows(inter_rows):
            inter_rows = attach_dynamic_pair_counts(
                summary_rows=inter_rows,
                identity_rows=identity_rows,
                context=context,
                include_method_axis=False,
            )
        if dynamic_rows:
            inter_rows = _with_cumulative_values(
                inter_rows,
                cumulative_inter_rows,
                context=context,
            )
        method_polar_rows = method_rows.iloc[0:0].copy()
        inter_polar_rows = inter_rows.iloc[0:0].copy()
        if context.polar_years:
            value_rows = value_rows_from_runs(context=context, identity_rows=identity_rows)
            method_polar_rows = collapsed_value_rows(
                rows=value_rows,
                context=context,
                include_method_axis=True,
            )
            inter_polar_rows = _collapsed_inter_method_value_rows(
                rows=value_rows,
                context=context,
            )
            method_polar_rows = _with_polar_summary_rows(
                rows=method_polar_rows,
                summary_rows=method_summary_rows,
                frequency_rows=method_frequency_rows,
            )
            inter_polar_rows = _with_polar_summary_rows(
                rows=inter_polar_rows,
                summary_rows=inter_summary_rows,
                frequency_rows=inter_frequency_rows,
            )
    else:
        method_rows = method_summary_rows
        identity_rows = (
            prepared_identity_rows(context=context, identity=identity)
            if _is_dynamic_rows(method_rows) or context.polar_years
            else identity.iloc[0:0].copy()
        )
        if _is_dynamic_rows(method_rows):
            method_rows = attach_dynamic_pair_counts(
                summary_rows=method_rows,
                identity_rows=identity_rows,
                context=context,
                include_method_axis=True,
            )
        method_rows = _with_frequency_summary(method_rows, method_frequency_rows)
        if dynamic_rows:
            method_rows = _with_cumulative_values(
                method_rows,
                cumulative_method_rows,
                context=context,
            )
        inter_rows = method_rows.iloc[0:0].copy()
        value_rows = (
            value_rows_from_runs(
                context=context,
                identity_rows=identity_rows,
            )
            if context.polar_years
            else identity.iloc[0:0].copy()
        )
        method_polar_rows = (
            collapsed_value_rows(rows=value_rows, context=context, include_method_axis=True)
            if context.polar_years and not value_rows.empty
            else value_rows.iloc[0:0].copy()
        )
        method_polar_rows = _with_polar_summary_rows(
            rows=method_polar_rows,
            summary_rows=method_summary_rows,
            frequency_rows=method_frequency_rows,
        )
        inter_polar_rows = value_rows.iloc[0:0].copy()
    scale_modes = _scale_modes_by_lcia(
        method_rows,
        inter_rows,
        cumulative_method_rows,
        cumulative_inter_rows,
        method_polar_rows,
        inter_polar_rows,
    )
    jobs = [
        *(
            _plan_polar_checkpoint_jobs(
                rows=inter_polar_rows,
                context=context,
                role="inter_method",
                label="inter_method",
                title_label="inter-method",
                family_label="ASR uncertainty",
                scale_modes=scale_modes,
            )
            if context.inter_method
            else []
        ),
        *(
            _plan_inter_method_jobs(
                rows=inter_rows,
                context=context,
                plotter=plot_band_scope,
                kind="multi_year",
                components=component_rows,
                global_ar6_source=global_ar6_source,
                scale_modes=scale_modes,
            )
            if context.inter_method
            else []
        ),
        *(
            _plan_multi_method_jobs(
                rows=method_rows,
                context=context,
                plotter=plot_mean_line_scope,
                kind="multi_year",
                mean_line=True,
                components=component_rows,
                global_ar6_source=global_ar6_source,
                scale_modes=scale_modes,
            )
            if context.multi_method
            else []
        ),
        *(
            _plan_per_method_jobs(
                rows=method_rows,
                context=context,
                plotter=plot_band_scope,
                kind="multi_year",
                components=component_rows,
                global_ar6_source=global_ar6_source,
                scale_modes=scale_modes,
            )
            if context.per_method
            else []
        ),
        *(
            _plan_polar_checkpoint_jobs(
                rows=method_polar_rows,
                context=context,
                role="per_method",
                label=None,
                title_label=None,
                family_label="ASR uncertainty",
                scale_modes=scale_modes,
            )
            if context.per_method
            else []
        ),
    ]
    return jobs


def _plan_per_method_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    plotter: Plotter,
    kind: str,
    product: str | None = None,
    family_label: str = "ASR uncertainty",
    polar_for_multi_impact: bool = False,
    components: ComponentDiagnosticRows | None = None,
    global_ar6_source: UncertaintyGlobalAR6Source | None = None,
    scale_modes: dict[str, ASRScaleMode],
) -> list[PlannedFigureJob]:
    jobs: list[PlannedFigureJob] = []
    for method in visible_values(rows, "__method"):
        method_rows = rows.loc[rows["__method"].astype(str).eq(str(method))].copy()
        jobs.extend(
            _plan_scope_jobs(
                rows=method_rows,
                context=context,
                role="per_method",
                label=str(method),
                title_label=str(method),
                plotter=plotter,
                kind=kind,
                product=product,
                family_label=family_label,
                group_legend=False,
                include_method_in_label=False,
                polar_for_multi_impact=polar_for_multi_impact,
                components=components,
                global_ar6_source=global_ar6_source,
                scale_modes=scale_modes,
            )
        )
    return jobs


def _collapsed_inter_method_value_rows(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for cc_type, cc_rows in rows.groupby("cc_type", dropna=False, sort=True):
        scoped_frames = (
            static_asocc_ssp_slices(
                cc_rows,
                requested_ssps=context.requested_asocc_ssps,
            )
            if str(cc_type) == "static"
            else [cc_rows.copy()]
        )
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


def _cumulative_rows(
    *,
    context: FigureContext,
    cumulative_identity: pd.DataFrame,
    cumulative_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cumulative_identity_rows = prepared_cumulative_identity_rows(
        context=context,
        cumulative_identity=cumulative_identity,
    )
    cumulative_frequency_rows = prepared_cumulative_frequency_rows(
        context=context,
        cumulative_summary=cumulative_summary,
    )
    method_frequency_rows = _summary_scope_rows(
        cumulative_frequency_rows,
        ASR_SUMMARY_SCOPE_PER_METHOD,
    )
    cumulative_value_rows = cumulative_value_rows_from_runs(
        context=context,
        cumulative_identity_rows=cumulative_identity_rows,
    )
    method_rows = collapsed_value_rows(
        rows=cumulative_value_rows,
        context=context,
        include_method_axis=True,
    )
    method_rows = _with_frequency_summary(method_rows, method_frequency_rows)
    if _inter_method_active(context):
        inter_frequency_rows = _summary_scope_rows(
            cumulative_frequency_rows,
            ASR_SUMMARY_SCOPE_INTER_METHOD,
        )
        inter_rows = collapsed_value_rows(
            rows=cumulative_value_rows,
            context=context,
            include_method_axis=False,
        )
        return method_rows, _with_frequency_summary(inter_rows, inter_frequency_rows)
    return method_rows, method_rows.iloc[0:0].copy()


def _plan_multi_method_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    plotter: Plotter,
    kind: str,
    mean_line: bool = False,
    product: str | None = None,
    family_label: str = "ASR uncertainty",
    components: ComponentDiagnosticRows | None = None,
    global_ar6_source: UncertaintyGlobalAR6Source | None = None,
    scale_modes: dict[str, ASRScaleMode],
) -> list[PlannedFigureJob]:
    if len(visible_values(rows, "__method")) <= 1:
        return []
    return _plan_scope_jobs(
        rows=rows,
        context=context,
        role="multi_method",
        label="multi_method",
        title_label=None,
        plotter=plotter,
        kind=kind,
        product=product,
        family_label=family_label,
        group_legend=True,
        include_method_in_label=True,
        split_multi_year_impacts=False,
        mean_line=mean_line,
        components=components,
        global_ar6_source=global_ar6_source,
        scale_modes=scale_modes,
    )


def _plan_inter_method_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    plotter: Plotter,
    kind: str,
    product: str | None = None,
    family_label: str = "ASR uncertainty",
    polar_for_multi_impact: bool = False,
    components: ComponentDiagnosticRows | None = None,
    global_ar6_source: UncertaintyGlobalAR6Source | None = None,
    scale_modes: dict[str, ASRScaleMode],
) -> list[PlannedFigureJob]:
    if rows.empty:
        return []
    return _plan_scope_jobs(
        rows=rows,
        context=context,
        role="inter_method",
        label="inter_method",
        title_label="inter-method",
        plotter=plotter,
        kind=kind,
        product=product,
        family_label=family_label,
        group_legend=False,
        include_method_in_label=False,
        polar_for_multi_impact=polar_for_multi_impact,
        components=components,
        global_ar6_source=global_ar6_source,
        scale_modes=scale_modes,
    )


def _plan_polar_checkpoint_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    role: str,
    label: str | None,
    title_label: str | None,
    family_label: str,
    scale_modes: dict[str, ASRScaleMode],
) -> list[PlannedFigureJob]:
    if rows.empty or not context.polar_years:
        return []
    jobs: list[PlannedFigureJob] = []
    method_values = visible_values(rows, "__method") if label is None else [label]
    for method in method_values:
        method_rows = (
            rows.loc[rows["__method"].astype(str).eq(str(method))].copy()
            if label is None
            else rows.copy()
        )
        scope_label = str(method)
        for scope in _figure_scopes(rows=method_rows, context=context):
            if visible_values(scope, "cc_type") != ["static"]:
                continue
            if len(ordered_impacts(scope)) <= 1:
                continue
            for selector_token, selector_title, selector_scope in selector_slices(scope):
                for year in context.polar_years:
                    year_scope = selector_scope.loc[
                        pd.Series(pd.to_numeric(selector_scope["year"], errors="raise"), copy=False)
                        .astype(int)
                        .eq(int(year))
                    ].copy()
                    _append_polar_scope_job(
                        jobs=jobs,
                        scope=year_scope,
                        context=context,
                        role=role,
                        label=scope_label,
                        title_label=title_label if label is not None else scope_label,
                        family_label=family_label,
                        studied_year=int(year),
                        selector_token=selector_token,
                        selector_title=selector_title,
                        scale_modes=scale_modes,
                    )
    return jobs


def _plan_scope_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    role: str,
    label: str,
    title_label: str | None,
    plotter: Plotter,
    kind: str,
    product: str | None,
    family_label: str,
    group_legend: bool,
    include_method_in_label: bool,
    split_multi_year_impacts: bool = False,
    mean_line: bool = False,
    polar_for_multi_impact: bool = False,
    components: ComponentDiagnosticRows | None = None,
    global_ar6_source: UncertaintyGlobalAR6Source | None = None,
    scale_modes: dict[str, ASRScaleMode],
) -> list[PlannedFigureJob]:
    jobs: list[PlannedFigureJob] = []
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
                _append_scope_job(
                    jobs=jobs,
                    scope=product_scope,
                    context=context,
                    role=role,
                    label=label,
                    title_label=title_label,
                    plotter=plotter,
                    kind=kind,
                    product=product,
                    family_label=family_label,
                    group_legend=group_legend,
                    include_method_in_label=include_method_in_label,
                    mean_line=mean_line,
                    studied_year=studied_year,
                    include_impact=split_impacts,
                    polar_for_multi_impact=polar_for_multi_impact,
                    components=components,
                    global_ar6_source=global_ar6_source,
                    scale_modes=scale_modes,
                    selector_token=selector_token,
                    selector_title=selector_title,
                )
    return jobs


def _append_scope_job(
    *,
    jobs: list[PlannedFigureJob],
    scope: pd.DataFrame,
    context: FigureContext,
    role: str,
    label: str,
    title_label: str | None,
    plotter: Plotter,
    kind: str,
    product: str | None,
    family_label: str,
    group_legend: bool,
    include_method_in_label: bool,
    mean_line: bool,
    studied_year: int | None,
    include_impact: bool,
    polar_for_multi_impact: bool,
    components: ComponentDiagnosticRows | None,
    global_ar6_source: UncertaintyGlobalAR6Source | None,
    scale_modes: dict[str, ASRScaleMode],
    selector_token: str,
    selector_title: str,
) -> None:
    if (
        polar_for_multi_impact
        and product is None
        and studied_year is not None
        and len(ordered_impacts(scope)) > 1
    ):
        _append_polar_scope_job(
            jobs=jobs,
            scope=scope,
            context=context,
            role=role,
            label=label,
            title_label=title_label,
            family_label=family_label,
            studied_year=studied_year,
            selector_token=selector_token,
            selector_title=selector_title,
            scale_modes=scale_modes,
        )
        return
    scale_mode = _scale_mode_for_frame(scope, scale_modes=scale_modes)
    output_base = (
        context.figures_root
        / role
        / asr_scope_stem(
            label,
            scope,
            product=product,
            include_impact=include_impact,
            studied_year=studied_year,
            selector_token=selector_token,
        )
    )
    title = asr_scope_title(
        family_label,
        title_label,
        scope,
        include_impact=include_impact or len(ordered_impacts(scope)) == 1,
        studied_year=studied_year,
        selector_title=selector_title,
    )
    jobs.append(
        PlannedFigureJob(
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
                components=components,
                global_ar6_source=global_ar6_source,
                scale_mode=scale_mode,
            ),
        )
    )


def _append_polar_scope_job(
    *,
    jobs: list[PlannedFigureJob],
    scope: pd.DataFrame,
    context: FigureContext,
    role: str,
    label: str,
    title_label: str | None,
    family_label: str,
    studied_year: int,
    selector_token: str,
    selector_title: str,
    scale_modes: dict[str, ASRScaleMode],
) -> None:
    scale_mode = _scale_mode_for_frame(scope, scale_modes=scale_modes)
    title = asr_scope_title(
        family_label,
        title_label,
        scope,
        include_impact=False,
        studied_year=studied_year,
        selector_title=selector_title,
    )
    for style in _polar_output_styles(context.polar_style):
        output_base = (
            context.figures_root
            / role
            / asr_scope_stem(
                f"polar_{style}_{label}",
                scope,
                studied_year=studied_year,
                selector_token=selector_token,
            )
        )
        jobs.append(
            PlannedFigureJob(
                kind=f"polar_{style}",
                label=output_base.name,
                render=lambda style_value=style, base=output_base: plot_polar_scope(
                    frame=scope,
                    output_stem=base,
                    title=title,
                    polar_style=style_value,
                    scale_mode=scale_mode,
                    dpi=context.figure_dpi,
                    output_format=context.figure_output_format,
                ),
            )
        )


def _polar_output_styles(style: str) -> tuple[str, ...]:
    return ("violin", "whisker") if str(style).strip() == "both" else (str(style).strip(),)


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
    components: ComponentDiagnosticRows | None,
    global_ar6_source: UncertaintyGlobalAR6Source | None,
    scale_mode: ASRScaleMode,
):
    def _render() -> list[Path]:
        extra: dict[str, Any] = {"components": components} if components is not None else {}
        if global_ar6_source is not None:
            extra["global_ar6_source"] = global_ar6_source
        extra["scale_mode"] = scale_mode
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
                **extra,
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
            **extra,
        )

    return _render


def _scale_modes_by_lcia(*frames: pd.DataFrame) -> dict[str, ASRScaleMode]:
    modes: dict[str, ASRScaleMode] = {}
    values_by_lcia: dict[str, list[np.ndarray]] = {}
    for frame in frames:
        if frame.empty or "lcia_method" not in frame.columns:
            continue
        for lcia_method, group in frame.groupby("lcia_method", dropna=False, sort=True):
            values_by_lcia.setdefault(str(lcia_method), []).append(_scale_values_from_frame(group))
    for lcia_method, value_arrays in values_by_lcia.items():
        modes[lcia_method] = asr_scale_mode_for_values(*value_arrays)
    return modes


def _scale_mode_for_frame(
    frame: pd.DataFrame,
    *,
    scale_modes: dict[str, ASRScaleMode],
) -> ASRScaleMode:
    lcia_methods = visible_values(frame, "lcia_method")
    return scale_modes.get(lcia_methods[0], ASR_NORMAL_SCALE) if lcia_methods else ASR_NORMAL_SCALE


def _scale_values_from_frame(frame: pd.DataFrame) -> np.ndarray:
    arrays: list[np.ndarray] = []
    if VALUE_ARRAY_COLUMN in frame.columns:
        arrays.extend(np.asarray(values, dtype=np.float64) for values in frame[VALUE_ARRAY_COLUMN])
    summary_columns = [column for column in SUMMARY_STAT_COLUMNS if column in frame.columns]
    if summary_columns:
        arrays.append(
            frame.loc[:, summary_columns]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(
                dtype=np.float64,
                copy=False,
            )
        )
    return np.concatenate([np.asarray(array, dtype=np.float64).ravel() for array in arrays])


def _with_polar_summary_rows(
    *,
    rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
    frequency_rows: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = _summary_merge_columns(rows=rows, summary_rows=summary_rows)
    summary = summary_rows.loc[:, [*key_columns, *SUMMARY_STAT_COLUMNS]].drop_duplicates(
        subset=key_columns,
        ignore_index=True,
    )
    frequency = frequency_rows.loc[:, [*key_columns, ASR_FREQUENCY_VALUE_COLUMN]].drop_duplicates(
        subset=key_columns,
        ignore_index=True,
    )
    frequency = frequency.rename(columns={ASR_FREQUENCY_VALUE_COLUMN: FNT_FRACTION_COLUMN})
    out = rows.merge(summary, on=key_columns, how="left")
    out = out.merge(frequency, on=key_columns, how="left")
    return out


def _summary_merge_columns(*, rows: pd.DataFrame, summary_rows: pd.DataFrame) -> list[str]:
    excluded = {
        VALUE_ARRAY_COLUMN,
        "public_row_id",
        FNT_FRACTION_COLUMN,
        "l1_l2_method",
        "l1_method",
        "l2_method",
        ASR_SUMMARY_METRIC_COLUMN,
        *SUMMARY_STAT_COLUMNS,
    }
    return [
        column
        for column in rows.columns
        if column in summary_rows.columns
        and column not in excluded
        and not (rows[column].dropna().empty and summary_rows[column].dropna().empty)
    ]


def _with_frequency_summary(rows: pd.DataFrame, frequency_rows: pd.DataFrame) -> pd.DataFrame:
    drop_columns = {
        ASR_FREQUENCY_VALUE_COLUMN,
        ASR_CUMULATIVE_FREQUENCY_VALUE_COLUMN,
        "mean",
        "std",
        "min",
        "p5",
        "p25",
        "median",
        "p75",
        "p95",
        "max",
        ASR_SUMMARY_METRIC_COLUMN,
    }
    key_columns = [
        column
        for column in rows.columns
        if column in frequency_rows.columns and column not in drop_columns
    ]
    value_column = _frequency_value_column(frequency_rows)
    frequency = frequency_rows.loc[:, [*key_columns, value_column]].rename(
        columns={value_column: FNT_FRACTION_COLUMN}
    )
    return rows.merge(frequency, on=key_columns, how="left")


def _summary_scope_rows(rows: pd.DataFrame, scope: str) -> pd.DataFrame:
    scoped = rows.loc[rows[ASR_SUMMARY_SCOPE_COLUMN].astype(str).eq(scope)].copy()
    return scoped.drop(columns=[ASR_SUMMARY_SCOPE_COLUMN])


def _with_cumulative_values(
    rows: pd.DataFrame,
    cumulative_rows: pd.DataFrame,
    *,
    context: FigureContext,
) -> pd.DataFrame:
    del context
    drop_columns = {
        VALUE_ARRAY_COLUMN,
        FNT_FRACTION_COLUMN,
        "public_row_id",
        "year",
        ASOCC_SSP_SCENARIO_COLUMN,
        "lca_ssp_scenario",
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
        ASR_FREQUENCY_VALUE_COLUMN,
        ASR_CUMULATIVE_FREQUENCY_VALUE_COLUMN,
        "mean",
        "std",
        "min",
        "p5",
        "p25",
        "median",
        "p75",
        "p95",
        "max",
        ASR_SUMMARY_METRIC_COLUMN,
    }
    key_columns = [
        column
        for column in rows.columns
        if column in cumulative_rows.columns and column not in drop_columns
    ]
    cumulative = cumulative_rows.loc[:, [*key_columns, VALUE_ARRAY_COLUMN, FNT_FRACTION_COLUMN]]
    cumulative = cumulative.rename(
        columns={
            VALUE_ARRAY_COLUMN: "__cumulative_values",
            FNT_FRACTION_COLUMN: CUMULATIVE_FNT_FRACTION_COLUMN,
        }
    )
    out = rows.merge(cumulative, on=key_columns, how="left")
    return out


def _frequency_value_column(rows: pd.DataFrame) -> str:
    if ASR_CUMULATIVE_FREQUENCY_VALUE_COLUMN in rows.columns:
        return ASR_CUMULATIVE_FREQUENCY_VALUE_COLUMN
    return ASR_FREQUENCY_VALUE_COLUMN


def _figure_scopes(*, rows: pd.DataFrame, context: FigureContext) -> list[pd.DataFrame]:
    scopes: list[pd.DataFrame] = []
    for cc_type, cc_rows in rows.groupby("cc_type", dropna=False, sort=True):
        if str(cc_type) == "static":
            ssp_slices = static_asocc_ssp_slices(
                cc_rows,
                requested_ssps=context.requested_asocc_ssps,
            )
            for ssp_rows in ssp_slices:
                scopes.extend(scope_slices(ssp_rows, ("lcia_method", "cc_bound")))
            continue
        dynamic_columns = [
            column
            for column in ("lcia_method", *DYNAMIC_SCOPE_COLUMNS)
            if column in cc_rows.columns
        ]
        scopes.extend(scope_slices(cc_rows, tuple(dynamic_columns)))
    return scopes


def _inter_method_active(context: FigureContext) -> bool:
    active = set(context.active_sources)
    return context.inter_method and (
        ASOCC_INTER_METHOD_SOURCE in active
        or any(source.endswith(ASOCC_INTER_METHOD_SOURCE) for source in active)
    )


def _is_dynamic_rows(rows: pd.DataFrame) -> bool:
    return "cc_type" in rows.columns and visible_values(rows, "cc_type") == [DYNAMIC_AR6_CC_TYPE]


def _raw_identity_is_dynamic(identity: pd.DataFrame) -> bool:
    values = {
        str(value).strip()
        for value in identity["cc_type"].dropna().astype(str).tolist()
        if str(value).strip()
    }
    return values == {DYNAMIC_AR6_CC_TYPE}

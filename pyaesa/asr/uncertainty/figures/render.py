"""ASR uncertainty figure orchestration."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
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
from pyaesa.asr.figures.axis import (
    ASR_NORMAL_SCALE,
    ASRScaleMode,
    asr_zero_log_scale_warning_message,
    asr_zero_log_scale_warning_needed,
)
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
    value_rows_from_runs,
    SUMMARY_STAT_COLUMNS,
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
from pyaesa.shared.figures.trajectory_bands import SUMMARY_COLUMNS as TRAJECTORY_SUMMARY_COLUMNS
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest

CUMULATIVE_VALUE_ARRAY_COLUMN = "__cumulative_values"

Plotter = Callable[..., list[Path]]


@dataclass(frozen=True)
class ASRUncertaintyFigureResult:
    """Rendered ASR uncertainty figure paths and summary warnings."""

    paths: list[Path]
    warning_messages: tuple[str, ...]


def render_asr_uncertainty_figures(
    *,
    manifest: UncertaintyManifest,
    paths: ASRUncertaintyRunPaths,
    figure_options: dict | None,
    figure_format: dict | None,
    status: StatusSink | None = None,
) -> ASRUncertaintyFigureResult:
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
        return ASRUncertaintyFigureResult(paths=[], warning_messages=())
    tables = read_figure_tables(
        context=context,
        include_cumulative=single_requested_year(context) is None,
    )
    clear_uncertainty_figure_scope(paths=paths)
    planned_jobs = tuple(
        _uncertainty_jobs(
            context=context,
            identity=tables.identity,
            summary=tables.summary,
            cumulative_identity=tables.cumulative_identity,
            cumulative_summary=tables.cumulative_summary,
        )
    )
    return ASRUncertaintyFigureResult(
        paths=render_figure_jobs(
            source="uncertainty_asr",
            jobs=lambda: iter(planned_jobs),
            status=status,
        ),
        warning_messages=_summary_warning_messages(planned_jobs),
    )


def _uncertainty_jobs(
    *,
    context: FigureContext,
    identity: pd.DataFrame,
    summary: pd.DataFrame,
    cumulative_identity: pd.DataFrame,
    cumulative_summary: pd.DataFrame,
) -> Iterator[PlannedFigureJob]:
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
) -> Iterator[PlannedFigureJob]:
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
        yield from _plan_inter_method_jobs(
            rows=inter_rows,
            context=context,
            plotter=plot_violin_scope,
            kind="single_year",
            polar_for_multi_impact=bool(context.polar_years),
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
            polar_for_multi_impact=bool(context.polar_years),
        )


def _multi_year_jobs(
    *,
    context: FigureContext,
    identity: pd.DataFrame,
    summary: pd.DataFrame,
    cumulative_identity: pd.DataFrame,
    cumulative_summary: pd.DataFrame,
) -> Iterator[PlannedFigureJob]:
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
            value_rows = _polar_value_rows(context=context, identity_rows=identity_rows)
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
            del value_rows
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
            _polar_value_rows(context=context, identity_rows=identity_rows)
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
    if context.inter_method:
        yield from _plan_polar_checkpoint_jobs(
            rows=inter_polar_rows,
            context=context,
            role="inter_method",
            label="inter_method",
            title_label="inter-method",
            family_label="ASR uncertainty",
        )
        yield from _plan_inter_method_jobs(
            rows=inter_rows,
            context=context,
            plotter=plot_band_scope,
            kind="multi_year",
            components=component_rows,
            global_ar6_source=global_ar6_source,
        )
    if context.multi_method:
        yield from _plan_multi_method_jobs(
            rows=method_rows,
            context=context,
            plotter=plot_mean_line_scope,
            kind="multi_year",
            mean_line=True,
            components=component_rows,
            global_ar6_source=global_ar6_source,
        )
    if context.per_method:
        yield from _plan_per_method_jobs(
            rows=method_rows,
            context=context,
            plotter=plot_band_scope,
            kind="multi_year",
            components=component_rows,
            global_ar6_source=global_ar6_source,
        )
        yield from _plan_polar_checkpoint_jobs(
            rows=method_polar_rows,
            context=context,
            role="per_method",
            label=None,
            title_label=None,
            family_label="ASR uncertainty",
        )


def _polar_value_rows(*, context: FigureContext, identity_rows: pd.DataFrame) -> pd.DataFrame:
    """Read ASR run distributions only for years requested by polar figures."""
    years = pd.Series(pd.to_numeric(identity_rows["year"], errors="raise"), copy=False).astype(int)
    polar_identity = identity_rows.loc[years.isin(list(context.polar_years))].copy()
    return value_rows_from_runs(context=context, identity_rows=polar_identity)


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
            product=product,
            family_label=family_label,
            group_legend=False,
            include_method_in_label=False,
            polar_for_multi_impact=polar_for_multi_impact,
            components=components,
            global_ar6_source=global_ar6_source,
        )


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
        product=product,
        family_label=family_label,
        group_legend=True,
        include_method_in_label=True,
        split_multi_year_impacts=False,
        mean_line=mean_line,
        components=components,
        global_ar6_source=global_ar6_source,
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
        product=product,
        family_label=family_label,
        group_legend=False,
        include_method_in_label=False,
        polar_for_multi_impact=polar_for_multi_impact,
        components=components,
        global_ar6_source=global_ar6_source,
    )


def _plan_polar_checkpoint_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    role: str,
    label: str | None,
    title_label: str | None,
    family_label: str,
) -> Iterator[PlannedFigureJob]:
    if rows.empty or not context.polar_years:
        return
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
                    yield from _polar_scope_jobs(
                        scope=year_scope,
                        context=context,
                        role=role,
                        label=scope_label,
                        title_label=title_label if label is not None else scope_label,
                        family_label=family_label,
                        studied_year=int(year),
                        selector_token=selector_token,
                        selector_title=selector_title,
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
    product: str | None,
    family_label: str,
    group_legend: bool,
    include_method_in_label: bool,
    split_multi_year_impacts: bool = False,
    mean_line: bool = False,
    polar_for_multi_impact: bool = False,
    components: ComponentDiagnosticRows | None = None,
    global_ar6_source: UncertaintyGlobalAR6Source | None = None,
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
                yield from _scope_jobs(
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
                    selector_token=selector_token,
                    selector_title=selector_title,
                )


def _scope_jobs(
    *,
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
    selector_token: str,
    selector_title: str,
) -> Iterator[PlannedFigureJob]:
    """Yield one or more ASR uncertainty jobs for a final figure scope."""
    if (
        polar_for_multi_impact
        and product is None
        and studied_year is not None
        and len(ordered_impacts(scope)) > 1
    ):
        yield from _polar_scope_jobs(
            scope=scope,
            context=context,
            role=role,
            label=label,
            title_label=title_label,
            family_label=family_label,
            studied_year=studied_year,
            selector_token=selector_token,
            selector_title=selector_title,
        )
        return
    scale_mode = _scale_mode_for_scope(scope, mean_line=mean_line)
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
    yield PlannedFigureJob(
        kind=kind,
        label=output_base.name,
        planned_outputs=_planned_scope_output_count(
            scope=scope,
            plotter=plotter,
        ),
        warning_contexts=_scale_warning_contexts(
            scope=scope,
            label=output_base.name,
            mean_line=mean_line,
        ),
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


def _polar_scope_jobs(
    *,
    scope: pd.DataFrame,
    context: FigureContext,
    role: str,
    label: str,
    title_label: str | None,
    family_label: str,
    studied_year: int,
    selector_token: str,
    selector_title: str,
) -> Iterator[PlannedFigureJob]:
    """Yield ASR uncertainty polar jobs for one selector and year scope."""
    scale_mode = _scale_mode_for_scope(scope, mean_line=False)
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
        yield PlannedFigureJob(
            kind=f"polar_{style}",
            label=output_base.name,
            warning_contexts=_scale_warning_contexts(
                scope=scope,
                label=output_base.name,
                mean_line=False,
            ),
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


def _planned_scope_output_count(*, scope: pd.DataFrame, plotter: Plotter) -> int:
    """Return the number of figure files rendered by one ASR uncertainty job."""
    has_cumulative = CUMULATIVE_VALUE_ARRAY_COLUMN in scope.columns
    if plotter is plot_band_scope:
        return 2 if len(ordered_impacts(scope)) > 1 or has_cumulative else 1
    if plotter is plot_mean_line_scope:
        dynamic_multiplier = 2 if has_cumulative else 1
        return max(1, len(ordered_impacts(scope))) * dynamic_multiplier
    return 1


def _scale_mode_for_scope(frame: pd.DataFrame, *, mean_line: bool) -> ASRScaleMode:
    values = _scale_values_from_frame(frame, mean_line=mean_line)
    return asr_scale_mode_for_values(values) if values.size else ASR_NORMAL_SCALE


def _scale_values_from_frame(frame: pd.DataFrame, *, mean_line: bool) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for value_column in (VALUE_ARRAY_COLUMN, CUMULATIVE_VALUE_ARRAY_COLUMN):
        if value_column in frame.columns:
            arrays.extend(np.asarray(values, dtype=np.float64) for values in frame[value_column])
    summary_source = ("mean",) if mean_line else TRAJECTORY_SUMMARY_COLUMNS
    summary_columns = [column for column in summary_source if column in frame.columns]
    if summary_columns:
        arrays.append(
            frame.loc[:, summary_columns]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(
                dtype=np.float64,
                copy=False,
            )
        )
    if not arrays:
        return np.empty(0, dtype=np.float64)
    return np.concatenate([np.asarray(array, dtype=np.float64).ravel() for array in arrays])


def _scale_warning_contexts(
    *,
    scope: pd.DataFrame,
    label: str,
    mean_line: bool,
) -> tuple[str, ...]:
    values = _scale_values_from_frame(scope, mean_line=mean_line)
    return (label,) if asr_zero_log_scale_warning_needed(values) else ()


def _summary_warning_messages(jobs: tuple[PlannedFigureJob, ...]) -> tuple[str, ...]:
    labels = tuple(
        dict.fromkeys(label for job in jobs for label in job.warning_contexts if str(label).strip())
    )
    if not labels:
        return ()
    return (asr_zero_log_scale_warning_message(labels=labels),)


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


def _figure_scopes(*, rows: pd.DataFrame, context: FigureContext) -> Iterator[pd.DataFrame]:
    """Yield one ASR uncertainty figure scope at a time."""
    for cc_type, cc_rows in rows.groupby("cc_type", dropna=False, sort=True):
        if str(cc_type) == "static":
            for ssp_rows in static_asocc_ssp_slices(
                cc_rows,
                requested_ssps=context.requested_asocc_ssps,
            ):
                yield from scope_slices(ssp_rows, ("lcia_method", "cc_bound"))
            continue
        dynamic_columns = [
            column
            for column in ("lcia_method", *DYNAMIC_SCOPE_COLUMNS)
            if column in cc_rows.columns
        ]
        yield from scope_slices(cc_rows, tuple(dynamic_columns))


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

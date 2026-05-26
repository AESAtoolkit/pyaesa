"""Run-level allocation orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyaesa.external_inputs.asocc.templates.templates import ensure_external_asocc_templates
from pyaesa.external_inputs.lca.paths import external_lca_root
from pyaesa.external_inputs.lca.templates import ensure_external_lca_templates
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
from pyaesa.shared.runtime.reporting.progress import StatusProgressPrinter
from pyaesa.shared.runtime.reporting.summary_log import write_summary_log

from ..io.logging import close_loggers_for_scope
from pyaesa.asocc.runtime.scope.branch_resolution import outputs_project_root
from pyaesa.asocc.runtime.paths.external import get_asocc_external_root
from pyaesa.asocc.runtime.paths.deterministic import _get_allocate_summary_log_path
from pyaesa.asocc.runtime.reporting.family import sync_asocc_branch_figures
from .run_allocate_support import (
    AllocateReport,
    build_run_summary_lines,
    clear_year_buffers,
    deterministic_output_root,
    emit_runtime_message,
    format_branch_label,
    format_indices_label,
    prune_year_scoped_caches,
    runtime_prefix,
)
from .setup.run_setup import PrepareContextRequest, _prepare_context
from .write.run_write import _write_outputs
from .yearly.run_year import _process_year


@dataclass(frozen=True)
class _RunCommonInputs:
    """Inputs shared by all per mode run branches."""

    project_name: str
    source: str
    agg_version: str | None
    agg_reg: bool | None
    agg_sec: bool | None
    years: int | list[int] | range | None
    historical_year_cap: int | None
    refresh: bool
    lcia_method: str | list[str] | None
    fu_code: str
    r_p: list[str] | None
    s_p: list[str] | None
    r_c: list[str] | None
    r_f: list[str] | None
    reference_years: int | list[int] | range | None
    ssp_scenario: str | list[str] | None
    projection_mode: str | None
    reg_window: list[int] | range | None
    l2_reuse_years: int | list[int] | range | None
    output_format: str
    intermediate_outputs: bool
    output_source_label: str | None = None


@dataclass(frozen=True)
class _ModeRunResult:
    """Resolved result payload for one deterministic branch."""

    skipped: bool
    proj_base: Any
    output_source_label: str
    fu_code: str
    requested_years: list[int]
    resolved_years: list[int]
    lcia_methods: list[str] | None
    run_signature: dict[str, Any]
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None
    summary_lines: list[str]
    reuse_status: str
    figure_paths: list[Path]
    output_paths: list[str]
    output_root: Path


def _mode_result(
    *,
    context,
    skipped: bool,
    summary_lines: list[str] | None = None,
    reuse_status: str,
    figure_paths: list[Path] | None = None,
    output_paths: list[str] | None = None,
) -> _ModeRunResult:
    """Build the deterministic branch result from the resolved run context."""
    return _ModeRunResult(
        skipped=skipped,
        proj_base=context.proj_base,
        output_source_label=context.output_source,
        fu_code=context.fu_code,
        requested_years=list(context.requested_years),
        resolved_years=list(context.resolved_years),
        lcia_methods=context.lcia_methods,
        run_signature=dict(context.run_signature),
        ssp_scenario_options_by_year=context.ssp_scenario_options_by_year,
        summary_lines=[] if summary_lines is None else list(summary_lines),
        reuse_status=reuse_status,
        figure_paths=[] if figure_paths is None else list(figure_paths),
        output_paths=list(getattr(context, "metadata_prior_outputs", None) or [])
        if output_paths is None
        else list(output_paths),
        output_root=deterministic_output_root(context=context, source=context.output_source),
    )


def _run_allocate_family(
    *,
    common: _RunCommonInputs,
    mode: str,
    group_indices: bool,
    l1_override: list[str] | None,
    combined_override: list[tuple[str, str]] | None,
    l2_one_step_override: list[str] | None,
    figures: bool,
    refresh: bool,
    figure_external_method: dict[str, list[str]] | None,
    figure_options: dict[str, bool],
    figure_output_format: str,
    figure_dpi: int,
    phase: PhasePrinter | NullPhasePrinter,
) -> AllocateReport:
    """Run the deterministic aSoCC family branch loop and aggregate the report."""
    project_root = outputs_project_root(project_name=common.project_name)
    try:
        figure_paths: list[Path] = []
        mode_result = _run_mode(
            common=common,
            mode=mode,
            show_mode_tag=False,
            group_indices=group_indices,
            l1_override=l1_override,
            combined_override=combined_override,
            l2_one_step_override=l2_one_step_override,
            variant_tag=None,
            figures=figures,
            refresh=refresh,
            figure_external_method=figure_external_method,
            figure_options=figure_options,
            figure_output_format=figure_output_format,
            figure_dpi=figure_dpi,
            phase=phase,
        )
        mode_figure_paths = list(mode_result.figure_paths)
        figure_paths.extend(mode_figure_paths)
        mode_lines = list(mode_result.summary_lines)
        report = AllocateReport(
            source=common.source,
            summaries=[mode_lines],
            figure_paths=sorted({Path(path) for path in figure_paths}),
            reuse_status=mode_result.reuse_status,
            output_root=mode_result.output_root,
        )
        write_summary_log(
            path=_get_allocate_summary_log_path(
                mode_result.proj_base,
                source=mode_result.output_source_label,
                agg_version=common.agg_version,
            ),
            summary=str(report),
        )
        return report
    finally:
        close_loggers_for_scope(project_root)


def _run_mode(
    *,
    common: _RunCommonInputs,
    mode: str,
    show_mode_tag: bool,
    group_indices: bool,
    l1_override: list[str] | None,
    combined_override: list[tuple[str, str]] | None,
    l2_one_step_override: list[str] | None,
    variant_tag: str | None,
    figures: bool,
    refresh: bool,
    figure_external_method: dict[str, list[str]] | None,
    figure_options: dict[str, bool],
    figure_output_format: str,
    figure_dpi: int,
    phase: PhasePrinter | NullPhasePrinter,
) -> _ModeRunResult:
    """Execute one deterministic branch for one resolved aggregation identity."""
    # Setup owns public validation and exact completed scope checks.
    context, state, skipped = _prepare_context(
        request=PrepareContextRequest(
            project_name=common.project_name,
            source=common.source,
            agg_version=common.agg_version,
            agg_reg=common.agg_reg,
            agg_sec=common.agg_sec,
            years=common.years,
            historical_year_cap=common.historical_year_cap,
            refresh=common.refresh,
            lcia_method=common.lcia_method,
            fu_code=common.fu_code,
            r_p=common.r_p,
            s_p=common.s_p,
            r_c=common.r_c,
            r_f=common.r_f,
            l_1=l1_override,
            l_2_combined_with_l_1=combined_override,
            l_2_one_step=l2_one_step_override,
            reference_years=common.reference_years,
            ssp_scenario=common.ssp_scenario,
            projection_mode=common.projection_mode,
            reg_window=common.reg_window,
            l2_reuse_years=common.l2_reuse_years,
            l1_reg_aggreg=mode,
            variant_tag=variant_tag,
            group_indices=group_indices,
            output_format=common.output_format,
            intermediate_outputs=common.intermediate_outputs,
            output_source_label=common.output_source_label,
        )
    )
    ensure_external_asocc_templates(
        external_dir=get_asocc_external_root(proj_base=context.proj_base)
    )
    ensure_external_lca_templates(external_dir=external_lca_root(project_base=context.proj_base))
    if skipped:
        reused_result = _mode_result(
            context=context,
            skipped=True,
            reuse_status="reused_exact",
        )
        figure_sync = sync_asocc_branch_figures(
            mode_result=reused_result,
            figures=figures,
            refresh=refresh,
            figure_external_method=figure_external_method,
            figure_options=figure_options,
            figure_output_format=figure_output_format,
            figure_dpi=figure_dpi,
            status_source="deterministic_asocc",
            status=phase,
        )
        summary_lines = build_run_summary_lines(
            context=context,
            state=state,
            show_mode_tag=show_mode_tag,
            figure_paths=list(figure_sync.figure_paths),
        )
        return _mode_result(
            context=context,
            skipped=True,
            summary_lines=summary_lines,
            reuse_status="reused_exact",
            figure_paths=list(figure_sync.figure_paths),
        )

    raw_compute_years = getattr(context, "compute_years", None)
    compute_years = list(raw_compute_years or context.resolved_years)
    persisted_years = {int(year) for year in context.resolved_years}

    branch_label = format_branch_label(
        context=context,
        mode=mode,
        grouped_mode=group_indices,
    )
    indices_label = format_indices_label(context.filters)
    branch_source_prefix = runtime_prefix(context=context, show_mode_tag=show_mode_tag)
    prefix = (
        f"{branch_source_prefix} Starting branch: fu={context.fu_code}, indices={indices_label}"
    )
    progress_action = f"[{mode}] computing" if show_mode_tag else "computing"
    compute_progress = StatusProgressPrinter(
        source="deterministic_asocc",
        action=progress_action,
        total=len(compute_years),
        status=phase,
    )
    setattr(state, "runtime_progress", compute_progress)
    setattr(state, "runtime_source_prefix", branch_source_prefix)
    emit_runtime_message(
        state=state,
        message=f"{prefix}, {branch_label}" if branch_label else prefix,
    )

    def _write_branch_outputs(
        *,
        write_metadata: bool = True,
        show_progress: bool = True,
        progress_label: str | None = None,
    ) -> None:
        _write_outputs(
            context=context,
            state=state,
            refresh=common.refresh,
            write_metadata=write_metadata,
            show_progress=show_progress,
            progress_label=progress_label,
            progress_prefix=branch_source_prefix,
        )

    try:
        for year in compute_years:
            # Year processing may skip when required MRIO enacting metrics
            # are not available.
            processed = _process_year(
                context=context,
                state=state,
                year=year,
                progress=compute_progress,
            )
            if not processed:
                compute_progress.skip_year()
                continue
            if int(year) not in persisted_years:
                clear_year_buffers(
                    context=context,
                    state=state,
                    preserve_l2_buckets={"l2_in_l1"},
                )
                continue
            # Keep year scoped compute caches bounded regardless of write cadence.
            prune_year_scoped_caches(state=state)
        _write_branch_outputs(
            show_progress=bool(state.processed_years),
        )
        computed_result = _mode_result(
            context=context,
            skipped=False,
            reuse_status="computed",
            output_paths=list(
                dict.fromkeys(
                    [
                        *(getattr(context, "metadata_prior_outputs", None) or []),
                        *state.outputs_all,
                    ]
                )
            ),
        )
        clear_year_buffers(context=context, state=state)
        figure_sync = sync_asocc_branch_figures(
            mode_result=computed_result,
            figures=figures,
            refresh=refresh,
            figure_external_method=figure_external_method,
            figure_options=figure_options,
            figure_output_format=figure_output_format,
            figure_dpi=figure_dpi,
            status_source="deterministic_asocc",
            status=phase or compute_progress,
        )
    finally:
        compute_progress.finish()
        delattr(state, "runtime_progress")
        delattr(state, "runtime_source_prefix")
    summary_lines = build_run_summary_lines(
        context=context,
        state=state,
        show_mode_tag=show_mode_tag,
        figure_paths=list(figure_sync.figure_paths),
    )
    requested_years = {int(year) for year in context.requested_years}
    computed_requested = {int(year) for year in context.resolved_years}
    skipped_already_saved = bool(requested_years - computed_requested)
    reuse_status = "partially_reused" if skipped_already_saved else "computed"
    return _mode_result(
        context=context,
        skipped=False,
        summary_lines=summary_lines,
        reuse_status=reuse_status,
        figure_paths=list(figure_sync.figure_paths),
    )

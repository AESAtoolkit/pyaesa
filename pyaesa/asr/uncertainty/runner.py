"""Public ASR uncertainty runtime orchestration."""

from dataclasses import replace
from typing import Any

from pyaesa.asr.uncertainty.figures.render import render_asr_uncertainty_figures
from pyaesa.asr.uncertainty.figures.reuse import render_reusable_asr_figures_if_requested
from pyaesa.asr.uncertainty.sources.lca_inputs import base_io_lca_args_from_allocate_args
from pyaesa.asr.uncertainty.runtime.component_inputs import (
    initial_asr_components,
    io_lca_progress_enabled,
)
from pyaesa.asr.uncertainty.runtime.checkpoints import run_asr_checkpoints
from pyaesa.asr.uncertainty.runtime.scope import build_asr_uncertainty_scope
from pyaesa.asr.uncertainty.sources.config import split_asr_uncertainty_config
from pyaesa.asr.uncertainty.io.manifest_payloads import (
    build_completed_asr_manifest,
    build_asr_manifest_context,
    initial_asr_sobol_status,
)
from pyaesa.asr.uncertainty.io.paths import build_asr_uncertainty_run_paths
from pyaesa.asr.uncertainty.evaluation.planning import build_asr_uncertainty_plan
from pyaesa.asr.uncertainty.sobol.runner import run_asr_sobol
from pyaesa.asr.uncertainty.io.source_methods import (
    build_asr_source_methods,
    write_asr_results_readme,
    write_asr_source_methods,
)
from pyaesa.shared.acc_asr_common.scope.composite import (
    base_asocc_kwargs_from_allocate_args,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    build_manifest,
    write_manifest,
)
from pyaesa.shared.uncertainty_assessment.run_state.figure_artifacts import (
    manifest_with_figure_artifacts,
)
from pyaesa.shared.uncertainty_assessment.run_state.report import (
    UncertaintyRunReport,
    uncertainty_report,
)
from pyaesa.shared.uncertainty_assessment.request.core import (
    BatchMemoryBlock,
    memory_bounded_batch_size,
    normalize_uncertainty_request,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.convergence import (
    convergence_run_checkpoints,
)
from pyaesa.shared.uncertainty_assessment.run_state.runs import (
    cleanup_monte_carlo_runs_for_refresh,
    compatible_completed_runs,
    complete_run_with_requested_runs,
)
from pyaesa.shared.uncertainty_assessment.sobol.plan import normalize_sobol_plan
from pyaesa.shared.uncertainty_assessment.io.tables import write_uncertainty_table
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    PHASE_A_LCA,
    PHASE_B2_ACC,
    PHASE_C_ASR,
    phase_reused_detail,
    phase_uncertainty_done_detail,
)
from pyaesa.shared.runtime.reporting.run_progress import (
    monte_carlo_run_progress,
)
from pyaesa.shared.runtime.reporting.phase import PhasePrinter
from pyaesa.shared.figures.request_validation import (
    normalize_figure_options,
    resolve_nested_polar_years,
)
from pyaesa.shared.selectors.time_selectors import normalize_requested_years
from pyaesa.shared.uncertainty_assessment.orchestration import (
    component_checkpoint_figures,
    manifest_output_root,
    phase_index_entry,
    progress_complete,
    uncertainty_phase_index_entry,
    write_uncertainty_phase_index,
)


def run_uncertainty_asr(
    *,
    project_name: str,
    source: str,
    group_reg: bool,
    group_sec: bool,
    group_version: str,
    years: int | list[int] | range,
    fu_code: str,
    r_p: str | list[str] | None,
    s_p: str | list[str] | None,
    r_c: str | list[str] | None,
    r_f: str | list[str] | None,
    aggreg_indices: bool,
    lcia_method: str | list[str],
    base_asocc_args: dict[str, Any] | None,
    external_method: dict[str, Any] | None,
    base_cc_args: dict[str, Any],
    lca_args: dict[str, Any],
    uncertainty_config: dict[str, Any],
    sobol_parameters: dict[str, Any] | None,
    output_format: str,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    subfigures: bool,
    refresh: bool,
) -> UncertaintyRunReport:
    """Run one public ASR uncertainty request."""
    phase = PhasePrinter("uncertainty_asr")
    config = dict(uncertainty_config)
    source_config = split_asr_uncertainty_config(config)
    runtime = normalize_uncertainty_request(
        family="asr",
        output_format=output_format,
        mc_parameters=config.get("mc_parameters"),
    )
    requested_years = normalize_requested_years(years)
    figure_options_norm = None
    if figures:
        figure_options_norm = normalize_figure_options(
            figure_options,
            allow_single_year_style=False,
            allow_polar_years=False,
            allow_per_method=True,
            allow_multi_method=True,
            allow_inter_method=True,
            allow_polar=True,
        )
        polar = dict(figure_options_norm["polar"])
        resolve_nested_polar_years(
            studied_years=requested_years,
            polar=polar,
            argument_name="figure_options.polar",
        )
    sobol_plan = normalize_sobol_plan(
        sobol_parameters=sobol_parameters,
        available_years=requested_years,
    )
    scope = build_asr_uncertainty_scope(
        project_name=project_name,
        source=source,
        group_reg=group_reg,
        group_sec=group_sec,
        group_version=group_version,
        years=years,
        fu_code=fu_code,
        r_p=r_p,
        s_p=s_p,
        r_c=r_c,
        r_f=r_f,
        aggreg_indices=aggreg_indices,
        lcia_method=lcia_method,
        base_asocc_args=base_asocc_args,
        external_method=external_method,
        base_cc_args=base_cc_args,
        lca_args=lca_args,
        sobol_enabled=sobol_plan.enabled,
    )
    checkpoints = convergence_run_checkpoints(runtime=runtime)
    render_subfigures = component_checkpoint_figures(
        runtime_mode=runtime.mode,
        subfigures=subfigures,
    )
    current_run_id: str | None = None
    acc_progress = monte_carlo_run_progress(source="uncertainty_acc", status=phase)
    lca_progress = monte_carlo_run_progress(
        source="uncertainty_io_lca",
        enabled=io_lca_progress_enabled(scope=scope, source_config=source_config),
        status=phase,
    )
    asr_progress = monte_carlo_run_progress(source="uncertainty_asr", status=phase)
    components = initial_asr_components(
        phase=phase,
        scope=scope,
        source_config=source_config,
        project_name=project_name,
        years=years,
        fu_code=fu_code,
        r_p=r_p,
        s_p=s_p,
        r_c=r_c,
        r_f=r_f,
        base_cc_args=base_cc_args,
        refresh=refresh,
        output_format=runtime.output_format,
        target_runs=checkpoints[0],
        parent_mode=runtime.mode,
        parent_max_runs=runtime.n_runs,
        render_subfigures=render_subfigures,
        figure_options=figure_options_norm,
        figure_format=figure_format,
        acc_progress=acc_progress,
        lca_progress=lca_progress,
    )
    acc_manifest = components.acc_manifest
    lca_input = components.lca_input
    current_run_id = components.run_id
    phase_entries = [
        uncertainty_phase_index_entry(
            phase_label=PHASE_B2_ACC,
            function_name="uncertainty_acc",
            manifest=acc_manifest,
            reuse_status=components.acc_reuse_status,
        ),
        phase_index_entry(
            phase_label=PHASE_A_LCA,
            function_name=lca_input.phase_function,
            reuse_status=lca_input.phase_reuse_status,
            output_root=lca_input.phase_output_root,
        ),
    ]
    phase.status("Building ASR uncertainty plan", owner="uncertainty_asr")
    plan = build_asr_uncertainty_plan(
        acc_manifest=acc_manifest,
        lca_input=lca_input,
        output_format=runtime.output_format,
    )
    plan = replace(plan, source_method_rows=build_asr_source_methods(plan=plan))
    runtime = replace(
        runtime,
        batch_size=memory_bounded_batch_size(
            runtime=runtime,
            row_count=len(plan.identity),
            extra_blocks=_asr_batch_memory_blocks(plan=plan),
        ),
    )
    context = build_asr_manifest_context(
        base_args=scope.base_args,
        runtime=runtime,
        plan=plan,
        sobol_status=initial_asr_sobol_status(
            sobol_plan=sobol_plan,
            active_sources=plan.active_sources,
        ),
        acc_reuse_status=components.acc_reuse_status,
    )
    phase.announce(PHASE_C_ASR, "uncertainty_asr")
    if refresh:
        cleanup_monte_carlo_runs_for_refresh(
            monte_carlo_root=scope.root,
            compatibility_key=context["compatibility_key"],
            run_id=None,
            arguments=context["arguments"],
            component_inventory=None,
        )
    reusable = complete_run_with_requested_runs(
        compatible=compatible_completed_runs(
            monte_carlo_root=scope.root,
            compatibility_key=context["compatibility_key"],
        ),
        requested_runs=runtime.n_runs,
        mode=runtime.mode,
        mc_parameters=context["mc_parameters"],
    )
    if reusable is not None and not refresh:
        progress_complete(
            progress=asr_progress,
            completed=reusable.manifest.completed_runs,
            max_runs=runtime.n_runs,
            mode=runtime.mode,
        )
        acc_progress.finish()
        lca_progress.finish()
        asr_progress.finish()
        reusable_manifest = render_reusable_asr_figures_if_requested(
            manifest=reusable.manifest,
            root=scope.root,
            figures=figures,
            figure_options=figure_options_norm,
            figure_format=figure_format,
            status=asr_progress,
        )
        phase.complete(
            phase_reused_detail(
                scope_name="ASR uncertainty",
                output_root=manifest_output_root(reusable.manifest),
            ),
            owner="uncertainty_asr",
        )
        write_uncertainty_phase_index(
            manifest=reusable_manifest,
            entries=[
                *phase_entries,
                uncertainty_phase_index_entry(
                    phase_label=PHASE_C_ASR,
                    function_name="uncertainty_asr",
                    manifest=reusable_manifest,
                    reuse_status="reused_exact",
                ),
            ],
        )
        phase.finish()
        return uncertainty_report(manifest=reusable_manifest, reuse_status="reused_exact")
    manifest = build_manifest(
        family="asr",
        mode=runtime.mode,
        output_format=runtime.output_format,
        active_sources=plan.active_sources,
        completed_runs=0,
        status="running",
        requested_runs=runtime.n_runs,
        mc_parameters=context["mc_parameters"],
        source_parameters=context["source_parameters"],
        arguments=context["arguments"],
        deterministic_prerequisites=context["deterministic_prerequisites"],
        external_inputs=context["external_inputs"],
        sobol=context["sobol"],
        component_inventory=context["component_inventory"],
        compatibility_key=context["compatibility_key"],
        compatibility_context=context["compatibility_context"],
        run_id=current_run_id,
    )
    paths = build_asr_uncertainty_run_paths(
        monte_carlo_root=scope.root,
        run_id=manifest.run_id,
        output_format=runtime.output_format,
    )
    write_manifest(path=paths.scope_manifest, manifest=manifest)
    write_uncertainty_table(
        path=paths.public_row_identity,
        frame=plan.identity,
        output_format=runtime.output_format,
    )
    if plan.has_cumulative_outputs:
        write_uncertainty_table(
            path=paths.cumulative_row_identity,
            frame=plan.cumulative_identity,
            output_format=runtime.output_format,
        )
    write_asr_source_methods(path=paths.source_methods, rows=plan.source_method_rows)
    write_asr_results_readme(
        path=paths.results_readme,
        active_sources=plan.active_sources,
        run_layout=plan.asr_run_layout,
        include_cumulative=plan.has_cumulative_outputs,
    )
    try:
        try:
            checkpoint_result = run_asr_checkpoints(
                paths=paths,
                runtime=runtime,
                checkpoints=checkpoints,
                initial_plan=plan,
                initial_acc_manifest=acc_manifest,
                initial_acc_session=components.acc_session,
                initial_lca_input=lca_input,
                initial_lca_session=components.lca_session,
                project_name=project_name,
                years=years,
                shared_methods=scope.shared_methods,
                fu_code=fu_code,
                r_p=r_p,
                s_p=s_p,
                r_c=r_c,
                r_f=r_f,
                mrio_scope=scope.mrio_scope,
                asocc_config=scope.asocc_config,
                base_cc_args=base_cc_args,
                source_config=source_config,
                external_method=scope.external_method,
                proj_base=scope.proj_base,
                source_label=scope.source_label,
                lca_type=scope.lca_type,
                lca_version_name=scope.lca_version_name,
                base_allocate_args=scope.base_allocate_args,
                output_format=runtime.output_format,
                phase=phase,
                render_subfigures=render_subfigures,
                subfigures=subfigures,
                figure_options=figure_options_norm,
                figure_format=figure_format,
                run_id=current_run_id,
                acc_progress=acc_progress,
                lca_progress=lca_progress,
                asr_progress=asr_progress,
                progress_mode=runtime.mode,
                progress_max_runs=runtime.n_runs,
                progress_component=False,
            )
        finally:
            acc_progress.finish()
            lca_progress.finish()
        plan = checkpoint_result.plan
        acc_manifest = checkpoint_result.acc_manifest
        lca_input = checkpoint_result.lca_input
        completed_runs = checkpoint_result.completed_runs
        convergence = checkpoint_result.convergence
        context = build_asr_manifest_context(
            base_args=scope.base_args,
            runtime=runtime,
            plan=plan,
            sobol_status=initial_asr_sobol_status(
                sobol_plan=sobol_plan,
                active_sources=plan.active_sources,
            ),
            acc_reuse_status=components.acc_reuse_status,
        )
        sobol_result = run_asr_sobol(
            paths=paths,
            runtime=runtime,
            branches=scope.branches,
            base_asocc_args=base_asocc_kwargs_from_allocate_args(
                base_allocate_args=scope.base_allocate_args
            ),
            base_io_lca_args=base_io_lca_args_from_allocate_args(
                base_allocate_args={**scope.base_allocate_args, "lcia_method": scope.shared_methods}
            ),
            acc_uncertainty_config=source_config.acc_config,
            lca_uncertainty_config=source_config.lca_config,
            external_method=scope.external_method,
            lca_input=lca_input,
            full_years=years,
            sobol_plan=sobol_plan,
            status=phase,
        )
        complete = build_completed_asr_manifest(
            paths=paths,
            runtime=runtime,
            plan=plan,
            context=context,
            run_id=manifest.run_id,
            completed_runs=completed_runs,
            convergence=convergence,
            sobol_status=sobol_result.status or context["sobol"],
        )
        if figures:
            figure_paths = render_asr_uncertainty_figures(
                manifest=complete,
                paths=paths,
                figure_options=figure_options_norm,
                figure_format=figure_format,
                status=asr_progress,
            )
            complete = manifest_with_figure_artifacts(
                manifest=complete,
                figure_paths=figure_paths,
                figure_options=figure_options_norm,
                figure_format=figure_format,
            )
        write_manifest(path=paths.scope_manifest, manifest=complete)
        asr_progress.finish()
        phase.complete(
            phase_uncertainty_done_detail(
                scope_name="ASR uncertainty",
                mode=complete.mode,
                convergence=complete.convergence,
                output_root=manifest_output_root(complete),
            ),
            owner="uncertainty_asr",
        )
        write_uncertainty_phase_index(
            manifest=complete,
            entries=[
                *phase_entries,
                uncertainty_phase_index_entry(
                    phase_label=PHASE_C_ASR,
                    function_name="uncertainty_asr",
                    manifest=complete,
                    reuse_status="computed",
                ),
            ],
        )
        phase.finish()
        return uncertainty_report(manifest=complete, reuse_status="computed")
    finally:
        asr_progress.finish()


def _asr_batch_memory_blocks(*, plan) -> tuple[BatchMemoryBlock, ...]:
    blocks = [
        BatchMemoryBlock("yearly_summary_values", len(plan.summary_identity)),
        BatchMemoryBlock("lca_input_values", len(plan.identity)),
        BatchMemoryBlock("acc_input_values", len(plan.identity)),
    ]
    if plan.has_cumulative_outputs:
        blocks.extend(
            [
                BatchMemoryBlock("cumulative_output_values", len(plan.cumulative_identity), 3),
                BatchMemoryBlock(
                    "cumulative_summary_values",
                    len(plan.cumulative_summary_identity),
                ),
            ]
        )
    return tuple(blocks)

"""Public aCC uncertainty runtime orchestration."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from pyaesa.acc.uncertainty.io.paths import build_acc_uncertainty_run_paths
from pyaesa.acc.uncertainty.figures.render import render_acc_uncertainty_figures
from pyaesa.acc.uncertainty.figures.reuse import render_reusable_acc_figures_if_requested
from pyaesa.acc.uncertainty.io.manifest_payloads import (
    acc_outputs_payload,
    acc_public_output_payload,
    build_completed_acc_manifest,
    build_acc_manifest_context,
    initial_acc_sobol_status,
)
from pyaesa.acc.uncertainty.runtime.models import (
    ACCAsoccInput,
    ACCDynamicCCInput,
    ACCUncertaintyPlan,
)
from pyaesa.acc.uncertainty.runtime.component_inputs import (
    ACCInitialComponents,
    initial_acc_components,
)
from pyaesa.acc.uncertainty.runtime.checkpoints import run_acc_checkpoints
from pyaesa.acc.uncertainty.runtime.scope import (
    ACCUncertaintyScope,
    build_acc_uncertainty_scope,
)
from pyaesa.acc.uncertainty.request.normalization import (
    AR6_DYNAMIC_CC_SOURCE,
    asocc_uncertainty_config_for_acc,
    dynamic_cc_source_parameters,
    normalize_acc_uncertainty_config,
)
from pyaesa.acc.uncertainty.evaluation.planning import build_acc_uncertainty_plan
from pyaesa.acc.uncertainty.sobol.runner import run_acc_sobol
from pyaesa.acc.uncertainty.io.source_methods import (
    write_acc_results_readme,
    write_acc_source_methods,
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
    memory_bounded_batch_size,
    normalize_uncertainty_request,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.composite import (
    ComponentRun,
    component_inventory_finalizes,
    component_inventory_parent_convergence,
    component_inventory_progress_parameters,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.convergence import (
    convergence_run_checkpoints,
)
from pyaesa.shared.uncertainty_assessment.run_state.runs import (
    appendable_completed_run,
    cleanup_monte_carlo_runs_for_refresh,
    compatible_completed_runs,
    compatible_completed_run_for_id,
    complete_run_with_requested_runs,
)
from pyaesa.shared.uncertainty_assessment.sobol.plan import normalize_sobol_plan
from pyaesa.shared.uncertainty_assessment.io.tables import write_uncertainty_table
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    CompositePhaseIndexEntry,
    PHASE_B1_AR6_DYNAMIC_CC,
    PHASE_B1_ASOCC,
    PHASE_B2_ACC,
    phase_reused_detail,
    phase_uncertainty_done_detail,
)
from pyaesa.shared.runtime.reporting.run_progress import (
    RunProgressPrinter,
    monte_carlo_run_progress,
    visible_status_for_run_work,
)
from pyaesa.shared.runtime.reporting.phase import PhasePrinter
from pyaesa.shared.selectors.time_selectors import normalize_requested_years
from pyaesa.shared.uncertainty_assessment.orchestration import (
    component_checkpoint_figures,
    deterministic_phase_index_entry,
    manifest_output_root,
    progress_complete,
    uncertainty_phase_index_entry,
    write_uncertainty_phase_index,
)


@dataclass(frozen=True)
class ACCComponentSession:
    """Prepared aCC state reused by parent convergence checkpoints."""

    scope: ACCUncertaintyScope
    plan: ACCUncertaintyPlan
    asocc_session: Any | None
    dynamic_cc_session: Any | None
    run_id: str | None
    output_state: Any | None = None


def run_uncertainty_acc(
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
    uncertainty_config: dict[str, Any],
    sobol_parameters: dict[str, Any] | None,
    output_format: str,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    subfigures: bool,
    refresh: bool,
    phase: PhasePrinter | None = None,
    component_inventory: dict[str, Any] | None = None,
    run_id: str | None = None,
    show_progress: bool = True,
    show_component_progress: bool = True,
    progress: RunProgressPrinter | None = None,
) -> UncertaintyRunReport:
    """Run one public aCC uncertainty request."""
    return run_uncertainty_acc_component(
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
        uncertainty_config=uncertainty_config,
        sobol_parameters=sobol_parameters,
        output_format=output_format,
        figures=figures,
        figure_options=figure_options,
        figure_format=figure_format,
        subfigures=subfigures,
        refresh=refresh,
        phase=phase,
        component_inventory=component_inventory,
        run_id=run_id,
        show_progress=show_progress,
        show_component_progress=show_component_progress,
        progress=progress,
    ).report


def run_uncertainty_acc_component(
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
    uncertainty_config: dict[str, Any],
    sobol_parameters: dict[str, Any] | None,
    output_format: str,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    subfigures: bool,
    refresh: bool,
    phase: PhasePrinter | None,
    component_inventory: dict[str, Any] | None,
    run_id: str | None,
    show_progress: bool,
    show_component_progress: bool,
    progress: RunProgressPrinter | None,
    component_session: ACCComponentSession | None = None,
    finalize_component_inventory: bool = False,
) -> ComponentRun:
    """Run or append one aCC component inventory using local component sessions."""
    owns_phase = phase is None
    phase_owner = PhasePrinter("uncertainty_acc") if owns_phase else phase
    config = normalize_acc_uncertainty_config(uncertainty_config)
    runtime = normalize_uncertainty_request(
        family="acc",
        output_format=output_format,
        mc_parameters=config.get("mc_parameters"),
    )
    component_parent_convergence = component_inventory_parent_convergence(
        component_inventory=component_inventory
    )
    finalize_outputs = component_inventory_finalizes(
        component_inventory=component_inventory,
        finalize_component_inventory=finalize_component_inventory,
    )
    had_component_session = component_session is not None
    progress_parameters = component_inventory_progress_parameters(
        component_inventory=component_inventory,
        runtime_mode=runtime.mode,
        runtime_n_runs=runtime.n_runs,
    )
    current_run_id = None if run_id is None else str(run_id)
    requested_years = normalize_requested_years(years)
    sobol_plan = normalize_sobol_plan(
        sobol_parameters=sobol_parameters,
        available_years=requested_years,
    )
    scope = (
        build_acc_uncertainty_scope(
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
            base_cc_args=base_cc_args,
            external_method=external_method,
        )
        if component_session is None
        else component_session.scope
    )
    checkpoints = convergence_run_checkpoints(runtime=runtime)
    render_component_checkpoint_figures = component_checkpoint_figures(
        runtime_mode=runtime.mode,
        subfigures=subfigures,
    )
    asocc_progress = monte_carlo_run_progress(
        source="uncertainty_asocc",
        enabled=show_component_progress,
        status=phase_owner,
    )
    dynamic_cc_progress = monte_carlo_run_progress(
        source="uncertainty_ar6_cc",
        enabled=show_component_progress,
        status=phase_owner,
    )
    external_progress = progress is not None
    acc_progress = progress or monte_carlo_run_progress(
        source="uncertainty_acc",
        enabled=show_progress,
        status=phase_owner,
    )
    acc_figure_status = visible_status_for_run_work(
        progress=acc_progress,
        fallback=phase_owner,
        progress_enabled=show_progress or external_progress,
    )
    component_run_id = (
        current_run_id
        if current_run_id is not None or component_session is None
        else component_session.run_id
    )
    if component_session is not None and component_session.output_state is not None:
        components = ACCInitialComponents(
            asocc_input=component_session.plan.asocc_input,
            asocc_session=component_session.asocc_session,
            dynamic_cc_input=component_session.plan.dynamic_cc_input,
            dynamic_cc_session=component_session.dynamic_cc_session,
            run_id=component_run_id,
        )
    else:
        components = initial_acc_components(
            phase=phase_owner,
            scope=scope,
            config=config,
            external_method=external_method,
            output_format=runtime.output_format,
            target_runs=checkpoints[0],
            parent_mode=progress_parameters["mode"],
            parent_max_runs=progress_parameters["max_runs"],
            component_figures=render_component_checkpoint_figures,
            figure_options=figure_options,
            figure_format=figure_format,
            current_run_id=component_run_id,
            refresh=refresh if component_session is None else False,
            asocc_progress=asocc_progress,
            dynamic_cc_progress=dynamic_cc_progress,
            asocc_session=None if component_session is None else component_session.asocc_session,
            dynamic_cc_session=(
                None if component_session is None else component_session.dynamic_cc_session
            ),
            finalize_component_inventory=bool(finalize_outputs and runtime.mode == "fixed"),
        )
    asocc_input = components.asocc_input
    dynamic_cc_input = components.dynamic_cc_input
    asocc_session = components.asocc_session
    dynamic_cc_session = components.dynamic_cc_session
    current_run_id = components.run_id
    phase_entries = [
        *_asocc_phase_entries(asocc_input=asocc_input),
        *_dynamic_cc_phase_entries(dynamic_cc_input=dynamic_cc_input),
    ]
    if component_session is None:
        phase_owner.status("Building aCC uncertainty plan", owner="uncertainty_acc")
        plan = build_acc_uncertainty_plan(
            asocc_input=asocc_input,
            dynamic_cc_input=dynamic_cc_input,
            branches=scope.branches,
            output_format=runtime.output_format,
        )
    else:
        plan = replace(
            component_session.plan,
            asocc_input=asocc_input,
            dynamic_cc_input=dynamic_cc_input,
        )
    runtime = replace(
        runtime,
        batch_size=memory_bounded_batch_size(runtime=runtime, row_count=len(plan.identity)),
    )
    context = build_acc_manifest_context(
        base_args=scope.base_args,
        runtime=runtime,
        plan=plan,
        sobol_status=initial_acc_sobol_status(
            sobol_plan=sobol_plan,
            active_sources=plan.active_sources,
        ),
        component_inventory=component_inventory,
    )
    phase_owner.announce(PHASE_B2_ACC, "uncertainty_acc")
    if refresh:
        cleanup_monte_carlo_runs_for_refresh(
            monte_carlo_root=scope.root,
            compatibility_key=context["compatibility_key"],
            run_id=run_id,
            arguments=context["arguments"],
            component_inventory=component_inventory,
        )
    required_run = compatible_completed_run_for_id(
        monte_carlo_root=scope.root,
        run_id=current_run_id,
    )
    compatible = (
        compatible_completed_runs(
            monte_carlo_root=scope.root,
            compatibility_key=context["compatibility_key"],
        )
        if run_id is None
        else (() if required_run is None else (required_run,))
    )
    reusable = complete_run_with_requested_runs(
        compatible=compatible,
        requested_runs=runtime.n_runs,
        mode=runtime.mode,
        mc_parameters=context["mc_parameters"],
    )
    if reusable is not None and not refresh:
        progress_complete(
            progress=acc_progress,
            completed=reusable.manifest.completed_runs,
            max_runs=max(progress_parameters["max_runs"], reusable.manifest.completed_runs),
            mode=progress_parameters["mode"],
            component=progress_parameters["component"],
            visible=not all((had_component_session, component_parent_convergence)),
        )
        asocc_progress.finish()
        dynamic_cc_progress.finish()
        acc_progress.finish()
        reused_manifest = render_reusable_acc_figures_if_requested(
            manifest=reusable.manifest,
            figures=figures,
            figure_options=figure_options,
            figure_format=figure_format,
            status=acc_figure_status,
        )
        if not component_parent_convergence:
            phase_owner.complete(
                phase_reused_detail(
                    scope_name="aCC uncertainty",
                    output_root=manifest_output_root(reused_manifest),
                )
            )
        write_uncertainty_phase_index(
            manifest=reused_manifest,
            entries=[
                *phase_entries,
                uncertainty_phase_index_entry(
                    phase_label=PHASE_B2_ACC,
                    function_name="uncertainty_acc",
                    manifest=reused_manifest,
                    reuse_status="reused_exact",
                ),
            ],
        )
        if owns_phase:
            phase_owner.finish()
        return ComponentRun(
            report=uncertainty_report(
                manifest=reused_manifest,
                reuse_status="reused_exact",
            ),
            session=ACCComponentSession(
                scope=scope,
                plan=plan,
                asocc_session=asocc_session,
                dynamic_cc_session=dynamic_cc_session,
                run_id=reused_manifest.run_id,
            ),
        )
    append_compatible = compatible
    if component_inventory is not None:
        required_append_run = compatible_completed_run_for_id(
            monte_carlo_root=scope.root,
            run_id=current_run_id,
            include_running_component_inventory=True,
        )
        append_compatible = (
            compatible_completed_runs(
                monte_carlo_root=scope.root,
                compatibility_key=context["compatibility_key"],
                include_running_component_inventory=True,
            )
            if current_run_id is None
            else (() if required_append_run is None else (required_append_run,))
        )
    append_run = (
        appendable_completed_run(
            compatible=append_compatible,
            mode=runtime.mode,
            max_completed_runs=runtime.n_runs,
        )
        if (
            component_inventory is not None
            and (component_session is None or component_session.output_state is None)
        )
        else None
    )
    manifest = build_manifest(
        family="acc",
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
        sobol=context["sobol"],
        component_inventory=context["component_inventory"],
        compatibility_key=context["compatibility_key"],
        compatibility_context=context["compatibility_context"],
        run_id=(append_run.manifest.run_id if append_run is not None else current_run_id),
    )
    paths = build_acc_uncertainty_run_paths(
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
    write_acc_source_methods(path=paths.source_methods, rows=plan.source_method_rows)
    write_acc_results_readme(
        path=paths.results_readme,
        active_sources=plan.active_sources,
        run_layout=plan.acc_run_layout,
    )
    try:
        try:
            checkpoint_result = run_acc_checkpoints(
                paths=paths,
                runtime=runtime,
                checkpoints=_acc_component_checkpoints(
                    checkpoints=checkpoints,
                    component_session=component_session,
                ),
                initial_plan=plan,
                initial_asocc_input=asocc_input,
                initial_asocc_session=asocc_session,
                initial_dynamic_cc_input=dynamic_cc_input,
                initial_dynamic_cc_session=dynamic_cc_session,
                start_completed_runs=(
                    0 if append_run is None else append_run.manifest.completed_runs
                ),
                base_allocate_args=scope.base_allocate_args,
                external_lcia_methods=scope.shared_methods,
                config=config,
                external_method=external_method,
                output_format=runtime.output_format,
                dynamic_branches=scope.dynamic_branches,
                years=years,
                render_component_checkpoint_figures=render_component_checkpoint_figures,
                subfigures=subfigures,
                figure_options=figure_options,
                figure_format=figure_format,
                run_id=current_run_id,
                asocc_progress=asocc_progress,
                dynamic_cc_progress=dynamic_cc_progress,
                acc_progress=acc_progress,
                progress_mode=progress_parameters["mode"],
                progress_max_runs=progress_parameters["max_runs"],
                progress_component=progress_parameters["component"],
                initial_output_state=(
                    None if component_session is None else component_session.output_state
                ),
                finalize_outputs=finalize_outputs,
            )
        finally:
            asocc_progress.finish()
            dynamic_cc_progress.finish()
        plan = checkpoint_result.plan
        asocc_input = checkpoint_result.asocc_input
        dynamic_cc_input = checkpoint_result.dynamic_cc_input
        asocc_session = checkpoint_result.asocc_session
        dynamic_cc_session = checkpoint_result.dynamic_cc_session
        output_state = checkpoint_result.output_state
        completed_runs = checkpoint_result.completed_runs
        convergence = checkpoint_result.convergence
        context = build_acc_manifest_context(
            base_args=scope.base_args,
            runtime=runtime,
            plan=plan,
            sobol_status=initial_acc_sobol_status(
                sobol_plan=sobol_plan,
                active_sources=plan.active_sources,
            ),
            component_inventory=component_inventory,
        )
        sobol_result = run_acc_sobol(
            paths=paths,
            runtime=runtime,
            branches=scope.branches,
            base_asocc_args=base_asocc_kwargs_from_allocate_args(
                base_allocate_args=scope.base_allocate_args
            ),
            asocc_uncertainty_config=asocc_uncertainty_config_for_acc(config),
            external_method=external_method,
            dynamic_cc_config=dynamic_cc_source_parameters(config.get(AR6_DYNAMIC_CC_SOURCE)),
            full_years=years,
            sobol_plan=sobol_plan,
            status=phase_owner,
        )
        if finalize_outputs:
            public_output = acc_public_output_payload(
                paths=paths,
                output_format=runtime.output_format,
                run_layout=plan.acc_run_layout,
            )
            complete = build_completed_acc_manifest(
                paths=paths,
                runtime=runtime,
                plan=plan,
                context=context,
                run_id=manifest.run_id,
                completed_runs=completed_runs,
                convergence=convergence,
                sobol_status=sobol_result.status or context["sobol"],
                public_output=public_output,
            )
        else:
            artifacts = acc_outputs_payload(paths=paths, output_format=runtime.output_format)
            artifacts["public_output"] = {"acc_runs": {"layout": plan.acc_run_layout}}
            complete = build_manifest(
                family="acc",
                mode=runtime.mode,
                output_format=runtime.output_format,
                active_sources=plan.active_sources,
                completed_runs=completed_runs,
                status="running",
                requested_runs=runtime.n_runs,
                mc_parameters=context["mc_parameters"],
                source_parameters=context["source_parameters"],
                arguments=context["arguments"],
                deterministic_prerequisites=context["deterministic_prerequisites"],
                artifacts=artifacts,
                sobol=context["sobol"],
                component_inventory=context["component_inventory"],
                compatibility_key=context["compatibility_key"],
                compatibility_context=context["compatibility_context"],
                run_id=manifest.run_id,
            )
        if figures:
            figure_paths = render_acc_uncertainty_figures(
                manifest=complete,
                paths=paths,
                figure_options=figure_options,
                figure_format=figure_format,
                status=acc_figure_status,
            )
            complete = manifest_with_figure_artifacts(
                manifest=complete,
                figure_paths=figure_paths,
                figure_options=figure_options,
                figure_format=figure_format,
            )
        write_manifest(path=paths.scope_manifest, manifest=complete)
        acc_progress.finish()
        if not component_parent_convergence:
            phase_owner.complete(
                phase_uncertainty_done_detail(
                    scope_name="aCC uncertainty",
                    mode=complete.mode,
                    convergence=complete.convergence,
                    output_root=manifest_output_root(complete),
                )
            )
        if finalize_outputs:
            write_uncertainty_phase_index(
                manifest=complete,
                entries=[
                    *phase_entries,
                    uncertainty_phase_index_entry(
                        phase_label=PHASE_B2_ACC,
                        function_name="uncertainty_acc",
                        manifest=complete,
                        reuse_status="computed",
                    ),
                ],
            )
        if owns_phase:
            phase_owner.finish()
    finally:
        acc_progress.finish()
    return ComponentRun(
        report=uncertainty_report(
            manifest=complete,
            reuse_status="computed",
        ),
        session=ACCComponentSession(
            scope=scope,
            plan=plan,
            asocc_session=asocc_session,
            dynamic_cc_session=dynamic_cc_session,
            run_id=complete.run_id,
            output_state=output_state,
        ),
    )


def _acc_component_checkpoints(
    *,
    checkpoints: tuple[int, ...],
    component_session: ACCComponentSession | None,
) -> tuple[int, ...]:
    if component_session is None or component_session.output_state is None:
        return checkpoints
    completed = int(component_session.output_state.completed_runs)
    return (completed, *(checkpoint for checkpoint in checkpoints if int(checkpoint) != completed))


def _asocc_phase_entries(*, asocc_input: ACCAsoccInput) -> list[CompositePhaseIndexEntry]:
    """Return the completed aSoCC component phase entry for one aCC run."""
    if asocc_input.manifest is not None:
        return [
            uncertainty_phase_index_entry(
                phase_label=PHASE_B1_ASOCC,
                function_name="uncertainty_asocc",
                manifest=asocc_input.manifest,
                reuse_status=asocc_input.reuse_status,
            )
        ]
    return [
        deterministic_phase_index_entry(
            phase_label=PHASE_B1_ASOCC,
            function_name="deterministic_asocc",
            metadata_path=cast(Path, asocc_input.deterministic_manifest_path),
            reuse_status=asocc_input.reuse_status,
        )
    ]


def _dynamic_cc_phase_entries(
    *,
    dynamic_cc_input: ACCDynamicCCInput | None,
) -> list[CompositePhaseIndexEntry]:
    """Return the completed dynamic AR6 CC component phase entry for one aCC run."""
    if dynamic_cc_input is None:
        return []
    if dynamic_cc_input.manifest is not None:
        return [
            uncertainty_phase_index_entry(
                phase_label=PHASE_B1_AR6_DYNAMIC_CC,
                function_name="uncertainty_ar6_cc",
                manifest=dynamic_cc_input.manifest,
                reuse_status=dynamic_cc_input.reuse_status,
            )
        ]
    return [
        deterministic_phase_index_entry(
            phase_label=PHASE_B1_AR6_DYNAMIC_CC,
            function_name="deterministic_ar6_cc",
            metadata_path=cast(Path, dynamic_cc_input.deterministic_manifest_path),
            reuse_status=dynamic_cc_input.reuse_status,
        )
    ]

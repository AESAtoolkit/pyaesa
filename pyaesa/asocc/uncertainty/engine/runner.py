"""aSoCC uncertainty public runtime orchestration."""

from dataclasses import dataclass, replace
from typing import Any

from pyaesa.asocc.inter_method_tools.tree_artifacts import write_inter_method_tree_artifacts
from pyaesa.asocc.uncertainty.engine.planning import (
    build_asocc_sampling_scope,
    build_asocc_source_scope,
    runtime_batch_size,
)
from pyaesa.asocc.uncertainty.engine.monte_carlo.run_execution import (
    write_monte_carlo_run_outputs,
)
from pyaesa.asocc.uncertainty.engine.reuse.prerequisites import (
    prepare_asocc_deterministic_prerequisite,
)
from pyaesa.asocc.uncertainty.engine.phase_reporting import (
    PHASE_B1_ASOCC,
    AsoccPhasePrinter,
    asocc_deterministic_phase_entries,
    asocc_phase_owner,
    complete_asocc_uncertainty_phase,
    complete_deterministic_asocc_phase,
    write_asocc_phase_index,
)
from pyaesa.asocc.uncertainty.engine.reuse.reuse import (
    appendable_run_for_runtime,
    compatible_complete_run,
    compatible_complete_sobol_run,
    compatible_completed_run_id_for_context,
    compatible_completed_runs_for_context,
)
from pyaesa.asocc.uncertainty.engine.sobol.runner import run_asocc_sobol
from pyaesa.asocc.uncertainty.engine.sobol.scope import selected_sobol_years
from pyaesa.asocc.uncertainty.figures.reuse import (
    render_reusable_asocc_figures_if_requested,
)
from pyaesa.asocc.uncertainty.figures.render import render_asocc_uncertainty_figures
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import (
    load_final_deterministic_asocc_rows,
    resolve_final_deterministic_asocc_row_scope,
)
from pyaesa.asocc.uncertainty.inputs.external_rows import (
    external_plan_for_years,
)
from pyaesa.asocc.uncertainty.io.manifest_payloads import (
    build_completed_asocc_manifest,
    manifest_context,
    outputs_payload,
)
from pyaesa.asocc.uncertainty.io.paths import (
    asocc_monte_carlo_root,
    build_asocc_uncertainty_run_paths,
)
from pyaesa.asocc.uncertainty.io.run_logs import write_run_logs
from pyaesa.asocc.uncertainty.sources.names import (
    ASOCC_UNCERTAINTY_SOURCES,
    DEFAULT_ASOCC_UNCERTAINTY_SOURCES,
    INTER_METHOD_SOURCE,
    REFERENCE_YEAR_SOURCE,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import run_seed_from_run_id
from pyaesa.shared.uncertainty_assessment.monte_carlo.composite import (
    ComponentRun,
    component_inventory_finalizes,
    component_inventory_parent_convergence,
    component_inventory_progress_parameters,
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
from pyaesa.shared.uncertainty_assessment.run_state.runs import (
    cleanup_monte_carlo_runs_for_refresh,
)
from pyaesa.shared.uncertainty_assessment.request.core import (
    normalize_uncertainty_request,
)
from pyaesa.shared.uncertainty_assessment.sobol.plan import normalize_sobol_plan
from pyaesa.shared.uncertainty_assessment.request.sources import (
    build_source_activation_plan,
)
from pyaesa.shared.runtime.reporting.run_progress import (
    RunProgressPrinter,
    monte_carlo_run_progress,
    visible_status_for_run_work,
)
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter
from pyaesa.shared.uncertainty_assessment.orchestration import progress_complete


@dataclass(frozen=True)
class ASOCCComponentSession:
    """Prepared aSoCC scopes reused by parent convergence checkpoints."""

    source_scope: Any
    phase_entries: list[Any]
    monte_carlo_root: Any
    sampling_scope: Any | None = None
    run_result: Any | None = None


def run_uncertainty_asocc(
    *,
    base_asocc_args: dict[str, Any],
    uncertainty_config: dict[str, Any],
    sobol_parameters: dict[str, Any] | None,
    external_method: dict[str, Any] | None,
    output_format: str,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    refresh: bool,
    component_inventory: dict[str, Any] | None = None,
    run_id: str | None = None,
    show_progress: bool = True,
    phase: AsoccPhasePrinter | None = None,
    progress: RunProgressPrinter | None = None,
) -> UncertaintyRunReport:
    """Run one aSoCC uncertainty request."""
    return run_uncertainty_asocc_component(
        base_asocc_args=base_asocc_args,
        uncertainty_config=uncertainty_config,
        sobol_parameters=sobol_parameters,
        external_method=external_method,
        output_format=output_format,
        figures=figures,
        figure_options=figure_options,
        figure_format=figure_format,
        refresh=refresh,
        component_inventory=component_inventory,
        external_lcia_methods=None,
        run_id=run_id,
        show_progress=show_progress,
        phase=phase,
        progress=progress,
    ).report


def run_uncertainty_asocc_component(
    *,
    base_asocc_args: dict[str, Any],
    uncertainty_config: dict[str, Any],
    sobol_parameters: dict[str, Any] | None,
    external_method: dict[str, Any] | None,
    output_format: str,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    refresh: bool,
    component_inventory: dict[str, Any] | None,
    external_lcia_methods: list[str] | None,
    run_id: str | None,
    show_progress: bool,
    phase: AsoccPhasePrinter | None,
    progress: RunProgressPrinter | None,
    component_session: ASOCCComponentSession | None = None,
    finalize_component_inventory: bool = False,
) -> ComponentRun:
    """Run or append one aSoCC component inventory using a local session."""
    owns_phase = phase is None
    phase_owner = asocc_phase_owner(phase)
    config = dict(uncertainty_config)
    runtime = normalize_uncertainty_request(
        family="asocc",
        output_format=output_format,
        mc_parameters=config.pop("mc_parameters", None),
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
    completion_phase = NullPhasePrinter() if component_parent_convergence else phase_owner
    sources = build_source_activation_plan(
        uncertainty_config=config,
        allowed_sources=ASOCC_UNCERTAINTY_SOURCES,
        default_sources=DEFAULT_ASOCC_UNCERTAINTY_SOURCES,
    )
    sobol_plan = normalize_sobol_plan(sobol_parameters=sobol_parameters)
    external_progress = progress is not None
    progress = progress or monte_carlo_run_progress(
        source="uncertainty_asocc",
        enabled=show_progress,
        status=phase_owner,
    )
    figure_status = visible_status_for_run_work(
        progress=progress,
        fallback=phase_owner,
        progress_enabled=show_progress or external_progress,
    )
    if component_session is None:
        phase_owner.announce(PHASE_B1_ASOCC, "deterministic_asocc")
        prerequisite = prepare_asocc_deterministic_prerequisite(
            base_asocc_args=base_asocc_args,
            refresh=refresh,
            reference_year_uncertainty_active=sources.is_active(REFERENCE_YEAR_SOURCE),
            phase=phase_owner,
        )
        complete_deterministic_asocc_phase(phase=phase_owner, prerequisite=prerequisite)
        phase_entries = asocc_deterministic_phase_entries(prerequisite=prerequisite)
        row_scope = resolve_final_deterministic_asocc_row_scope(prerequisite=prerequisite)
        phase_owner.status("Loading deterministic aSoCC outputs", owner="deterministic_asocc")
        loaded = load_final_deterministic_asocc_rows(
            prerequisite=prerequisite,
            row_scope=row_scope,
        )
        phase_owner.status("Building aSoCC uncertainty source scope", owner="uncertainty_asocc")
        source_scope = build_asocc_source_scope(
            loaded=loaded,
            external_method=external_method,
            external_lcia_methods=external_lcia_methods,
            runtime=runtime,
            sources=sources,
            phase=phase_owner,
        )
        component_session = ASOCCComponentSession(
            source_scope=source_scope,
            phase_entries=phase_entries,
            monte_carlo_root=asocc_monte_carlo_root(
                deterministic_manifest_path=source_scope.loaded.deterministic_manifest_path
            ),
        )
    source_scope = component_session.source_scope
    phase_entries = component_session.phase_entries
    loaded = source_scope.loaded
    external_plan = source_scope.external_plan
    sources = source_scope.sources
    inter_method_plan = source_scope.inter_method_plan
    inter_mrio_plan = source_scope.inter_mrio_plan
    run_context = manifest_context(
        base_asocc_args=base_asocc_args,
        loaded=loaded,
        runtime=runtime,
        sources=sources,
        external_plan=external_plan,
        inter_method_plan=inter_method_plan,
        inter_mrio_plan=inter_mrio_plan,
        component_inventory=component_inventory,
    )
    monte_carlo_root = component_session.monte_carlo_root
    if refresh:
        cleanup_monte_carlo_runs_for_refresh(
            monte_carlo_root=monte_carlo_root,
            compatibility_key=run_context["compatibility_key"],
            run_id=run_id if component_inventory is None else None,
            arguments=run_context["arguments"],
            component_inventory=component_inventory,
        )
    required_run = compatible_completed_run_id_for_context(
        deterministic_manifest_path=loaded.deterministic_manifest_path,
        run_id=run_id,
    )
    compatible = (
        compatible_completed_runs_for_context(
            deterministic_manifest_path=loaded.deterministic_manifest_path,
            compatibility_key=run_context["compatibility_key"],
        )
        if run_id is None
        else (() if required_run is None else (required_run,))
    )
    if sobol_plan.enabled:
        reusable = compatible_complete_sobol_run(
            compatible=compatible,
            runtime=runtime,
            mc_parameters=run_context["mc_parameters"],
            sources=sources,
            external_plan=external_plan,
            sobol_plan=sobol_plan,
            requested_years=tuple(int(year) for year in loaded.requested_years),
        )
        if reusable is not None:
            progress_complete(
                progress=progress,
                completed=reusable.manifest.completed_runs,
                max_runs=max(
                    progress_parameters["max_runs"],
                    reusable.manifest.completed_runs,
                ),
                mode=progress_parameters["mode"],
                component=progress_parameters["component"],
                visible=not all((had_component_session, component_parent_convergence)),
            )
            progress.finish()
            reused_manifest = render_reusable_asocc_figures_if_requested(
                manifest=reusable.manifest,
                figures=figures,
                figure_options=figure_options,
                figure_format=figure_format,
                status=figure_status,
            )
            phase_owner.announce(PHASE_B1_ASOCC, "uncertainty_asocc")
            complete_asocc_uncertainty_phase(
                phase=completion_phase,
                manifest=reused_manifest,
                reuse_status="reused_exact",
            )
            write_asocc_phase_index(
                manifest=reused_manifest,
                phase_entries=phase_entries,
                reuse_status="reused_exact",
            )
            if owns_phase:
                phase_owner.finish()
            return ComponentRun(
                report=uncertainty_report(
                    manifest=reused_manifest,
                    reuse_status="reused_exact",
                ),
                session=component_session,
            )
    else:
        reusable = compatible_complete_run(
            compatible=compatible,
            runtime=runtime,
            mc_parameters=run_context["mc_parameters"],
        )
        if reusable is not None:
            progress_complete(
                progress=progress,
                completed=reusable.manifest.completed_runs,
                max_runs=max(
                    progress_parameters["max_runs"],
                    reusable.manifest.completed_runs,
                ),
                mode=progress_parameters["mode"],
                component=progress_parameters["component"],
                visible=not all((had_component_session, component_parent_convergence)),
            )
            progress.finish()
            reused_manifest = render_reusable_asocc_figures_if_requested(
                manifest=reusable.manifest,
                figures=figures,
                figure_options=figure_options,
                figure_format=figure_format,
                status=figure_status,
            )
            phase_owner.announce(PHASE_B1_ASOCC, "uncertainty_asocc")
            complete_asocc_uncertainty_phase(
                phase=completion_phase,
                manifest=reused_manifest,
                reuse_status="reused_exact",
            )
            write_asocc_phase_index(
                manifest=reused_manifest,
                phase_entries=phase_entries,
                reuse_status="reused_exact",
            )
            if owns_phase:
                phase_owner.finish()
            return ComponentRun(
                report=uncertainty_report(
                    manifest=reused_manifest,
                    reuse_status="reused_exact",
                ),
                session=component_session,
            )
    append_compatible = compatible
    if component_inventory is not None:
        required_append_run = compatible_completed_run_id_for_context(
            deterministic_manifest_path=loaded.deterministic_manifest_path,
            run_id=run_id,
            include_running_component_inventory=True,
        )
        append_compatible = (
            compatible_completed_runs_for_context(
                deterministic_manifest_path=loaded.deterministic_manifest_path,
                compatibility_key=run_context["compatibility_key"],
                include_running_component_inventory=True,
            )
            if run_id is None
            else (() if required_append_run is None else (required_append_run,))
        )
    append_run = (
        appendable_run_for_runtime(
            compatible=append_compatible,
            runtime=runtime,
        )
        if component_inventory is not None and component_session.run_result is None
        else None
    )
    if component_session.sampling_scope is None:
        phase_owner.status("Building aSoCC uncertainty sampling scope", owner="uncertainty_asocc")
        sampling_scope = build_asocc_sampling_scope(
            source_scope=source_scope,
            runtime=runtime,
            append_existing=append_run is not None,
        )
        component_session = replace(component_session, sampling_scope=sampling_scope)
        runtime = sampling_scope.runtime
    else:
        sampling_scope = component_session.sampling_scope
        runtime = replace(
            runtime,
            batch_size=runtime_batch_size(
                runtime=runtime,
                source_scope=source_scope,
                lcia_plan=sampling_scope.lcia_plan,
                projection_plan=sampling_scope.projection_plan,
                inter_method_execution_plan=sampling_scope.inter_method_execution_plan,
            ),
        )
    lcia_plan = sampling_scope.lcia_plan
    projection_plan = sampling_scope.projection_plan
    inter_method_execution_plan = sampling_scope.inter_method_execution_plan
    manifest = build_manifest(
        family="asocc",
        mode=runtime.mode,
        output_format=runtime.output_format,
        active_sources=sources.names,
        completed_runs=0,
        status="running",
        requested_runs=runtime.n_runs,
        mc_parameters=run_context["mc_parameters"],
        source_parameters=run_context["source_parameters"],
        arguments=run_context["arguments"],
        deterministic_prerequisites=run_context["deterministic_prerequisites"],
        external_inputs=run_context["external_inputs"],
        artifacts=run_context["artifacts"],
        lineage=run_context["lineage"],
        component_inventory=run_context["component_inventory"],
        compatibility_key=run_context["compatibility_key"],
        compatibility_context=run_context["compatibility_context"],
        run_id=(append_run.manifest.run_id if append_run is not None else run_id),
    )
    paths = build_asocc_uncertainty_run_paths(
        deterministic_manifest_path=loaded.deterministic_manifest_path,
        run_id=manifest.run_id,
        output_format=runtime.output_format,
        inter_method_parameters=(
            sources.parameters_for(INTER_METHOD_SOURCE) if inter_method_plan is not None else None
        ),
    )
    run_seed = run_seed_from_run_id(run_id=manifest.run_id)
    if (
        inter_method_plan is not None
        and append_run is None
        and component_session.run_result is None
    ):
        phase_owner.status("Writing inter method tree artifacts", owner="uncertainty_asocc")
        write_inter_method_tree_artifacts(
            tree_csv_path=paths.inter_method_tree_csv,
            figure_base_path=paths.inter_method_tree_figure_base,
            frame=inter_method_plan.tree_frame,
            candidates=inter_method_plan.candidates,
            figure_format=figure_format,
        )
    write_manifest(path=paths.scope_manifest, manifest=manifest)
    phase_owner.announce(PHASE_B1_ASOCC, "uncertainty_asocc")
    run_result = write_monte_carlo_run_outputs(
        paths=paths,
        loaded=loaded,
        inter_method_plan=inter_method_plan,
        inter_method_execution_plan=inter_method_execution_plan,
        inter_mrio_plan=inter_mrio_plan,
        lcia_plan=lcia_plan,
        projection_plan=projection_plan,
        runtime=runtime,
        sources=sources,
        external_plan=external_plan,
        append_run=append_run,
        run_seed=run_seed,
        show_progress=show_progress,
        progress=progress,
        progress_mode=progress_parameters["mode"],
        progress_max_runs=progress_parameters["max_runs"],
        progress_component=progress_parameters["component"],
        previous_result=component_session.run_result,
    )
    if finalize_outputs and not component_parent_convergence:
        phase_owner.status("Writing uncertainty run logs", owner="uncertainty_asocc")
    write_run_logs(
        paths=paths,
        loaded=loaded,
        inter_method_plan=inter_method_plan,
        inter_mrio_plan=inter_mrio_plan,
        lcia_plan=lcia_plan,
        projection_plan=projection_plan,
        output_format=runtime.output_format,
        sources=sources,
        summary_run_count=run_result.summary_run_count,
        public_runs_sparse=run_result.public_runs_sparse,
        write_summary=finalize_outputs,
    )
    if sobol_plan.enabled:
        external_plan = external_plan_for_years(
            plan=external_plan,
            years=selected_sobol_years(
                plan=sobol_plan,
                requested_years=tuple(int(year) for year in loaded.requested_years),
            ),
        )
    sobol_result = run_asocc_sobol(
        paths=paths,
        loaded=loaded,
        inter_mrio_plan=inter_mrio_plan,
        runtime=runtime,
        sources=sources,
        external_plan=external_plan,
        sobol_plan=sobol_plan,
        status=phase_owner,
    )
    if finalize_outputs:
        complete = build_completed_asocc_manifest(
            paths=paths,
            runtime=runtime,
            sources=sources,
            run_context=run_context,
            run_id=manifest.run_id,
            completed_runs=run_result.completed_runs,
            convergence=run_result.convergence,
            sobol_status=sobol_result.status,
        )
    else:
        artifacts = outputs_payload(paths=paths, output_format=runtime.output_format)
        artifacts["public_output"] = {
            "asocc_runs": {
                "layout": (
                    "sparse_selected_rows"
                    if inter_method_plan is not None
                    else "compact_run_matrix"
                )
            }
        }
        complete = build_manifest(
            family="asocc",
            mode=runtime.mode,
            output_format=runtime.output_format,
            active_sources=sources.names,
            completed_runs=run_result.completed_runs,
            status="running",
            requested_runs=runtime.n_runs,
            mc_parameters=run_context["mc_parameters"],
            source_parameters=run_context["source_parameters"],
            arguments=run_context["arguments"],
            deterministic_prerequisites=run_context["deterministic_prerequisites"],
            external_inputs=run_context["external_inputs"],
            artifacts=artifacts,
            lineage=run_context["lineage"],
            component_inventory=run_context["component_inventory"],
            convergence=run_result.convergence,
            sobol=sobol_result.status,
            compatibility_key=run_context["compatibility_key"],
            compatibility_context=run_context["compatibility_context"],
            run_id=manifest.run_id,
        )
    if figures:
        figure_paths = render_asocc_uncertainty_figures(
            manifest=complete,
            paths=paths,
            figure_options=figure_options,
            figure_format=figure_format,
            status=figure_status,
        )
        complete = manifest_with_figure_artifacts(
            manifest=complete,
            figure_paths=figure_paths,
            figure_options=figure_options,
            figure_format=figure_format,
        )
    write_manifest(path=paths.scope_manifest, manifest=complete)
    progress.finish()
    if finalize_outputs:
        complete_asocc_uncertainty_phase(
            phase=completion_phase,
            manifest=complete,
            reuse_status="computed",
        )
        write_asocc_phase_index(
            manifest=complete,
            phase_entries=phase_entries,
            reuse_status="computed",
        )
    if owns_phase:
        phase_owner.finish()
    return ComponentRun(
        report=uncertainty_report(
            manifest=complete,
            reuse_status="computed",
        ),
        session=replace(component_session, run_result=None if finalize_outputs else run_result),
    )

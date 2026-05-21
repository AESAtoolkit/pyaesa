"""Public IO-LCA uncertainty runtime orchestration."""

from dataclasses import dataclass, replace
from typing import Any

from pyaesa.io_lca.uncertainty.runtime.models import (
    IOLCADeterministicScope,
)
from pyaesa.io_lca.uncertainty.figures.render import render_io_lca_uncertainty_figures
from pyaesa.io_lca.uncertainty.figures.reuse import (
    render_reusable_io_lca_figures_if_requested,
)
from pyaesa.io_lca.uncertainty.io.paths import (
    build_io_lca_uncertainty_run_paths,
    io_lca_monte_carlo_root,
)
from pyaesa.io_lca.uncertainty.runtime.prerequisites import (
    load_deterministic_public_rows,
    prepare_io_lca_deterministic_prerequisite,
)
from pyaesa.io_lca.uncertainty.request.normalization import normalize_io_lca_uncertainty_request
from pyaesa.io_lca.uncertainty.evaluation.sampling import (
    build_io_lca_lcia_plan,
)
from pyaesa.io_lca.uncertainty.io.manifest_payloads import (
    build_completed_io_lca_manifest,
    build_io_lca_manifest_context,
    io_lca_outputs_payload,
    io_lca_public_output_payload,
)
from pyaesa.io_lca.uncertainty.io.run_outputs import append_io_lca_run_outputs
from pyaesa.io_lca.uncertainty.io.source_methods import (
    write_io_lca_results_readme,
    write_io_lca_source_methods,
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
from pyaesa.shared.uncertainty_assessment.run_state.runs import (
    appendable_completed_run,
    compatible_completed_runs,
    compatible_completed_run_for_id,
    complete_run_with_requested_runs,
    cleanup_monte_carlo_runs_for_refresh,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.composite import (
    ComponentRun,
    component_inventory_finalizes,
    component_inventory_parent_convergence,
    component_inventory_progress_parameters,
)
from pyaesa.shared.uncertainty_assessment.io.downstream_run_outputs import (
    DownstreamRunOutputPaths,
    DownstreamRunOutputState,
    close_downstream_run_output_state,
    new_downstream_run_output_state,
)
from pyaesa.shared.uncertainty_assessment.request.sources import build_source_activation_plan
from pyaesa.shared.lcia.uncertainty_source import LCIA_SOURCE
from pyaesa.shared.uncertainty_assessment.io.tables import (
    write_uncertainty_table,
)
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    PHASE_A_LCA,
    phase_ready_detail,
    phase_reused_detail,
    phase_uncertainty_done_detail,
)
from pyaesa.shared.runtime.reporting.phase import PhasePrinter
from pyaesa.shared.runtime.reporting.run_progress import (
    RunProgressPrinter,
    monte_carlo_run_progress,
    visible_status_for_run_work,
)
from pyaesa.shared.uncertainty_assessment.orchestration import (
    deterministic_phase_index_entry,
    manifest_output_root,
    output_root_from_path,
    progress_complete,
    uncertainty_phase_index_entry,
    write_uncertainty_phase_index,
)

IO_LCA_UNCERTAINTY_SOURCES = (LCIA_SOURCE,)


@dataclass(frozen=True)
class IOLCAComponentSession:
    """Prepared IO-LCA sampling plan reused by parent convergence checkpoints."""

    request: Any
    prerequisite: Any
    phase_entries: list[Any]
    plan: Any
    monte_carlo_root: Any
    output_state: DownstreamRunOutputState | None = None


def run_uncertainty_io_lca(
    *,
    base_io_lca_args: dict[str, Any],
    uncertainty_config: dict[str, Any],
    output_format: str,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    refresh: bool,
    phase: PhasePrinter | None = None,
    component_inventory: dict[str, Any] | None = None,
    run_id: str | None = None,
    show_progress: bool = True,
    progress: RunProgressPrinter | None = None,
) -> UncertaintyRunReport:
    """Run one IO-LCA uncertainty request."""
    return run_uncertainty_io_lca_component(
        base_io_lca_args=base_io_lca_args,
        uncertainty_config=uncertainty_config,
        output_format=output_format,
        figures=figures,
        figure_options=figure_options,
        figure_format=figure_format,
        refresh=refresh,
        phase=phase,
        component_inventory=component_inventory,
        run_id=run_id,
        show_progress=show_progress,
        progress=progress,
    ).report


def run_uncertainty_io_lca_component(
    *,
    base_io_lca_args: dict[str, Any],
    uncertainty_config: dict[str, Any],
    output_format: str,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    refresh: bool,
    phase: PhasePrinter | None,
    component_inventory: dict[str, Any] | None,
    run_id: str | None,
    show_progress: bool,
    progress: RunProgressPrinter | None,
    component_session: IOLCAComponentSession | None = None,
    finalize_component_inventory: bool = False,
) -> ComponentRun:
    """Run or append one IO-LCA component inventory using a local session."""
    owns_phase = phase is None
    phase_owner = PhasePrinter("uncertainty_io_lca") if owns_phase else phase
    config = dict(uncertainty_config)
    runtime = normalize_uncertainty_request(
        family="io_lca",
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
    sources = build_source_activation_plan(
        uncertainty_config=config,
        allowed_sources=IO_LCA_UNCERTAINTY_SOURCES,
    )
    if not sources.is_active(LCIA_SOURCE):
        raise ValueError(f"uncertainty_io_lca requires {LCIA_SOURCE} to be active.")
    if component_session is None:
        request = normalize_io_lca_uncertainty_request(
            base_io_lca_args=base_io_lca_args,
            lcia_parameters=sources.parameters_for(LCIA_SOURCE),
        )
        phase_owner.announce(PHASE_A_LCA, "deterministic_io_lca")
        prerequisite = prepare_io_lca_deterministic_prerequisite(
            request=request,
            refresh=refresh,
            status=phase_owner,
        )
        _complete_deterministic_lca_phase(phase=phase_owner, prerequisite=prerequisite)
        phase_entries = [
            deterministic_phase_index_entry(
                phase_label=PHASE_A_LCA,
                function_name="deterministic_io_lca",
                metadata_path=prerequisite.metadata_path,
                reuse_status=prerequisite.reuse_status,
            )
        ]
        phase_owner.status("Loading deterministic IO-LCA outputs", owner="deterministic_io_lca")
        public_rows = load_deterministic_public_rows(request=request, scope=prerequisite)
        phase_owner.status("Building IO-LCA sampling plan", owner="uncertainty_io_lca")
        plan = build_io_lca_lcia_plan(request=request, public_rows=public_rows)
        component_session = IOLCAComponentSession(
            request=request,
            prerequisite=prerequisite,
            phase_entries=phase_entries,
            plan=plan,
            monte_carlo_root=io_lca_monte_carlo_root(
                deterministic_manifest_path=prerequisite.metadata_path
            ),
        )
    request = component_session.request
    prerequisite = component_session.prerequisite
    phase_entries = component_session.phase_entries
    plan = component_session.plan
    runtime = replace(
        runtime,
        batch_size=memory_bounded_batch_size(runtime=runtime, row_count=len(plan.identity)),
    )
    context = build_io_lca_manifest_context(
        request=request,
        runtime=runtime,
        prerequisite=prerequisite,
        component_inventory=component_inventory,
    )
    phase_owner.announce(PHASE_A_LCA, "uncertainty_io_lca")
    external_progress = progress is not None
    progress = progress or monte_carlo_run_progress(
        source="uncertainty_io_lca",
        enabled=show_progress,
        status=phase_owner,
    )
    figure_status = visible_status_for_run_work(
        progress=progress,
        fallback=phase_owner,
        progress_enabled=show_progress or external_progress,
    )
    monte_carlo_root = component_session.monte_carlo_root
    if refresh:
        cleanup_monte_carlo_runs_for_refresh(
            monte_carlo_root=monte_carlo_root,
            compatibility_key=context["compatibility_key"],
            run_id=run_id if component_inventory is None else None,
            arguments=context["arguments"],
            component_inventory=component_inventory,
        )
    required_run = compatible_completed_run_for_id(
        monte_carlo_root=monte_carlo_root,
        run_id=run_id,
    )
    compatible = (
        compatible_completed_runs(
            monte_carlo_root=monte_carlo_root,
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
    if reusable is not None:
        progress_complete(
            progress=progress,
            completed=reusable.manifest.completed_runs,
            max_runs=max(progress_parameters["max_runs"], reusable.manifest.completed_runs),
            mode=progress_parameters["mode"],
            component=progress_parameters["component"],
            visible=not all((had_component_session, component_parent_convergence)),
        )
        progress.finish()
        reused_manifest = render_reusable_io_lca_figures_if_requested(
            manifest=reusable.manifest,
            request=request,
            figures=figures,
            figure_options=figure_options,
            figure_format=figure_format,
            status=figure_status,
        )
        if not component_parent_convergence:
            phase_owner.complete(
                phase_reused_detail(
                    scope_name="LCA uncertainty",
                    output_root=manifest_output_root(reused_manifest),
                )
            )
        write_uncertainty_phase_index(
            manifest=reused_manifest,
            entries=[
                *phase_entries,
                uncertainty_phase_index_entry(
                    phase_label=PHASE_A_LCA,
                    function_name="uncertainty_io_lca",
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
            session=component_session,
        )
    append_compatible = compatible
    if component_inventory is not None:
        required_append_run = compatible_completed_run_for_id(
            monte_carlo_root=monte_carlo_root,
            run_id=run_id,
            include_running_component_inventory=True,
        )
        append_compatible = (
            compatible_completed_runs(
                monte_carlo_root=monte_carlo_root,
                compatibility_key=context["compatibility_key"],
                include_running_component_inventory=True,
            )
            if run_id is None
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
            and runtime.mode == "fixed"
            and component_session.output_state is None
        )
        else None
    )
    manifest = build_manifest(
        family="io_lca",
        mode=runtime.mode,
        output_format=runtime.output_format,
        active_sources=sources.names,
        completed_runs=0,
        status="running",
        requested_runs=runtime.n_runs,
        mc_parameters=context["mc_parameters"],
        source_parameters=context["source_parameters"],
        arguments=context["arguments"],
        deterministic_prerequisites=context["deterministic_prerequisites"],
        component_inventory=context["component_inventory"],
        compatibility_key=context["compatibility_key"],
        compatibility_context=context["compatibility_context"],
        run_id=(append_run.manifest.run_id if append_run is not None else run_id),
    )
    paths = build_io_lca_uncertainty_run_paths(
        deterministic_manifest_path=prerequisite.metadata_path,
        run_id=manifest.run_id,
        output_format=runtime.output_format,
    )
    write_manifest(path=paths.scope_manifest, manifest=manifest)
    try:
        output_state = component_session.output_state or new_downstream_run_output_state(
            paths=DownstreamRunOutputPaths(
                run_root=paths.run_root,
                public_runs=paths.public_runs,
                summary_stats_runs=paths.summary_stats_runs,
            ),
            completed_runs=0 if append_run is None else append_run.manifest.completed_runs,
        )
        output_state, convergence = append_io_lca_run_outputs(
            paths=paths,
            plan=plan,
            runtime=runtime,
            seed_run_id=manifest.run_id,
            state=output_state,
            target_runs=runtime.n_runs,
            final_checkpoint=finalize_outputs,
            show_progress=show_progress,
            progress=progress,
            progress_mode=progress_parameters["mode"],
            progress_max_runs=progress_parameters["max_runs"],
            progress_component=progress_parameters["component"],
        )
        completed_runs = output_state.completed_runs
        write_uncertainty_table(
            path=paths.public_row_identity,
            frame=plan.identity,
            output_format=runtime.output_format,
        )
        write_io_lca_source_methods(path=paths.source_methods, rows=plan.source_method_rows)
        write_io_lca_results_readme(
            paths=paths,
            request=request,
        )
        if finalize_outputs:
            public_output = io_lca_public_output_payload(
                paths=paths,
                output_format=runtime.output_format,
            )
            complete = build_completed_io_lca_manifest(
                paths=paths,
                runtime=runtime,
                context=context,
                run_id=manifest.run_id,
                completed_runs=completed_runs,
                convergence=convergence,
                public_output=public_output,
            )
        else:
            complete = build_manifest(
                family="io_lca",
                mode=runtime.mode,
                output_format=runtime.output_format,
                active_sources=sources.names,
                completed_runs=completed_runs,
                status="running",
                requested_runs=runtime.n_runs,
                mc_parameters=context["mc_parameters"],
                source_parameters=context["source_parameters"],
                arguments=context["arguments"],
                deterministic_prerequisites=context["deterministic_prerequisites"],
                component_inventory=context["component_inventory"],
                artifacts=io_lca_outputs_payload(
                    paths=paths,
                    output_format=runtime.output_format,
                ),
                compatibility_key=context["compatibility_key"],
                compatibility_context=context["compatibility_context"],
                run_id=manifest.run_id,
            )
        if figures:
            figure_paths = render_io_lca_uncertainty_figures(
                manifest=complete,
                paths=paths,
                request=request,
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
        if finalize_outputs:
            close_downstream_run_output_state(state=output_state)
            output_state = None
        progress.finish()
        if not component_parent_convergence:
            phase_owner.complete(
                phase_uncertainty_done_detail(
                    scope_name="LCA uncertainty",
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
                        phase_label=PHASE_A_LCA,
                        function_name="uncertainty_io_lca",
                        manifest=complete,
                        reuse_status="computed",
                    ),
                ],
            )
        if owns_phase:
            phase_owner.finish()
        return ComponentRun(
            report=uncertainty_report(
                manifest=complete,
                reuse_status="computed",
            ),
            session=replace(component_session, output_state=output_state),
        )
    finally:
        progress.finish()


def _complete_deterministic_lca_phase(
    *,
    phase: PhasePrinter,
    prerequisite: IOLCADeterministicScope,
) -> None:
    """Print deterministic LCA prerequisite completion for IO-LCA uncertainty."""
    detail_builder = (
        phase_reused_detail if prerequisite.reuse_status == "reused_exact" else phase_ready_detail
    )
    phase.expect_visible(PHASE_A_LCA)
    phase.complete(
        detail_builder(
            scope_name="LCA deterministic",
            output_root=output_root_from_path(prerequisite.metadata_path),
        )
    )

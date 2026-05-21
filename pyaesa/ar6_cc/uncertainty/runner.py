"""Public AR6 CC uncertainty runtime orchestration."""

from dataclasses import dataclass, replace
from typing import Any

from pyaesa.ar6_cc.uncertainty.figures.render import render_ar6_cc_uncertainty_figures
from pyaesa.ar6_cc.uncertainty.figures.reuse import (
    render_reusable_ar6_cc_figures_if_requested,
)
from pyaesa.ar6_cc.uncertainty.io.paths import (
    ar6_cc_monte_carlo_root,
    build_ar6_cc_uncertainty_run_paths,
)
from pyaesa.ar6_cc.uncertainty.runtime.prerequisites import (
    load_deterministic_ar6_cc_rows,
    load_deterministic_ar6_cc_post_study_rows,
    prepare_ar6_cc_deterministic_prerequisite,
)
from pyaesa.ar6_cc.uncertainty.runtime.rows import combine_study_post_rows, post_study_years
from pyaesa.ar6_cc.uncertainty.request.normalization import (
    AR6_CC_UNCERTAINTY_SOURCES,
    AR6_DYNAMIC_CC_SOURCE,
    normalize_ar6_cc_uncertainty_request,
)
from pyaesa.ar6_cc.uncertainty.evaluation.sampling import (
    build_ar6_cc_sampling_plan,
)
from pyaesa.ar6_cc.uncertainty.io.manifest_payloads import (
    ar6_cc_outputs_payload,
    build_ar6_cc_manifest_context,
    build_completed_ar6_cc_manifest,
    ar6_cc_public_output_payload,
)
from pyaesa.ar6_cc.uncertainty.io.run_outputs import write_ar6_cc_study_post_outputs
from pyaesa.ar6_cc.uncertainty.io.source_methods import (
    write_ar6_cc_results_readme,
    write_ar6_cc_source_methods,
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
from pyaesa.shared.uncertainty_assessment.request.sources import build_source_activation_plan
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    PHASE_B1_AR6_DYNAMIC_CC,
    phase_ready_detail,
    phase_reused_detail,
    phase_uncertainty_done_detail,
)
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
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


@dataclass(frozen=True)
class AR6CCComponentSession:
    """Prepared AR6 CC sampling plan reused by parent convergence checkpoints."""

    request: Any
    prerequisite: Any
    phase_entries: list[Any]
    plan: Any
    post_study_scope_years: list[int]
    monte_carlo_root: Any
    run_id: str | None = None
    output_states: dict[str, dict[str, Any]] | None = None
    completed_runs: int = 0


def run_uncertainty_ar6_cc(
    *,
    base_ar6_cc_args: dict[str, Any],
    uncertainty_config: dict[str, Any] | None,
    output_format: str,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    refresh: bool,
    component_inventory: dict[str, Any] | None = None,
    run_id: str | None = None,
    show_progress: bool = True,
    phase: PhasePrinter | NullPhasePrinter | None = None,
    progress: RunProgressPrinter | None = None,
) -> UncertaintyRunReport:
    """Run one AR6 CC uncertainty request."""
    return run_uncertainty_ar6_cc_component(
        base_ar6_cc_args=base_ar6_cc_args,
        uncertainty_config=uncertainty_config,
        output_format=output_format,
        figures=figures,
        figure_options=figure_options,
        figure_format=figure_format,
        refresh=refresh,
        component_inventory=component_inventory,
        run_id=run_id,
        show_progress=show_progress,
        phase=phase,
        progress=progress,
    ).report


def run_uncertainty_ar6_cc_component(
    *,
    base_ar6_cc_args: dict[str, Any],
    uncertainty_config: dict[str, Any] | None,
    output_format: str,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    refresh: bool,
    component_inventory: dict[str, Any] | None,
    run_id: str | None,
    show_progress: bool,
    phase: PhasePrinter | NullPhasePrinter | None,
    progress: RunProgressPrinter | None,
    component_session: AR6CCComponentSession | None = None,
    finalize_component_inventory: bool = False,
) -> ComponentRun:
    """Run or append one AR6 CC component inventory using a local session."""
    owns_phase = phase is None
    phase_owner = PhasePrinter("uncertainty_ar6_cc") if owns_phase else phase
    config = dict(uncertainty_config or {})
    runtime = normalize_uncertainty_request(
        family="ar6_cc",
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
        uncertainty_config=None if not config else config,
        allowed_sources=AR6_CC_UNCERTAINTY_SOURCES,
        default_sources=AR6_CC_UNCERTAINTY_SOURCES,
    )
    if not sources.is_active(AR6_DYNAMIC_CC_SOURCE):
        raise ValueError(f"uncertainty_ar6_cc requires {AR6_DYNAMIC_CC_SOURCE} to be active.")
    if component_session is None:
        request = normalize_ar6_cc_uncertainty_request(
            base_ar6_cc_args=base_ar6_cc_args,
            source_parameters=sources.parameters_for(AR6_DYNAMIC_CC_SOURCE),
        )
        phase_owner.expect_visible(PHASE_B1_AR6_DYNAMIC_CC)
        phase_owner.announce(PHASE_B1_AR6_DYNAMIC_CC, "deterministic_ar6_cc")
        prerequisite = prepare_ar6_cc_deterministic_prerequisite(
            request=request,
            refresh=refresh,
            figures=figures,
            figure_format=figure_format,
            status=phase_owner,
        )
        detail_builder = (
            phase_reused_detail
            if prerequisite.reuse_status == "reused_exact"
            else phase_ready_detail
        )
        phase_owner.complete(
            detail_builder(
                scope_name="dynamic AR6 CC deterministic",
                output_root=output_root_from_path(prerequisite.metadata_path),
            ),
            owner="deterministic_ar6_cc",
        )
        phase_entries = [
            deterministic_phase_index_entry(
                phase_label=PHASE_B1_AR6_DYNAMIC_CC,
                function_name="deterministic_ar6_cc",
                metadata_path=prerequisite.metadata_path,
                reuse_status=prerequisite.reuse_status,
            )
        ]
        phase_owner.status("Loading deterministic AR6 CC outputs", owner="deterministic_ar6_cc")
        deterministic_rows = load_deterministic_ar6_cc_rows(
            request=request,
            scope=prerequisite,
        )
        post_study_rows = load_deterministic_ar6_cc_post_study_rows(
            request=request,
            scope=prerequisite,
        )
        post_study_scope_years = (
            [] if post_study_rows is None else post_study_years(post_study_rows)
        )
        full_rows = combine_study_post_rows(
            study_rows=deterministic_rows,
            post_study_rows=post_study_rows,
            post_study_years=post_study_scope_years,
        )
        phase_owner.status("Building AR6 CC sampling plan", owner="uncertainty_ar6_cc")
        plan = build_ar6_cc_sampling_plan(
            request=replace(request, years=[*request.years, *post_study_scope_years]),
            deterministic_rows=full_rows,
        )
        component_session = AR6CCComponentSession(
            request=request,
            prerequisite=prerequisite,
            phase_entries=phase_entries,
            plan=plan,
            post_study_scope_years=post_study_scope_years,
            monte_carlo_root=ar6_cc_monte_carlo_root(
                deterministic_manifest_path=prerequisite.metadata_path
            ),
        )
    elif figures:
        component_session = replace(
            component_session,
            prerequisite=prepare_ar6_cc_deterministic_prerequisite(
                request=component_session.request,
                refresh=False,
                figures=True,
                figure_format=figure_format,
                status=phase_owner,
            ),
        )
    request = component_session.request
    prerequisite = component_session.prerequisite
    phase_entries = component_session.phase_entries
    plan = component_session.plan
    post_study_scope_years = component_session.post_study_scope_years
    runtime = replace(
        runtime,
        batch_size=memory_bounded_batch_size(runtime=runtime, row_count=len(plan.identity)),
    )
    context = build_ar6_cc_manifest_context(
        request=request,
        runtime=runtime,
        prerequisite=prerequisite,
        plan=plan,
        component_inventory=component_inventory,
    )
    phase_owner.announce(PHASE_B1_AR6_DYNAMIC_CC, "uncertainty_ar6_cc")
    external_progress = progress is not None
    progress = progress or monte_carlo_run_progress(
        source="uncertainty_ar6_cc",
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
        reused_manifest = (
            render_reusable_ar6_cc_figures_if_requested(
                manifest=reusable.manifest,
                figure_options=figure_options,
                figure_format=figure_format,
                status=figure_status,
            )
            if figures
            else reusable.manifest
        )
        if not component_parent_convergence:
            phase_owner.complete(
                phase_reused_detail(
                    scope_name="dynamic AR6 CC uncertainty",
                    output_root=manifest_output_root(reused_manifest),
                )
            )
        write_uncertainty_phase_index(
            manifest=reused_manifest,
            entries=[
                *phase_entries,
                uncertainty_phase_index_entry(
                    phase_label=PHASE_B1_AR6_DYNAMIC_CC,
                    function_name="uncertainty_ar6_cc",
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
            and component_session.output_states is None
        )
        else None
    )
    manifest = build_manifest(
        family="ar6_cc",
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
        lineage=context["lineage"],
        component_inventory=context["component_inventory"],
        compatibility_key=context["compatibility_key"],
        compatibility_context=context["compatibility_context"],
        run_id=(
            component_session.run_id
            if component_session.output_states is not None
            else append_run.manifest.run_id
            if append_run is not None
            else run_id
        ),
    )
    paths = build_ar6_cc_uncertainty_run_paths(
        deterministic_manifest_path=prerequisite.metadata_path,
        run_id=manifest.run_id,
        output_format=runtime.output_format,
    )
    write_manifest(path=paths.scope_manifest, manifest=manifest)
    try:
        completed_runs, convergence, has_post_study, output_states = (
            write_ar6_cc_study_post_outputs(
                paths=paths,
                plan=plan,
                study_years=request.years,
                post_study_years=post_study_scope_years,
                runtime=runtime,
                seed_run_id=manifest.run_id,
                start_run_index=(
                    component_session.completed_runs
                    if component_session.output_states is not None
                    else 0
                    if append_run is None
                    else append_run.manifest.completed_runs
                ),
                progress=progress,
                progress_mode=progress_parameters["mode"],
                progress_max_runs=progress_parameters["max_runs"],
                progress_component=progress_parameters["component"],
                states=component_session.output_states,
                final_checkpoint=finalize_outputs,
            )
        )
        write_ar6_cc_source_methods(path=paths.source_methods, rows=plan.source_method_rows)
        write_ar6_cc_results_readme(
            paths=paths,
            request=request,
            availability_messages=plan.availability_messages,
        )
        if finalize_outputs:
            public_output = ar6_cc_public_output_payload(
                paths=paths,
                output_format=runtime.output_format,
            )
            complete = build_completed_ar6_cc_manifest(
                paths=paths,
                runtime=runtime,
                context=context,
                run_id=manifest.run_id,
                completed_runs=completed_runs,
                convergence=convergence,
                include_post_study=has_post_study,
                public_output=public_output,
            )
        else:
            artifacts = ar6_cc_outputs_payload(
                paths=paths,
                include_post_study=has_post_study,
                output_format=runtime.output_format,
            )
            artifacts["public_output"] = {"cc_runs": {"layout": "sparse_selected_rows"}}
            complete = build_manifest(
                family="ar6_cc",
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
                lineage=context["lineage"],
                component_inventory=context["component_inventory"],
                artifacts=artifacts,
                convergence=convergence,
                compatibility_key=context["compatibility_key"],
                compatibility_context=context["compatibility_context"],
                run_id=manifest.run_id,
            )
        if figures:
            figure_paths = render_ar6_cc_uncertainty_figures(
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
        if not component_parent_convergence:
            phase_owner.complete(
                phase_uncertainty_done_detail(
                    scope_name="dynamic AR6 CC uncertainty",
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
                        phase_label=PHASE_B1_AR6_DYNAMIC_CC,
                        function_name="uncertainty_ar6_cc",
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
            session=replace(
                component_session,
                run_id=manifest.run_id,
                output_states=output_states,
                completed_runs=0 if output_states is None else completed_runs,
            ),
        )
    finally:
        progress.finish()

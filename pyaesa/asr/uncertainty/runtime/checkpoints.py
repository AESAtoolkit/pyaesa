"""Checkpoint execution for uncertainty ASR runs."""

from dataclasses import dataclass, replace
from typing import Any

from pyaesa.asr.uncertainty.io.run_outputs import (
    append_asr_run_outputs,
    close_asr_run_output_state,
    new_asr_run_output_state,
)
from pyaesa.asr.uncertainty.runtime.component_inputs import (
    acc_inventory_report,
    lca_component_inventory,
)
from pyaesa.asr.uncertainty.runtime.models import (
    ASRUncertaintyPlan,
    ASRUncertaintyRunPaths,
    LCAUncertaintyInput,
)
from pyaesa.io_lca.data.contracts import IO_LCA_FAMILY
from pyaesa.asr.uncertainty.sources.lca_inputs import resolve_lca_uncertainty_component_input
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
from pyaesa.shared.runtime.reporting.run_progress import (
    RunProgressPrinter,
    monte_carlo_run_progress,
)
from pyaesa.shared.runtime.reporting.composite_phase_index import PHASE_A_LCA, PHASE_B2_ACC
from pyaesa.shared.uncertainty_assessment.orchestration import progress_begin, progress_complete
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


@dataclass(frozen=True)
class ASRCheckpointResult:
    """Final component and output state after one ASR checkpoint sequence."""

    plan: ASRUncertaintyPlan
    acc_manifest: UncertaintyManifest
    acc_session: Any | None
    lca_input: LCAUncertaintyInput
    lca_session: Any | None
    completed_runs: int
    convergence: dict[str, Any] | None


def run_asr_checkpoints(
    *,
    paths: ASRUncertaintyRunPaths,
    runtime: UncertaintyRuntimeRequest,
    checkpoints: tuple[int, ...],
    initial_plan: ASRUncertaintyPlan,
    initial_acc_manifest: UncertaintyManifest,
    initial_acc_session: Any | None,
    initial_lca_input: LCAUncertaintyInput,
    initial_lca_session: Any | None,
    project_name: str,
    years: int | list[int] | range,
    shared_methods: list[str],
    fu_code: str,
    r_p: str | list[str] | None,
    s_p: str | list[str] | None,
    r_c: str | list[str] | None,
    r_f: str | list[str] | None,
    mrio_scope: dict[str, Any],
    asocc_config: dict[str, Any],
    base_cc_args: dict[str, Any],
    source_config,
    external_method: dict[str, Any] | None,
    proj_base,
    source_label: str,
    lca_type: str,
    lca_version_name: str | None,
    base_allocate_args: dict[str, Any],
    output_format: str,
    phase: PhasePrinter,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    run_id: str | None,
    acc_progress: RunProgressPrinter,
    lca_progress: RunProgressPrinter,
    asr_progress: RunProgressPrinter,
    progress_mode: str,
    progress_max_runs: int,
    progress_component: bool,
) -> ASRCheckpointResult:
    """Append ASR checkpoint outputs and refresh component inventories as needed."""
    plan = initial_plan
    acc_manifest = initial_acc_manifest
    acc_session = initial_acc_session
    lca_input = initial_lca_input
    lca_session = initial_lca_session
    state = new_asr_run_output_state(paths=paths, plan=plan)
    convergence = None
    try:
        for index, checkpoint in enumerate(checkpoints):
            if index:
                acc_manifest, acc_session = _checkpoint_acc(
                    project_name=project_name,
                    years=years,
                    shared_methods=shared_methods,
                    fu_code=fu_code,
                    r_p=r_p,
                    s_p=s_p,
                    r_c=r_c,
                    r_f=r_f,
                    mrio_scope=mrio_scope,
                    asocc_config=asocc_config,
                    base_allocate_args=base_allocate_args,
                    base_cc_args=base_cc_args,
                    source_config=source_config.acc_config,
                    external_method=external_method,
                    output_format=output_format,
                    phase=phase,
                    target_runs=checkpoint,
                    parent_mode=runtime.mode,
                    parent_max_runs=runtime.n_runs,
                    figures=False,
                    subfigures=False,
                    figure_options=figure_options,
                    figure_format=figure_format,
                    run_id=run_id,
                    acc_progress=acc_progress,
                    component_session=acc_session,
                    finalize_component_inventory=False,
                )
                lca_input, lca_session = _checkpoint_lca(
                    lca_input=lca_input,
                    proj_base=proj_base,
                    source_label=source_label,
                    lca_type=lca_type,
                    lca_version_name=lca_version_name,
                    base_allocate_args=base_allocate_args,
                    shared_methods=shared_methods,
                    source_config=source_config.lca_config,
                    output_format=output_format,
                    refresh=False,
                    phase=phase,
                    target_runs=checkpoint,
                    parent_mode=runtime.mode,
                    parent_max_runs=runtime.n_runs,
                    figures=False,
                    figure_format=figure_format,
                    run_id=run_id,
                    lca_progress=lca_progress,
                    component_session=lca_session,
                    finalize_component_inventory=False,
                )
                plan = replace(plan, acc_manifest=acc_manifest, lca_input=lca_input)
            progress_begin(
                progress=asr_progress,
                completed=state.completed_runs,
                max_runs=progress_max_runs,
                target_runs=checkpoint,
                mode=progress_mode,
                component=progress_component,
            )
            state, convergence = append_asr_run_outputs(
                paths=paths,
                plan=plan,
                runtime=runtime,
                state=state,
                target_runs=checkpoint,
                final_checkpoint=checkpoint == runtime.n_runs,
                show_progress=False,
                status=phase,
            )
            progress_complete(
                progress=asr_progress,
                completed=state.completed_runs,
                max_runs=progress_max_runs,
                mode=progress_mode,
                component=progress_component,
            )
            if convergence is not None and bool(convergence.get("reached")):
                break
        finalize_components = state.completed_runs > 0 and (
            (acc_session is not None and acc_session.requires_finalization())
            or (lca_session is not None and lca_session.requires_finalization())
        )
        if finalize_components:
            silent_phase: Any = NullPhasePrinter()
            silent_acc_progress = monte_carlo_run_progress(
                source="uncertainty_acc",
                enabled=False,
            )
            silent_lca_progress = monte_carlo_run_progress(
                source="uncertainty_io_lca",
                enabled=False,
            )
            acc_manifest, acc_session = _checkpoint_acc(
                project_name=project_name,
                years=years,
                shared_methods=shared_methods,
                fu_code=fu_code,
                r_p=r_p,
                s_p=s_p,
                r_c=r_c,
                r_f=r_f,
                mrio_scope=mrio_scope,
                asocc_config=asocc_config,
                base_allocate_args=base_allocate_args,
                base_cc_args=base_cc_args,
                source_config=source_config.acc_config,
                external_method=external_method,
                output_format=output_format,
                phase=silent_phase,
                target_runs=state.completed_runs,
                parent_mode=runtime.mode,
                parent_max_runs=runtime.n_runs,
                figures=False,
                subfigures=False,
                figure_options=figure_options,
                figure_format=figure_format,
                run_id=run_id,
                acc_progress=silent_acc_progress,
                component_session=acc_session,
                finalize_component_inventory=finalize_components,
            )
            lca_input, lca_session = _checkpoint_lca(
                lca_input=lca_input,
                proj_base=proj_base,
                source_label=source_label,
                lca_type=lca_type,
                lca_version_name=lca_version_name,
                base_allocate_args=base_allocate_args,
                shared_methods=shared_methods,
                source_config=source_config.lca_config,
                output_format=output_format,
                refresh=False,
                phase=silent_phase,
                target_runs=state.completed_runs,
                parent_mode=runtime.mode,
                parent_max_runs=runtime.n_runs,
                figures=False,
                figure_format=figure_format,
                run_id=run_id,
                lca_progress=silent_lca_progress,
                component_session=lca_session,
                finalize_component_inventory=finalize_components,
            )
            plan = replace(plan, acc_manifest=acc_manifest, lca_input=lca_input)
        return ASRCheckpointResult(
            plan=plan,
            acc_manifest=acc_manifest,
            acc_session=acc_session,
            lca_input=lca_input,
            lca_session=lca_session,
            completed_runs=state.completed_runs,
            convergence=convergence,
        )
    finally:
        close_asr_run_output_state(state=state)


def _checkpoint_acc(
    *,
    project_name: str,
    years: int | list[int] | range,
    shared_methods: list[str],
    fu_code: str,
    r_p: str | list[str] | None,
    s_p: str | list[str] | None,
    r_c: str | list[str] | None,
    r_f: str | list[str] | None,
    mrio_scope: dict[str, Any],
    asocc_config: dict[str, Any],
    base_allocate_args: dict[str, Any],
    base_cc_args: dict[str, Any],
    source_config: dict[str, Any],
    external_method: dict[str, Any] | None,
    output_format: str,
    phase,
    target_runs: int,
    parent_mode: str,
    parent_max_runs: int,
    figures: bool,
    subfigures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    run_id: str | None,
    acc_progress: RunProgressPrinter,
    component_session: Any | None,
    finalize_component_inventory: bool,
) -> tuple[UncertaintyManifest, Any | None]:
    """Run or reuse the aCC component at one ASR checkpoint."""
    phase.announce(PHASE_B2_ACC, "uncertainty_acc")
    report = acc_inventory_report(
        project_name=project_name,
        years=years,
        shared_methods=shared_methods,
        fu_code=fu_code,
        r_p=r_p,
        s_p=s_p,
        r_c=r_c,
        r_f=r_f,
        mrio_scope=mrio_scope,
        asocc_config=asocc_config,
        base_allocate_args=base_allocate_args,
        base_cc_args=base_cc_args,
        source_config=source_config,
        external_method=external_method,
        output_format=output_format,
        phase=phase,
        target_runs=target_runs,
        parent_mode=parent_mode,
        parent_max_runs=parent_max_runs,
        figures=figures,
        figure_options=figure_options,
        figure_format=figure_format,
        subfigures=subfigures,
        show_progress=True,
        show_component_progress=True,
        run_id=run_id,
        refresh=False,
        progress=acc_progress,
        component_session=component_session,
        finalize_component_inventory=finalize_component_inventory,
    )
    return report.report.manifest, report.session


def _checkpoint_lca(
    *,
    lca_input: LCAUncertaintyInput,
    proj_base,
    source_label: str,
    lca_type: str,
    lca_version_name: str | None,
    base_allocate_args: dict[str, Any],
    shared_methods: list[str],
    source_config: dict[str, Any],
    output_format: str,
    refresh: bool,
    phase,
    target_runs: int,
    parent_mode: str,
    parent_max_runs: int,
    figures: bool,
    figure_format: dict[str, Any] | None,
    run_id: str | None,
    lca_progress: RunProgressPrinter,
    component_session: Any | None,
    finalize_component_inventory: bool,
) -> tuple[LCAUncertaintyInput, Any | None]:
    """Run or reuse the LCA component at one ASR checkpoint."""
    # External LCA inventories are immutable within one ASR run; IO-LCA owns
    # checkpoint scoped component execution.
    if lca_type != IO_LCA_FAMILY and not figures:
        return lca_input, component_session
    phase.announce(
        PHASE_A_LCA,
        "uncertainty_io_lca" if lca_type == "io_lca" else "external_lca",
    )
    resolved = resolve_lca_uncertainty_component_input(
        proj_base=proj_base,
        source_label=source_label,
        lca_type=lca_type,
        lca_version_name=lca_version_name,
        base_allocate_args=base_allocate_args,
        lcia_methods=shared_methods,
        uncertainty_config=source_config,
        output_format=output_format,
        refresh=refresh,
        phase=phase,
        component_inventory=lca_component_inventory(
            lca_type=lca_type,
            target_runs=target_runs,
            parent_mode=parent_mode,
            parent_max_runs=parent_max_runs,
        ),
        figures=figures,
        figure_format=figure_format,
        show_progress=True,
        run_id=run_id,
        status=lca_progress,
        progress=lca_progress,
        component_session=component_session,
        finalize_component_inventory=finalize_component_inventory,
        figure_run_count=target_runs,
    )
    return resolved.input, resolved.session

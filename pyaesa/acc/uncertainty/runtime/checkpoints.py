"""Checkpoint execution for uncertainty aCC runs."""

from dataclasses import dataclass, replace
from collections.abc import Sequence
from typing import Any

from pyaesa.acc.uncertainty.io.run_outputs import (
    append_acc_run_outputs,
    close_acc_run_output_state,
    new_acc_run_output_state,
)
from pyaesa.acc.uncertainty.runtime.component_inputs import (
    asocc_inventory_report,
    dynamic_cc_input as resolve_dynamic_cc_input,
)
from pyaesa.acc.uncertainty.runtime.models import (
    ACCAsoccInput,
    ACCDynamicCCInput,
    ACCUncertaintyPlan,
    ACCUncertaintyRunPaths,
)
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter
from pyaesa.shared.runtime.reporting.run_progress import RunProgressPrinter
from pyaesa.shared.uncertainty_assessment.orchestration import progress_begin, progress_complete
from pyaesa.shared.uncertainty_assessment.monte_carlo.composite import (
    ComponentInput,
)
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest


@dataclass(frozen=True)
class ACCCheckpointResult:
    """Final component and output state after one aCC checkpoint sequence."""

    plan: ACCUncertaintyPlan
    asocc_input: ACCAsoccInput
    asocc_session: Any | None
    dynamic_cc_input: ACCDynamicCCInput | None
    dynamic_cc_session: Any | None
    completed_runs: int
    convergence: dict[str, Any] | None
    output_state: Any | None = None


def run_acc_checkpoints(
    *,
    paths: ACCUncertaintyRunPaths,
    runtime: UncertaintyRuntimeRequest,
    checkpoints: Sequence[int],
    initial_plan: ACCUncertaintyPlan,
    initial_asocc_input: ACCAsoccInput,
    initial_asocc_session: Any | None,
    initial_dynamic_cc_input: ACCDynamicCCInput | None,
    initial_dynamic_cc_session: Any | None,
    start_completed_runs: int,
    base_allocate_args: dict[str, Any],
    external_lcia_methods: list[str] | None,
    config: dict[str, Any],
    external_method: dict[str, Any] | None,
    output_format: str,
    dynamic_branch: dict[str, Any] | None,
    years: int | list[int] | range,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    run_id: str | None,
    asocc_progress: RunProgressPrinter,
    dynamic_cc_progress: RunProgressPrinter,
    acc_progress: RunProgressPrinter,
    progress_mode: str,
    progress_max_runs: int,
    progress_component: bool,
    initial_output_state: Any | None = None,
    finalize_outputs: bool = True,
) -> ACCCheckpointResult:
    """Append aCC checkpoint outputs and refresh component inventories as needed."""
    plan = initial_plan
    asocc_input = initial_asocc_input
    asocc_session = initial_asocc_session
    dynamic_cc = initial_dynamic_cc_input
    dynamic_cc_session = initial_dynamic_cc_session
    state = (
        new_acc_run_output_state(paths=paths, completed_runs=start_completed_runs)
        if initial_output_state is None
        else initial_output_state
    )
    convergence = None
    final_summary_written = False
    try:
        for index, checkpoint in enumerate(checkpoints):
            if int(checkpoint) <= state.completed_runs:
                continue
            if index:
                asocc_input, asocc_session = _checkpoint_asocc_input(
                    asocc_input=asocc_input,
                    base_allocate_args=base_allocate_args,
                    external_lcia_methods=external_lcia_methods,
                    config=config,
                    external_method=external_method,
                    output_format=output_format,
                    target_runs=checkpoint,
                    parent_mode=progress_mode,
                    parent_max_runs=progress_max_runs,
                    figures=False,
                    figure_options=figure_options,
                    figure_format=figure_format,
                    run_id=run_id,
                    progress=asocc_progress,
                    component_session=asocc_session,
                    finalize_component_inventory=False,
                )
                dynamic_component = _checkpoint_dynamic_cc(
                    dynamic_cc=dynamic_cc,
                    dynamic_branch=dynamic_branch,
                    years=years,
                    config=config,
                    output_format=output_format,
                    target_runs=checkpoint,
                    parent_mode=progress_mode,
                    parent_max_runs=progress_max_runs,
                    figures=False,
                    figure_format=figure_format,
                    progress=dynamic_cc_progress,
                    run_id=run_id,
                    component_session=dynamic_cc_session,
                    finalize_component_inventory=False,
                )
                dynamic_cc = dynamic_component.input
                dynamic_cc_session = dynamic_component.session
                plan = replace(
                    plan,
                    asocc_input=asocc_input,
                    dynamic_cc_input=dynamic_cc,
                )
            progress_begin(
                progress=acc_progress,
                completed=state.completed_runs,
                max_runs=progress_max_runs,
                target_runs=checkpoint,
                mode=progress_mode,
                component=progress_component,
            )
            state, convergence = append_acc_run_outputs(
                paths=paths,
                plan=plan,
                runtime=runtime,
                state=state,
                target_runs=checkpoint,
                final_checkpoint=bool(finalize_outputs and checkpoint == runtime.n_runs),
                show_progress=False,
            )
            final_summary_written = bool(
                final_summary_written
                or (finalize_outputs and checkpoint == runtime.n_runs)
                or convergence is not None
            )
            progress_complete(
                progress=acc_progress,
                completed=state.completed_runs,
                max_runs=progress_max_runs,
                mode=progress_mode,
                component=progress_component,
            )
            if convergence is not None and bool(convergence.get("reached")):
                break
        if finalize_outputs and state.completed_runs > 0 and not final_summary_written:
            state, final_convergence = append_acc_run_outputs(
                paths=paths,
                plan=plan,
                runtime=runtime,
                state=state,
                target_runs=state.completed_runs,
                final_checkpoint=True,
                show_progress=False,
            )
            convergence = final_convergence
        finalize_components = (
            bool(finalize_outputs)
            and state.completed_runs > 0
            and (
                (asocc_session is not None and asocc_session.requires_finalization())
                or (dynamic_cc_session is not None and dynamic_cc_session.requires_finalization())
            )
        )
        if finalize_components:
            asocc_input, asocc_session = _checkpoint_asocc_input(
                asocc_input=asocc_input,
                base_allocate_args=base_allocate_args,
                external_lcia_methods=external_lcia_methods,
                config=config,
                external_method=external_method,
                output_format=output_format,
                target_runs=state.completed_runs,
                parent_mode=progress_mode,
                parent_max_runs=progress_max_runs,
                figures=False,
                figure_options=figure_options,
                figure_format=figure_format,
                run_id=run_id,
                progress=asocc_progress,
                component_session=asocc_session,
                finalize_component_inventory=finalize_components,
            )
            dynamic_component = _checkpoint_dynamic_cc(
                dynamic_cc=dynamic_cc,
                dynamic_branch=dynamic_branch,
                years=years,
                config=config,
                output_format=output_format,
                target_runs=state.completed_runs,
                parent_mode=progress_mode,
                parent_max_runs=progress_max_runs,
                figures=False,
                figure_format=figure_format,
                progress=dynamic_cc_progress,
                run_id=run_id,
                component_session=dynamic_cc_session,
                finalize_component_inventory=finalize_components,
            )
            dynamic_cc = dynamic_component.input
            dynamic_cc_session = dynamic_component.session
            plan = replace(
                plan,
                asocc_input=asocc_input,
                dynamic_cc_input=dynamic_cc,
            )
        return ACCCheckpointResult(
            plan=plan,
            asocc_input=asocc_input,
            asocc_session=asocc_session,
            dynamic_cc_input=dynamic_cc,
            dynamic_cc_session=dynamic_cc_session,
            completed_runs=state.completed_runs,
            convergence=convergence,
            output_state=None if finalize_outputs else state,
        )
    finally:
        if finalize_outputs:
            close_acc_run_output_state(state=state)


def _checkpoint_dynamic_cc(
    *,
    dynamic_cc: ACCDynamicCCInput | None,
    dynamic_branch: dict[str, Any] | None,
    years: int | list[int] | range,
    config: dict[str, Any],
    output_format: str,
    target_runs: int,
    parent_mode: str,
    parent_max_runs: int,
    figures: bool,
    figure_format: dict[str, Any] | None,
    progress: RunProgressPrinter,
    run_id: str | None,
    component_session: Any | None,
    finalize_component_inventory: bool,
) -> ComponentInput[ACCDynamicCCInput | None]:
    if dynamic_branch is None:
        return ComponentInput(input=None, session=None)
    return resolve_dynamic_cc_input(
        branch=dynamic_branch,
        years=years,
        config=config,
        output_format=output_format,
        target_runs=target_runs,
        parent_mode=parent_mode,
        parent_max_runs=parent_max_runs,
        figures=figures,
        figure_format=figure_format,
        show_progress=False,
        progress=progress,
        run_id=run_id,
        phase=NullPhasePrinter(),
        refresh=False,
        component_session=component_session,
        finalize_component_inventory=finalize_component_inventory,
    )


def _checkpoint_asocc_input(
    *,
    asocc_input: ACCAsoccInput,
    base_allocate_args: dict[str, Any],
    external_lcia_methods: list[str] | None,
    config: dict[str, Any],
    external_method: dict[str, Any] | None,
    output_format: str,
    target_runs: int,
    parent_mode: str,
    parent_max_runs: int,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    run_id: str | None,
    progress: RunProgressPrinter,
    component_session: Any | None,
    finalize_component_inventory: bool,
) -> tuple[ACCAsoccInput, Any | None]:
    if asocc_input.manifest is None:
        return asocc_input, component_session
    run = asocc_inventory_report(
        base_allocate_args=base_allocate_args,
        external_lcia_methods=external_lcia_methods,
        config=config,
        external_method=external_method,
        output_format=output_format,
        target_runs=target_runs,
        parent_mode=parent_mode,
        parent_max_runs=parent_max_runs,
        figures=figures,
        figure_options=figure_options,
        figure_format=figure_format,
        show_progress=False,
        phase=NullPhasePrinter(),
        run_id=run_id,
        refresh=False,
        progress=progress,
        component_session=component_session,
        finalize_component_inventory=finalize_component_inventory,
    )
    return (
        ACCAsoccInput(
            identity=None,
            deterministic_values=None,
            manifest=run.report.manifest,
            deterministic_manifest_path=None,
            reuse_status=run.report.reuse_status,
        ),
        run.session,
    )

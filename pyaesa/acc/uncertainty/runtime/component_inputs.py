"""Component inputs for uncertainty aCC runs."""

from dataclasses import dataclass
from typing import Any, cast

from pyaesa.acc.uncertainty.request.normalization import (
    AR6_DYNAMIC_CC_SOURCE,
    asocc_uncertainty_config_for_acc,
    dynamic_cc_source_parameters,
)
from pyaesa.acc.uncertainty.runtime.models import ACCAsoccInput, ACCDynamicCCInput
from pyaesa.acc.uncertainty.runtime.scope import ACCUncertaintyScope
from pyaesa.acc.uncertainty.sources.dynamic_cc import (
    deterministic_dynamic_cc_input,
    dynamic_ar6_cc_uncertainty_input,
)
from pyaesa.asocc.uncertainty.engine.reuse.prerequisites import (
    prepare_asocc_deterministic_prerequisite,
)
from pyaesa.asocc.uncertainty.engine.runner import run_uncertainty_asocc_component
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import (
    ASOCC_VALUE_COLUMN,
    load_final_deterministic_asocc_rows,
)
from pyaesa.asocc.uncertainty.inputs.external_rows import (
    external_asocc_has_monte_carlo_rows,
    resolve_external_asocc_rows,
)
from pyaesa.asocc.uncertainty.schema.public_rows import finalize_asocc_public_row_identity
from pyaesa.asocc.uncertainty.sources.names import (
    ASOCC_UNCERTAINTY_SOURCES,
    DEFAULT_ASOCC_UNCERTAINTY_SOURCES,
)
from pyaesa.shared.acc_asr_common.scope.composite import base_asocc_kwargs_from_allocate_args
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    PHASE_B1_AR6_DYNAMIC_CC,
    PHASE_B1_ASOCC,
    phase_ready_detail,
    phase_reused_detail,
)
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
from pyaesa.shared.runtime.reporting.run_progress import RunProgressPrinter
from pyaesa.shared.uncertainty_assessment.monte_carlo.composite import (
    ComponentInput,
    ComponentRun,
    component_inventory_payload,
    fixed_inventory_mc_parameters,
)
from pyaesa.shared.uncertainty_assessment.orchestration import (
    output_root_from_path,
)
from pyaesa.shared.uncertainty_assessment.request.sources import (
    SourceActivationPlan,
    build_source_activation_plan,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


@dataclass(frozen=True)
class ACCInitialComponents:
    """Initial pyaesa owned component state for an aCC uncertainty run."""

    asocc_input: ACCAsoccInput
    asocc_session: Any | None
    dynamic_cc_input: ACCDynamicCCInput | None
    dynamic_cc_session: Any | None
    run_id: str | None


def initial_acc_components(
    *,
    phase: PhasePrinter,
    scope: ACCUncertaintyScope,
    config: dict[str, Any],
    external_method: dict[str, Any] | None,
    output_format: str,
    target_runs: int,
    parent_mode: str,
    parent_max_runs: int,
    component_figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    current_run_id: str | None,
    refresh: bool,
    asocc_progress: RunProgressPrinter,
    dynamic_cc_progress: RunProgressPrinter,
    asocc_session: Any | None = None,
    dynamic_cc_session: Any | None = None,
    finalize_component_inventory: bool = False,
) -> ACCInitialComponents:
    """Resolve the first aSoCC and dynamic AR6 CC component inventories."""
    asocc_component = initial_asocc_input(
        phase=phase,
        base_allocate_args=scope.base_allocate_args,
        external_lcia_methods=scope.shared_methods,
        config=config,
        external_method=external_method,
        output_format=output_format,
        target_runs=target_runs,
        parent_mode=parent_mode,
        parent_max_runs=parent_max_runs,
        figures=component_figures,
        figure_options=figure_options,
        figure_format=figure_format,
        progress=asocc_progress,
        run_id=current_run_id,
        refresh=refresh,
        component_session=asocc_session,
        finalize_component_inventory=finalize_component_inventory,
    )
    asocc_input = asocc_component.input
    run_id = _component_run_id(current_run_id=current_run_id, asocc_input=asocc_input)
    dynamic_component = _initial_dynamic_cc_input(
        phase=phase,
        scope=scope,
        years=scope.base_args["years"],
        config=config,
        output_format=output_format,
        target_runs=target_runs,
        parent_mode=parent_mode,
        parent_max_runs=parent_max_runs,
        component_figures=component_figures,
        figure_format=figure_format,
        progress=dynamic_cc_progress,
        run_id=run_id,
        refresh=refresh,
        component_session=dynamic_cc_session,
        finalize_component_inventory=finalize_component_inventory,
    )
    dynamic_input = dynamic_component.input
    if run_id is None and dynamic_input is not None and dynamic_input.manifest is not None:
        run_id = dynamic_input.manifest.run_id
    return ACCInitialComponents(
        asocc_input=asocc_input,
        asocc_session=asocc_component.session,
        dynamic_cc_input=dynamic_input,
        dynamic_cc_session=dynamic_component.session,
        run_id=run_id,
    )


def initial_asocc_input(
    *,
    phase: PhasePrinter,
    base_allocate_args: dict[str, Any],
    external_lcia_methods: list[str],
    config: dict[str, Any],
    external_method: dict[str, Any] | None,
    output_format: str,
    target_runs: int,
    parent_mode: str,
    parent_max_runs: int,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    progress: RunProgressPrinter,
    run_id: str | None,
    refresh: bool,
    component_session: Any | None = None,
    finalize_component_inventory: bool = False,
) -> ComponentInput[ACCAsoccInput]:
    """Resolve deterministic or stochastic aSoCC input for aCC."""
    sources = _asocc_source_plan(config=config)
    base_asocc_args = base_asocc_kwargs_from_allocate_args(base_allocate_args=base_allocate_args)
    if not sources.names:
        deterministic = deterministic_asocc_input(
            phase=phase,
            base_asocc_args=base_asocc_args,
            external_lcia_methods=external_lcia_methods,
            external_method=external_method,
            figures=figures,
            figure_options=figure_options,
            figure_format=figure_format,
            refresh=refresh,
        )
        if deterministic is not None:
            return ComponentInput(input=deterministic, session=None)
    phase.announce(PHASE_B1_ASOCC, "uncertainty_asocc")
    asocc_run = asocc_inventory_report(
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
        show_progress=True,
        phase=phase,
        run_id=run_id,
        refresh=refresh,
        progress=progress,
        component_session=component_session,
        finalize_component_inventory=finalize_component_inventory,
    )
    return ComponentInput(
        input=ACCAsoccInput(
            identity=None,
            deterministic_values=None,
            manifest=asocc_run.report.manifest,
            deterministic_manifest_path=None,
            reuse_status=asocc_run.report.reuse_status,
        ),
        session=asocc_run.session,
    )


def deterministic_asocc_input(
    *,
    phase: Any,
    base_asocc_args: dict[str, Any],
    external_lcia_methods: list[str],
    external_method: dict[str, Any] | None,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    refresh: bool,
) -> ACCAsoccInput | None:
    """Load deterministic aSoCC rows as fixed values or report a stochastic external lane."""
    phase.announce(PHASE_B1_ASOCC, "deterministic_asocc")
    phase.status("Resolving deterministic aSoCC prerequisite", owner="deterministic_asocc")
    prerequisite = prepare_asocc_deterministic_prerequisite(
        base_asocc_args=base_asocc_args,
        refresh=refresh,
        figures=figures,
        figure_options=None
        if figure_options is None
        else {
            "per_method": figure_options["per_method"],
            "multi_method": figure_options["multi_method"],
        },
        figure_format=figure_format,
        figure_external_method=external_method if figures else None,
        phase=phase,
    )
    phase.status("Loading deterministic aSoCC outputs", owner="deterministic_asocc")
    loaded = load_final_deterministic_asocc_rows(prerequisite=prerequisite)
    if external_asocc_has_monte_carlo_rows(
        loaded=loaded,
        external_method=external_method,
        external_lcia_methods=external_lcia_methods,
    ):
        return None
    loaded, _external_plan = resolve_external_asocc_rows(
        loaded=loaded,
        external_method=external_method,
        required_runs=None,
        external_lcia_methods=external_lcia_methods,
    )
    detail_builder = (
        phase_reused_detail if prerequisite.reuse_status == "reused_exact" else phase_ready_detail
    )
    phase.complete(
        detail_builder(
            scope_name="aSoCC deterministic",
            output_root=output_root_from_path(prerequisite.deterministic_manifest_path),
        )
    )
    return ACCAsoccInput(
        identity=finalize_asocc_public_row_identity(
            frame=loaded.rows,
            value_column=ASOCC_VALUE_COLUMN,
        ),
        deterministic_values=loaded.rows[ASOCC_VALUE_COLUMN].to_numpy(dtype="float64"),
        manifest=None,
        deterministic_manifest_path=prerequisite.deterministic_manifest_path,
        reuse_status=prerequisite.reuse_status,
    )


def asocc_inventory_report(
    *,
    base_allocate_args: dict[str, Any],
    external_lcia_methods: list[str],
    config: dict[str, Any],
    external_method: dict[str, Any] | None,
    output_format: str,
    target_runs: int,
    parent_mode: str,
    parent_max_runs: int,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    show_progress: bool,
    phase: PhasePrinter | NullPhasePrinter,
    run_id: str | None,
    refresh: bool,
    progress: RunProgressPrinter | None = None,
    component_session: Any | None = None,
    finalize_component_inventory: bool = False,
) -> ComponentRun:
    """Run or reuse the aSoCC component inventory for one aCC checkpoint."""
    asocc_config = {
        **asocc_uncertainty_config_for_acc(config),
        "mc_parameters": fixed_inventory_mc_parameters(target_runs=target_runs),
    }
    return run_uncertainty_asocc_component(
        base_asocc_args=base_asocc_kwargs_from_allocate_args(base_allocate_args=base_allocate_args),
        uncertainty_config=asocc_config,
        sobol_parameters={"active": False},
        external_method=external_method,
        output_format=output_format,
        figures=figures,
        figure_options=figure_options,
        figure_format=figure_format,
        refresh=refresh,
        component_inventory=component_inventory_payload(
            composite_family="acc",
            component_name="asocc",
            target_runs=target_runs,
            parent_mode=parent_mode,
            parent_max_runs=parent_max_runs,
        ),
        external_lcia_methods=external_lcia_methods,
        run_id=run_id,
        show_progress=show_progress,
        phase=phase,
        progress=progress,
        component_session=component_session,
        finalize_component_inventory=finalize_component_inventory,
    )


def _initial_dynamic_cc_input(
    *,
    phase: PhasePrinter,
    scope: ACCUncertaintyScope,
    years: int | list[int] | range,
    config: dict[str, Any],
    output_format: str,
    target_runs: int,
    parent_mode: str,
    parent_max_runs: int,
    component_figures: bool,
    figure_format: dict[str, Any] | None,
    progress: RunProgressPrinter,
    run_id: str | None,
    refresh: bool,
    component_session: Any | None = None,
    finalize_component_inventory: bool = False,
) -> ComponentInput[ACCDynamicCCInput | None]:
    if not scope.dynamic_branches:
        return ComponentInput(input=None, session=None)
    phase.expect_visible(PHASE_B1_AR6_DYNAMIC_CC)
    source_parameters = dynamic_cc_source_parameters(config.get(AR6_DYNAMIC_CC_SOURCE))
    if source_parameters is None:
        phase.announce(PHASE_B1_AR6_DYNAMIC_CC, "deterministic_ar6_cc")
    try:
        dynamic_input = dynamic_cc_input(
            branch=scope.dynamic_branches[0],
            years=years,
            config=config,
            output_format=output_format,
            target_runs=target_runs,
            parent_mode=parent_mode,
            parent_max_runs=parent_max_runs,
            figures=component_figures,
            figure_format=figure_format,
            show_progress=False,
            progress=progress,
            run_id=run_id,
            phase=phase,
            refresh=refresh,
            component_session=component_session,
            finalize_component_inventory=finalize_component_inventory,
        )
    finally:
        progress.finish()
    if source_parameters is None:
        complete_dynamic_cc_phase(
            phase=phase,
            dynamic_cc_input=cast(ACCDynamicCCInput, dynamic_input.input),
        )
    return dynamic_input


def dynamic_cc_input(
    *,
    branch: dict[str, Any],
    years: int | list[int] | range,
    config: dict[str, Any],
    output_format: str,
    target_runs: int,
    parent_mode: str,
    parent_max_runs: int,
    figures: bool,
    figure_format: dict[str, Any] | None,
    show_progress: bool,
    progress: RunProgressPrinter,
    run_id: str | None,
    phase: PhasePrinter | NullPhasePrinter,
    refresh: bool,
    component_session: Any | None = None,
    finalize_component_inventory: bool = False,
) -> ComponentInput[ACCDynamicCCInput | None]:
    """Resolve deterministic or uncertain dynamic AR6 CC input for aCC."""
    source_parameters = dynamic_cc_source_parameters(config.get(AR6_DYNAMIC_CC_SOURCE))
    if source_parameters is None:
        return ComponentInput(
            input=deterministic_dynamic_cc_input(
                branch=branch,
                years=years,
                figures=figures,
                figure_format=figure_format,
                status=phase,
                refresh=refresh,
            ),
            session=None,
        )
    phase.announce(PHASE_B1_AR6_DYNAMIC_CC, "uncertainty_ar6_cc")
    dynamic_input, session = dynamic_ar6_cc_uncertainty_input(
        branch=branch,
        years=years,
        source_parameters=source_parameters,
        mc_parameters=fixed_inventory_mc_parameters(target_runs=target_runs),
        output_format=output_format,
        figures=figures,
        figure_format=figure_format,
        component_inventory=component_inventory_payload(
            composite_family="acc",
            component_name="ar6_cc",
            target_runs=target_runs,
            parent_mode=parent_mode,
            parent_max_runs=parent_max_runs,
        ),
        run_id=run_id,
        show_progress=show_progress,
        phase=phase,
        refresh=refresh,
        progress=progress,
        component_session=component_session,
        finalize_component_inventory=finalize_component_inventory,
    )
    return ComponentInput(input=dynamic_input, session=session)


def complete_dynamic_cc_phase(
    *,
    phase: PhasePrinter,
    dynamic_cc_input: ACCDynamicCCInput,
) -> None:
    """Print phase completion for deterministic dynamic CC input."""
    detail_builder = (
        phase_reused_detail
        if dynamic_cc_input.reuse_status == "reused_exact"
        else phase_ready_detail
    )
    manifest_path = dynamic_cc_input.deterministic_manifest_path
    phase.complete(
        detail_builder(
            scope_name="dynamic AR6 CC deterministic",
            output_root=None if manifest_path is None else output_root_from_path(manifest_path),
        ),
        owner="deterministic_ar6_cc",
    )


def _component_run_id(
    *,
    current_run_id: str | None,
    asocc_input: ACCAsoccInput,
) -> str | None:
    if asocc_input.manifest is not None:
        return str(cast(UncertaintyManifest, asocc_input.manifest).run_id)
    return current_run_id


def _asocc_source_plan(*, config: dict[str, Any]) -> SourceActivationPlan:
    payload = {
        key: value
        for key, value in asocc_uncertainty_config_for_acc(config).items()
        if key != "mc_parameters"
    }
    return build_source_activation_plan(
        uncertainty_config=payload,
        allowed_sources=ASOCC_UNCERTAINTY_SOURCES,
        default_sources=DEFAULT_ASOCC_UNCERTAINTY_SOURCES,
    )

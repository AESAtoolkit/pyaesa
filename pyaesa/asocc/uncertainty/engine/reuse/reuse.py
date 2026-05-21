"""aSoCC uncertainty completed run reuse policy."""

from pathlib import Path
from typing import Any, Mapping, cast

from pyaesa.asocc.uncertainty.inputs.external_rows import (
    EXTERNAL_ASOCC_RUN_SOURCE,
    ExternalAsoccRowsPlan,
    external_plan_for_years,
)
from pyaesa.asocc.uncertainty.io.paths import asocc_monte_carlo_root
from pyaesa.asocc.uncertainty.engine.sobol.scope import selected_sobol_years
from pyaesa.shared.uncertainty_assessment.run_state.runs import (
    CompatibleMonteCarloRun,
    appendable_completed_run,
    compatible_completed_runs,
    compatible_completed_run_for_id,
    complete_run_with_requested_runs,
    complete_run_with_requested_sobol,
)
from pyaesa.shared.uncertainty_assessment.sobol.plan import SobolPlan, sobol_plan_payload


def compatible_completed_runs_for_context(
    *,
    deterministic_manifest_path: Path,
    compatibility_key: str,
    include_running_component_inventory: bool = False,
) -> tuple[CompatibleMonteCarloRun, ...]:
    """Return compatible aSoCC runs for one manifest context."""
    return compatible_completed_runs(
        monte_carlo_root=asocc_monte_carlo_root(
            deterministic_manifest_path=deterministic_manifest_path
        ),
        compatibility_key=compatibility_key,
        include_running_component_inventory=include_running_component_inventory,
    )


def compatible_completed_run_id_for_context(
    *,
    deterministic_manifest_path: Path,
    run_id: str | None,
    include_running_component_inventory: bool = False,
) -> CompatibleMonteCarloRun | None:
    """Return one required pyaesa owned aSoCC run id when present."""
    return compatible_completed_run_for_id(
        monte_carlo_root=asocc_monte_carlo_root(
            deterministic_manifest_path=deterministic_manifest_path
        ),
        run_id=run_id,
        include_running_component_inventory=include_running_component_inventory,
    )


def compatible_complete_run(
    *,
    compatible: tuple[CompatibleMonteCarloRun, ...],
    runtime: Any,
    mc_parameters: Mapping[str, Any] | None,
) -> CompatibleMonteCarloRun | None:
    """Return a completed aSoCC run satisfying the current run request."""
    return complete_run_with_requested_runs(
        compatible=compatible,
        requested_runs=runtime.n_runs,
        mode=runtime.mode,
        mc_parameters=mc_parameters,
    )


def compatible_complete_sobol_run(
    *,
    compatible: tuple[CompatibleMonteCarloRun, ...],
    runtime: Any,
    mc_parameters: Mapping[str, Any] | None,
    sources: Any,
    external_plan: ExternalAsoccRowsPlan,
    sobol_plan: SobolPlan,
    requested_years: tuple[int, ...],
) -> CompatibleMonteCarloRun | None:
    """Return a completed aSoCC run satisfying run and Sobol requests."""
    sobol_parameters = sobol_plan_payload(plan=sobol_plan)
    selected_years = selected_sobol_years(plan=sobol_plan, requested_years=requested_years)
    source_dimensions = _sobol_source_dimensions(
        sources=sources,
        external_plan=external_plan_for_years(
            plan=external_plan,
            years=selected_years,
        ),
    )
    if len(source_dimensions) < 2:
        return complete_run_with_requested_runs(
            compatible=compatible,
            requested_runs=runtime.n_runs,
            mode=runtime.mode,
            mc_parameters=mc_parameters,
        )
    for candidate in compatible:
        reusable = complete_run_with_requested_sobol(
            compatible=(candidate,),
            requested_runs=runtime.n_runs,
            mode=runtime.mode,
            mc_parameters=mc_parameters,
            sobol_parameters=sobol_parameters,
        )
        if (
            reusable is not None
            and _same_sobol_source_dimensions(
                run=reusable,
                source_dimensions=source_dimensions,
            )
            and _same_sobol_selected_years(run=reusable, selected_years=selected_years)
        ):
            return reusable
    return None


def _sobol_source_dimensions(
    *,
    sources: Any,
    external_plan: ExternalAsoccRowsPlan,
) -> tuple[str, ...]:
    names = list(sources.names)
    if external_plan.monte_carlo_sources:
        names.append(EXTERNAL_ASOCC_RUN_SOURCE)
    return tuple(names)


def _same_sobol_source_dimensions(
    *,
    run: CompatibleMonteCarloRun,
    source_dimensions: tuple[str, ...],
) -> bool:
    sobol = run.manifest.sobol or {}
    method = sobol.get("method")
    if not isinstance(method, dict):
        return False
    return tuple(method.get("source_dimensions", ())) == source_dimensions


def _same_sobol_selected_years(
    *,
    run: CompatibleMonteCarloRun,
    selected_years: tuple[int, ...],
) -> bool:
    sobol = run.manifest.sobol or {}
    method = cast(Mapping[str, Any], sobol.get("method"))
    expected = list(selected_years)
    return (
        sobol.get("selected_output_years") == expected
        and method.get("selected_output_years") == expected
    )


def appendable_run_for_runtime(
    *,
    compatible: tuple[CompatibleMonteCarloRun, ...],
    runtime: Any,
) -> CompatibleMonteCarloRun | None:
    """Return the completed run that owns the missing run index prefix."""
    return appendable_completed_run(
        compatible=compatible,
        mode=runtime.mode,
        max_completed_runs=runtime.n_runs,
    )

"""Shared Monte Carlo run discovery and append helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    read_manifest,
)
from pyaesa.shared.runtime.reuse.branch_reuse import cleanup_refresh_scope_targets


@dataclass(frozen=True)
class CompatibleMonteCarloRun:
    """One completed Monte Carlo run compatible with the current request."""

    run_root: Path
    manifest: UncertaintyManifest


def compatible_completed_runs(
    *,
    monte_carlo_root: Path,
    compatibility_key: str,
    include_running_component_inventory: bool = False,
) -> tuple[CompatibleMonteCarloRun, ...]:
    """Return compatible runs that can supply already written run indices."""
    root = Path(monte_carlo_root)
    if not root.exists():
        return ()
    matches: list[CompatibleMonteCarloRun] = []
    for manifest_path in sorted(root.glob("mc_*/logs/scope_manifest.json")):
        manifest = read_manifest(path=manifest_path)
        if not _run_can_supply_indices(
            manifest=manifest,
            include_running_component_inventory=include_running_component_inventory,
        ):
            continue
        if manifest.compatibility_key != compatibility_key:
            continue
        matches.append(
            CompatibleMonteCarloRun(run_root=manifest_path.parents[1], manifest=manifest)
        )
    return tuple(sorted(matches, key=lambda item: item.manifest.completed_runs, reverse=True))


def compatible_completed_run_for_id(
    *,
    monte_carlo_root: Path,
    run_id: str | None,
    include_running_component_inventory: bool = False,
) -> CompatibleMonteCarloRun | None:
    """Return one compatible run id that can supply already written run indices."""
    if run_id is None:
        return None
    manifest_path = Path(monte_carlo_root) / str(run_id) / "logs" / "scope_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_manifest(path=manifest_path)
    if not _run_can_supply_indices(
        manifest=manifest,
        include_running_component_inventory=include_running_component_inventory,
    ):
        return None
    return CompatibleMonteCarloRun(run_root=manifest_path.parents[1], manifest=manifest)


def _run_can_supply_indices(
    *,
    manifest: UncertaintyManifest,
    include_running_component_inventory: bool,
) -> bool:
    if manifest.completed_runs <= 0:
        return False
    if manifest.status == "complete":
        return True
    return bool(
        include_running_component_inventory
        and manifest.status == "running"
        and manifest.component_inventory is not None
    )


def cleanup_monte_carlo_runs_for_refresh(
    *,
    monte_carlo_root: Path,
    compatibility_key: str,
    run_id: str | None,
    arguments: Mapping[str, Any] | None = None,
    component_inventory: Mapping[str, Any] | None = None,
) -> None:
    """Delete the resolved Monte Carlo run directories for one refresh request."""
    root = Path(monte_carlo_root)
    if run_id is not None:
        cleanup_refresh_scope_targets(targets=(root / str(run_id),))
        return
    if not root.exists():
        return
    targets: list[Path] = []
    for manifest_path in sorted(root.glob("mc_*/logs/scope_manifest.json")):
        manifest = read_manifest(path=manifest_path)
        if manifest.compatibility_key == compatibility_key or _same_refresh_request(
            manifest=manifest,
            arguments=arguments,
            component_inventory=component_inventory,
        ):
            targets.append(manifest_path.parents[1])
    cleanup_refresh_scope_targets(targets=tuple(targets))


def _same_refresh_request(
    *,
    manifest: UncertaintyManifest,
    arguments: Mapping[str, Any] | None,
    component_inventory: Mapping[str, Any] | None,
) -> bool:
    """Return whether a manifest belongs to the same public refresh request."""
    if arguments is None:
        return False
    expected_inventory = None if component_inventory is None else dict(component_inventory)
    return (
        manifest.arguments == dict(arguments) and manifest.component_inventory == expected_inventory
    )


def complete_run_with_requested_runs(
    *,
    compatible: tuple[CompatibleMonteCarloRun, ...],
    requested_runs: int,
    mode: str,
    mc_parameters: Mapping[str, Any] | None = None,
) -> CompatibleMonteCarloRun | None:
    """Return a compatible completed run that satisfies the current request."""
    for run in compatible:
        if run.manifest.mode != mode:
            continue
        if mode == "convergence":
            same_criteria = _same_convergence_criteria(
                run_parameters=cast(Mapping[str, Any], run.manifest.mc_parameters),
                requested_parameters=cast(Mapping[str, Any], mc_parameters),
            )
            if same_criteria and (
                run.manifest.completed_runs == requested_runs
                or (_convergence_reached(run=run) and run.manifest.completed_runs <= requested_runs)
            ):
                return run
            continue
        if run.manifest.completed_runs >= requested_runs:
            return run
    return None


def complete_run_with_requested_sobol(
    *,
    compatible: tuple[CompatibleMonteCarloRun, ...],
    requested_runs: int,
    mode: str,
    mc_parameters: Mapping[str, Any] | None,
    sobol_parameters: Mapping[str, Any],
) -> CompatibleMonteCarloRun | None:
    """Return a completed run that satisfies Monte Carlo and Sobol requests."""
    for run in compatible:
        completed_run = complete_run_with_requested_runs(
            compatible=(run,),
            requested_runs=requested_runs,
            mode=mode,
            mc_parameters=mc_parameters,
        )
        if completed_run is None:
            continue
        sobol = run.manifest.sobol or {}
        if not bool(sobol.get("ran")):
            continue
        if sobol.get("parameters") != dict(sobol_parameters):
            continue
        if str(sobol_parameters.get("mode")) == "convergence" and not bool(sobol.get("reached")):
            continue
        return run
    return None


def appendable_completed_run(
    *,
    compatible: tuple[CompatibleMonteCarloRun, ...],
    mode: str,
    max_completed_runs: int,
) -> CompatibleMonteCarloRun | None:
    """Return the largest compatible run that can supply the requested run index prefix."""
    parent_modes = {"fixed"} if mode == "fixed" else {"fixed", "convergence"}
    for run in compatible:
        completed = int(run.manifest.completed_runs)
        equal_unfinished_component = (
            completed == int(max_completed_runs)
            and run.manifest.status == "running"
            and run.manifest.component_inventory is not None
        )
        if run.manifest.mode in parent_modes and (
            completed < int(max_completed_runs) or equal_unfinished_component
        ):
            return run
    return None


def _convergence_reached(*, run: CompatibleMonteCarloRun) -> bool:
    convergence = run.manifest.convergence or {}
    return bool(convergence.get("reached"))


def _same_convergence_criteria(
    *,
    run_parameters: Mapping[str, Any],
    requested_parameters: Mapping[str, Any],
) -> bool:
    return all(
        run_parameters.get(key) == requested_parameters.get(key)
        for key in ("rtol", "stable_runs", "convergence_statistics")
    )

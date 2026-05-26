"""Composite Monte Carlo convergence checkpoints and inventory metadata."""

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from pyaesa.shared.uncertainty_assessment.request.core import (
    DEFAULT_CONVERGENCE_MAX_RUNS,
    DEFAULT_CONVERGENCE_RTOL,
    DEFAULT_CONVERGENCE_STABLE_RUNS,
    DEFAULT_CONVERGENCE_STATISTICS,
)
from pyaesa.shared.uncertainty_assessment.run_state.report import UncertaintyRunReport

# ``PUBLIC_RUN_ROLE`` marks a normal direct public uncertainty run in
# compatibility metadata. ``COMPONENT_INVENTORY_ROLE`` marks an upstream fixed
# run inventory written for a downstream ACC or ASR convergence run. The role
# prevents the reuse resolver from treating those two run types as equivalent
# when their scientific request axes are otherwise identical.
COMPONENT_INVENTORY_ROLE = "component_inventory"
PUBLIC_RUN_ROLE = "public"
TComponentInput = TypeVar("TComponentInput")


@dataclass(frozen=True)
class ComponentInput(Generic[TComponentInput]):
    """Resolved component input and local session returned to a parent run."""

    input: TComponentInput
    session: Any | None


@dataclass(frozen=True)
class ComponentRun:
    """Component inventory report and local session returned to a parent run."""

    report: UncertaintyRunReport
    session: Any


def fixed_inventory_mc_parameters(*, target_runs: int) -> dict[str, dict[str, object]]:
    """Return fixed Monte Carlo parameters for a component inventory checkpoint."""
    return {
        "fixed": {"active": True, "n_runs": int(target_runs)},
        "convergence": {
            "active": False,
            "max_runs": DEFAULT_CONVERGENCE_MAX_RUNS,
            "rtol": DEFAULT_CONVERGENCE_RTOL,
            "stable_runs": DEFAULT_CONVERGENCE_STABLE_RUNS,
            "convergence_statistics": list(DEFAULT_CONVERGENCE_STATISTICS),
        },
    }


def component_inventory_payload(
    *,
    composite_family: str,
    component_name: str,
    target_runs: int,
    parent_mode: str = "fixed",
    parent_max_runs: int | None = None,
) -> dict[str, Any]:
    """Return manifest metadata for pyaesa owned component inventories."""
    target = int(target_runs)
    return {
        "role": COMPONENT_INVENTORY_ROLE,
        "composite_family": str(composite_family),
        "component_name": str(component_name),
        "target_runs": target,
        "parent_mode": str(parent_mode),
        "parent_max_runs": target if parent_max_runs is None else int(parent_max_runs),
    }


def run_role_payload(
    *,
    component_inventory: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the compatibility run role payload for public and inventory runs."""
    if component_inventory is None:
        return {"role": PUBLIC_RUN_ROLE}
    payload = dict(component_inventory)
    payload.pop("target_runs", None)
    payload.pop("parent_mode", None)
    payload.pop("parent_max_runs", None)
    return payload


def component_inventory_parent_convergence(
    *,
    component_inventory: dict[str, Any] | None,
) -> bool:
    """Return whether this inventory belongs to a parent convergence run."""
    return (
        component_inventory is not None and str(component_inventory["parent_mode"]) == "convergence"
    )


def component_inventory_finalizes(
    *,
    component_inventory: dict[str, Any] | None,
    finalize_component_inventory: bool,
) -> bool:
    """Return whether a component call must write final public artifacts."""
    return component_inventory is None or bool(finalize_component_inventory)


def initial_component_inventory_finalizes(
    *,
    checkpoints: Sequence[int],
    finalize_outputs: bool = True,
) -> bool:
    """Return whether the first component checkpoint is also final.

    A parent convergence request with one checkpoint has no later component
    append step, so the upstream inventory should close its output state before
    the downstream plan is built.
    """
    return bool(finalize_outputs and len(checkpoints) == 1)


def component_inventory_progress_parameters(
    *,
    component_inventory: dict[str, Any] | None,
    runtime_mode: str,
    runtime_n_runs: int,
) -> dict[str, Any]:
    """Return the progress label parameters for public or component run output."""
    if component_inventory is None or not component_inventory_parent_convergence(
        component_inventory=component_inventory
    ):
        return {"mode": runtime_mode, "max_runs": int(runtime_n_runs), "component": False}
    return {
        "mode": "convergence",
        "max_runs": int(component_inventory["parent_max_runs"]),
        "component": True,
    }

"""Rebuild ASR subfigure inputs from completed dependency manifests."""

from dataclasses import replace
from typing import Any

from pyaesa.acc.uncertainty.runtime.reuse_dependencies import plan_from_reused_acc_dependencies
from pyaesa.shared.uncertainty_assessment.orchestration import manifest_output_root
from pyaesa.shared.uncertainty_assessment.run_state.completed_dependencies import (
    optional_completed_dependency_manifest,
    required_completed_dependency_manifest,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def plan_from_reused_asr_dependencies(*, plan, manifest: UncertaintyManifest):
    """Return an ASR subfigure plan rooted in completed manifest dependencies."""
    acc_manifest, _acc_reuse = required_completed_dependency_manifest(
        manifest=manifest,
        base_function_source="uncertainty_acc",
        parent_scope_name="ASR",
    )
    lca_input = plan.lca_input
    io_lca_dependency = optional_completed_dependency_manifest(
        manifest=manifest,
        base_function_source="uncertainty_io_lca",
        parent_scope_name="ASR",
    )
    if io_lca_dependency is not None:
        lca_manifest, lca_reuse_status = io_lca_dependency
        lca_input = replace(
            lca_input,
            manifest=lca_manifest,
            phase_reuse_status=lca_reuse_status,
            phase_output_root=manifest_output_root(lca_manifest),
        )
    return replace(plan, acc_manifest=acc_manifest, lca_input=lca_input)


def completed_acc_component_session(*, acc_session: Any, acc_manifest: UncertaintyManifest):
    """Return an aCC session rooted in completed ASR dependency manifests."""
    return replace(
        acc_session,
        plan=plan_from_reused_acc_dependencies(
            plan=acc_session.plan,
            manifest=acc_manifest,
        ),
        asocc_session=None,
        dynamic_cc_session=None,
        run_id=acc_manifest.run_id,
        output_state=None,
    )

"""Rebuild aCC subfigure inputs from completed dependency manifests."""

from dataclasses import replace

from pyaesa.acc.uncertainty.runtime.models import (
    ACCAsoccInput,
    ACCDynamicCCInput,
    ACCUncertaintyPlan,
)
from pyaesa.shared.uncertainty_assessment.run_state.completed_dependencies import (
    required_completed_dependency_manifest,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def plan_from_reused_acc_dependencies(
    *,
    plan: ACCUncertaintyPlan,
    manifest: UncertaintyManifest,
) -> ACCUncertaintyPlan:
    """Return an aCC subfigure plan rooted in completed manifest dependencies."""
    asocc_input = plan.asocc_input
    if plan.asocc_input.manifest is not None:
        asocc_manifest, asocc_reuse = required_completed_dependency_manifest(
            manifest=manifest,
            base_function_source="uncertainty_asocc",
            parent_scope_name="aCC",
        )
        asocc_input = ACCAsoccInput(
            identity=None,
            deterministic_values=None,
            manifest=asocc_manifest,
            deterministic_manifest_path=None,
            reuse_status=asocc_reuse,
        )
    dynamic_cc_input = plan.dynamic_cc_input
    if dynamic_cc_input is not None and dynamic_cc_input.manifest is not None:
        dynamic_manifest, dynamic_reuse = required_completed_dependency_manifest(
            manifest=manifest,
            base_function_source="uncertainty_ar6_cc",
            parent_scope_name="aCC",
        )
        dynamic_cc_input = ACCDynamicCCInput(
            identity=None,
            deterministic_values=None,
            manifest=dynamic_manifest,
            deterministic_manifest_path=None,
            reuse_status=dynamic_reuse,
        )
    return replace(
        plan,
        asocc_input=asocc_input,
        dynamic_cc_input=dynamic_cc_input,
    )

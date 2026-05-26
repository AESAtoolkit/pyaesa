"""aCC uncertainty manifest and public output payload assembly."""

from typing import Any

from pyaesa.ar6_cc.uncertainty.request.normalization import AR6_DYNAMIC_CC_SOURCE
from pyaesa.acc.uncertainty.runtime.models import ACCUncertaintyPlan, ACCUncertaintyRunPaths
from pyaesa.shared.runtime.manifest_contract import manifest_json_value
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    build_compatibility_key,
    build_manifest,
)
from pyaesa.shared.uncertainty_assessment.run_state.compatibility import (
    strip_reporting_only_fields,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.composite import run_role_payload
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest
from pyaesa.shared.uncertainty_assessment.sobol.plan import SobolPlan, sobol_plan_payload
from pyaesa.shared.uncertainty_assessment.run_state.manifest_payloads import (
    mc_parameters_payload,
    optional_sobol_artifact_paths,
    public_run_artifact_paths,
    public_run_output_payload,
)

ACC_ARTIFACT_CONTRACT = "acc_runs_preserve_upstream_layout"


def build_acc_manifest_context(
    *,
    base_args: dict[str, Any],
    runtime: UncertaintyRuntimeRequest,
    plan: ACCUncertaintyPlan,
    sobol_status: dict[str, Any],
    component_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build compatibility and manifest payloads for one aCC uncertainty run."""
    prerequisites = tuple(
        item
        for item in (
            _upstream_payload(
                name="uncertainty_asocc",
                manifest=plan.asocc_input.manifest,
                reuse_status=plan.asocc_input.reuse_status,
            ),
            _asocc_prerequisite_payload(plan=plan),
            _upstream_payload(
                name="uncertainty_ar6_cc",
                manifest=None if plan.dynamic_cc_input is None else plan.dynamic_cc_input.manifest,
                reuse_status=(
                    None if plan.dynamic_cc_input is None else plan.dynamic_cc_input.reuse_status
                ),
            ),
            _dynamic_cc_prerequisite_payload(plan=plan),
        )
        if item is not None
    )
    compatibility_prerequisites = tuple(
        item
        for item in (
            _upstream_compatibility_payload(
                name="uncertainty_asocc",
                manifest=plan.asocc_input.manifest,
            ),
            _asocc_prerequisite_payload(plan=plan),
            _upstream_compatibility_payload(
                name="uncertainty_ar6_cc",
                manifest=None if plan.dynamic_cc_input is None else plan.dynamic_cc_input.manifest,
            ),
            _dynamic_cc_prerequisite_payload(plan=plan),
        )
        if item is not None
    )
    compatibility_payload = {
        "family": "acc",
        "artifact_contract": ACC_ARTIFACT_CONTRACT,
        "run_role": run_role_payload(component_inventory=component_inventory),
        "output_format": runtime.output_format,
        "arguments": base_args,
        "active_sources": list(plan.active_sources),
        "deterministic_prerequisites": strip_reporting_only_fields(
            list(compatibility_prerequisites)
        ),
    }
    return {
        "mc_parameters": mc_parameters_payload(runtime=runtime),
        "source_parameters": {
            "upstream_sources": list(plan.active_sources),
            "dynamic_cc_category_uncertainty": plan.dynamic_category_uncertainty_active,
            "dynamic_cc_sampling_method": _dynamic_cc_sampling_method(plan=plan),
        },
        "arguments": manifest_json_value(base_args),
        "deterministic_prerequisites": prerequisites,
        "sobol": sobol_status,
        "component_inventory": component_inventory,
        "compatibility_key": build_compatibility_key(compatibility_payload),
        "compatibility_context": {
            "active_sources": list(plan.active_sources),
            "artifact_contract": ACC_ARTIFACT_CONTRACT,
            "run_role": run_role_payload(component_inventory=component_inventory),
        },
    }


def initial_acc_sobol_status(
    *,
    sobol_plan: SobolPlan,
    active_sources: tuple[str, ...],
) -> dict[str, Any]:
    """Return the initial Sobol manifest status before optional evaluation."""
    if not sobol_plan.enabled:
        return {"ran": False, "reason": "not_requested"}
    return {
        "ran": False,
        "reason": "pending",
        "active_sources": list(active_sources),
        "parameters": sobol_plan_payload(plan=sobol_plan),
    }


def acc_outputs_payload(*, paths: ACCUncertaintyRunPaths, output_format: str) -> dict[str, Any]:
    """Return persisted output paths for the completed aCC run manifest."""
    return {
        **public_run_artifact_paths(
            paths=paths,
            run_key="acc_runs",
            output_format=output_format,
        ),
        **optional_sobol_artifact_paths(paths=paths),
    }


def acc_public_output_payload(
    *,
    paths: ACCUncertaintyRunPaths,
    output_format: str,
    run_layout: str,
) -> dict[str, Any]:
    """Return public table column metadata for the completed aCC run manifest."""
    return public_run_output_payload(
        paths=paths,
        output_format=output_format,
        run_key="acc_runs",
        metric="acc",
        layout=run_layout,
    )


def build_completed_acc_manifest(
    *,
    paths: ACCUncertaintyRunPaths,
    runtime: UncertaintyRuntimeRequest,
    plan: ACCUncertaintyPlan,
    context: dict[str, Any],
    run_id: str,
    completed_runs: int,
    convergence: dict[str, Any] | None,
    sobol_status: dict[str, Any],
    public_output: dict[str, Any],
) -> UncertaintyManifest:
    """Build the completed aCC uncertainty run manifest."""
    return build_manifest(
        family="acc",
        mode=runtime.mode,
        output_format=runtime.output_format,
        active_sources=plan.active_sources,
        completed_runs=completed_runs,
        status="complete",
        run_id=run_id,
        requested_runs=runtime.n_runs,
        mc_parameters=context["mc_parameters"],
        source_parameters=context["source_parameters"],
        arguments=context["arguments"],
        deterministic_prerequisites=context["deterministic_prerequisites"],
        artifacts={
            **acc_outputs_payload(paths=paths, output_format=runtime.output_format),
            "public_output": public_output,
        },
        convergence=convergence,
        sobol=sobol_status,
        component_inventory=context["component_inventory"],
        compatibility_key=context["compatibility_key"],
        compatibility_context=context["compatibility_context"],
    )


def _upstream_payload(
    *,
    name: str,
    manifest: UncertaintyManifest | None,
    reuse_status: str | None,
) -> dict[str, Any] | None:
    if manifest is None:
        return None
    payload = {
        "base_function_source": name,
        "run_id": manifest.run_id,
        "completed_runs": manifest.completed_runs,
        "scope_manifest": manifest.artifacts.get("scope_manifest"),
        "reuse_status": reuse_status,
    }
    if manifest.component_inventory is not None:
        payload["component_inventory"] = dict(manifest.component_inventory)
    return payload


def _upstream_compatibility_payload(
    *,
    name: str,
    manifest: UncertaintyManifest | None,
) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {
        "base_function_source": name,
        "component_inventory": run_role_payload(component_inventory=manifest.component_inventory),
        "compatibility_key": manifest.compatibility_key,
    }


def _dynamic_cc_prerequisite_payload(*, plan: ACCUncertaintyPlan) -> dict[str, Any] | None:
    dynamic_cc_input = plan.dynamic_cc_input
    if dynamic_cc_input is None or dynamic_cc_input.deterministic_manifest_path is None:
        return None
    return {
        "base_function_source": "deterministic_ar6_cc",
        "scope_manifest": str(dynamic_cc_input.deterministic_manifest_path),
        "reuse_status": dynamic_cc_input.reuse_status,
        "process_ar6": dynamic_cc_input.process_ar6,
    }


def _dynamic_cc_sampling_method(*, plan: ACCUncertaintyPlan) -> str | None:
    if plan.dynamic_cc_input is None or plan.dynamic_cc_input.manifest is None:
        return None
    source_parameters = dict(plan.dynamic_cc_input.manifest.source_parameters)
    return str(dict(source_parameters[AR6_DYNAMIC_CC_SOURCE])["sampling_method"])


def _asocc_prerequisite_payload(*, plan: ACCUncertaintyPlan) -> dict[str, str] | None:
    if plan.asocc_input.deterministic_manifest_path is None:
        return None
    return {
        "base_function_source": "deterministic_asocc",
        "scope_manifest": str(plan.asocc_input.deterministic_manifest_path),
        "reuse_status": plan.asocc_input.reuse_status,
    }

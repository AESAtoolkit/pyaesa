"""ASR uncertainty manifest payload builders."""

from typing import Any, cast

from pyaesa.asr.uncertainty.runtime.models import ASRUncertaintyPlan, ASRUncertaintyRunPaths
from pyaesa.acc.uncertainty.evaluation.summary import (
    acc_dynamic_category_uncertainty_active_from_manifest,
)
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
from pyaesa.shared.uncertainty_assessment.sobol.plan import SobolPlan, sobol_plan_payload
from pyaesa.shared.uncertainty_assessment.io.tables import (
    public_run_artifact_contract,
    uncertainty_table_columns,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest_payloads import (
    mc_parameters_payload,
    optional_sobol_artifact_paths,
    public_run_artifact_paths,
    public_run_output_payload,
)

ASR_ARTIFACT_CONTRACT = "asr_uncertainty_runs_with_frequency_of_no_transgression"


def build_asr_manifest_context(
    *,
    base_args: dict[str, Any],
    runtime,
    plan: ASRUncertaintyPlan,
    sobol_status: dict[str, Any],
    component_inventory: dict[str, Any] | None = None,
    acc_reuse_status: str = "computed",
) -> dict[str, Any]:
    """Build manifest context for one ASR uncertainty run."""
    acc_source_parameters = cast(dict[str, Any], plan.acc_manifest.source_parameters or {})
    source_parameters = {
        "active_sources": list(plan.active_sources),
        "dynamic_cc_category_uncertainty": acc_dynamic_category_uncertainty_active_from_manifest(
            manifest=plan.acc_manifest
        ),
        "dynamic_cc_sampling_method": acc_source_parameters.get("dynamic_cc_sampling_method"),
    }
    prerequisites = (
        _component_prerequisite_payload(
            base_function_source="uncertainty_acc",
            manifest=plan.acc_manifest,
            reuse_status=acc_reuse_status,
        ),
    )
    compatibility_prerequisites = (_upstream_compatibility_payload(plan=plan),)
    if plan.lca_input.manifest is not None:
        prerequisites = (
            *prerequisites,
            _component_prerequisite_payload(
                base_function_source="uncertainty_io_lca",
                manifest=plan.lca_input.manifest,
                reuse_status=plan.lca_input.phase_reuse_status,
            ),
        )
        compatibility_prerequisites = (
            *compatibility_prerequisites,
            _lca_compatibility_payload(manifest=plan.lca_input.manifest),
        )
    compatibility_payload = {
        "family": "asr",
        "artifact_contract": ASR_ARTIFACT_CONTRACT,
        "run_role": run_role_payload(component_inventory=component_inventory),
        "output_format": runtime.output_format,
        "arguments": base_args,
        "active_sources": list(plan.active_sources),
        "source_parameters": source_parameters,
        "has_cumulative_outputs": plan.has_cumulative_outputs,
        "deterministic_prerequisites": list(compatibility_prerequisites),
        "external_inputs": strip_reporting_only_fields(list(plan.lca_input.external_inputs)),
        "sobol": sobol_status,
    }
    return {
        "mc_parameters": mc_parameters_payload(runtime=runtime),
        "source_parameters": source_parameters,
        "arguments": manifest_json_value(base_args),
        "deterministic_prerequisites": prerequisites,
        "external_inputs": plan.lca_input.external_inputs,
        "sobol": sobol_status,
        "component_inventory": component_inventory,
        "compatibility_key": build_compatibility_key(compatibility_payload),
        "compatibility_context": {
            "active_sources": list(plan.active_sources),
            "artifact_contract": ASR_ARTIFACT_CONTRACT,
            "run_layout": plan.asr_run_layout,
            "has_cumulative_outputs": plan.has_cumulative_outputs,
            "run_role": run_role_payload(component_inventory=component_inventory),
        },
    }


def _component_prerequisite_payload(
    base_function_source: str,
    *,
    manifest: UncertaintyManifest,
    reuse_status: str,
) -> dict[str, Any]:
    artifacts = cast(dict[str, Any], manifest.artifacts)
    payload = {
        "base_function_source": base_function_source,
        "run_id": manifest.run_id,
        "completed_runs": manifest.completed_runs,
        "scope_manifest": artifacts["scope_manifest"],
        "reuse_status": reuse_status,
    }
    if manifest.component_inventory is not None:
        payload["component_inventory"] = dict(manifest.component_inventory)
    return payload


def _upstream_compatibility_payload(*, plan: ASRUncertaintyPlan) -> dict[str, Any]:
    return {
        "base_function_source": "uncertainty_acc",
        "component_inventory": run_role_payload(
            component_inventory=plan.acc_manifest.component_inventory
        ),
        "compatibility_key": plan.acc_manifest.compatibility_key,
    }


def _lca_compatibility_payload(*, manifest: UncertaintyManifest) -> dict[str, Any]:
    return {
        "base_function_source": "uncertainty_io_lca",
        "component_inventory": run_role_payload(component_inventory=manifest.component_inventory),
        "compatibility_key": manifest.compatibility_key,
    }


def initial_asr_sobol_status(
    *,
    sobol_plan: SobolPlan,
    active_sources: tuple[str, ...],
) -> dict[str, Any]:
    """Return the initial Sobol status for one manifest."""
    if not sobol_plan.enabled:
        return {"ran": False, "reason": "not_requested"}
    if len(active_sources) < 2:
        return {
            "ran": False,
            "reason": "requires_at_least_two_active_sources",
            "active_source_count": len(active_sources),
            "parameters": sobol_plan_payload(plan=sobol_plan),
        }
    return {"ran": False, "reason": "pending", "parameters": sobol_plan_payload(plan=sobol_plan)}


def asr_outputs_payload(
    *,
    paths: ASRUncertaintyRunPaths,
    include_cumulative: bool,
    output_format: str,
) -> dict[str, Any]:
    """Return complete ASR output artifact paths."""
    payload = public_run_artifact_paths(
        paths=paths,
        run_key="asr_runs",
        output_format=output_format,
    )
    if include_cumulative:
        cumulative = public_run_artifact_contract(
            path=paths.cumulative_runs,
            output_format=output_format,
        )
        payload.update(
            {
                "cumulative_row_identity": str(paths.cumulative_row_identity),
                "cumulative_asr_runs": str(paths.cumulative_runs),
                "cumulative_asr_runs_artifact_kind": cumulative["artifact_kind"],
                "cumulative_asr_runs_interval_index": cumulative["interval_index_path"],
                "cumulative_asr_runs_interval_index_kind": cumulative["interval_index_kind"],
                "cumulative_summary_stats_runs": str(paths.cumulative_summary_stats_runs),
            }
        )
    payload.update(optional_sobol_artifact_paths(paths=paths))
    return payload


def asr_public_output_payload(
    *,
    paths: ASRUncertaintyRunPaths,
    output_format: str,
    run_layout: str,
    include_cumulative: bool,
) -> dict[str, Any]:
    """Return public ASR output column metadata."""
    payload = public_run_output_payload(
        paths=paths,
        output_format=output_format,
        run_key="asr_runs",
        metric="asr",
        layout=run_layout,
        summary_metrics=["asr", "frequency_of_no_transgression"],
    )
    if include_cumulative:
        cumulative_run_columns = uncertainty_table_columns(
            path=paths.cumulative_runs,
            output_format=output_format,
        )
        cumulative_summary_columns = uncertainty_table_columns(
            path=paths.cumulative_summary_stats_runs,
            output_format=output_format,
        )
        cumulative_identity_columns = uncertainty_table_columns(
            path=paths.cumulative_row_identity,
            output_format=output_format,
        )
        payload.update(
            {
                "cumulative_identity_columns": cumulative_identity_columns,
                "cumulative_asr_runs": {
                    **public_run_artifact_contract(
                        path=paths.cumulative_runs,
                        output_format=output_format,
                    ),
                    "layout": "compact_run_matrix",
                    "metric": "cumulative_asr",
                    "columns_preview": cumulative_run_columns[
                        : min(5, len(cumulative_run_columns))
                    ],
                },
                "cumulative_summary_metrics": [
                    "cumulative_asr",
                    "cumulative_frequency_of_no_transgression",
                ],
                "cumulative_summary_columns": cumulative_summary_columns,
            }
        )
    return payload


def build_completed_asr_manifest(
    *,
    paths: ASRUncertaintyRunPaths,
    runtime,
    plan: ASRUncertaintyPlan,
    context: dict[str, Any],
    run_id: str,
    completed_runs: int,
    convergence: dict[str, Any] | None,
    sobol_status: dict[str, Any],
) -> UncertaintyManifest:
    """Build the completed ASR uncertainty run manifest."""
    artifacts: dict[str, Any] = asr_outputs_payload(
        paths=paths,
        include_cumulative=plan.has_cumulative_outputs,
        output_format=runtime.output_format,
    )
    artifacts["public_output"] = asr_public_output_payload(
        paths=paths,
        output_format=runtime.output_format,
        run_layout=plan.asr_run_layout,
        include_cumulative=plan.has_cumulative_outputs,
    )
    return build_manifest(
        family="asr",
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
        external_inputs=context["external_inputs"],
        artifacts=artifacts,
        convergence=convergence,
        sobol=sobol_status,
        component_inventory=context["component_inventory"],
        compatibility_key=context["compatibility_key"],
        compatibility_context=context["compatibility_context"],
    )

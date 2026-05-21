"""AR6 CC uncertainty manifest and public output payload assembly."""

from typing import Any

from pyaesa.ar6_cc.uncertainty.runtime.models import (
    AR6CCDeterministicScope,
    AR6CCUncertaintyPlan,
    AR6CCUncertaintyRequest,
    AR6CCUncertaintyRunPaths,
)
from pyaesa.ar6_cc.uncertainty.request.normalization import AR6_DYNAMIC_CC_SOURCE
from pyaesa.shared.runtime.manifest_contract import manifest_json_value
from pyaesa.shared.uncertainty_assessment.io.tables import public_run_artifact_contract
from pyaesa.shared.uncertainty_assessment.monte_carlo.composite import run_role_payload
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest
from pyaesa.shared.uncertainty_assessment.run_state.manifest_payloads import (
    mc_parameters_payload,
    public_run_artifact_paths,
    public_run_output_payload,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    build_compatibility_key,
    build_manifest,
)
from pyaesa.shared.uncertainty_assessment.run_state.compatibility import (
    strip_reporting_only_fields,
)

AR6_CC_ARTIFACT_CONTRACT = "ar6_cc_sparse_selected_trajectory_flow_runs_with_post_study_v1"


def build_ar6_cc_manifest_context(
    *,
    request: AR6CCUncertaintyRequest,
    runtime: UncertaintyRuntimeRequest,
    prerequisite: AR6CCDeterministicScope,
    plan: AR6CCUncertaintyPlan,
    component_inventory: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build compatibility and manifest payloads for one AR6 CC uncertainty run."""
    source_parameters = {AR6_DYNAMIC_CC_SOURCE: dict(request.source_parameters)}
    prerequisites = (_prerequisite_payload(prerequisite=prerequisite),)
    compatibility_payload = {
        "family": "ar6_cc",
        "artifact_contract": AR6_CC_ARTIFACT_CONTRACT,
        "run_role": run_role_payload(component_inventory=component_inventory),
        "output_format": runtime.output_format,
        "arguments": dict(request.base_ar6_cc_args),
        "active_sources": [AR6_DYNAMIC_CC_SOURCE],
        "source_parameters": source_parameters,
        "deterministic_prerequisites": strip_reporting_only_fields(list(prerequisites)),
    }
    return {
        "mc_parameters": mc_parameters_payload(runtime=runtime),
        "source_parameters": source_parameters,
        "arguments": manifest_json_value({"base_ar6_cc_args": dict(request.base_ar6_cc_args)}),
        "deterministic_prerequisites": prerequisites,
        "lineage": _lineage_payload(plan=plan),
        "component_inventory": component_inventory,
        "compatibility_key": build_compatibility_key(compatibility_payload),
        "compatibility_context": {
            "active_sources": [AR6_DYNAMIC_CC_SOURCE],
            "artifact_contract": AR6_CC_ARTIFACT_CONTRACT,
            "run_role": run_role_payload(component_inventory=component_inventory),
        },
    }


def build_completed_ar6_cc_manifest(
    *,
    paths: AR6CCUncertaintyRunPaths,
    runtime: UncertaintyRuntimeRequest,
    context: dict[str, Any],
    run_id: str,
    completed_runs: int,
    convergence: dict[str, Any] | None,
    include_post_study: bool,
    public_output: dict[str, Any],
) -> UncertaintyManifest:
    """Build the completed AR6 CC uncertainty run manifest."""
    return build_manifest(
        family="ar6_cc",
        mode=runtime.mode,
        output_format=runtime.output_format,
        active_sources=(AR6_DYNAMIC_CC_SOURCE,),
        completed_runs=completed_runs,
        status="complete",
        run_id=run_id,
        requested_runs=runtime.n_runs,
        mc_parameters=context["mc_parameters"],
        source_parameters=context["source_parameters"],
        arguments=context["arguments"],
        deterministic_prerequisites=context["deterministic_prerequisites"],
        artifacts={
            **ar6_cc_outputs_payload(
                paths=paths,
                include_post_study=include_post_study,
                output_format=runtime.output_format,
            ),
            "public_output": public_output,
        },
        lineage=context["lineage"],
        component_inventory=context["component_inventory"],
        convergence=convergence,
        compatibility_key=context["compatibility_key"],
        compatibility_context=context["compatibility_context"],
    )


def ar6_cc_outputs_payload(
    *,
    paths: AR6CCUncertaintyRunPaths,
    include_post_study: bool,
    output_format: str,
) -> dict[str, Any]:
    """Return persisted output paths for the completed AR6 CC manifest."""
    budget = public_run_artifact_contract(
        path=paths.budget_runs,
        output_format=output_format,
    )
    payload: dict[str, Any] = {
        **public_run_artifact_paths(
            paths=paths,
            run_key="cc_runs",
            output_format=output_format,
        ),
        "study_and_post_study_period_budget_row_identity": str(paths.budget_row_identity),
        "study_and_post_study_period_budget_runs": str(paths.budget_runs),
        "study_and_post_study_period_budget_runs_artifact_kind": budget["artifact_kind"],
        "study_and_post_study_period_budget_runs_interval_index": budget["interval_index_path"],
        "study_and_post_study_period_budget_runs_interval_index_kind": budget[
            "interval_index_kind"
        ],
        "study_and_post_study_period_budget_summary_stats": str(paths.budget_summary_stats_runs),
    }
    if include_post_study:
        post_study = public_run_artifact_contract(
            path=paths.post_study_public_runs,
            output_format=output_format,
        )
        payload.update(
            {
                "post_study_period_public_row_identity": str(paths.post_study_public_row_identity),
                "post_study_period_cc_runs": str(paths.post_study_public_runs),
                "post_study_period_cc_runs_artifact_kind": post_study["artifact_kind"],
                "post_study_period_cc_runs_interval_index": post_study["interval_index_path"],
                "post_study_period_cc_runs_interval_index_kind": post_study["interval_index_kind"],
                "post_study_period_summary_stats_runs": str(paths.post_study_summary_stats_runs),
            }
        )
    return payload


def ar6_cc_public_output_payload(
    *,
    paths: AR6CCUncertaintyRunPaths,
    output_format: str,
) -> dict[str, Any]:
    """Return public table column metadata for the completed AR6 CC manifest."""
    return public_run_output_payload(
        paths=paths,
        output_format=output_format,
        run_key="cc_runs",
        metric="cc",
        layout="sparse_selected_rows",
    )


def _prerequisite_payload(*, prerequisite: AR6CCDeterministicScope) -> dict[str, Any]:
    return {
        "base_function_source": "deterministic_ar6_cc",
        "scope_key": prerequisite.scope_key,
        "output_format": prerequisite.output_format,
        "metadata_path": str(prerequisite.metadata_path),
        "reuse_status": prerequisite.reuse_status,
        "output_file": str(prerequisite.output_file),
        "post_study_output_file": (
            None
            if prerequisite.post_study_output_file is None
            else str(prerequisite.post_study_output_file)
        ),
        "emission_type": prerequisite.emission_type,
        "include_afolu": prerequisite.include_afolu,
        "variable": prerequisite.variable,
        "emissions_mode": prerequisite.emissions_mode,
        "categories": list(prerequisite.categories),
        "ssp_scenarios": list(prerequisite.ssp_scenarios),
        "subset_version": prerequisite.subset_version,
        "pathway_counts": list(prerequisite.pathway_counts),
        "missing_pathway_combinations": list(prerequisite.missing_pathway_combinations),
        "process_ar6": prerequisite.process_ar6,
    }


def _lineage_payload(*, plan: AR6CCUncertaintyPlan) -> dict[str, Any]:
    return {
        "source_inventory": {
            "candidate_trajectories": len(
                plan.source_method_rows[
                    ["cc_category", "ssp_scenario", "cc_model", "cc_scenario"]
                ].drop_duplicates()
            ),
            "public_rows": len(plan.identity),
            "category_pools": [
                {
                    "ssp_scenario": pool.ssp_scenario,
                    "candidate_categories": [
                        plan.groups[index].category for index in pool.group_indices
                    ],
                }
                for pool in plan.category_pools
            ],
            "scope_availability_messages": list(plan.availability_messages),
        }
    }

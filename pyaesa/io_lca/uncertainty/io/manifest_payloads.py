"""IO-LCA uncertainty manifest and public output payload assembly."""

from typing import Any

from pyaesa.io_lca.uncertainty.runtime.models import (
    IOLCADeterministicScope,
    IOLCAUncertaintyRequest,
    IOLCAUncertaintyRunPaths,
)
from pyaesa.shared.lcia.uncertainty_source import LCIA_SOURCE
from pyaesa.shared.runtime.manifest_contract import manifest_json_value
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

IO_LCA_ARTIFACT_CONTRACT = "io_lca_lca_runs_with_public_identity_v1"


def build_io_lca_manifest_context(
    *,
    request: IOLCAUncertaintyRequest,
    runtime: UncertaintyRuntimeRequest,
    prerequisite: IOLCADeterministicScope,
    component_inventory: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build compatibility and manifest payloads for one IO-LCA uncertainty run."""
    source_parameters = {LCIA_SOURCE: dict(request.source_parameters)}
    prerequisites = (
        {
            "base_function_source": "deterministic_io_lca",
            "source": prerequisite.source,
            "scope_key": prerequisite.scope_key,
            "reuse_status": prerequisite.reuse_status,
            "completed_years_by_method": {
                key: list(value) for key, value in prerequisite.completed_years_by_method.items()
            },
            "output_format": prerequisite.output_format,
            "metadata_path": str(prerequisite.metadata_path),
            "deterministic_paths": list(prerequisite.deterministic_paths),
        },
    )
    compatibility_payload = {
        "family": "io_lca",
        "artifact_contract": IO_LCA_ARTIFACT_CONTRACT,
        "run_role": run_role_payload(component_inventory=component_inventory),
        "output_format": runtime.output_format,
        "arguments": dict(request.base_io_lca_args),
        "active_sources": [LCIA_SOURCE],
        "source_parameters": source_parameters,
        "deterministic_prerequisites": strip_reporting_only_fields(list(prerequisites)),
    }
    return {
        "mc_parameters": mc_parameters_payload(runtime=runtime),
        "source_parameters": source_parameters,
        "arguments": manifest_json_value({"base_io_lca_args": dict(request.base_io_lca_args)}),
        "deterministic_prerequisites": prerequisites,
        "component_inventory": component_inventory,
        "compatibility_key": build_compatibility_key(compatibility_payload),
        "compatibility_context": {
            "active_sources": [LCIA_SOURCE],
            "run_role": run_role_payload(component_inventory=component_inventory),
        },
    }


def build_completed_io_lca_manifest(
    *,
    paths: IOLCAUncertaintyRunPaths,
    runtime: UncertaintyRuntimeRequest,
    context: dict[str, Any],
    run_id: str,
    completed_runs: int,
    convergence: dict[str, Any] | None,
    public_output: dict[str, Any],
) -> UncertaintyManifest:
    """Build the completed IO-LCA uncertainty run manifest."""
    return build_manifest(
        family="io_lca",
        mode=runtime.mode,
        output_format=runtime.output_format,
        active_sources=(LCIA_SOURCE,),
        completed_runs=completed_runs,
        status="complete",
        run_id=run_id,
        requested_runs=runtime.n_runs,
        mc_parameters=context["mc_parameters"],
        source_parameters=context["source_parameters"],
        arguments=context["arguments"],
        deterministic_prerequisites=context["deterministic_prerequisites"],
        component_inventory=context["component_inventory"],
        artifacts={
            **io_lca_outputs_payload(paths=paths, output_format=runtime.output_format),
            "public_output": public_output,
        },
        convergence=convergence,
        compatibility_key=context["compatibility_key"],
        compatibility_context=context["compatibility_context"],
    )


def io_lca_outputs_payload(
    *,
    paths: IOLCAUncertaintyRunPaths,
    output_format: str,
) -> dict[str, Any]:
    """Return persisted output paths for the completed IO-LCA run manifest."""
    return public_run_artifact_paths(
        paths=paths,
        run_key="lca_runs",
        output_format=output_format,
    )


def io_lca_public_output_payload(
    *,
    paths: IOLCAUncertaintyRunPaths,
    output_format: str,
) -> dict[str, Any]:
    """Return public table column metadata for the completed IO-LCA manifest."""
    return public_run_output_payload(
        paths=paths,
        output_format=output_format,
        run_key="lca_runs",
        metric="lca",
        layout="compact_run_matrix",
    )

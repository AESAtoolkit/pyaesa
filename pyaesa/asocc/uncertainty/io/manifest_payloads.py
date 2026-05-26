"""Manifest payload construction for aSoCC Monte Carlo runs."""

from typing import Any

from pyaesa.asocc.io.metadata import _load_run_metadata
from pyaesa.asocc.uncertainty.inputs.external_rows import ExternalAsoccRowsPlan
from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.asocc.uncertainty.schema.public_rows import ASOCC_UNCERTAINTY_CSV_DTYPES
from pyaesa.asocc.uncertainty.sources.names import INTER_METHOD_SOURCE
from pyaesa.asocc.orchestration.common_formatting import format_year_ranges
from pyaesa.external_inputs.asocc.monte_carlo.files import external_monte_carlo_manifest_payload
from pyaesa.shared.runtime.io.file_identity import file_identity_payload
from pyaesa.shared.runtime.manifest_contract import manifest_json_value
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    build_compatibility_key,
    build_manifest,
)
from pyaesa.shared.uncertainty_assessment.run_state.compatibility import (
    strip_reporting_only_fields,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.composite import run_role_payload
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan
from pyaesa.shared.uncertainty_assessment.io.tables import (
    read_uncertainty_table,
    uncertainty_table_columns,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest_payloads import (
    mc_parameters_payload,
    optional_sobol_artifact_paths,
    public_run_artifact_paths,
    public_run_output_payload,
)


def manifest_context(
    *,
    base_asocc_args: dict[str, Any],
    loaded,
    runtime,
    sources: SourceActivationPlan,
    external_plan: ExternalAsoccRowsPlan,
    inter_method_plan=None,
    inter_mrio_plan=None,
    component_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build manifest context and compatibility key for one run."""
    source_parameters = {source.name: source.parameters for source in sources.sources}
    if inter_method_plan is not None:
        inter_method_parameters = dict(source_parameters.get(INTER_METHOD_SOURCE, {}))
        inter_method_parameters["probability_tree_key"] = build_compatibility_key(
            {
                "candidate_labels": list(inter_method_plan.candidate_labels),
                "probabilities": [
                    float(value) for value in inter_method_plan.selection_probabilities
                ],
            }
        )
        inter_method_parameters["candidate_count"] = len(inter_method_plan.candidate_labels)
        source_parameters[INTER_METHOD_SOURCE] = inter_method_parameters
    summary_records = _inter_mrio_summary_records(inter_mrio_plan=inter_mrio_plan)
    lineage = {"summary_records": summary_records} if summary_records else None
    prerequisites = tuple(
        {
            "base_function_source": "deterministic_asocc",
            "source": str(loaded.base_asocc_args["source"]),
            "output_source_label": str(loaded.path_scope.source_label),
            "proj_base": str(loaded.path_scope.proj_base),
            "scope_key": str(scope.scope_key),
            "scope_manifest": str(loaded.deterministic_manifest_path),
            "reuse_status": str(loaded.reuse_status),
            "completed_years": list(scope.completed_years),
            "output_format": str(scope.output_format),
            "summary_records": _deterministic_summary_records(
                path=loaded.deterministic_manifest_path
            ),
        }
        for scope in loaded.persisted_scopes
    )
    external_inputs = external_input_payload(plan=external_plan)
    compatibility_payload = {
        "family": "asocc",
        "artifact_contract": "asocc_runs_with_public_identity_v1",
        "run_role": run_role_payload(component_inventory=component_inventory),
        "output_format": runtime.output_format,
        "arguments": dict(loaded.base_asocc_args),
        "active_sources": list(sources.names),
        "source_parameters": source_parameters,
        "deterministic_prerequisites": strip_reporting_only_fields(list(prerequisites)),
        "external_inputs": strip_reporting_only_fields(list(external_inputs)),
    }
    compatibility_key = build_compatibility_key(compatibility_payload)
    return {
        "mc_parameters": mc_parameters_payload(runtime=runtime),
        "source_parameters": source_parameters,
        "arguments": manifest_json_value(dict(base_asocc_args)),
        "deterministic_prerequisites": prerequisites,
        "external_inputs": external_inputs,
        "artifacts": {},
        "lineage": lineage,
        "component_inventory": component_inventory,
        "compatibility_key": compatibility_key,
        "compatibility_context": {
            "active_sources": list(sources.names),
            "external_input_count": len(external_inputs),
            "run_role": run_role_payload(component_inventory=component_inventory),
        },
    }


def external_input_payload(*, plan: ExternalAsoccRowsPlan) -> tuple[dict[str, Any], ...]:
    """Return manifest rows for external aSoCC inputs."""
    rows: list[dict[str, Any]] = []
    for source in plan.deterministic_sources:
        rows.append(
            {
                "storage_mode": "deterministic",
                "selection": source.selection.asocc_method_label,
                "files": [
                    {
                        **file_identity_payload(path=file_selection.path),
                        "lcia_method": file_selection.lcia_method,
                        "requested_years": list(file_selection.requested_years),
                        ASOCC_SSP_SCENARIO_COLUMN: file_selection.ssp_scenario,
                    }
                    for file_selection in source.file_selections
                ],
            }
        )
    for source in plan.monte_carlo_sources:
        rows.append(external_monte_carlo_manifest_payload(source=source))
    return tuple(rows)


def _inter_mrio_summary_records(*, inter_mrio_plan) -> list[dict[str, str]]:
    """Return reporting records for inter-MRIO route skips."""
    if inter_mrio_plan is None:
        return []
    route_report = getattr(inter_mrio_plan, "route_report", None)
    skipped_years = tuple(int(year) for year in getattr(route_report, "skipped_years", ()) or ())
    if not skipped_years:
        return []
    pairs = tuple(str(value) for value in getattr(route_report, "skipped_route_pairs", ()) or ())
    scopes = tuple(str(value) for value in getattr(route_report, "skipped_scopes", ()) or ())
    detail_parts = [f"years {format_year_ranges(list(skipped_years))}"]
    if pairs:
        detail_parts.append("route pairs " + ", ".join(pairs))
    if scopes:
        detail_parts.append("method year scopes " + "; ".join(scopes))
    return [
        {
            "severity": "WARNING",
            "message": (
                "Inter-MRIO uncertainty was not applied for "
                + "; ".join(detail_parts)
                + " because deterministic aSoCC time routes did not match."
            ),
        }
    ]


def _deterministic_summary_records(*, path) -> list[dict[str, str]]:
    """Return reporting records persisted by deterministic aSoCC when available."""
    payload = _load_run_metadata(path)
    records = payload.get("summary_records") or ()
    out: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        severity = str(record.get("severity", "")).strip().upper()
        message = str(record.get("message", "")).strip()
        if severity in {"INFO", "WARNING"} and message:
            out.append({"severity": severity, "message": message})
    return out


def outputs_payload(*, paths: AsoccUncertaintyRunPaths, output_format: str) -> dict[str, Any]:
    """Return canonical manifest output paths."""
    payload = public_run_artifact_paths(
        paths=paths,
        run_key="asocc_runs",
        output_format=output_format,
    )
    if paths.inter_method_tree_csv.exists():
        payload["inter_method_tree_csv"] = str(paths.inter_method_tree_csv)
        payload["inter_method_tree_figure"] = str(
            sorted(
                paths.inter_method_tree_figure_base.parent.glob(
                    f"{paths.inter_method_tree_figure_base.name}.*"
                )
            )[0]
        )
    payload.update(optional_sobol_artifact_paths(paths=paths))
    return payload


def public_output_payload(*, paths: AsoccUncertaintyRunPaths, output_format: str) -> dict[str, Any]:
    """Return compact public output description for the manifest."""
    identity = read_uncertainty_table(
        path=paths.public_row_identity,
        output_format=output_format,
        csv_dtypes=ASOCC_UNCERTAINTY_CSV_DTYPES,
    )
    matrix_columns = uncertainty_table_columns(path=paths.public_runs, output_format=output_format)
    sparse = matrix_columns[:3] == ["run_index", "public_row_id", "asocc"]
    return public_run_output_payload(
        paths=paths,
        output_format=output_format,
        run_key="asocc_runs",
        metric="asocc",
        layout="sparse_selected_rows" if sparse else "compact_run_matrix",
        identity_columns=list(identity.columns),
        public_row_count=int(len(identity)),
    )


def build_completed_asocc_manifest(
    *,
    paths: AsoccUncertaintyRunPaths,
    runtime,
    sources: SourceActivationPlan,
    run_context: dict[str, Any],
    run_id: str,
    completed_runs: int,
    convergence: dict[str, Any] | None,
    sobol_status: dict[str, Any] | None,
) -> UncertaintyManifest:
    """Build one completed aSoCC uncertainty manifest."""
    artifacts = outputs_payload(paths=paths, output_format=runtime.output_format)
    artifacts["public_output"] = public_output_payload(
        paths=paths,
        output_format=runtime.output_format,
    )
    return build_manifest(
        family="asocc",
        mode=runtime.mode,
        output_format=runtime.output_format,
        active_sources=sources.names,
        completed_runs=completed_runs,
        status="complete",
        run_id=run_id,
        requested_runs=runtime.n_runs,
        mc_parameters=run_context["mc_parameters"],
        source_parameters=run_context["source_parameters"],
        arguments=run_context["arguments"],
        deterministic_prerequisites=run_context["deterministic_prerequisites"],
        external_inputs=run_context["external_inputs"],
        artifacts=artifacts,
        lineage=run_context["lineage"],
        component_inventory=run_context["component_inventory"],
        convergence=convergence,
        sobol=sobol_status,
        compatibility_key=run_context["compatibility_key"],
        compatibility_context=run_context["compatibility_context"],
    )

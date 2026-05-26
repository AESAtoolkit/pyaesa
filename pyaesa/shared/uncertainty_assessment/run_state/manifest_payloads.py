"""Shared manifest payload helpers for uncertainty run artifacts."""

from typing import Any

from pyaesa.shared.uncertainty_assessment.io.run_artifacts import public_run_artifact_contract
from pyaesa.shared.uncertainty_assessment.io.tables import uncertainty_table_columns
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest


def mc_parameters_payload(*, runtime: UncertaintyRuntimeRequest) -> dict[str, Any]:
    """Return the canonical Monte Carlo execution parameters payload."""
    return {
        "mode": runtime.mode,
        "requested_runs": runtime.n_runs,
        "max_runs": runtime.max_runs,
        "rtol": runtime.rtol,
        "stable_runs": runtime.stable_runs,
        "convergence_statistics": list(runtime.convergence_statistics),
    }


def public_run_artifact_paths(*, paths: Any, run_key: str, output_format: str) -> dict[str, Any]:
    """Return common public run artifact paths for one uncertainty family."""
    artifact = public_run_artifact_contract(
        path=paths.public_runs,
        output_format=output_format,
    )
    return {
        "public_row_identity": str(paths.public_row_identity),
        "run_values": str(paths.public_runs),
        "run_values_artifact_kind": artifact["artifact_kind"],
        "run_values_interval_index": artifact["interval_index_path"],
        "run_values_interval_index_kind": artifact["interval_index_kind"],
        run_key: str(paths.public_runs),
        f"{run_key}_artifact_kind": artifact["artifact_kind"],
        f"{run_key}_interval_index": artifact["interval_index_path"],
        f"{run_key}_interval_index_kind": artifact["interval_index_kind"],
        "summary_stats_runs": str(paths.summary_stats_runs),
        "results_readme": str(paths.results_readme),
        "source_methods": str(paths.source_methods),
        "scope_manifest": str(paths.scope_manifest),
    }


def optional_sobol_artifact_paths(*, paths: Any) -> dict[str, str]:
    """Return Sobol artifact paths when one family wrote Sobol outputs."""
    sobol_indices = getattr(paths, "sobol_indices", None)
    if sobol_indices is None or not sobol_indices.exists():
        return {}
    return {
        "sobol_indices": str(sobol_indices),
        "sobol_source_summary": str(paths.sobol_source_summary),
        "sobol_readme": str(paths.sobol_readme),
    }


def public_run_output_payload(
    *,
    paths: Any,
    output_format: str,
    run_key: str,
    metric: str,
    layout: str,
    identity_columns: list[str] | None = None,
    public_row_count: int | None = None,
    summary_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Return common public output column metadata for one run artifact."""
    run_columns = uncertainty_table_columns(path=paths.public_runs, output_format=output_format)
    summary_columns = uncertainty_table_columns(
        path=paths.summary_stats_runs,
        output_format=output_format,
    )
    identity = (
        uncertainty_table_columns(path=paths.public_row_identity, output_format=output_format)
        if identity_columns is None
        else identity_columns
    )
    run_payload: dict[str, Any] = {
        **public_run_artifact_contract(path=paths.public_runs, output_format=output_format),
        "layout": layout,
        "metric": metric,
        "columns_preview": run_columns[: min(5, len(run_columns))],
    }
    if public_row_count is not None:
        run_payload["public_row_count"] = public_row_count
    payload: dict[str, Any] = {
        "identity_columns": identity,
        run_key: run_payload,
        "summary_columns": summary_columns,
    }
    if summary_metrics is not None:
        payload["summary_metrics"] = summary_metrics
    payload.update(optional_sobol_public_output_payload(paths=paths, output_format=output_format))
    return payload


def optional_sobol_public_output_payload(*, paths: Any, output_format: str) -> dict[str, Any]:
    """Return Sobol column metadata when one family wrote Sobol outputs."""
    sobol_indices = getattr(paths, "sobol_indices", None)
    if sobol_indices is None or not sobol_indices.exists():
        return {}
    return {
        "sobol_indices": {
            "columns": uncertainty_table_columns(
                path=sobol_indices,
                output_format=output_format,
            )
        },
        "sobol_source_summary": {
            "columns": uncertainty_table_columns(
                path=paths.sobol_source_summary,
                output_format=output_format,
            )
        },
    }

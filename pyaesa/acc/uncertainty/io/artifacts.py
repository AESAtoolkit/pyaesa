"""Persisted aCC uncertainty artifact contracts."""

from pathlib import Path
from typing import Any, cast

from pyaesa.acc.uncertainty.io.paths import build_acc_uncertainty_run_paths
from pyaesa.acc.uncertainty.runtime.models import ACCUncertaintyRunPaths
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def acc_run_paths_from_manifest(*, manifest: UncertaintyManifest) -> ACCUncertaintyRunPaths:
    """Return typed aCC uncertainty artifact paths from a run manifest."""
    artifacts = cast(dict[str, Any], manifest.artifacts)
    run_root = Path(artifacts["scope_manifest"]).parents[1]
    paths = build_acc_uncertainty_run_paths(
        monte_carlo_root=run_root.parent,
        run_id=run_root.name,
        output_format=manifest.output_format,
    )
    has_cumulative_outputs = "cumulative_acc_runs" in cast(
        dict[str, Any],
        artifacts["public_output"],
    )
    return ACCUncertaintyRunPaths(
        run_root=paths.run_root,
        public_row_identity=Path(artifacts["public_row_identity"]),
        public_runs=Path(artifacts["acc_runs"]),
        summary_stats_runs=Path(artifacts.get("summary_stats_runs", paths.summary_stats_runs)),
        cumulative_row_identity=_cumulative_artifact_path(
            artifacts=artifacts,
            key="cumulative_row_identity",
            default=paths.cumulative_row_identity,
            has_cumulative_outputs=has_cumulative_outputs,
        ),
        cumulative_runs=_cumulative_artifact_path(
            artifacts=artifacts,
            key="cumulative_acc_runs",
            default=paths.cumulative_runs,
            has_cumulative_outputs=has_cumulative_outputs,
        ),
        cumulative_summary_stats_runs=_cumulative_artifact_path(
            artifacts=artifacts,
            key="cumulative_summary_stats_runs",
            default=paths.cumulative_summary_stats_runs,
            has_cumulative_outputs=has_cumulative_outputs,
        ),
        results_readme=Path(artifacts["results_readme"]),
        source_methods=Path(artifacts["source_methods"]),
        sobol_indices=Path(artifacts.get("sobol_indices", paths.sobol_indices)),
        sobol_source_summary=Path(
            artifacts.get("sobol_source_summary", paths.sobol_source_summary)
        ),
        sobol_readme=Path(artifacts.get("sobol_readme", paths.sobol_readme)),
        scope_manifest=Path(artifacts["scope_manifest"]),
    )


def _cumulative_artifact_path(
    *,
    artifacts: dict[str, Any],
    key: str,
    default: Path,
    has_cumulative_outputs: bool,
) -> Path:
    """Return a cumulative artifact path or the canonical static path slot."""
    if has_cumulative_outputs:
        return Path(artifacts[key])
    return default


def acc_run_layout_from_manifest(*, manifest: UncertaintyManifest) -> str:
    """Return the persisted aCC run table layout from a completed manifest."""
    public = cast(dict[str, Any], manifest.artifacts["public_output"])
    return str(cast(dict[str, Any], public["acc_runs"])["layout"])

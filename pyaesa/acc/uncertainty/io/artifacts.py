"""Persisted aCC uncertainty artifact contracts."""

from pathlib import Path
from typing import Any, cast

from pyaesa.acc.uncertainty.runtime.models import ACCUncertaintyRunPaths
from pyaesa.shared.uncertainty_assessment.io.formats import suffix_for_uncertainty_output
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def acc_run_paths_from_manifest(*, manifest: UncertaintyManifest) -> ACCUncertaintyRunPaths:
    """Return typed aCC uncertainty artifact paths from a run manifest."""
    artifacts = cast(dict[str, Any], manifest.artifacts)
    run_root = Path(artifacts["scope_manifest"]).parents[1]
    suffix = suffix_for_uncertainty_output(manifest.output_format)
    summary_stats_runs = artifacts.get(
        "summary_stats_runs",
        run_root / "results" / f"summary_stats_runs{suffix}",
    )
    return ACCUncertaintyRunPaths(
        run_root=run_root,
        public_row_identity=Path(artifacts["public_row_identity"]),
        public_runs=Path(artifacts["acc_runs"]),
        summary_stats_runs=Path(str(summary_stats_runs)),
        results_readme=Path(artifacts["results_readme"]),
        source_methods=Path(artifacts["source_methods"]),
        sobol_indices=Path(artifacts.get("sobol_indices", "")),
        sobol_source_summary=Path(artifacts.get("sobol_source_summary", "")),
        sobol_readme=Path(artifacts.get("sobol_readme", "")),
        scope_manifest=Path(artifacts["scope_manifest"]),
    )


def acc_run_layout_from_manifest(*, manifest: UncertaintyManifest) -> str:
    """Return the persisted aCC run table layout from a completed manifest."""
    public = cast(dict[str, Any], manifest.artifacts["public_output"])
    return str(cast(dict[str, Any], public["acc_runs"])["layout"])

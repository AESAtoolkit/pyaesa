"""Persisted aSoCC uncertainty artifact contracts."""

from pathlib import Path
from typing import Any, cast

from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def asocc_run_paths_from_manifest(*, manifest: UncertaintyManifest) -> AsoccUncertaintyRunPaths:
    """Return typed aSoCC uncertainty artifact paths from a completed manifest."""
    artifacts = cast(dict[str, Any], manifest.artifacts)
    return AsoccUncertaintyRunPaths(
        run_root=Path(artifacts["scope_manifest"]).parents[1],
        public_row_identity=Path(artifacts["public_row_identity"]),
        public_runs=Path(artifacts["asocc_runs"]),
        summary_stats_runs=Path(artifacts["summary_stats_runs"]),
        results_readme=Path(artifacts["results_readme"]),
        source_methods=Path(artifacts["source_methods"]),
        inter_method_tree_csv=Path(artifacts.get("inter_method_tree_csv", "")),
        inter_method_tree_figure_base=Path(artifacts.get("inter_method_tree_figure_base", "")),
        sobol_indices=Path(artifacts.get("sobol_indices", "")),
        sobol_source_summary=Path(artifacts.get("sobol_source_summary", "")),
        sobol_readme=Path(artifacts.get("sobol_readme", "")),
        scope_manifest=Path(artifacts["scope_manifest"]),
    )


def asocc_run_layout_from_manifest(*, manifest: UncertaintyManifest) -> str:
    """Return the persisted aSoCC run table layout from a completed manifest."""
    public = cast(dict[str, Any], manifest.artifacts["public_output"])
    return str(cast(dict[str, Any], public["asocc_runs"])["layout"])

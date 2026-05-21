"""Persisted IO-LCA uncertainty artifact contracts."""

from pathlib import Path
from typing import Any, cast

from pyaesa.io_lca.uncertainty.runtime.models import IOLCAUncertaintyRunPaths
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def io_lca_run_paths_from_manifest(*, manifest: UncertaintyManifest) -> IOLCAUncertaintyRunPaths:
    """Return typed IO-LCA uncertainty artifact paths from a completed manifest."""
    artifacts = cast(dict[str, Any], manifest.artifacts)
    return IOLCAUncertaintyRunPaths(
        run_root=Path(artifacts["scope_manifest"]).parents[1],
        public_row_identity=Path(artifacts["public_row_identity"]),
        public_runs=Path(artifacts["lca_runs"]),
        summary_stats_runs=Path(artifacts["summary_stats_runs"]),
        results_readme=Path(artifacts["results_readme"]),
        source_methods=Path(artifacts["source_methods"]),
        scope_manifest=Path(artifacts["scope_manifest"]),
    )

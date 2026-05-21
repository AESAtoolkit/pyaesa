"""Metadata and cleanup ownership for IO-LCA uncertainty figures."""

from pathlib import Path
from typing import Any

from pyaesa.io_lca.uncertainty.runtime.models import IOLCAUncertaintyRunPaths
from pyaesa.shared.uncertainty_assessment.run_state.figure_artifacts import (
    clear_manifest_figure_paths,
    write_manifest_figure_paths,
)


def clear_uncertainty_figure_scope(*, paths: IOLCAUncertaintyRunPaths) -> None:
    """Delete persisted IO-LCA uncertainty figure files for one run."""
    clear_manifest_figure_paths(manifest_path=paths.scope_manifest)


def write_run_figure_paths(
    *,
    paths: IOLCAUncertaintyRunPaths,
    figure_paths: list[Path],
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
) -> None:
    """Persist IO-LCA uncertainty figure paths in the run manifest artifacts."""
    write_manifest_figure_paths(
        manifest_path=paths.scope_manifest,
        figure_paths=figure_paths,
        figure_options=figure_options,
        figure_format=figure_format,
    )

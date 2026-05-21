"""Metadata and cleanup ownership for aSoCC uncertainty figures."""

from pathlib import Path
from typing import Any

from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.shared.uncertainty_assessment.run_state.figure_artifacts import (
    clear_manifest_figure_paths,
    write_manifest_figure_paths,
)


def clear_uncertainty_figure_scope(*, paths: AsoccUncertaintyRunPaths) -> None:
    """Delete persisted uncertainty figure files for one run."""
    clear_manifest_figure_paths(manifest_path=paths.scope_manifest)


def write_run_figure_paths(
    *,
    paths: AsoccUncertaintyRunPaths,
    figure_paths: list[Path],
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
) -> None:
    """Persist uncertainty figure paths in the run manifest artifacts."""
    write_manifest_figure_paths(
        manifest_path=paths.scope_manifest,
        figure_paths=figure_paths,
        figure_options=figure_options,
        figure_format=figure_format,
    )

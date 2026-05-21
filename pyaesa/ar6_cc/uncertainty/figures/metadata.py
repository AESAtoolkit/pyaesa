"""Metadata and cleanup ownership for AR6 CC uncertainty figures."""

from pathlib import Path
from typing import Any

from pyaesa.ar6_cc.uncertainty.runtime.models import AR6CCUncertaintyRunPaths
from pyaesa.shared.uncertainty_assessment.run_state.figure_artifacts import (
    clear_manifest_figure_paths,
    write_manifest_figure_paths,
)


def clear_uncertainty_figure_scope(*, paths: AR6CCUncertaintyRunPaths) -> None:
    """Delete persisted AR6 CC uncertainty figure files for one run."""
    clear_manifest_figure_paths(manifest_path=paths.scope_manifest)


def write_run_figure_paths(
    *,
    paths: AR6CCUncertaintyRunPaths,
    figure_paths: list[Path],
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
) -> None:
    """Persist AR6 CC uncertainty figure paths in the run manifest artifacts."""
    write_manifest_figure_paths(
        manifest_path=paths.scope_manifest,
        figure_paths=figure_paths,
        figure_options=figure_options,
        figure_format=figure_format,
    )

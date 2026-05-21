"""Reused aSoCC uncertainty figure rendering."""

from typing import Any

from pyaesa.asocc.uncertainty.figures.metadata import write_run_figure_paths
from pyaesa.asocc.uncertainty.figures.render import render_asocc_uncertainty_figures
from pyaesa.asocc.uncertainty.io.artifacts import asocc_run_paths_from_manifest
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    read_manifest,
)
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.uncertainty_assessment.run_state.figure_artifacts import (
    manifest_figure_artifacts_current,
)


def render_reusable_asocc_figures_if_requested(
    *,
    manifest: UncertaintyManifest,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    status: StatusSink | None = None,
) -> UncertaintyManifest:
    """Render figures for a reused complete aSoCC run when requested."""
    if not figures:
        return manifest
    if manifest_figure_artifacts_current(
        manifest=manifest,
        figure_options=figure_options,
        figure_format=figure_format,
    ):
        return manifest
    paths = asocc_run_paths_from_manifest(manifest=manifest)
    figure_paths = render_asocc_uncertainty_figures(
        manifest=manifest,
        paths=paths,
        figure_options=figure_options,
        figure_format=figure_format,
        status=status,
    )
    write_run_figure_paths(
        paths=paths,
        figure_paths=figure_paths,
        figure_options=figure_options,
        figure_format=figure_format,
    )
    return read_manifest(path=paths.scope_manifest)

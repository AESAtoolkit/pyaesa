"""Figure rerendering for reused uncertainty IO-LCA runs."""

from typing import Any

from pyaesa.io_lca.uncertainty.figures.metadata import write_run_figure_paths
from pyaesa.io_lca.uncertainty.figures.render import render_io_lca_uncertainty_figures
from pyaesa.io_lca.uncertainty.io.artifacts import io_lca_run_paths_from_manifest
from pyaesa.io_lca.uncertainty.runtime.models import IOLCAUncertaintyRequest
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    read_manifest,
)
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.uncertainty_assessment.run_state.figure_artifacts import (
    manifest_figure_artifacts_current,
)


def render_reusable_io_lca_figures_if_requested(
    *,
    manifest: UncertaintyManifest,
    request: IOLCAUncertaintyRequest,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    status: StatusSink | None = None,
) -> UncertaintyManifest:
    """Render figures for a reused complete IO-LCA uncertainty run when requested."""
    if not figures:
        return manifest
    if manifest_figure_artifacts_current(
        manifest=manifest,
        figure_options=figure_options,
        figure_format=figure_format,
    ):
        return manifest
    paths = io_lca_run_paths_from_manifest(manifest=manifest)
    figure_paths = render_io_lca_uncertainty_figures(
        manifest=manifest,
        paths=paths,
        request=request,
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

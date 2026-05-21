"""Figure rerendering for reused uncertainty AR6 CC runs."""

from typing import Any

from pyaesa.ar6_cc.uncertainty.figures.metadata import write_run_figure_paths
from pyaesa.ar6_cc.uncertainty.figures.render import render_ar6_cc_uncertainty_figures
from pyaesa.ar6_cc.uncertainty.io.artifacts import ar6_cc_run_paths_from_manifest
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    read_manifest,
)
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.uncertainty_assessment.run_state.figure_artifacts import (
    manifest_figure_artifacts_current,
)


def render_reusable_ar6_cc_figures_if_requested(
    *,
    manifest: UncertaintyManifest,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    status: StatusSink | None = None,
) -> UncertaintyManifest:
    """Render figures for a reused complete AR6 CC uncertainty run when requested."""
    if manifest_figure_artifacts_current(
        manifest=manifest,
        figure_options=figure_options,
        figure_format=figure_format,
    ):
        return manifest
    paths = ar6_cc_run_paths_from_manifest(manifest=manifest)
    figure_paths = render_ar6_cc_uncertainty_figures(
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

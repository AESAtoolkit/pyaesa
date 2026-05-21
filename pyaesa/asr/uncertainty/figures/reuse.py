"""Figure rerendering for reused uncertainty ASR runs."""

from pathlib import Path
from typing import Any

from pyaesa.asr.uncertainty.figures.metadata import write_run_figure_paths
from pyaesa.asr.uncertainty.figures.render import render_asr_uncertainty_figures
from pyaesa.asr.uncertainty.io.paths import build_asr_uncertainty_run_paths
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    read_manifest,
)
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.uncertainty_assessment.run_state.figure_artifacts import (
    manifest_figure_artifacts_current,
)


def render_reusable_asr_figures_if_requested(
    *,
    manifest: UncertaintyManifest,
    root: Path,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    status: StatusSink | None = None,
) -> UncertaintyManifest:
    """Render figures for a reused complete ASR uncertainty run when requested."""
    if not figures:
        return manifest
    if manifest_figure_artifacts_current(
        manifest=manifest,
        figure_options=figure_options,
        figure_format=figure_format,
    ):
        return manifest
    paths = build_asr_uncertainty_run_paths(
        monte_carlo_root=root,
        run_id=manifest.run_id,
        output_format=manifest.output_format,
    )
    figure_paths = render_asr_uncertainty_figures(
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

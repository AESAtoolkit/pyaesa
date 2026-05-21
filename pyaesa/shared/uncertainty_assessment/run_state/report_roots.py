"""Output root helpers for uncertainty run reports."""

from pathlib import Path

from pyaesa.shared.runtime.reporting.output_roots import public_output_root_from_path
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def uncertainty_manifest_output_root(manifest: UncertaintyManifest) -> Path:
    """Return the user facing output root for one uncertainty manifest."""
    return public_output_root_from_path(Path(str(manifest.artifacts["scope_manifest"])))

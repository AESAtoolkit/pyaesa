"""Manifest argument extraction for uncertainty reports."""

from typing import Any

from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def scope_arguments(*, manifest: UncertaintyManifest) -> dict[str, Any]:
    """Return the canonical public selector arguments for one manifest."""
    arguments = manifest.arguments or {}
    base_key_by_family = {
        "ar6_cc": "base_ar6_cc_args",
        "io_lca": "base_io_lca_args",
    }
    base_key = base_key_by_family.get(manifest.family)
    if base_key is None:
        return dict(arguments)
    return dict(arguments[base_key])

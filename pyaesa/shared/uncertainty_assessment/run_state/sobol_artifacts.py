"""Sobol artifact updates for completed uncertainty manifests."""

from dataclasses import replace
from typing import Any, Mapping

from pyaesa.shared.runtime.manifest_contract import manifest_json_value
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    write_manifest,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest_payloads import (
    optional_sobol_artifact_paths,
    optional_sobol_public_output_payload,
)


def manifest_with_sobol_artifacts(
    *,
    manifest: UncertaintyManifest,
    paths: Any,
    output_format: str,
    sobol_status: Mapping[str, Any] | None,
) -> UncertaintyManifest:
    """Return ``manifest`` with current Sobol status and output metadata."""
    artifacts = dict(manifest.artifacts)
    public_output = dict(artifacts.get("public_output") or {})
    public_output.update(
        optional_sobol_public_output_payload(paths=paths, output_format=output_format)
    )
    artifacts.update(optional_sobol_artifact_paths(paths=paths))
    artifacts["public_output"] = public_output
    return replace(
        manifest,
        artifacts=manifest_json_value(artifacts),
        sobol=None if sobol_status is None else manifest_json_value(dict(sobol_status)),
    )


def write_manifest_with_sobol_artifacts(
    *,
    manifest: UncertaintyManifest,
    paths: Any,
    output_format: str,
    sobol_status: Mapping[str, Any] | None,
) -> UncertaintyManifest:
    """Persist ``manifest`` after adding requested Sobol artifact metadata."""
    updated = manifest_with_sobol_artifacts(
        manifest=manifest,
        paths=paths,
        output_format=output_format,
        sobol_status=sobol_status,
    )
    write_manifest(path=paths.scope_manifest, manifest=updated)
    return updated

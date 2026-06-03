"""Resolve completed upstream manifests recorded by reused uncertainty runs."""

from pathlib import Path
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    read_manifest,
)


def required_completed_dependency_manifest(
    *,
    manifest: UncertaintyManifest,
    base_function_source: str,
    parent_scope_name: str,
) -> tuple[UncertaintyManifest, str]:
    """Return one required completed dependency manifest from a parent manifest."""
    dependency = optional_completed_dependency_manifest(
        manifest=manifest,
        base_function_source=base_function_source,
        parent_scope_name=parent_scope_name,
    )
    if dependency is None:
        raise ValueError(
            f"Completed {parent_scope_name} manifest is missing its "
            f"{base_function_source} dependency manifest."
        )
    return dependency


def optional_completed_dependency_manifest(
    *,
    manifest: UncertaintyManifest,
    base_function_source: str,
    parent_scope_name: str,
) -> tuple[UncertaintyManifest, str] | None:
    """Return one optional completed dependency manifest from a parent manifest."""
    matches = [
        item
        for item in manifest.deterministic_prerequisites
        if str(item.get("base_function_source")) == base_function_source
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Completed {parent_scope_name} manifest contains multiple "
            f"{base_function_source} dependency manifests."
        )
    dependency = matches[0]
    dependency_manifest = read_manifest(path=Path(str(dependency["scope_manifest"])))
    if dependency_manifest.status != "complete":
        raise ValueError(
            f"Completed {parent_scope_name} manifest points to an incomplete "
            f"{base_function_source} dependency manifest."
        )
    return dependency_manifest, str(dependency.get("reuse_status", "computed"))

"""Canonical uncertainty run manifest state."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import secrets
from typing import Any, Mapping, cast

from pyaesa.shared.runtime.manifest_contract import manifest_digest, manifest_json_value
from pyaesa.shared.runtime.metadata.json import read_json_dict, write_json_dict
from pyaesa.shared.uncertainty_assessment.io.formats import (
    normalize_uncertainty_output_format,
)

RUN_ID_PREFIX = "mc_"


@dataclass(frozen=True)
class UncertaintyManifest:
    """Persisted uncertainty run state."""

    run_id: str
    family: str
    mode: str
    output_format: str
    active_sources: tuple[str, ...]
    status: str
    completed_runs: int
    created_at: str
    requested_runs: int = 0
    mc_parameters: dict[str, Any] | None = None
    source_parameters: dict[str, Any] | None = None
    arguments: dict[str, Any] | None = None
    deterministic_prerequisites: tuple[dict[str, Any], ...] = ()
    external_inputs: tuple[dict[str, Any], ...] = ()
    artifacts: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] | None = None
    component_inventory: dict[str, Any] | None = None
    convergence: dict[str, Any] | None = None
    sobol: dict[str, Any] | None = None
    compatibility_key: str | None = None
    compatibility_context: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest payload."""
        return {
            "function": f"uncertainty_{self.family}",
            "arguments": self.arguments,
            "execution": {
                "run_id": self.run_id,
                "family": self.family,
                "mode": self.mode,
                "output_format": self.output_format,
                "active_sources": list(self.active_sources),
                "status": self.status,
                "completed_runs": self.completed_runs,
                "created_at": self.created_at,
                "requested_runs": self.requested_runs,
                "mc_parameters": self.mc_parameters,
                "convergence": self.convergence,
                "sobol": self.sobol,
            },
            "reuse": {
                "compatibility_key": self.compatibility_key,
                "compatibility_context": self.compatibility_context,
                "component_inventory": self.component_inventory,
            },
            "artifacts": self.artifacts,
            "provenance": {
                "source_parameters": self.source_parameters,
                "deterministic_prerequisites": list(self.deterministic_prerequisites),
                "external_inputs": list(self.external_inputs),
                "lineage": self.lineage,
            },
        }


def allocate_run_id() -> str:
    """Allocate one user-invisible Monte Carlo run id."""
    return f"{RUN_ID_PREFIX}{secrets.token_hex(8)}"


def build_compatibility_key(payload: Mapping[str, Any]) -> str:
    """Return one stable digest key for exact Monte Carlo run reuse."""
    return manifest_digest(payload)


def build_manifest(
    *,
    family: str,
    mode: str,
    output_format: str,
    active_sources: tuple[str, ...],
    completed_runs: int = 0,
    status: str = "prepared",
    run_id: str | None = None,
    requested_runs: int = 0,
    mc_parameters: Mapping[str, Any] | None = None,
    source_parameters: Mapping[str, Any] | None = None,
    arguments: Mapping[str, Any] | None = None,
    deterministic_prerequisites: tuple[Mapping[str, Any], ...] = (),
    external_inputs: tuple[Mapping[str, Any], ...] = (),
    artifacts: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    component_inventory: Mapping[str, Any] | None = None,
    convergence: Mapping[str, Any] | None = None,
    sobol: Mapping[str, Any] | None = None,
    compatibility_key: str | None = None,
    compatibility_context: Mapping[str, Any] | None = None,
) -> UncertaintyManifest:
    """Build a canonical uncertainty manifest payload."""
    return UncertaintyManifest(
        run_id=allocate_run_id() if run_id is None else run_id,
        family=family,
        mode=mode,
        output_format=normalize_uncertainty_output_format(output_format),
        active_sources=tuple(active_sources),
        status=status,
        completed_runs=int(completed_runs),
        created_at=datetime.now().isoformat(),
        requested_runs=int(requested_runs),
        mc_parameters=_json_mapping(mc_parameters),
        source_parameters=_json_mapping(source_parameters),
        arguments=_json_mapping(arguments),
        deterministic_prerequisites=tuple(
            _json_mapping(item) or {} for item in deterministic_prerequisites
        ),
        external_inputs=tuple(_json_mapping(item) or {} for item in external_inputs),
        artifacts=_json_mapping(artifacts) or {},
        lineage=_json_mapping(lineage),
        component_inventory=_json_mapping(component_inventory),
        convergence=_json_mapping(convergence),
        sobol=_json_mapping(sobol),
        compatibility_key=compatibility_key,
        compatibility_context=_json_mapping(compatibility_context),
    )


def write_manifest(*, path: Path, manifest: UncertaintyManifest) -> None:
    """Write one uncertainty manifest."""
    write_json_dict(path, manifest.as_dict())


def read_manifest(*, path: Path) -> UncertaintyManifest:
    """Read one uncertainty manifest."""
    payload = read_json_dict(path)
    execution = manifest_json_value(dict(cast(Mapping[str, Any], payload["execution"])))
    reuse = manifest_json_value(dict(cast(Mapping[str, Any], payload["reuse"])))
    artifacts = manifest_json_value(dict(cast(Mapping[str, Any], payload["artifacts"])))
    provenance = manifest_json_value(dict(cast(Mapping[str, Any], payload["provenance"])))
    prerequisites = provenance["deterministic_prerequisites"]
    external_inputs = provenance["external_inputs"]
    return UncertaintyManifest(
        run_id=str(execution["run_id"]),
        family=str(execution["family"]),
        mode=str(execution["mode"]),
        output_format=normalize_uncertainty_output_format(str(execution["output_format"])),
        active_sources=tuple(str(source).strip() for source in execution["active_sources"]),
        status=str(execution["status"]),
        completed_runs=int(execution["completed_runs"]),
        created_at=str(execution["created_at"]),
        requested_runs=int(execution["requested_runs"]),
        mc_parameters=_optional_mapping(execution["mc_parameters"]),
        source_parameters=_optional_mapping(provenance["source_parameters"]),
        arguments=_optional_mapping(payload["arguments"]),
        deterministic_prerequisites=tuple(
            manifest_json_value(dict(item)) for item in prerequisites
        ),
        external_inputs=tuple(manifest_json_value(dict(item)) for item in external_inputs),
        artifacts=manifest_json_value(artifacts),
        lineage=_optional_mapping(provenance["lineage"]),
        component_inventory=_optional_mapping(reuse["component_inventory"]),
        convergence=_optional_mapping(execution["convergence"]),
        sobol=_optional_mapping(execution["sobol"]),
        compatibility_key=(
            None if reuse["compatibility_key"] is None else str(reuse["compatibility_key"])
        ),
        compatibility_context=_optional_mapping(reuse["compatibility_context"]),
    )


def _json_mapping(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    return manifest_json_value(dict(cast(Mapping[str, Any], value)))


def _optional_mapping(value: object) -> dict[str, Any] | None:
    """Return one package-owned manifest mapping field."""
    if value is None:
        return None
    return manifest_json_value(dict(cast(Mapping[str, Any], value)))

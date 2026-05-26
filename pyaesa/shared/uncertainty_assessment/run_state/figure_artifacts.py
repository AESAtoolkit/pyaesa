"""Figure artifact updates and reuse checks for uncertainty scope manifests."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from pyaesa.shared.figures.request_validation import normalize_figure_format
from pyaesa.shared.figures.persisted_outputs import delete_persisted_figure_paths
from pyaesa.shared.runtime.manifest_contract import manifest_digest, manifest_json_value
from pyaesa.shared.runtime.metadata.json import read_json_dict, write_json_dict
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    read_manifest,
)

_FIGURE_REQUEST_KEY = "figure_request"
_FIGURE_PATHS_KEY = "figure_paths"
_FIGURE_WARNING_SOURCE = "figures"


def clear_manifest_figure_paths(*, manifest_path: Path) -> None:
    """Delete figure files recorded in one uncertainty scope manifest."""
    payload = read_json_dict(manifest_path)
    artifacts = dict(payload["artifacts"])
    delete_persisted_figure_paths(raw_paths=artifacts.get(_FIGURE_PATHS_KEY))


def write_manifest_figure_paths(
    *,
    manifest_path: Path,
    figure_paths: list[Path],
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    warning_messages: tuple[str, ...] = (),
) -> None:
    """Persist figure paths and request signature in one uncertainty scope manifest."""
    payload = read_json_dict(manifest_path)
    manifest = read_manifest(path=manifest_path)
    artifacts = _figure_artifacts(
        manifest=manifest,
        figure_paths=figure_paths,
        figure_options=figure_options,
        figure_format=figure_format,
    )
    payload["artifacts"] = artifacts
    payload["provenance"]["lineage"] = _lineage_with_figure_warnings(
        lineage=manifest.lineage,
        warning_messages=warning_messages,
    )
    write_json_dict(manifest_path, payload)


def manifest_with_figure_artifacts(
    *,
    manifest: UncertaintyManifest,
    figure_paths: list[Path],
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    warning_messages: tuple[str, ...] = (),
) -> UncertaintyManifest:
    """Return ``manifest`` with current figure paths and request signature."""
    return replace(
        manifest,
        artifacts=_figure_artifacts(
            manifest=manifest,
            figure_paths=figure_paths,
            figure_options=figure_options,
            figure_format=figure_format,
        ),
        lineage=_lineage_with_figure_warnings(
            lineage=manifest.lineage,
            warning_messages=warning_messages,
        ),
    )


def manifest_figure_artifacts_current(
    *,
    manifest: UncertaintyManifest,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
) -> bool:
    """Return whether manifest figures match the requested visual contract."""
    artifacts = dict(manifest.artifacts)
    stored_request = artifacts.get(_FIGURE_REQUEST_KEY)
    if stored_request != _figure_request_payload(
        manifest=manifest,
        figure_options=figure_options,
        figure_format=figure_format,
    ):
        return False
    figure_paths = artifacts.get(_FIGURE_PATHS_KEY)
    if not isinstance(figure_paths, list) or not figure_paths:
        return False
    return all(Path(str(path)).exists() for path in figure_paths)


def _figure_artifacts(
    *,
    manifest: UncertaintyManifest,
    figure_paths: list[Path],
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
) -> dict[str, Any]:
    artifacts = dict(manifest.artifacts)
    artifacts[_FIGURE_PATHS_KEY] = sorted({str(path) for path in figure_paths})
    artifacts[_FIGURE_REQUEST_KEY] = _figure_request_payload(
        manifest=manifest,
        figure_options=figure_options,
        figure_format=figure_format,
    )
    return manifest_json_value(artifacts)


def _figure_request_payload(
    *,
    manifest: UncertaintyManifest,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = {
        "family": manifest.family,
        "compatibility_key": manifest.compatibility_key,
        "completed_runs": manifest.completed_runs,
        "arguments": manifest.arguments,
        "active_sources": list(manifest.active_sources),
        "figure_options": {} if figure_options is None else dict(figure_options),
        "figure_format": normalize_figure_format(figure_format),
        "public_output": dict(manifest.artifacts.get("public_output") or {}),
    }
    return {
        "signature": manifest_digest(normalized),
        "payload": manifest_json_value(normalized),
    }


def _lineage_with_figure_warnings(
    *,
    lineage: dict[str, Any] | None,
    warning_messages: tuple[str, ...],
) -> dict[str, Any] | None:
    base = {} if lineage is None else dict(lineage)
    records = [
        record
        for record in base.get("summary_records") or ()
        if not (isinstance(record, dict) and record.get("source") == _FIGURE_WARNING_SOURCE)
    ]
    records.extend(
        {
            "severity": "WARNING",
            "source": _FIGURE_WARNING_SOURCE,
            "message": message,
        }
        for message in dict.fromkeys(str(message).strip() for message in warning_messages)
        if message
    )
    if records:
        base["summary_records"] = records
    else:
        base.pop("summary_records", None)
    return manifest_json_value(base) if base else None

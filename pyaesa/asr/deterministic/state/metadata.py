"""Deterministic ASR metadata state."""

from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.manifest_contract import manifest_digest
from pyaesa.shared.runtime.metadata.contracts import FIGURE_MANIFEST_FILENAME
from pyaesa.shared.runtime.metadata.json import read_optional_json_dict, write_json_dict
from pyaesa.shared.runtime.reuse.derived_state import set_request_state

_FIGURE_STATE_KEY = "figure_state"


def load_run_metadata(path: Path) -> dict[str, Any]:
    """Load one deterministic ASR metadata snapshot."""
    return read_optional_json_dict(path)


def cached_manifest_value(
    *,
    existing_metadata: dict[str, Any],
    field_name: str,
) -> Any:
    """Return one required value from a cached deterministic ASR manifest."""
    if field_name in {
        "output_dirs",
        "output_files",
        "figure_paths",
        "dynamic_component_rows",
    }:
        return existing_metadata["artifacts"][field_name]
    if field_name in {"n_acc_files_matched", "n_asr_files_written"}:
        return existing_metadata["execution"][field_name]
    return existing_metadata["provenance"][field_name]


def identity_matches(
    *, existing_metadata: dict[str, Any] | None, identity_payload: dict[str, Any]
) -> bool:
    """Return whether a deterministic ASR manifest matches one identity payload."""
    if not existing_metadata:
        return False
    return str(existing_metadata["reuse"]["identity_key"]) == manifest_digest(identity_payload)


def cached_text_list(
    *,
    existing_metadata: dict[str, Any],
    field_name: str,
) -> list[str]:
    """Return one canonical text-list field from a cached ASR manifest."""
    value = cached_manifest_value(
        existing_metadata=existing_metadata,
        field_name=field_name,
    )
    return [str(item).strip() for item in value]


def cached_path_list(
    *,
    existing_metadata: dict[str, Any],
    field_name: str,
) -> list[Path]:
    """Return one canonical path-list field from a cached ASR manifest."""
    return [
        Path(path)
        for path in cached_text_list(
            existing_metadata=existing_metadata,
            field_name=field_name,
        )
    ]


def cached_int(
    *,
    existing_metadata: dict[str, Any],
    field_name: str,
) -> int:
    """Return one canonical integer field from a cached ASR manifest."""
    value = cached_manifest_value(
        existing_metadata=existing_metadata,
        field_name=field_name,
    )
    return int(value)


def save_run_metadata(path: Path, payload: dict[str, Any]) -> None:
    """Persist one deterministic ASR metadata snapshot."""
    write_json_dict(path, payload)


def figure_metadata_path_for_scope_manifest(path: Path) -> Path:
    """Return deterministic ASR figure metadata beside one scope manifest."""
    return path.parent / FIGURE_MANIFEST_FILENAME


def load_figure_metadata(path: Path) -> dict[str, Any]:
    """Load deterministic ASR figure request metadata."""
    return read_optional_json_dict(figure_metadata_path_for_scope_manifest(path))


def save_figure_metadata(path: Path, payload: dict[str, Any]) -> None:
    """Persist deterministic ASR figure request metadata."""
    write_json_dict(figure_metadata_path_for_scope_manifest(path), payload)


def set_figure_state(
    *,
    payload: dict[str, Any],
    signature: dict[str, Any],
    compute_signature: dict[str, Any],
    paths: list[Path],
) -> dict[str, Any]:
    """Store deterministic ASR figure request state in one metadata payload."""
    set_request_state(
        payload=payload,
        state_key=_FIGURE_STATE_KEY,
        request_signature=signature,
        compute_signature=compute_signature,
        paths=paths,
    )
    return payload

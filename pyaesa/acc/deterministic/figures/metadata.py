"""Metadata and cleanup ownership for deterministic aCC figures."""

from pathlib import Path
from typing import Any

from pyaesa.acc.deterministic.state.metadata import load_run_metadata, save_run_metadata
from pyaesa.shared.acc_asr_common.deterministic.state.scope_guard import (
    coverage_signature_covers,
)
from pyaesa.shared.figures.persisted_outputs import delete_persisted_figure_paths
from pyaesa.shared.runtime.manifest_contract import path_list
from pyaesa.shared.runtime.metadata.contracts import FIGURE_MANIFEST_FILENAME
from pyaesa.shared.runtime.metadata.json import read_optional_json_dict, write_json_dict
from pyaesa.shared.runtime.reuse.derived_state import request_state_matches, set_request_state

_FIGURE_STATE_KEY = "figure_state"


def figure_request_signature(
    *,
    dpi: int,
    output_format: str,
    figure_options: dict[str, bool],
) -> dict[str, Any]:
    """Return the deterministic aCC figure request payload."""
    return {
        "function": "deterministic_acc_figures",
        "figure_format": {"dpi": int(dpi), "format": str(output_format)},
        "figure_options": dict(figure_options),
    }


def recorded_figure_paths(*, payload: dict[str, Any]) -> list[Path]:
    """Return deterministic aCC figure paths recorded in figure metadata."""
    block = payload[_FIGURE_STATE_KEY]
    return [Path(str(path)) for path in block.get("paths", [])]


def figure_metadata_path_for_scope_manifest(*, metadata_path: Path) -> Path:
    """Return the deterministic aCC figure state path beside one scope manifest."""
    return metadata_path.parent / FIGURE_MANIFEST_FILENAME


def load_figure_metadata(*, metadata_path: Path) -> dict[str, Any]:
    """Load deterministic aCC figure request metadata."""
    return read_optional_json_dict(
        figure_metadata_path_for_scope_manifest(metadata_path=metadata_path)
    )


def save_figure_metadata(*, metadata_path: Path, payload: dict[str, Any]) -> None:
    """Persist deterministic aCC figure request metadata."""
    write_json_dict(figure_metadata_path_for_scope_manifest(metadata_path=metadata_path), payload)


def figure_state_matches(
    *,
    payload: dict[str, Any],
    request_signature: dict[str, Any],
    compute_signature: dict[str, Any],
) -> bool:
    """Return whether persisted deterministic aCC figures satisfy the current request."""
    return request_state_matches(
        payload=payload,
        state_key=_FIGURE_STATE_KEY,
        request_signature=request_signature,
        compute_signature=compute_signature,
        compute_compatible=coverage_signature_covers,
    )


def clear_deterministic_figure_scope(*, metadata_path: Path) -> None:
    """Delete persisted deterministic aCC figure files recorded in figure metadata."""
    figure_path = figure_metadata_path_for_scope_manifest(metadata_path=metadata_path)
    payload = read_optional_json_dict(figure_path)
    delete_persisted_figure_paths(
        raw_paths=payload.get(_FIGURE_STATE_KEY, {}).get("paths"),
    )
    figure_path.unlink(missing_ok=True)


def write_branch_figure_paths(
    *,
    metadata_path: Path,
    figure_paths: list[Path],
    request_signature: dict[str, Any],
    compute_signature: dict[str, Any],
) -> None:
    """Persist deterministic aCC figure paths in branch metadata."""
    payload = load_run_metadata(metadata_path)
    payload["artifacts"]["figure_paths"] = path_list(figure_paths)
    save_run_metadata(metadata_path, payload)
    figure_payload: dict[str, Any] = {}
    set_request_state(
        payload=figure_payload,
        state_key=_FIGURE_STATE_KEY,
        request_signature=request_signature,
        compute_signature=compute_signature,
        paths=figure_paths,
    )
    save_figure_metadata(metadata_path=metadata_path, payload=figure_payload)

"""Metadata ownership for deterministic aCC runs."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.manifest_contract import manifest_digest, path_list
from pyaesa.shared.runtime.metadata.json import read_optional_json_dict, write_json_dict


def load_run_metadata(path: Path) -> dict[str, Any]:
    """Load one deterministic aCC metadata snapshot."""
    return read_optional_json_dict(path)


def save_run_metadata(path: Path, payload: dict[str, Any]) -> None:
    """Persist one deterministic aCC metadata snapshot."""
    write_json_dict(path, payload)


def load_recorded_output_files(*, metadata_path: Path) -> list[Path]:
    """Return aCC output files recorded in the branch manifest."""
    metadata = load_run_metadata(metadata_path)
    return [Path(str(value)) for value in metadata["artifacts"]["output_files"]]


def build_run_metadata_payload(
    *,
    arguments: dict[str, Any],
    identity_payload: dict[str, Any],
    coverage: dict[str, list[Any]],
    cc_source: str,
    cc_type: str,
    cc_bounds: list[str],
    n_share_files_processed: int,
    n_acc_files_written: int,
    impacts: list[str],
    requested_years: list[int],
    share_transition_meta: dict[str, dict[str, object]],
    output_dirs: list[Path],
    output_files: list[Path],
    cc_input_path: Path,
    scope_label: str,
) -> dict[str, Any]:
    """Build the canonical deterministic aCC scope-manifest payload."""
    return {
        "function": "deterministic_acc",
        "arguments": deepcopy(arguments),
        "execution": {
            "status": "complete",
            "n_share_files_processed": int(n_share_files_processed),
            "n_acc_files_written": int(n_acc_files_written),
        },
        "reuse": {
            "identity_key": manifest_digest(identity_payload),
            "coverage": deepcopy(coverage),
        },
        "artifacts": {
            "output_dirs": path_list(tuple(output_dirs)),
            "output_files": path_list(tuple(output_files)),
            "figure_paths": [],
        },
        "provenance": {
            "scope_label": scope_label,
            "cc_source": cc_source,
            "cc_type": cc_type,
            "cc_bounds": list(cc_bounds),
            "impacts": list(impacts),
            "requested_years": list(requested_years),
            "share_transition_meta": deepcopy(share_transition_meta),
            "cc_input_path": str(cc_input_path),
        },
    }

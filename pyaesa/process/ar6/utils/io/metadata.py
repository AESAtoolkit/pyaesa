"""Metadata ownership for processed AR6 climate outputs."""

from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.manifest_contract import manifest_digest
from pyaesa.shared.runtime.metadata.json import read_json_dict, write_json_dict
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from .report_summaries import serialize_variable_coverage_summary_counts


def read_json(path: Path) -> dict[str, Any] | None:
    """Return JSON payload from ``path`` or ``None`` when missing."""
    try:
        payload = read_json_dict(path)
    except FileNotFoundError:
        return None
    return payload or None


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Persist ``payload`` as JSON and return ``path``."""
    path = ensure_file_parent(path)
    write_json_dict(path, dict(payload))
    return path


def signature_matches(existing_meta: dict[str, Any] | None, signature: dict[str, Any]) -> bool:
    """Return whether stored metadata matches ``signature`` exactly."""
    if existing_meta is None:
        return False
    return existing_meta.get("arguments") == signature


def variable_coverage_summary_payload(payload: dict[str, Any]) -> object:
    """Return the serialized process-owned variable coverage payload."""
    provenance = payload["provenance"]
    return provenance.get("variable_coverage_summary_counts", [])


def build_process_metadata_payload(
    *,
    signature: dict[str, object],
    categories: list[str],
    ssps: list[int],
    harmonization: bool,
    harmonization_method: str | None,
    latest_historical_year: int | None,
    requested_harmonization_year: int | None,
    harmonization_year: int | None,
    harmonization_message: str | None,
    processed_dir: Path,
    logs_dir: Path,
    figures_dir: Path,
    output_file: Path,
    log_file: Path | None,
    dropped_rows_csv_file: Path,
    variable_coverage_summary_counts: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Build the persisted process metadata payload for one AR6 run."""
    return {
        "function": "process_ar6",
        "arguments": dict(signature),
        "execution": {"status": "complete"},
        "reuse": {"identity_key": manifest_digest(signature)},
        "artifacts": {
            "output_file": str(output_file),
            "log_file": None if log_file is None else str(log_file),
            "dropped_rows_csv_file": str(dropped_rows_csv_file),
            "processed_dir": str(processed_dir),
            "logs_dir": str(logs_dir),
            "figures_dir": str(figures_dir),
        },
        "provenance": {
            "categories": list(categories),
            "ssps": [int(value) for value in ssps],
            "harmonization": bool(harmonization),
            "harmonization_method": harmonization_method,
            "latest_historical_year": latest_historical_year,
            "harmonization_year_requested": requested_harmonization_year,
            "harmonization_year": harmonization_year,
            "harmonization_year_message": harmonization_message,
            "variable_coverage_summary_counts": serialize_variable_coverage_summary_counts(
                variable_coverage_summary_counts
            ),
        },
    }


def build_figure_metadata_payload(
    *,
    signature: dict[str, object],
    figure_files: list[str],
    generation_complete: bool,
    sampling_log_csv_file: Path | None = None,
    sampling_log_columns_txt_file: Path | None = None,
) -> dict[str, object]:
    """Build the persisted figure metadata payload for one AR6 run."""
    return {
        "function": "process_ar6.figures",
        "arguments": dict(signature),
        "execution": {
            "status": "complete" if bool(generation_complete) else "running",
            "complete": bool(generation_complete),
        },
        "reuse": {"identity_key": manifest_digest(signature)},
        "artifacts": {
            "figure_files": list(figure_files),
            "sampling_convergence_log_csv": (
                None if sampling_log_csv_file is None else str(sampling_log_csv_file)
            ),
            "sampling_convergence_log_columns_txt": (
                None
                if sampling_log_columns_txt_file is None
                else str(sampling_log_columns_txt_file)
            ),
        },
        "provenance": {},
    }


def write_figure_metadata(
    *,
    figures_metadata_file: Path,
    signature: dict[str, object],
    figure_files: list[str],
    generation_complete: bool,
    sampling_log_csv_file: Path | None = None,
    sampling_log_columns_txt_file: Path | None = None,
) -> None:
    """Persist figure metadata for one AR6 run."""
    write_json(
        figures_metadata_file,
        build_figure_metadata_payload(
            signature=signature,
            figure_files=figure_files,
            generation_complete=generation_complete,
            sampling_log_csv_file=sampling_log_csv_file,
            sampling_log_columns_txt_file=sampling_log_columns_txt_file,
        ),
    )

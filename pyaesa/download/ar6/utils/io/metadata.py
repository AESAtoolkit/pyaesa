"""Metadata ownership for AR6 raw climate change downloads."""

from datetime import datetime
from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.metadata.json import read_json_dict, write_json_dict

from pyaesa.download.ar6.utils.io.paths import get_metadata_path
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent


def read_download_metadata() -> dict[str, Any] | None:
    """Return stored raw metadata, or ``None`` when absent."""
    try:
        payload = read_json_dict(get_metadata_path())
    except FileNotFoundError:
        return None
    return payload or None


def build_download_metadata_payload(
    *,
    signature: dict[str, Any],
    raw_root: Path,
    explorer_csv_file: Path,
    citation_txt_file: Path,
    ar6_public_explorer: dict[str, str],
    historical_sources: dict[str, Any],
) -> dict[str, Any]:
    """Build the persisted AR6 raw download metadata payload."""
    return {
        "function": "download_ar6",
        "signature": signature,
        "raw_root": str(raw_root),
        "explorer_csv_file": str(explorer_csv_file),
        "citation_txt_file": str(citation_txt_file),
        "ar6_public_explorer": dict(ar6_public_explorer),
        "historical_sources": historical_sources,
    }


def write_download_metadata(payload: dict[str, Any]) -> Path:
    """Persist raw metadata JSON and return its path."""
    path = ensure_file_parent(get_metadata_path())
    full_payload = dict(payload)
    full_payload["created_at"] = datetime.now().isoformat()
    write_json_dict(path, full_payload)
    return path


def signature_matches(existing_meta: dict[str, Any] | None, signature: dict[str, Any]) -> bool:
    """Return whether stored metadata matches ``signature`` exactly."""
    if existing_meta is None:
        return False
    return existing_meta.get("signature") == signature


def require_metadata_for_existing_output(
    *, metadata: dict[str, Any] | None, paths: list[Path]
) -> None:
    """Fail when raw outputs exist without the metadata used for reuse validation."""
    if metadata is not None:
        return
    if not any(path.exists() for path in paths):
        return
    examples = [str(path) for path in paths if path.exists()][:3]
    raise RuntimeError(
        "AR6 raw files exist but their metadata JSON is missing. "
        f"Existing raw files include {examples}. Rerun with refresh=True to rebuild "
        "consistent raw outputs."
    )

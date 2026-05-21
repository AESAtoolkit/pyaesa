"""Metadata ownership for processed MRIO saves."""

from datetime import datetime
import json
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, cast

from pyaesa.process.mrios.utils.io.paths import _get_metadata_path
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent


def _read_metadata(source_key: str, *, matrix_version: Optional[str]) -> Dict[str, Any]:
    """Return stored metadata for ``source_key`` and version.

    Args:
        source_key: MRIO source identifier.

    Returns:
        Dict[str, Any]: Parsed JSON payload or an empty template when missing.
    """
    path = _get_metadata_path(source_key, matrix_version=matrix_version)
    if not path.exists():
        return {
            "source": source_key,
            "version_tag": None,
            "grouping": {},
            "labels": {},
            "years": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = cast(Dict[str, Any], data)
    payload["source"] = source_key
    return payload


def read_processed_mrio_regions(
    source_key: str,
    *,
    matrix_version: Optional[str],
) -> list[str]:
    """Return the canonical ordered region labels for one processed MRIO scope.

    Args:
        source_key: MRIO source identifier.
        matrix_version: Optional grouped-matrix version token for grouped scopes.

    Returns:
        list[str]: Ordered processed-region labels from the metadata payload.

    """
    payload = _read_metadata(source_key, matrix_version=matrix_version)
    labels = cast(Mapping[str, Any], payload["labels"])
    regions = cast(Sequence[Any], labels["regions_used"])
    return [str(value) for value in regions]


def _write_metadata(
    source_key: str,
    payload: Mapping[str, Any],
    *,
    matrix_version: Optional[str],
) -> None:
    """Persist ``payload`` for ``source_key`` and version.

    Args:
        source_key: MRIO source identifier.
        payload: Mapping to serialize into the metadata file.
    """
    data = dict(payload)
    data["source"] = source_key
    data["timestamp"] = datetime.now().isoformat()
    path = ensure_file_parent(_get_metadata_path(source_key, matrix_version=matrix_version))
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_year_entry(
    meta: Mapping[str, Any],
    year: int,
) -> Optional[Dict[str, Any]]:
    """Return metadata entry for ``year``.

    Args:
        meta: Metadata payload returned by :func:`_read_metadata`.
        year: Year requested.

    Returns:
        Optional[Dict[str, Any]]: Entry describing saved matrices or None.
    """
    years = cast(Mapping[str, Any], meta["years"])
    key = str(int(year))
    entry = years.get(key)
    if entry is None:
        return None
    return dict(cast(Mapping[str, Any], entry))


def _set_year_entry(meta: MutableMapping[str, Any], year: int, entry: Mapping[str, Any]) -> None:
    """Update ``meta`` with ``entry`` for ``year``.

    Args:
        meta: Metadata payload being mutated.
        year: Year key to update.
        entry: Entry describing matrices/extensions.
    """
    years = cast(MutableMapping[str, Any], meta.setdefault("years", {}))
    years[str(int(year))] = dict(entry)


def _remove_year_entry(meta: MutableMapping[str, Any], year: int) -> None:
    """Remove the metadata entry for ``year`` when a refresh clears it."""
    years = cast(MutableMapping[str, Any], meta.setdefault("years", {}))
    years.pop(str(int(year)), None)


def _metadata_satisfies(
    entry: Mapping[str, Any] | None,
    *,
    saved_exists: bool,
    required_core: Sequence[str],
    required_extensions: Sequence[str],
    required_lcia_method: Optional[str],
    required_lcia_methods: Optional[Sequence[str]] = None,
) -> bool:
    """Return True when ``entry`` satisfies requested requirements.

    Args:
        entry: Metadata entry describing saved matrices/extensions.
        saved_exists: Whether the on disk saved directory exists.
        required_core: Core matrix names required by the caller.
        required_extensions: Extension names required by the caller.
        required_lcia_method: Optional LCIA method that must be present.
        required_lcia_methods: Optional LCIA methods that must be present.

    Returns:
        bool: True when all requirements are satisfied.
    """
    if entry is None or not saved_exists:
        return False
    raw_core = cast(Sequence[str], entry["core"])
    extensions = cast(Mapping[str, Any], entry["extensions"])
    core = set(raw_core)
    extension_keys = set(extensions.keys())

    # Core matrices must always be present.
    if required_core and not set(required_core).issubset(core):
        return False

    if required_extensions and not set(required_extensions).issubset(extension_keys):
        return False

    if required_lcia_methods is None and required_lcia_method is not None:
        required_lcia_methods = [required_lcia_method]

    if required_lcia_methods:
        for lcia_method in required_lcia_methods:
            if lcia_method not in extension_keys:
                return False
    return True

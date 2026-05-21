"""Metadata ownership for processed population and GDP datasets.

Provides utilities to persist begin/end year coverage and timestamps for the
processed artefacts so later runs can detect whether regeneration is needed.
"""

from datetime import datetime
import json
from typing import Any, Dict, Optional, cast

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.process.pop_gdp.io.paths import _get_metadata_path


def _read_meta(name: str) -> Optional[Dict[str, Any]]:
    """Return stored metadata for ``name`` or ``None`` when missing.

    Args:
        name (str): Identifier passed to :func:`write_meta`.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON payload or ``None`` when missing.
    """
    path = _get_metadata_path(name)
    if not path.exists():
        return None
    return cast(Dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_meta(name: str, begin_year: int, end_year: int) -> None:
    """Persist begin/end year coverage for ``name``.

    Args:
        name (str): Identifier for the processed dataset.
        begin_year (int): Earliest year represented in the processed file.
        end_year (int): Latest year represented in the processed file.
    """
    path = _get_metadata_path(name)
    path = ensure_file_parent(path)
    payload = {
        "begin_year": int(begin_year),
        "end_year": int(end_year),
        "timestamp": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _meta_covers(meta: Dict[str, Any], begin_year: int, end_year: int) -> bool:
    """Return ``True`` when metadata range covers [begin_year, end_year].

    Args:
        meta (Dict[str, Any]): Metadata payload returned by :func:`read_meta`.
        begin_year (int): Requested begin year.
        end_year (int): Requested end year.

    Returns:
        bool: True when the stored range encompasses the requested span.
    """
    stored_begin = int(meta["begin_year"])
    stored_end = int(meta["end_year"])
    return stored_begin <= int(begin_year) and stored_end >= int(end_year)

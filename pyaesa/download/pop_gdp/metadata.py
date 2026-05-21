"""Metadata ownership for population and GDP downloaders.

This module provides ownership for reading, writing, and validating small JSON
metadata files that accompany the raw CSV outputs produced by the
population/GDP downloaders.
"""

from datetime import datetime
import json
from typing import Any, Dict, Iterable, Optional

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.download.pop_gdp.raw_paths import _get_meta_path


def _read_meta(output_filename: str) -> Optional[Dict[str, Any]]:
    """Read metadata JSON for a dataset.

    Attempts to read and parse the metadata JSON for ``output_filename``.
    If the file is missing the function returns ``None``. Invalid/corrupted
    metadata raises a clear runtime error.

    Args:
        output_filename (str): base name used to compose the metadata file.

    Returns:
        Optional[dict]: Parsed metadata mapping or ``None`` when missing.
    """
    p = _get_meta_path(output_filename)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _write_meta(
    output_filename: str,
    begin_year: int,
    end_year: int,
    variables: Iterable[str],
) -> None:
    """Write metadata JSON for ``output_filename``.

    Args:
        output_filename (str): base name used to compose the metadata file.
        begin_year (int): first year covered by the dataset.
        end_year (int): last year covered by the dataset.
        variables (Iterable[str]): list of variable names included in the CSV.
    """
    p = _get_meta_path(output_filename)
    p = ensure_file_parent(p)
    timestamp = datetime.now().isoformat()
    payload: Dict[str, Any] = {
        "begin_year": int(begin_year),
        "end_year": int(end_year),
        "variables": list(variables),
        "timestamp": timestamp,
    }
    json_text = json.dumps(payload, indent=2)
    p.write_text(json_text, encoding="utf-8")


def _meta_covers(
    meta: Dict[str, Any],
    begin_year: int,
    end_year: int,
    variables: Iterable[str],
) -> bool:
    """Return True if ``meta`` covers the requested year range and variables.

    Args:
        meta (dict): Parsed metadata mapping.
        begin_year (int): First requested year.
        end_year (int): Last requested year.
        variables (Iterable[str]): Variables required.

    Returns:
        bool: True when the metadata covers the requested years and all
            requested variables, False otherwise.
    """
    if not meta:
        return False
    if int(meta["begin_year"]) > int(begin_year):
        return False
    if int(meta["end_year"]) < int(end_year):
        return False
    meta_vars = [str(variable) for variable in meta["variables"]]
    return {str(variable) for variable in variables}.issubset(set(meta_vars))

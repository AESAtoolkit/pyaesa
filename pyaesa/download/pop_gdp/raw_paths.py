"""Path ownership for raw population and GDP data.

This module provides ownership for output and metadata file paths for
population and GDP raw CSVs. Paths are composed relative to the package
repository root configured by :func:`pyaesa.set_workspace`.
"""

from pathlib import Path
from pyaesa.workspace_initialisation.workspace import (
    get_default_repo_root as _get_repo_root,
)
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent


def _get_output_path(output_filename: str) -> Path:
    """Return the full path to the CSV output for ``output_filename``.

    Args:
        output_filename (str): base name used to compose the output CSV
            (without suffix).

    Returns:
        pathlib.Path: Full path to the CSV file under
            ``<repo_root>/data_raw/pop_gdp/{output_filename}_raw.csv``.
    """
    return ensure_file_parent(
        _get_repo_root().joinpath("data_raw", "pop_gdp", f"{output_filename}_raw.csv")
    )


def _get_meta_path(output_filename: str) -> Path:
    """Return the path to the metadata JSON for ``output_filename``.

    Args:
        output_filename (str): base name used to compose the metadata file
            (without suffix).

    Returns:
        pathlib.Path: Full path to JSON metadata under
            ``<repo_root>/data_raw/logs/{output_filename}_meta.json``.
    """
    return ensure_file_parent(
        _get_repo_root().joinpath("data_raw", "logs", f"{output_filename}_meta.json")
    )


def _clear_raw_output_scope(output_filename: str) -> None:
    """Delete one population GDP raw CSV and its coverage metadata."""
    _get_output_path(output_filename).unlink(missing_ok=True)
    _get_meta_path(output_filename).unlink(missing_ok=True)

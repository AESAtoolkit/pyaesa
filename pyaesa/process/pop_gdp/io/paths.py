"""Path ownership for processed pop_gdp outputs.

This module centralizes file system conventions for ``data_processed`` outputs,
logs, metadata, and static MRIO matching files used during processing.
"""

from pathlib import Path

from pyaesa.download.mrios.utils.source_registry import get_mrio_entry
from pyaesa.workspace_initialisation.workspace import (
    get_default_repo_root as _get_repo_root,
)
from pyaesa.shared.runtime.io.filesystem import ensure_dir, ensure_file_parent


def _log_dir() -> Path:
    """Return the root log directory for processed pop/gdp artefacts."""
    return ensure_dir(_get_repo_root().joinpath("data_processed", "pop_gdp", "logs"))


def _get_processed_output_path(dataset: str) -> Path:
    """Return the CSV path that stores processed ``dataset`` outputs.

    Args:
        dataset (str): Short key such as ``"wb"`` or ``"ssp"``.

    Returns:
        pathlib.Path: Absolute path to ``<repo>/data_processed/pop_gdp/...``.
    """
    dataset = dataset.lower()
    return ensure_file_parent(
        _get_repo_root().joinpath("data_processed", "pop_gdp", f"{dataset}_processed.csv")
    )


def _get_log_path(filename: str) -> Path:
    """Return the log path under ``data_processed/pop_gdp/logs``.

    Args:
        filename (str): Base filename to append beneath the logs directory.

    Returns:
        pathlib.Path: Absolute path to the requested log file.
    """
    return ensure_file_parent(_log_dir().joinpath(filename))


def _get_metadata_path(name: str) -> Path:
    """Return the metadata JSON path for a processed dataset.

    Args:
        name (str): Identifier for the processed dataset (``"wb_processed"``).

    Returns:
        pathlib.Path: Path pointing to ``<logs>/<name>_meta.json``.
    """
    safe = name.lower()
    return ensure_file_parent(_log_dir().joinpath(f"{safe}_meta.json"))


def _clear_processed_dataset_scope(dataset: str) -> None:
    """Delete one processed pop GDP dataset CSV, metadata, and owned logs."""
    dataset_key = str(dataset).strip().lower()
    _get_processed_output_path(dataset_key).unlink(missing_ok=True)
    _get_metadata_path(f"{dataset_key}_processed").unlink(missing_ok=True)
    if dataset_key == "wb":
        _get_log_path("wb_fill_log.csv").unlink(missing_ok=True)


def _get_matching_path_for_dataset(*, dataset_key: str, source: str) -> Path:
    """Return the MRIO matching CSV for one canonical processed dataset key.

    Args:
        dataset_key (str): Canonical processed dataset key (``"wb"`` or ``"ssp"``).
        source (str): MRIO source key (EXIOBASE or OECD). EXIO variants share
            matching assets under ``data_raw/mrio/exiobase_3/reg_matching``.

    Returns:
        pathlib.Path: Absolute path to the canonical matching CSV.

    Raises:
        ValueError: If ``source`` is unsupported.
    """
    entry = get_mrio_entry(source)
    matching_source_token = (
        entry.shared_prereq_root if entry.family == "exiobase" else entry.source_key
    )
    return ensure_file_parent(
        _get_repo_root().joinpath(
            "data_raw",
            "mrio",
            matching_source_token,
            "reg_matching",
            f"{dataset_key}_{matching_source_token}_matching.csv",
        )
    )


def _get_wb_matching_path(source: str) -> Path:
    """Return the World Bank historical pop GDP MRIO matching CSV path."""
    return _get_matching_path_for_dataset(dataset_key="wb", source=source)


def _get_ssp_matching_path(source: str) -> Path:
    """Return the SSP prospective pop GDP MRIO matching CSV path."""
    return _get_matching_path_for_dataset(dataset_key="ssp", source=source)

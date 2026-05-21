"""Path ownership for MRIO raw downloads.

These functions build deterministic paths relative to the repository root
configured via ``set_workspace`` and expose a minimal public surface so the
rest of the codebase can ask for MRIO folders without reimplementing the
layout.
"""

from pathlib import Path
from typing import Tuple

from pyaesa.workspace_initialisation.workspace import (
    get_default_repo_root as _get_repo_root,
)
from pyaesa.shared.runtime.io.filesystem import ensure_dir, ensure_file_parent
from pyaesa.download.mrios.utils.source_registry import get_mrio_entry


def _resolve_source_layout(source_key: str) -> Tuple[str, str]:
    """Return raw folder layout for ``source_key``.

    Args:
        source_key: Known MRIO source identifier.

    Returns:
        Tuple ``(source_root, full_dir_name)``.

    Raises:
        KeyError: If ``source_key`` is not registered.
    """
    entry = get_mrio_entry(source_key)
    source_root = entry.raw_root
    if entry.family == "exiobase":
        source_root = str(Path(entry.shared_prereq_root) / entry.raw_root)
    return source_root, entry.raw_full_dir_name


def _get_source_root(source_key: str) -> Path:
    """Return the root directory for a MRIO source under ``data_raw``.

    Args:
        source_key: Known MRIO source identifier.

    Returns:
        Path to ``<repo>/data_raw/mrio/<source_root>``.
    """
    subdir, _ = _resolve_source_layout(source_key)
    return ensure_dir(_get_repo_root().joinpath("data_raw", "mrio", subdir))


def _get_full_dir(source_key: str) -> Path:
    """Return the directory where raw archives are stored for ``source_key``."""
    _, full_dir_name = _resolve_source_layout(source_key)
    return ensure_dir(_get_source_root(source_key).joinpath(full_dir_name))


def _get_metadata_path(source_key: str) -> Path:
    """Return the metadata JSON path for ``source_key`` downloads."""
    return ensure_file_parent(
        _get_repo_root().joinpath("data_raw", "logs", f"{source_key}_mrio_meta.json")
    )


def _get_exio_archive_path(full_dir: Path, year: int, *, system: str) -> Path:
    """Return the EXIOBASE archive path for ``year`` and ``system``."""
    system_clean = str(system).strip().lower()
    return ensure_file_parent(full_dir / f"IOT_{int(year)}_{system_clean}.zip")


def _get_exio_archive_temp_dir(full_dir: Path, year: int, *, system: str) -> Path:
    """Return the temporary EXIOBASE staging directory for one archive."""
    system_clean = str(system).strip().lower()
    return full_dir / f".tmp_IOT_{int(year)}_{system_clean}"


def _get_exio_archive_temp_path(full_dir: Path, year: int, *, system: str) -> Path:
    """Return the temporary EXIOBASE archive path for one staged download."""
    temp_dir = _get_exio_archive_temp_dir(full_dir, year, system=system)
    archive_name = _get_exio_archive_path(Path("."), year, system=system).name
    return ensure_file_parent(temp_dir / archive_name)


def _get_oecd_csv_path(full_dir: Path, year: int) -> Path:
    """Return the OECD ICIO CSV path for ``year`` inside ``full_dir``."""
    return ensure_file_parent(full_dir / f"ICIO2025_{int(year)}.csv")


def _get_oecd_bundle_zip_path(target_dir: Path, bundle: str, version: str = "v2025") -> Path:
    """Return the downloaded OECD bundle ZIP path for ``bundle``."""
    return ensure_file_parent(target_dir / f"ICIO{version.replace('v', '')}_{bundle}.zip")


def _get_oecd_bundle_temp_dir(target_dir: Path, bundle: str, version: str = "v2025") -> Path:
    """Return the temporary OECD staging directory for one bundle download."""
    bundle_name = _get_oecd_bundle_zip_path(Path("."), bundle, version).stem
    return target_dir / f".tmp_{bundle_name}"


def _get_oecd_bundle_temp_zip_path(target_dir: Path, bundle: str, version: str = "v2025") -> Path:
    """Return the temporary OECD ZIP path for one staged bundle download."""
    temp_dir = _get_oecd_bundle_temp_dir(target_dir, bundle, version)
    return ensure_file_parent(temp_dir / _get_oecd_bundle_zip_path(Path("."), bundle, version).name)

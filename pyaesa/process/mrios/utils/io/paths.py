"""Path ownership for processed MRIO saves."""

from pathlib import Path
import re
from typing import Literal, Optional

from pyaesa.download.mrios.utils.source_registry import (
    get_mrio_entry,
)
from pyaesa.workspace_initialisation.workspace import (
    get_default_repo_root as _get_repo_root,
)
from pyaesa.shared.lcia.paths import characterization_matrix_path
from pyaesa.shared.runtime.io.filesystem import ensure_dir, ensure_file_parent

_PRECLIP_DIR = "preclip"
_PRECLIP_EXTENSIONS_DIR = "extensions"
_PRECLIP_CALC_ALL_MARKER = "pymrio_calc_all_complete.json"
_ROOT_EXTENSIONS_DIR = "extensions"


def _resolve_version_tag(matrix_version: Optional[str]) -> str:
    """Return the version tag for matrices.

    Args:
        matrix_version (str | None): Optional version name.

    Returns:
        str: ``"original_classification"`` when ``matrix_version`` is None/empty,
            otherwise ``"custom_classification_{matrix_version}"``.
    """
    if matrix_version is None:
        return "original_classification"
    matrix_version = str(matrix_version).strip()
    if not matrix_version:
        return "original_classification"
    return f"custom_classification_{matrix_version}"


def _year_saved_folder_name(source_key: str, year: int) -> str:
    """Return the canonical processed year folder name for ``source_key``."""
    entry = get_mrio_entry(source_key)
    if entry.family == "exiobase":
        return f"IOT_{int(year)}_{entry.system}_calc"
    return f"ICIO{entry.version_token.removeprefix('v')}_{int(year)}_calc"


def _get_saved_dir(source_key: str, *, matrix_version: Optional[str] = None) -> Path:
    """Return ``data_processed/mrio/<source>/<version_tag>``.

    Args:
        source_key (str): MRIO source identifier.
        matrix_version (str | None): Optional grouping version.

    Returns:
        Path: Root folder for saved MRIO calc matrices.
    """
    version_tag = _resolve_version_tag(matrix_version)
    return _get_repo_root().joinpath(
        "data_processed",
        "mrio",
        get_mrio_entry(source_key).source_key,
        version_tag,
    )


def _get_year_saved_dir(
    source_key: str,
    year: int,
    *,
    matrix_version: Optional[str] = None,
) -> Path:
    """Return the canonical per year saved directory.

    Args:
        source_key (str): MRIO source identifier.
        year (int): Year for which the saved folder is built.
        matrix_version (str | None): Optional grouping version.

    Returns:
        Path: Absolute path to ``saved_/<year_folder>``.
    """
    return ensure_dir(_get_year_saved_path(source_key, year, matrix_version=matrix_version))


def _get_year_saved_path(
    source_key: str,
    year: int,
    *,
    matrix_version: Optional[str] = None,
) -> Path:
    """Return the canonical per year saved path without creating it."""
    return _get_saved_dir(source_key, matrix_version=matrix_version).joinpath(
        _year_saved_folder_name(source_key, year)
    )


def _get_metadata_path(source_key: str, *, matrix_version: Optional[str] = None) -> Path:
    """Return the metadata JSON path for ``source_key`` saves."""
    return ensure_file_parent(
        _get_saved_dir(
            source_key,
            matrix_version=matrix_version,
        ).joinpath("logs", "processed_metadata.json")
    )


def _get_mrio_calc_logs_dir(
    source_key: str,
    *,
    matrix_version: Optional[str] = None,
) -> Path:
    """Return the log directory for one MRIO source/version scope."""
    return ensure_dir(
        _get_saved_dir(
            source_key,
            matrix_version=matrix_version,
        ).joinpath("logs")
    )


def _get_mrio_calc_log_path(
    filename: str,
    *,
    source_key: str,
    matrix_version: Optional[str] = None,
) -> Path:
    """Return a log path under one source/version owned MRIO log folder."""
    return _get_mrio_calc_logs_dir(
        source_key,
        matrix_version=matrix_version,
    ).joinpath(filename)


def _get_mrio_clipping_log_path(
    source_key: str,
    *,
    matrix_version: Optional[str] = None,
) -> Path:
    """Return the shared clipping log CSV path for one MRIO source/version lane.

    Args:
        source_key: MRIO source identifier.
        matrix_version: Optional grouping version. ``None`` or blank maps to
            ``"original_classification"``.

    Returns:
        Absolute path to the clipping diagnostics CSV under the source owned
        processed log directory.
    """
    source_clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source_key).strip())
    version_label = "original_classification"
    if matrix_version is not None and str(matrix_version).strip():
        version_label = str(matrix_version).strip()
    version_clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", version_label)
    return ensure_file_parent(
        _get_mrio_calc_log_path(
            f"{source_clean}_{version_clean}_clipping_log.csv",
            source_key=source_key,
            matrix_version=matrix_version,
        )
    )


def _get_mrio_raw_corrected_values_log_path(
    source_key: str,
    *,
    matrix_version: Optional[str] = None,
) -> Path:
    """Return the shared raw corrected values log CSV path for one MRIO lane."""
    source_clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source_key).strip())
    version_label = "original_classification"
    if matrix_version is not None and str(matrix_version).strip():
        version_label = str(matrix_version).strip()
    version_clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", version_label)
    return ensure_file_parent(
        _get_mrio_calc_log_path(
            f"{source_clean}_{version_clean}_raw_corrected_values_log.csv",
            source_key=source_key,
            matrix_version=matrix_version,
        )
    )


def _get_mrio_clipping_log_columns_explanation_path(
    source_key: str,
    *,
    matrix_version: Optional[str] = None,
) -> Path:
    """Return the MRIO clipping log schema explanation TXT path."""
    return ensure_file_parent(
        _get_mrio_calc_log_path(
            "clipping_log_columns_explanation.txt",
            source_key=source_key,
            matrix_version=matrix_version,
        )
    )


def _get_characterization_matrix_path(*, source_key: str, lcia_method: str) -> Path:
    """Return LCIA characterization matrix CSV path for one method."""
    return characterization_matrix_path(source=source_key, lcia_method=lcia_method)


def _get_grouping_dir(source_key: str, *, kind: Literal["reg", "sec"]) -> Path:
    """Return grouping folder for ``source_key`` and grouping ``kind``.

    For EXIO variant sources:
    - region grouping (``kind="reg"``) is shared under ``exiobase_3``;
    - sector grouping (``kind="sec"``) is under
      ``exiobase_3/grouping/<system>`` where system is ``ixi`` or ``pxp``.
    """
    entry = get_mrio_entry(source_key)
    if entry.family == "exiobase":
        base = ensure_dir(
            _get_repo_root().joinpath("data_raw", "mrio", entry.shared_prereq_root, "grouping")
        )
        if kind == "reg":
            return base
        return ensure_dir(base.joinpath(str(entry.system)))
    return ensure_dir(_get_repo_root().joinpath("data_raw", "mrio", entry.source_key, "grouping"))


def _get_group_map_path(
    source_key: str,
    *,
    kind: Literal["reg", "sec"],
    group_version: str,
) -> Path:
    """Return path to a grouping map CSV.

    Args:
        source_key (str): MRIO source identifier.
        kind (str): ``"reg"`` or ``"sec"``.
        group_version (str): User specified grouping version name.
    """
    if kind == "reg":
        filename = f"group_reg_{group_version}.csv"
    else:
        filename = f"group_sec_{group_version}.csv"
    return ensure_file_parent(
        _get_grouping_dir(
            source_key,
            kind=kind,
        ).joinpath(filename)
    )


def _get_preclip_dir(saved_dir: Path) -> Path:
    """Return the explicit directory that stores optional preclip outputs."""
    return Path(saved_dir) / _PRECLIP_DIR


def _get_preclip_extensions_dir(saved_dir: Path) -> Path:
    """Return the preclip extension output directory."""
    return _get_preclip_dir(saved_dir) / _PRECLIP_EXTENSIONS_DIR


def _get_preclip_calc_all_marker_path(saved_dir: Path) -> Path:
    """Return the marker path indicating calc_all payload completion."""
    return _get_preclip_dir(saved_dir) / _PRECLIP_CALC_ALL_MARKER


def _get_root_extensions_dir(saved_dir: Path) -> Path:
    """Return the year root extension directory used by UNCASExt intermediates."""
    return Path(saved_dir) / _ROOT_EXTENSIONS_DIR

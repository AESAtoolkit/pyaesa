"""Path ownership for AR6 raw climate change assets."""

from pathlib import Path

from pyaesa.workspace_initialisation.workspace import (
    get_default_repo_root as _get_repo_root,
)
from pyaesa.shared.runtime.io.filesystem import ensure_dir, ensure_file_parent

_RAW_SUBDIR = ("data_raw", "carrying_capacities", "dynamic_climate_change_ar6")
_META_FILENAME = "dynamic_climate_change_ar6_meta.json"


def get_raw_dir() -> Path:
    """Return the AR6 raw asset directory."""
    return ensure_dir(_get_repo_root().joinpath(*_RAW_SUBDIR))


def get_logs_dir() -> Path:
    """Return the raw metadata directory."""
    return ensure_dir(_get_repo_root().joinpath("data_raw", "logs"))


def get_metadata_path() -> Path:
    """Return the raw metadata JSON path."""
    return ensure_file_parent(get_logs_dir().joinpath(_META_FILENAME))


def explorer_csv_path_for_raw_dir(raw_dir: Path, database: str) -> Path:
    """Return the IIASA explorer CSV path for ``database`` inside ``raw_dir``."""
    db_token = str(database).replace("-", "_")
    return ensure_file_parent(Path(raw_dir).joinpath(f"{db_token}_explorer.csv"))


def get_explorer_csv_path(database: str) -> Path:
    """Return the IIASA explorer CSV path for ``database``."""
    return explorer_csv_path_for_raw_dir(get_raw_dir(), database)


def citation_txt_path_for_raw_dir(raw_dir: Path) -> Path:
    """Return the citation/source usage TXT path inside one explicit raw directory."""
    return ensure_file_parent(
        Path(raw_dir).joinpath("recommended_citations_data_sources_and_usage.txt")
    )


def get_citation_txt_path() -> Path:
    """Return the raw citation/source usage TXT path."""
    return citation_txt_path_for_raw_dir(get_raw_dir())


def clear_download_output_scope(database: str) -> None:
    """Delete the AR6 explorer, citation text, and download metadata files."""
    get_explorer_csv_path(database).unlink(missing_ok=True)
    get_citation_txt_path().unlink(missing_ok=True)
    get_metadata_path().unlink(missing_ok=True)

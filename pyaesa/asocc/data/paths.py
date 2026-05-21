"""Processed input locators owned by the allocation data layer."""

from pathlib import Path

from pyaesa.process.mrios.utils.io.paths import _get_year_saved_path
from pyaesa.workspace_initialisation.workspace import (
    get_default_repo_root as _get_repo_root,
)


def _get_processed_pop_gdp_table_path(*, dataset: str) -> Path:
    """Return processed WB/SSP table path used by deterministic_asocc setup."""
    dataset_norm = str(dataset).strip().lower()
    return _get_repo_root().joinpath(
        "data_processed",
        "pop_gdp",
        f"{dataset_norm}_processed.csv",
    )


def _get_mrio_year_dir(
    *,
    source: str,
    year: int,
    group_version: str | None,
) -> Path:
    """Return the processed MRIO year directory."""
    return _get_year_saved_path(source, year, matrix_version=group_version)

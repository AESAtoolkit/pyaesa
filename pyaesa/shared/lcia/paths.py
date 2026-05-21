"""Canonical filesystem path ownership for bundled LCIA prerequisites."""

from pathlib import Path

from pyaesa.download.mrios.utils.source_registry import get_mrio_entry
from pyaesa.workspace_initialisation.workspace import (
    get_default_repo_root as _get_repo_root,
)
from pyaesa.shared.lcia.contracts import (
    CHARACTERIZATION_FACTORS_MATRICES_DIRNAME,
    RESPONSIBILITY_PERIODS_DIRNAME,
    STATIC_CC_SUFFIX,
    normalize_lcia_method_name,
)

_CARBON_ACCOUNT_COV_SUBDIR = ("data_raw", "mrio", "exiobase_3", "lcia", "carbon_accounts_covs")


def _shared_lcia_subdir(*, source: str) -> str:
    """Return the MRIO prerequisite subdir used by bundled LCIA assets.

    Args:
        source: MRIO source identifier.

    Returns:
        The canonical shared prerequisite subdirectory for the MRIO source.
    """
    source_key = str(source).strip().lower()
    return get_mrio_entry(source_key).shared_prereq_root


def bundled_static_cc_dir() -> Path:
    """Return the project local static carrying capacity prerequisite directory."""
    return _get_repo_root() / "data_raw" / "carrying_capacities"


def characterization_matrix_path(*, source: str, lcia_method: str) -> Path:
    """Return the LCIA characterization matrix path for one MRIO source and method."""
    return _get_repo_root().joinpath(
        "data_raw",
        "mrio",
        _shared_lcia_subdir(source=source),
        "lcia",
        CHARACTERIZATION_FACTORS_MATRICES_DIRNAME,
        f"{normalize_lcia_method_name(lcia_method)}.csv",
    )


def responsibility_periods_csv_path(*, source: str, lcia_method: str) -> Path:
    """Return the LCIA responsibility period CSV path for one MRIO source and method."""
    return _get_repo_root().joinpath(
        "data_raw",
        "mrio",
        _shared_lcia_subdir(source=source),
        "lcia",
        RESPONSIBILITY_PERIODS_DIRNAME,
        f"{normalize_lcia_method_name(lcia_method)}_rps.csv",
    )


def static_cc_csv_path(*, lcia_method: str) -> Path:
    """Return the expected bundled static carrying capacity CSV path for one LCIA method."""
    return bundled_static_cc_dir() / f"{normalize_lcia_method_name(lcia_method)}{STATIC_CC_SUFFIX}"


def carbon_account_cov_dir() -> Path:
    """Return the bundled Carbon consumption based accounts CoV asset directory."""
    return _get_repo_root().joinpath(*_CARBON_ACCOUNT_COV_SUBDIR)


def carbon_account_cov_path(*, asset_name: str) -> Path:
    """Return one bundled Carbon consumption based accounts CoV asset path."""
    return carbon_account_cov_dir() / str(asset_name).strip()

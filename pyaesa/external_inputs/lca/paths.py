"""Canonical path owners for project scoped external LCA storage."""

from pathlib import Path

from pyaesa.io_lca.data.contracts import EXTERNAL_LCA_DIRNAME
from pyaesa.shared.runtime.io.family_root_names import LCA_ROOT_DIRNAME


def external_lca_root(*, project_base: Path) -> Path:
    """Return the project scoped external LCA input root."""
    return Path(project_base) / LCA_ROOT_DIRNAME / EXTERNAL_LCA_DIRNAME


def external_lca_deterministic_dir(*, project_base: Path) -> Path:
    """Return the deterministic external LCA storage root."""
    return external_lca_root(project_base=project_base) / "deterministic"


def external_lca_monte_carlo_dir(*, project_base: Path) -> Path:
    """Return the Monte Carlo external LCA storage root."""
    return external_lca_root(project_base=project_base) / "monte_carlo"


def external_lca_deterministic_figures_dir(*, project_base: Path) -> Path:
    """Return the deterministic external LCA figure root."""
    return external_lca_deterministic_dir(project_base=project_base) / "figures"


def external_lca_monte_carlo_figures_dir(*, project_base: Path) -> Path:
    """Return the Monte Carlo external LCA figure root."""
    return external_lca_monte_carlo_dir(project_base=project_base) / "figures"

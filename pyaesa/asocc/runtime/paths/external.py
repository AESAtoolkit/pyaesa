"""Project scoped external aSoCC path ownership for the aSoCC family."""

from pathlib import Path

from pyaesa.asocc.io.contracts import EXTERNAL_ASOCC_DIRNAME

from .family_roots import _get_asocc_root


def external_asocc_relative_dir(*, level: str) -> Path:
    """Return the downstream public relative output path for one external aSoCC level."""
    return {"level_1": Path("results"), "level_2": Path("results_l2_vs_global")}[str(level)]


def get_asocc_external_root(*, proj_base: Path) -> Path:
    """Return the project scoped external aSoCC input root."""
    return _get_asocc_root(proj_base=proj_base) / EXTERNAL_ASOCC_DIRNAME


def get_asocc_external_method_level_dir(
    *,
    proj_base: Path,
    storage_mode: str,
    level: str,
) -> Path:
    """Return the project scoped flat external aSoCC directory for one storage mode."""
    del level
    return get_asocc_external_root(proj_base=proj_base) / storage_mode

"""Filesystem availability helpers for bundled LCIA prerequisites."""

import importlib.resources as resources
from pathlib import Path

from pyaesa.workspace_initialisation.packaged_prerequisites import packaged_prerequisites_root

from pyaesa.shared.lcia.paths import (
    bundled_static_cc_dir,
    characterization_matrix_path,
    responsibility_periods_csv_path,
    static_cc_csv_path,
)
from pyaesa.shared.lcia.contracts import STATIC_CC_SUFFIX


def has_characterization_matrix(*, source: str, lcia_method: str) -> bool:
    """Return whether a characterization matrix exists for one source and LCIA method."""
    return characterization_matrix_path(source=source, lcia_method=lcia_method).exists()


def has_rps(*, source: str, lcia_method: str) -> bool:
    """Return whether a responsibility period CSV exists for one source and LCIA method."""
    return responsibility_periods_csv_path(source=source, lcia_method=lcia_method).exists()


def has_static_cc(*, lcia_method: str) -> bool:
    """Return whether a bundled static carrying capacity CSV exists for one LCIA method."""
    return static_cc_csv_path(lcia_method=lcia_method).exists()


def require_static_cc_csv_path(*, lcia_method: str) -> Path:
    """Return one bundled static carrying capacity CSV path or fail with context."""
    path = static_cc_csv_path(lcia_method=lcia_method)
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled static carrying capacity CSV not found for lcia_method='{lcia_method}'. "
            f"Expected path: '{path}'."
        )
    return path


def _discover_static_cc_methods_in_dir(static_cc_dir: Path) -> tuple[str, ...]:
    """Return bundled static carrying capacity method names discovered in one directory."""
    methods: list[str] = []
    for path in sorted(static_cc_dir.glob(f"*{STATIC_CC_SUFFIX}")):
        stem = path.name[: -len(STATIC_CC_SUFFIX)]
        if stem.startswith("name_"):
            continue
        methods.append(stem)
    return tuple(methods)


def discover_static_cc_methods() -> tuple[str, ...]:
    """Return bundled static carrying capacity method names from workspace or package assets."""
    try:
        return _discover_static_cc_methods_in_dir(bundled_static_cc_dir())
    except RuntimeError:
        packaged_static_cc_dir = packaged_prerequisites_root().joinpath("carrying_capacities")
        with resources.as_file(packaged_static_cc_dir) as packaged_dir:
            return _discover_static_cc_methods_in_dir(Path(packaged_dir))

"""Argument validation and normalization for IO-LCA public entrypoints."""

from pyaesa.asocc.methods.lcia_inputs import normalize_lcia_methods
from pyaesa.asocc.runtime.selection.normalize import normalize_output_mode
from pyaesa.shared.figures.contracts import (
    FIGURE_OUTPUT_FORMAT_SET,
    normalize_figure_output_format as normalize_shared_figure_output_format,
)
from pyaesa.shared.tabular.contracts import normalize_tabular_output_format
from pyaesa.download.mrios.utils.source_registry import is_exio_mrio_source

_SUPPORTED_SOURCES = {
    "exiobase_396_ixi",
    "exiobase_396_pxp",
    "exiobase_3102_ixi",
    "exiobase_3102_pxp",
}


def normalize_supported_source(*, source: str, caller: str) -> str:
    """Validate source support for IO-LCA functions.

    Args:
        source: Source key.
        caller: Calling function label, for example ``"deterministic_io_lca"`` or
            ``"deterministic_io_lca figure generation"``.

    Returns:
        Normalized source key.

    Raises:
        ValueError: If source is missing or unsupported.
    """
    if source is None:
        raise ValueError(f"{caller} requires a non-empty source.")
    value = str(source).strip()
    if not value:
        raise ValueError(f"{caller} requires a non-empty source.")
    if value not in _SUPPORTED_SOURCES or not is_exio_mrio_source(value):
        supported = sorted(_SUPPORTED_SOURCES)
        raise ValueError(
            f"{caller} currently supports only EXIOBASE sources {supported}. Got '{value}'."
        )
    return value


def normalize_lcia_method_list(*, lcia_method: str | list[str]) -> list[str]:
    """Normalize and require non empty LCIA method list."""
    methods = normalize_lcia_methods(lcia_method)
    if not methods:
        raise ValueError("lcia_method must contain at least one method name.")
    return methods


def normalize_aggreg_indices_modes(aggreg_indices: bool) -> list[bool]:
    """Normalize aggreg_indices to one deterministic execution branch."""
    try:
        return [normalize_output_mode(aggreg_indices)]
    except ValueError as exc:
        raise ValueError("aggreg_indices must be a boolean.") from exc


def normalize_grouping(
    *,
    group_reg: bool,
    group_sec: bool,
    group_version: str | None,
) -> tuple[bool, bool, str | None]:
    """Normalize grouping flags and validate group_version contract.

    Args:
        group_reg: Region grouping flag.
        group_sec: Sector grouping flag.
        group_version: Optional grouping version.

    Returns:
        ``(group_reg, group_sec, group_version_clean)``.

    Raises:
        ValueError: If grouping arguments are inconsistent.
    """
    reg = bool(group_reg)
    sec = bool(group_sec)
    grouped = reg or sec
    cleaned = str(group_version).strip() if group_version is not None else ""
    if grouped and not cleaned:
        raise ValueError("group_version must be provided when group_reg or group_sec is True.")
    if (not grouped) and cleaned:
        raise ValueError(
            "group_version was provided but grouping is disabled. "
            "Either enable grouping or omit group_version."
        )
    return reg, sec, (cleaned if grouped else None)


def normalize_io_output_format(output_format: str) -> str:
    """Validate IO-LCA output format selector."""
    return normalize_tabular_output_format(output_format)


def normalize_figure_output_format(output_format: str) -> str:
    """Validate figure output format selector."""
    try:
        return normalize_shared_figure_output_format(output_format)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported figure output_format '{output_format}'. "
            f"Use one of: {sorted(FIGURE_OUTPUT_FORMAT_SET)}."
        ) from exc


def validate_upstream_stages(upstream_stages: int) -> int:
    """Validate stage depth selector used in upstream analysis."""
    try:
        value = int(upstream_stages)
    except (TypeError, ValueError) as exc:
        raise ValueError("upstream_stages must be a positive integer.") from exc
    if value <= 0:
        raise ValueError("upstream_stages must be a positive integer.")
    return value


def validate_dpi(dpi: int) -> int:
    """Validate figure DPI."""
    try:
        value = int(dpi)
    except (TypeError, ValueError) as exc:
        raise ValueError("dpi must be a positive integer.") from exc
    if value <= 0:
        raise ValueError("dpi must be a positive integer.")
    return value

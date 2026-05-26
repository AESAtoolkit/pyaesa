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


def normalize_group_indices_modes(group_indices: bool) -> list[bool]:
    """Normalize group_indices to one deterministic execution branch."""
    try:
        return [normalize_output_mode(group_indices)]
    except ValueError as exc:
        raise ValueError("group_indices must be a boolean.") from exc


def normalize_aggregation(
    *,
    agg_reg: bool,
    agg_sec: bool,
    agg_version: str | None,
) -> tuple[bool, bool, str | None]:
    """Normalize aggregation flags and validate agg_version contract.

    Args:
        agg_reg: Region aggregation flag.
        agg_sec: Sector MRIO aggregation and disaggregation flag.
        agg_version: Optional aggregation version.

    Returns:
        ``(agg_reg, agg_sec, agg_version_clean)``.

    Raises:
        ValueError: If MRIO aggregation and disaggregation arguments are inconsistent.
    """
    reg = bool(agg_reg)
    sec = bool(agg_sec)
    aggregated = reg or sec
    cleaned = str(agg_version).strip() if agg_version is not None else ""
    if aggregated and not cleaned:
        raise ValueError("agg_version must be provided when agg_reg or agg_sec is True.")
    if (not aggregated) and cleaned:
        raise ValueError(
            "agg_version was provided but aggregation is disabled. "
            "Either enable aggregation or omit agg_version."
        )
    return reg, sec, (cleaned if aggregated else None)


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

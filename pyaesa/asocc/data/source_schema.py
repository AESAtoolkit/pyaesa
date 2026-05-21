"""Source key contracts used by allocation data and orchestration layers."""

from pyaesa.download.mrios.utils.source_registry import (
    get_mrio_entry,
    is_exio_mrio_source,
    list_mrio_source_keys,
)

ISO3_SOURCE_KEY = "iso3"
_MRIO_SOURCE_KEYS = set(list_mrio_source_keys())
_SUPPORTED_SOURCES = {ISO3_SOURCE_KEY, *_MRIO_SOURCE_KEYS}


def _normalize_source_key(source_key: str) -> str:
    """Return the normalized source key and validate support."""
    key = str(source_key).strip().lower()
    if key not in _SUPPORTED_SOURCES:
        supported = sorted(_SUPPORTED_SOURCES)
        raise ValueError(f"Unsupported MRIO source '{source_key}'. Supported sources: {supported}")
    return key


def region_code_column_for_source(source_key: str) -> str:
    """Return the canonical region code column for one source."""
    key = _normalize_source_key(source_key)
    if key == ISO3_SOURCE_KEY:
        return "iso3_code"
    return get_mrio_entry(key).region_code_column


def is_exio_source(source_key: str) -> bool:
    """Return whether the MRIO source belongs to the EXIO family."""
    key = _normalize_source_key(source_key)
    return key != ISO3_SOURCE_KEY and is_exio_mrio_source(key)


def is_iso3_source(source_key: str) -> bool:
    """Return whether the source key points to ISO3-only mode."""
    return _normalize_source_key(source_key) == ISO3_SOURCE_KEY


def max_modeled_year_for_source(source_key: str) -> int | None:
    """Return the max modeled year for one source."""
    key = _normalize_source_key(source_key)
    if key == ISO3_SOURCE_KEY:
        return None
    return get_mrio_entry(key).modeled_year_max


def min_modeled_year_for_source(source_key: str) -> int | None:
    """Return the min modeled year for one source."""
    key = _normalize_source_key(source_key)
    if key == ISO3_SOURCE_KEY:
        return None
    return get_mrio_entry(key).modeled_year_min


def default_historical_cutoff_for_source(source_key: str) -> int | None:
    """Return the implicit historical fit/reference cutoff for one source."""
    key = _normalize_source_key(source_key)
    if key == ISO3_SOURCE_KEY:
        return None
    return get_mrio_entry(key).default_historical_cutoff


def default_regression_window_for_source(source_key: str) -> tuple[int, int] | None:
    """Return the default regression window for one source."""
    key = _normalize_source_key(source_key)
    if key == ISO3_SOURCE_KEY:
        return None
    entry = get_mrio_entry(key)
    return entry.modeled_year_min, entry.default_historical_cutoff

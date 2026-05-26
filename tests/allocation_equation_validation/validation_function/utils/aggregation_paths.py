"""Aggregation version helpers for validation utility paths."""

from pyaesa.process.mrios.utils.io.paths import _resolve_version_tag


def normalize_agg_version(value: object) -> str | None:
    """Normalize optional aggregate version values from config or metadata."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_agg_domain_tag(agg_version: object) -> str:
    """Resolve canonical output domain tag from optional aggregate version input."""
    return _resolve_version_tag(normalize_agg_version(agg_version))

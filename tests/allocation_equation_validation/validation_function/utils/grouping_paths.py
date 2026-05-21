"""Grouping version helpers for validation utility paths."""

from pyaesa.process.mrios.utils.io.paths import _resolve_version_tag


def normalize_group_version(value: object) -> str | None:
    """Normalize optional group version values from config or metadata."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_group_domain_tag(group_version: object) -> str:
    """Resolve canonical output domain tag from optional group version input."""
    return _resolve_version_tag(normalize_group_version(group_version))

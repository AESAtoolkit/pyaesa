"""Compatibility payload normalization for uncertainty run reuse."""

from typing import Any

_REPORTING_ONLY_KEYS = {
    "reuse_status",
    "summary_records",
    "summary_infos",
    "summary_warnings",
    "infos",
    "warnings",
}


def strip_reporting_only_fields(value: object) -> Any:
    """Return a compatibility payload without non scientific reporting fields."""
    if isinstance(value, dict):
        return {
            str(key): strip_reporting_only_fields(item)
            for key, item in value.items()
            if str(key) not in _REPORTING_ONLY_KEYS
        }
    if isinstance(value, list | tuple):
        return [strip_reporting_only_fields(item) for item in value]
    return value

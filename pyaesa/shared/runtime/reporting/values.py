"""Neutral value formatting helpers for public runtime reports."""

from collections.abc import Mapping, Sequence
from typing import Any

from pyaesa.shared.runtime.reporting.reuse_status import public_reuse_status
from pyaesa.shared.runtime.reporting.year_ranges import format_year_ranges


def as_sequence(value: object) -> tuple[object, ...]:
    """Return one public value as a tuple without splitting strings."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def format_values(values: Sequence[object]) -> str:
    """Return a readable comma separated public value list."""
    clean = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(clean) if clean else "none"


def format_report_value(value: Any) -> str:
    """Return a readable public representation for one summary value."""
    if isinstance(value, list | tuple):
        return format_values(tuple(value))
    if isinstance(value, Mapping):
        return "; ".join(f"{key}: {format_report_value(item)}" for key, item in value.items())
    return str(value)


def format_summary_value(*, key: str, value: Any) -> str:
    """Return one summary value with canonical handling for reused statuses and years."""
    if key == "reuse_status":
        return public_reuse_status(str(value))
    if is_year_summary_key(key=key):
        return format_year_summary_value(value=value)
    return format_report_value(value)


def is_year_summary_key(*, key: str) -> bool:
    """Return whether one summary key stores a year selector or period."""
    return key == "years" or key == "study_period" or key.endswith("_years")


def format_year_summary_value(*, value: Any) -> str:
    """Return compact public year ranges for summary values."""
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, list | tuple):
        return format_year_ranges([int(item) for item in value])
    return format_report_value(value)


def format_ssp_value(value: object) -> str:
    """Return one SSP value with the public SSP prefix."""
    text = str(value).strip()
    if text.upper().startswith("SSP"):
        return text
    return f"SSP{text}"

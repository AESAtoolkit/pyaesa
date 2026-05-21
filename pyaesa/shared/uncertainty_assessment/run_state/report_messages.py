"""Structured dependency message helpers for uncertainty reports."""

from collections.abc import Sequence
from typing import Any

from pyaesa.shared.runtime.reporting.summary import SummaryInfo, SummaryWarning, info, warning


def payload_infos(*, payload: dict[str, Any]) -> tuple[SummaryInfo, ...]:
    """Return structured information records carried by one dependency payload."""
    return tuple(info(message) for message in payload_messages(payload=payload, severity="INFO"))


def payload_warnings(*, payload: dict[str, Any]) -> tuple[SummaryWarning, ...]:
    """Return structured warning records carried by one dependency payload."""
    return tuple(
        warning(message) for message in payload_messages(payload=payload, severity="WARNING")
    )


def payload_messages(*, payload: dict[str, Any], severity: str) -> tuple[str, ...]:
    """Return unique payload messages matching one severity."""
    messages: list[str] = []
    severity_key = severity.lower()
    keyed_sources = (
        f"summary_{severity_key}s",
        f"{severity_key}s",
        f"{severity_key}_messages",
    )
    for key in keyed_sources:
        messages.extend(_record_messages(payload.get(key)))
    for record in payload.get("summary_records") or ():
        if not isinstance(record, dict):
            continue
        if str(record.get("severity", "")).strip().upper() != severity:
            continue
        messages.extend(_record_messages(record.get("message")))
    return tuple(dict.fromkeys(message for message in messages if message))


def _record_messages(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        return _record_messages(value.get("message"))
    if isinstance(value, Sequence):
        messages: list[str] = []
        for item in value:
            messages.extend(_record_messages(item))
        return messages
    text = str(value).strip()
    return [text] if text else []

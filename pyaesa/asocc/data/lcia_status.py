"""Shared LCIA status parsing."""

from typing import Any


def format_lcia_missing_reason(raw_reason: Any) -> str | None:
    """Normalize LCIA missing reason text from metadata payloads."""
    if raw_reason is None:
        return None
    if isinstance(raw_reason, list):
        items = [str(x).strip() for x in raw_reason if str(x).strip()]
        if not items:
            return None
        if len(items) == 1:
            return f"{items[0]} extension missing"
        return f"{', '.join(items)} extensions missing"
    text = str(raw_reason).strip()
    return text or None


def resolve_lcia_status(
    year_entry: dict[str, Any],
    lcia_method: str,
    *,
    metadata_path: str | None = None,
    year: int | None = None,
) -> tuple[bool, str | None]:
    """Resolve LCIA availability and reason from package-owned `lcia_status` metadata."""
    del metadata_path, year
    lcia_status = year_entry["lcia_status"]
    method_status = lcia_status[lcia_method]
    available = method_status.get("available")
    reason = format_lcia_missing_reason(method_status.get("missing") or method_status.get("reason"))
    if available is False:
        return False, (str(reason) if reason else "LCIA unavailable")
    return True, None

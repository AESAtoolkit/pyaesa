"""Public reuse status labels for reports and phase indexes."""

_PUBLIC_REUSE_STATUS = {
    "computed": "computed",
    "partially_reused": "updated",
    "reused_exact": "reused",
}


def public_reuse_status(status: str) -> str:
    """Return the public status label for one package reuse state."""
    return _PUBLIC_REUSE_STATUS[str(status).strip()]

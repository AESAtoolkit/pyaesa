"""Shared scalar and token formatting helpers for tabular and figure contracts."""

import re
from typing import Any

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")


def canonical_scalar_text(value: Any) -> str:
    """Return a deterministic string representation for one scalar."""
    if pd.isna(value):
        return "missing"
    text = str(value).strip()
    return text or "missing"


def is_display_missing(value: Any) -> bool:
    """Return whether one scalar should be omitted from user-facing display text."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    text = str(value).strip()
    return not text or text.lower() in {"nan", "none", "nat"}


def display_scalar(value: Any) -> str | None:
    """Return one cleaned user-facing scalar string."""
    if is_display_missing(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return str(numeric)
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def sanitize_token(value: Any) -> str:
    """Return one filesystem-safe token."""
    return _TOKEN_RE.sub("_", canonical_scalar_text(value)).strip("._-") or "item"

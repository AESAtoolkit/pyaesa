"""Shared visible scope value helpers for figure planning and stems."""

import pandas as pd

from pyaesa.shared.tabular.scalars import is_display_missing


def visible_scope_values(frame: pd.DataFrame, column: str) -> list[str]:
    """Return sorted nonmissing display values for one figure scope column."""
    if column not in frame.columns:
        return []
    return sorted(
        {
            str(value).strip()
            for value in frame[column].tolist()
            if not is_display_missing(value) and str(value).strip()
        }
    )

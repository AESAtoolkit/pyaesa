"""Shared helpers for removing fully empty tabular rows."""

import pandas as pd


def drop_fully_empty_rows(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without rows whose fields are all empty or missing.

    Args:
        frame: Loaded tabular data.

    Returns:
        Copy of ``frame`` without rows whose values are all missing after blank
        strings are normalized to missing scalars.
    """
    if frame.empty:
        return frame.copy()
    normalized = frame.copy()
    object_columns = list(normalized.select_dtypes(include=["object", "str"]).columns)
    for column in object_columns:
        values = pd.Series(normalized.loc[:, column], copy=False)
        normalized.loc[:, column] = values.map(
            lambda value: (
                pd.NA if value is None or (isinstance(value, str) and not value.strip()) else value
            )
        )
    keep_mask = ~normalized.isna().all(axis=1)
    if bool(keep_mask.all()):
        return frame.copy()
    return frame.loc[keep_mask].reset_index(drop=True)

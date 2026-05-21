"""Shared DataFrame operations for orchestration stages."""

import pandas as pd


def coalesce_unique_non_null(
    values: pd.Series,
    *,
    conflict_context: str,
) -> object:
    """Return one unique non null value or raise on conflicting duplicates."""
    non_null = values.dropna()
    if non_null.empty:
        return pd.NA
    unique = pd.unique(non_null)
    if len(unique) > 1:
        raise ValueError(
            f"Conflicting duplicate values for {conflict_context}. sample={unique.tolist()[:5]}"
        )
    return unique[0]

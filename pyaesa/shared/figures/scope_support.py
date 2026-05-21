"""Shared grouping helpers for ASR figure tables."""

import pandas as pd


def impact_column(frame: pd.DataFrame) -> str | None:
    """Return the impact like column available in a table."""
    for column in ("impact", "variable"):
        if column in frame.columns:
            return column
    return None


def repeat_generic_impacts(
    frame: pd.DataFrame,
    *,
    impact_name: str | None,
) -> pd.DataFrame:
    """Repeat rows without impact identity into each visible impact group."""
    if impact_name is None or impact_name not in frame.columns or frame.empty:
        return frame.copy()
    impact_series = pd.Series(frame.loc[:, impact_name], copy=False)
    visible_impacts = sorted(
        {str(value) for value in impact_series.dropna().tolist() if str(value).strip()}
    )
    if not visible_impacts:
        return frame.copy()
    generic = frame.loc[impact_series.isna()].copy()
    if generic.empty:
        return frame.copy()
    specific = frame.loc[impact_series.notna()].copy()
    expanded = [specific]
    for impact in visible_impacts:
        repeated = generic.copy()
        repeated.loc[:, impact_name] = impact
        expanded.append(repeated)
    return pd.concat(expanded, ignore_index=True)

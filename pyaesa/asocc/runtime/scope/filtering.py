"""Shared filter normalization and axis slicing."""

import pandas as pd


def normalize_filter_values(values: list[str] | None) -> set[str] | None:
    """Normalize optional filter values to a string set."""
    if not values:
        return None
    # Sets give O(1) membership checks during repeated axis filtering.
    return {str(value) for value in values}


def _mask_for_allowed(labels: pd.Index, allowed: set[str]):
    """Build membership mask for normalized textual labels."""
    return labels.isin(allowed)


def slice_frame_any_axis(
    frame: pd.DataFrame,
    *,
    axis_name: str,
    allowed: set[str] | None,
) -> pd.DataFrame:
    """Slice a DataFrame by axis name on index and columns."""
    if not allowed:
        return frame
    out = frame
    # Apply the same axis filter to index and columns so this logic works for
    # matrices where the target axis can appear on either side.
    if isinstance(out.index, pd.MultiIndex):
        idx_names = [str(name) for name in out.index.names]
        if axis_name in idx_names:
            mask = _mask_for_allowed(
                out.index.get_level_values(axis_name),
                allowed,
            )
            out = out.loc[mask]
    elif out.index.name is not None and str(out.index.name) == axis_name:
        mask = _mask_for_allowed(out.index, allowed)
        out = out.loc[mask]
    if isinstance(out.columns, pd.MultiIndex):
        col_names = [str(name) for name in out.columns.names]
        if axis_name in col_names:
            level = col_names.index(axis_name)
            mask = _mask_for_allowed(
                out.columns.get_level_values(level),
                allowed,
            )
            out = out.loc[:, mask]
    elif out.columns.name is not None and str(out.columns.name) == axis_name:
        mask = _mask_for_allowed(out.columns, allowed)
        out = out.loc[:, mask]
    return out


def slice_series_any_axis(
    series: pd.Series,
    *,
    axis_name: str,
    allowed: set[str] | None,
) -> pd.Series:
    """Slice a Series by axis name on index."""
    if not allowed:
        return series
    if isinstance(series.index, pd.MultiIndex):
        idx_names = [str(name) for name in series.index.names]
        if axis_name not in idx_names:
            return series
        mask = _mask_for_allowed(
            series.index.get_level_values(axis_name),
            allowed,
        )
        return series.loc[mask]
    if series.index.name is not None and str(series.index.name) == axis_name:
        mask = _mask_for_allowed(series.index, allowed)
        return series.loc[mask]
    return series

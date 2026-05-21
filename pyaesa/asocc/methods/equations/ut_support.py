"""Shared UT allocation support."""

from typing import cast

import numpy as np
import pandas as pd

from .share_math import safe_divide_frame


def _safe_divide_frame(numer: pd.DataFrame, denom: pd.Series) -> pd.DataFrame:
    """Divide a DataFrame by a Series with zero safe handling."""
    return safe_divide_frame(numer, denom, axis=1)


def _safe_divide_series(numer: pd.Series, denom: float | pd.Series) -> pd.Series:
    """Divide a Series by a scalar or Series with zero safe handling.

    Args:
        numer: Series numerator.
        denom: Scalar or Series denominator.

    Returns:
        Series with undefined divisions preserved as NaN.
    """
    if isinstance(denom, pd.Series):
        denom_series = pd.Series(denom, copy=False)
        zero_denominator = denom_series.eq(0)
        denom_safe = denom_series.where(~zero_denominator, np.nan)
        out = numer.div(denom_safe)
        zero_zero_mask = zero_denominator.reindex(out.index).fillna(False) & numer.eq(0)
        if bool(zero_zero_mask.any()):
            out = out.mask(zero_zero_mask, 0.0)
        return out
    if denom == 0:
        if bool(numer.eq(0).all()):
            return pd.Series(0.0, index=numer.index, dtype="float64")
        return pd.Series(np.nan, index=numer.index, dtype="float64")
    return numer / denom


def _stack_frame_to_series(
    frame: pd.DataFrame,
) -> pd.Series:
    """Stack a DataFrame to a Series."""
    return cast(pd.Series, frame.stack(future_stack=True))


def _get_x_vec(x_to_rc: pd.DataFrame) -> pd.Series:
    """Return total output by producer from x_to_rc.

    Args:
        x_to_rc: Producer by destination total demand matrix.

    Returns:
        Series of total output by producer.
    """
    return x_to_rc.sum(axis=1)


def _stack_to_year(
    weights: pd.DataFrame,
    year: int,
    col_name: str,
) -> pd.DataFrame:
    """Stack columns into index with a named level and add year column."""
    stacked_series = _stack_frame_to_series(weights)
    stacked_series.index = stacked_series.index.set_names([*weights.index.names, col_name])
    return pd.DataFrame({int(year): stacked_series})

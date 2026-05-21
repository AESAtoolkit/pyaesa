"""Shared ratio and share computations."""

from typing import Literal

import numpy as np
import pandas as pd

AxisLiteral = Literal[0, 1, "index", "columns"]


def _coerce_float_series(values: pd.Series) -> pd.Series:
    """Return one float Series with invalid entries coerced to NaN."""
    numeric = pd.Series(
        pd.to_numeric(values, errors="coerce"),
        index=values.index,
        copy=False,
    )
    return numeric.astype(float)


def safe_divide_frame(
    numer: pd.DataFrame,
    denom: pd.Series | pd.DataFrame,
    *,
    axis: AxisLiteral,
    level: str | None = None,
) -> pd.DataFrame:
    """Safely divide DataFrame values while preserving undefined values.

    This function is used by equation modules to keep division behavior
    consistent (zero denominator -> NA, inf -> NA).
    """
    denom_safe = denom.replace(0, pd.NA)
    out = numer.div(denom_safe, axis=axis, level=level)
    return out.replace([float("inf"), -float("inf")], pd.NA)


def safe_divide_series(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Safely divide a Series by another Series, preserving undefined values.

    Mirrors ``safe_divide_frame`` but for Series inputs.
    """
    if numer.index.equals(denom.index):
        index = numer.index
        numer_values = numer.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
        denom_values = denom.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    else:
        aligned_numer, aligned_denom = numer.align(denom)
        index = aligned_numer.index
        numer_values = aligned_numer.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
        denom_values = aligned_denom.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    values = np.full(len(index), np.nan, dtype=np.float64)
    valid = ~np.isnan(numer_values) & ~np.isnan(denom_values) & (denom_values != 0.0)
    np.divide(numer_values, denom_values, out=values, where=valid)
    values[~np.isfinite(values)] = np.nan
    return pd.Series(values, index=index, name=numer.name)


def normalize_share(values: pd.Series) -> pd.Series:
    """Normalize non missing values to sum to one while preserving missing entries.

    Behavior:
    - all NA input -> unchanged shape, float dtype
    - non NA sum == 0 -> explicit 0.0 on valid entries
    - otherwise -> divide by valid entry sum
    """
    valid = values.dropna()
    if valid.empty:
        return _coerce_float_series(values)
    total = valid.sum()
    if total == 0:
        out = _coerce_float_series(values).copy()
        out.loc[valid.index] = 0.0
        return out
    return values / total

"""Internal tabular ownership for processed pop/gdp datasets."""

import math
from typing import Any, Sequence, cast

import pandas as pd


def coerce_finite_float(value: Any) -> float | None:
    """Return ``value`` as finite float when possible, otherwise ``None``."""
    numeric = cast(pd.Series, pd.to_numeric(pd.Series([value]), errors="coerce"))
    candidate = numeric.iloc[0]
    if pd.isna(candidate):
        return None
    coerced = float(candidate)
    if not math.isfinite(coerced):
        return None
    return coerced


def requested_year_columns(years: Sequence[int]) -> list[str]:
    """Return canonical string year columns for one processing request."""
    return [str(year) for year in years]


def ensure_requested_year_columns(df: pd.DataFrame, year_cols: Sequence[str]) -> pd.DataFrame:
    """Return a copy with every requested year column present."""
    out = df.copy()
    for col in year_cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def wide_year_column_positions(df: pd.DataFrame, year_cols: Sequence[str]) -> dict[str, int]:
    """Return validated column positions for one wide year block."""
    positions: dict[str, int] = {}
    for col in year_cols:
        positions[col] = cast(int, df.columns.get_loc(col))
    return positions


def row_year_series(row: pd.Series, year_cols: Sequence[str]) -> pd.Series:
    """Return one wide row as a numeric series indexed by integer years."""
    year_values = {int(year_cols[i]): row[year_cols[i]] for i in range(len(year_cols))}
    return cast(pd.Series, pd.to_numeric(pd.Series(year_values), errors="raise"))


def write_row_year_series(
    df: pd.DataFrame,
    *,
    row_idx: int,
    year_cols: Sequence[str],
    year_ints: Sequence[int],
    series: pd.Series,
    column_positions: dict[str, int],
) -> None:
    """Write one integer-indexed year series back into a wide row in place."""
    for index, year_int in enumerate(year_ints):
        col = year_cols[index]
        df.iat[row_idx, column_positions[col]] = series.at[year_int]

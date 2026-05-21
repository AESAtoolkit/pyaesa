"""Explorer CSV transforms and raw data readers for AR6 downloads."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent


@dataclass
class ExplorerData:
    """Wide AR6 explorer table saved on disk."""

    data: pd.DataFrame


def drop_non_persisted_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop explorer columns that are outside the persisted raw-download contract."""
    out_df = frame.copy()
    if "Category_definition" in out_df.columns:
        out_df = out_df.drop(columns=["Category_definition"])
    return out_df


def _numeric_year_series(data_df: pd.DataFrame) -> pd.Series:
    """Return the explorer year column as a numeric pandas Series."""
    return pd.Series(pd.to_numeric(data_df["year"], errors="raise"), copy=False)


def to_wide_with_meta(data_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Return the wide explorer CSV merged with scenario metadata."""
    numeric_years = _numeric_year_series(data_df)
    years_order = sorted(numeric_years.dropna().astype(int).unique().tolist())
    wide_df = (
        data_df.assign(year=numeric_years.astype(int))
        .pivot_table(
            index=["model", "scenario", "variable", "unit", "region"],
            columns="year",
            values="value",
            aggfunc="first",
        )
        .reindex(columns=years_order)
        .reset_index()
    )
    wide_df.columns = [str(col) for col in wide_df.columns]
    meta_reset = meta_df.reset_index()
    return drop_non_persisted_columns(
        wide_df.merge(meta_reset, on=["model", "scenario"], how="left")
    )


def write_explorer_csv(*, csv_file: Path, data_df: pd.DataFrame, meta_df: pd.DataFrame) -> None:
    """Persist the wide explorer CSV."""
    csv_file = ensure_file_parent(csv_file)
    to_wide_with_meta(data_df=data_df, meta_df=meta_df).to_csv(csv_file, index=False)


def read_explorer_csv(csv_file: Path) -> ExplorerData:
    """Load a saved explorer CSV."""
    return ExplorerData(data=drop_non_persisted_columns(pd.read_csv(csv_file, low_memory=False)))

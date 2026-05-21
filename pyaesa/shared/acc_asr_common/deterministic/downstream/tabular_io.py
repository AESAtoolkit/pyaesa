"""Shared tabular I/O for deterministic ACC and ASR outputs."""

from pathlib import Path

import pandas as pd
from pyaesa.shared.tabular.contracts import normalize_tabular_output_format
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.tabular.wide_tables import (
    detect_year_columns as _detect_year_columns,
    id_columns as _id_columns,
    requested_year_columns as _requested_year_columns,
)


def normalize_downstream_output_format(output_format: str) -> str:
    """Validate one persisted tabular output format for downstream families."""
    return normalize_tabular_output_format(output_format)


def detect_year_columns(df: pd.DataFrame) -> list[str]:
    """Detect year-like column names in a deterministic output table."""
    return _detect_year_columns(df)


def requested_year_columns(
    df: pd.DataFrame,
    *,
    requested_years: list[int],
) -> list[str]:
    """Return requested year columns present in one deterministic table."""
    return _requested_year_columns(df, requested_years=requested_years)


def detect_id_columns(df: pd.DataFrame, year_cols: list[str]) -> list[str]:
    """Detect identifier columns in one deterministic output table."""
    return _id_columns(df, year_columns=year_cols)


def write_output_table(
    *,
    df: pd.DataFrame,
    output_path: Path,
    output_format: str,
) -> None:
    """Write one deterministic output table in the requested persisted format."""
    output_path = ensure_file_parent(output_path)
    output_format = normalize_downstream_output_format(output_format)
    if output_format == "csv":
        df.to_csv(output_path, index=False)
        return
    if output_format == "parquet":
        df.to_parquet(output_path, index=False)
        return
    df.to_pickle(output_path)

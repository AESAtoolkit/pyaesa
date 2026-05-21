"""Vectorized tabular readers for external input files."""

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from pyaesa.shared.tabular.empty_rows import drop_fully_empty_rows

_PC = cast(Any, pc)


def tabular_columns(path: Path) -> list[str]:
    """Return external file columns without loading full CSV or Parquet data."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        reader = pacsv.open_csv(path)
        try:
            return list(reader.schema.names)
        finally:
            reader.close()
    if suffix == ".parquet":
        return list(pq.ParquetFile(path).schema_arrow.names)
    frame = cast(pd.DataFrame, pd.read_pickle(path))
    return [str(column) for column in drop_fully_empty_rows(frame=frame).columns]


def year_columns_from_schema(path: Path) -> list[int]:
    """Return deterministic wide year columns from an external file schema."""
    return [int(column) for column in year_column_names(tabular_columns(path))]


def year_column_names(columns: list[str]) -> list[str]:
    """Return year-like column names from an external file schema."""
    years: list[str] = []
    for column in columns:
        try:
            year = int(column)
        except (TypeError, ValueError):
            continue
        if 1900 < year < 2200:
            years.append(str(column))
    return sorted(years, key=int)


def read_projected_table(path: Path, columns: list[str] | None = None) -> pa.Table:
    """Read one external table with an optional column projection."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        options = None if columns is None else pacsv.ConvertOptions(include_columns=columns)
        return pacsv.read_csv(path, convert_options=options)
    if suffix == ".parquet":
        return pq.read_table(path, columns=columns)
    frame = drop_fully_empty_rows(frame=cast(pd.DataFrame, pd.read_pickle(path)))
    if columns is not None:
        frame = frame.loc[:, columns]
    return pa.Table.from_pandas(frame, preserve_index=False)


def arrow_wide_to_long(
    *,
    table: pa.Table,
    identity_columns: list[str],
    requested_years: list[int],
    value_column: str = "value",
) -> pa.Table:
    """Unpivot requested wide year columns into canonical long Arrow rows."""
    parts = []
    for year in requested_years:
        year_column = str(int(year))
        values = pc.cast(table[year_column], pa.float64())
        if bool(_PC.any(_PC.is_null(values)).as_py()):
            raise ValueError(
                f"External wide input year column '{year_column}' contains missing values."
            )
        count = table.num_rows
        parts.append(
            pa.table(
                {
                    **{column: table[column] for column in identity_columns},
                    "year": pa.repeat(int(year), count).cast(pa.int64()),
                    value_column: values,
                }
            )
        )
    return pa.concat_tables(parts, promote_options="default")


def arrow_table_to_pandas(table: pa.Table) -> pd.DataFrame:
    """Convert the final selected external table to pandas without an index column."""
    return cast(pd.DataFrame, table.to_pandas(ignore_metadata=True))


def none_for_missing_series(series: pd.Series) -> pd.Series:
    """Return a pandas object series where missing values are canonical None."""
    out = series.astype(object)
    out.loc[pd.isna(out)] = None
    return pd.Series(out, index=series.index, dtype=object)

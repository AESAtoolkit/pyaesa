"""Complete uncertainty table IO."""

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq

from pyaesa.shared.runtime.io.filesystem import write_via_atomic_temp
from pyaesa.shared.uncertainty_assessment.io.csv_fragments import csv_run_fragment_input
from pyaesa.shared.uncertainty_assessment.io.formats import (
    is_csv_compact_output,
    normalize_uncertainty_output_format,
)
from pyaesa.shared.uncertainty_assessment.io.run_artifacts import first_run_fragment_path

UNCERTAINTY_CSV_FLOAT_FORMAT = "%.17g"


def write_uncertainty_table(*, path: Path, frame: pd.DataFrame, output_format: str) -> Path:
    """Write one complete uncertainty table file."""
    table_path = Path(path)
    fmt = normalize_uncertainty_output_format(output_format)
    out = _normalize_integer_identity_columns(frame=frame)
    if is_csv_compact_output(fmt):
        return write_via_atomic_temp(
            table_path,
            writer=lambda tmp_path: out.to_csv(
                tmp_path,
                index=False,
                float_format=UNCERTAINTY_CSV_FLOAT_FORMAT,
            ),
        )
    return write_via_atomic_temp(
        table_path,
        writer=lambda tmp_path: out.to_parquet(tmp_path, index=False),
    )


def read_uncertainty_table(
    *,
    path: Path,
    output_format: str,
    csv_dtypes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read one complete uncertainty table file."""
    table_path = Path(path)
    if is_csv_compact_output(output_format):
        return pd.read_csv(
            table_path,
            dtype=cast(Any, csv_dtypes),
            float_precision="round_trip",
            low_memory=False,
        )
    return pd.read_parquet(table_path)


def uncertainty_table_columns(*, path: Path, output_format: str) -> list[str]:
    """Return columns for one persisted uncertainty table."""
    table_path = Path(path)
    if is_csv_compact_output(output_format):
        if table_path.is_dir():
            table_path = first_run_fragment_path(path=table_path, output_format=output_format)
        with csv_run_fragment_input(path=table_path) as source:
            return list(pd.read_csv(source, nrows=0).columns)
    if table_path.is_dir():
        table_path = first_run_fragment_path(path=table_path, output_format=output_format)
    return list(pq.ParquetFile(table_path).schema_arrow.names)


def _normalize_integer_identity_columns(*, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy(deep=False)
    for column in out.columns:
        if _is_integer_identity_column(str(column)):
            series = pd.Series(out.loc[:, column], copy=False)
            numeric = pd.Series(pd.to_numeric(series, errors="raise"), index=out.index)
            out[column] = numeric.astype("Int64")
    return out


def _is_integer_identity_column(column: str) -> bool:
    return column == "year" or column.endswith("_year")

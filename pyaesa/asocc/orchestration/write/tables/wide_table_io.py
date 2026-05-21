"""Table readers and writers for deterministic wide output tables."""

import csv
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent

from ....runtime.output.contracts import IdentifierSchema
from pyaesa.asocc.orchestration.write.tables.wide_validation import (
    normalize_existing_wide_table,
    validate_wide_frame,
)

_CSV_WRITE_ROWS_PER_CHUNK = 50_000


def _format_csv_float_block(values: np.ndarray) -> np.ndarray:
    """Return CSV text values for one numeric year block."""
    out = np.empty(values.shape, dtype=object)
    missing = np.isnan(values)
    out[missing] = ""
    present = ~missing
    present_values = values[present]
    out[present] = np.fromiter(
        (format(float(value), ".12g") for value in present_values),
        dtype=object,
        count=int(present_values.size),
    )
    return out


def _write_csv_table(path: Path, frame: pd.DataFrame) -> None:
    """Write one public wide CSV table without pandas CSV serialization."""
    columns = [str(column) for column in frame.columns]
    year_positions = [idx for idx, column in enumerate(columns) if column.isdigit()]
    id_positions = [idx for idx, column in enumerate(columns) if not column.isdigit()]
    row_count = len(frame)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for start in range(0, row_count, _CSV_WRITE_ROWS_PER_CHUNK):
            stop = min(start + _CSV_WRITE_ROWS_PER_CHUNK, row_count)
            id_values = frame.iloc[start:stop, id_positions].to_numpy(dtype=object, copy=True)
            id_values[pd.isna(id_values)] = ""
            year_values = (
                frame.iloc[start:stop, year_positions].astype("float64").to_numpy(copy=False)
            )
            year_text = _format_csv_float_block(year_values)
            writer.writerows(
                (*identifier_row, *year_row)
                for identifier_row, year_row in zip(id_values, year_text, strict=True)
            )


def _write_parquet_table(path: Path, frame: pd.DataFrame) -> None:
    """Write one public wide parquet table directly from column arrays."""
    arrays: list[pa.Array] = []
    columns: list[str] = []
    for column in frame.columns:
        name = str(column)
        columns.append(name)
        if name.isdigit():
            values = frame[column].astype("float64").to_numpy(copy=False)
            arrays.append(pa.array(values, type=pa.float64(), from_pandas=True))
        else:
            values = frame[column].to_numpy(dtype=object, copy=True)
            arrays.append(pa.array(values, from_pandas=True))
    pq.write_table(pa.Table.from_arrays(arrays, names=columns), path)


def _read_wide_table(*, path: Path, output_format: str) -> pd.DataFrame:
    if output_format == "csv":
        return pd.read_csv(path)
    if output_format == "pickle":
        return cast(pd.DataFrame, pd.read_pickle(path))
    return cast(pd.DataFrame, pd.read_parquet(path))


def _write_table(*, path: Path, frame: pd.DataFrame, output_format: str) -> None:
    if output_format == "csv":
        _write_csv_table(path, frame)
        return
    if output_format == "pickle":
        frame.to_pickle(path)
        return
    _write_parquet_table(path, frame)


def upsert_wide_table(
    *,
    path: Path,
    frame: pd.DataFrame,
    schema: IdentifierSchema,
    refresh: bool,
    output_format: str = "csv",
) -> bool:
    """Persist one strict wide batch into a public output table."""
    batch_wide = cast(pd.DataFrame, validate_wide_frame(frame, schema))
    if batch_wide.empty:
        return False

    path = ensure_file_parent(path)
    if not path.exists() or refresh:
        _write_table(path=path, frame=batch_wide, output_format=output_format)
        return True

    existing = normalize_existing_wide_table(
        _read_wide_table(path=path, output_format=output_format)
    )

    existing_years = [str(c) for c in existing.columns if str(c).isdigit()]
    batch_years = [str(c) for c in batch_wide.columns if str(c).isdigit()]
    existing_id_cols = [str(c) for c in existing.columns if not str(c).isdigit()]
    batch_id_cols = [str(c) for c in batch_wide.columns if not str(c).isdigit()]
    all_id_cols = list(dict.fromkeys([*batch_id_cols, *existing_id_cols]))
    existing = existing.copy()
    batch_wide = batch_wide.copy()
    for column in all_id_cols:
        if column not in existing.columns:
            existing[column] = None
        if column not in batch_wide.columns:
            batch_wide[column] = None
    existing = cast(pd.DataFrame, existing[all_id_cols + existing_years])
    batch_wide = cast(pd.DataFrame, batch_wide[all_id_cols + batch_years])
    all_years = sorted(set(existing_years) | set(batch_years), key=int)
    existing_idx = existing.set_index(all_id_cols)
    batch_idx = batch_wide.set_index(all_id_cols)
    existing_aligned = existing_idx.reindex(columns=all_years)
    batch_aligned = batch_idx.reindex(columns=all_years)
    new_idx = batch_aligned.index.difference(existing_aligned.index)
    batch_ids_already_present = all(column in existing_id_cols for column in batch_id_cols)
    batch_years_already_present = all(column in existing_years for column in batch_years)
    if (
        batch_ids_already_present
        and batch_years_already_present
        and len(new_idx) == 0
        and np.array_equal(
            existing_aligned.loc[batch_aligned.index, batch_years].to_numpy(),
            batch_aligned.loc[:, batch_years].to_numpy(),
            equal_nan=True,
        )
    ):
        return False

    common = existing_aligned.index.intersection(batch_aligned.index)
    if len(common) > 0 and batch_years:
        batch_update = batch_aligned.loc[common, batch_years].copy()
        for col in batch_years:
            batch_update[col] = pd.to_numeric(batch_update[col], errors="raise")
        existing_aligned.loc[common, batch_years] = batch_update
    if len(new_idx) > 0:
        merged_idx = cast(
            pd.DataFrame,
            pd.concat([existing_aligned, batch_aligned.loc[new_idx]], axis=0),
        )
    else:
        merged_idx = existing_aligned

    merged_wide = cast(pd.DataFrame, merged_idx.reset_index())
    merged_wide = cast(pd.DataFrame, merged_wide[all_id_cols + all_years])
    _write_table(path=path, frame=merged_wide, output_format=output_format)
    return True

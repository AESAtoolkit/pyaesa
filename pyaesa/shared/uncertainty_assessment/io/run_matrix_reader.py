"""Chunked readers for public uncertainty run matrix artifacts."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pyaesa.shared.uncertainty_assessment.io.formats import is_csv_compact_output
from pyaesa.shared.uncertainty_assessment.io.summary_kernels import (
    summary_scan_max_numeric_cells,
)
from pyaesa.shared.uncertainty_assessment.io.run_writers import (
    SparseRunRows,
    sparse_run_row_numeric_bytes_per_row,
)
from pyaesa.shared.uncertainty_assessment.io.csv_fragments import csv_run_fragment_input
from pyaesa.shared.uncertainty_assessment.io.run_artifacts import read_run_interval_index
from pyaesa.shared.uncertainty_assessment.io.tables import uncertainty_table_columns
from pyaesa.shared.runtime.memory import memory_bounded_rows


def iter_compact_run_matrix(
    *,
    path: Path,
    output_format: str,
    column_count: int,
    start_run_index: int = 0,
    stop_run_index: int | None = None,
    max_rows_per_chunk: int | None = None,
):
    """Yield compact run matrix chunks as run indices and numeric values."""
    column_names = [str(index) for index in range(int(column_count))]
    yield from iter_compact_run_matrix_columns(
        path=path,
        output_format=output_format,
        column_names=column_names,
        start_run_index=start_run_index,
        stop_run_index=stop_run_index,
        max_rows_per_chunk=max_rows_per_chunk,
    )


def iter_compact_run_matrix_columns(
    *,
    path: Path,
    output_format: str,
    column_names: list[str],
    start_run_index: int = 0,
    stop_run_index: int | None = None,
    max_rows_per_chunk: int | None = None,
):
    """Yield selected compact run matrix columns as run indices and values."""
    batch_rows = max(
        1,
        summary_scan_max_numeric_cells(output_format=output_format) // max(1, len(column_names)),
    )
    if max_rows_per_chunk is not None:
        batch_rows = min(batch_rows, max(1, int(max_rows_per_chunk)))
    intervals = _run_intervals_in_range(
        path=path,
        output_format=output_format,
        start_run_index=start_run_index,
        stop_run_index=stop_run_index,
    )
    if is_csv_compact_output(output_format):
        chunks = _iter_csv_interval_frames(
            path=path,
            intervals=intervals,
            chunksize=batch_rows,
            usecols=["run_index", *column_names],
        )
        for chunk in chunks:
            result = _compact_chunk_in_range(
                run_index=chunk.loc[:, "run_index"].to_numpy(dtype=np.int64),
                values=chunk.loc[:, column_names].to_numpy(dtype=np.float64),
                start_run_index=start_run_index,
                stop_run_index=stop_run_index,
            )
            if result is not None:
                yield result
        return
    batches = _iter_parquet_interval_batches(
        path=path,
        intervals=intervals,
        batch_size=batch_rows,
        columns=["run_index", *column_names],
    )
    for batch in batches:
        result = _compact_chunk_in_range(
            run_index=batch.column("run_index")
            .to_numpy(zero_copy_only=False)
            .astype(
                np.int64,
                copy=False,
            ),
            values=np.column_stack(
                [batch.column(column).to_numpy(zero_copy_only=False) for column in column_names]
            ).astype(np.float64, copy=False),
            start_run_index=start_run_index,
            stop_run_index=stop_run_index,
        )
        if result is not None:
            yield result


def iter_sparse_run_rows(
    *,
    path: Path,
    output_format: str,
    start_run_index: int = 0,
    stop_run_index: int | None = None,
    max_rows_per_chunk: int | None = None,
):
    """Yield sparse run row chunks without splitting one run across chunks."""
    value_column = str(uncertainty_table_columns(path=path, output_format=output_format)[2])
    intervals = _run_intervals_in_range(
        path=path,
        output_format=output_format,
        start_run_index=start_run_index,
        stop_run_index=stop_run_index,
    )
    empty_rows = SparseRunRows(
        run_index=np.empty(0, dtype=np.int64),
        public_row_id=np.empty(0, dtype=np.int64),
        values=np.empty(0, dtype=np.float64),
        value_column=value_column,
    )
    pending = empty_rows
    if is_csv_compact_output(output_format):
        chunks = _iter_csv_sparse_interval_chunks(
            path=path,
            intervals=intervals,
            value_column=value_column,
            max_rows_per_chunk=max_rows_per_chunk,
        )
    else:
        chunks = _iter_parquet_sparse_interval_chunks(
            path=path,
            output_format=output_format,
            intervals=intervals,
            value_column=value_column,
            max_rows_per_chunk=max_rows_per_chunk,
        )
    for chunk in chunks:
        work = (
            _concat_sparse_rows(pieces=[pending, chunk], empty_rows=empty_rows)
            if pending.run_index.size
            else chunk
        )
        last_run = int(work.run_index[-1])
        ready = _mask_sparse_rows(rows=work, mask=work.run_index != last_run)
        pending = _mask_sparse_rows(rows=work, mask=work.run_index == last_run)
        if ready.run_index.size:
            rows = _sparse_rows_in_range(
                rows=ready,
                start_run_index=start_run_index,
                stop_run_index=stop_run_index,
            )
            if rows is not None:
                yield rows
    if pending.run_index.size:
        rows = _sparse_rows_in_range(
            rows=pending,
            start_run_index=start_run_index,
            stop_run_index=stop_run_index,
        )
        if rows is not None:
            yield rows


def _run_intervals_in_range(
    *,
    path: Path,
    output_format: str,
    start_run_index: int,
    stop_run_index: int | None,
) -> pd.DataFrame:
    intervals = read_run_interval_index(path=path, output_format=output_format)
    mask = intervals["run_stop"].to_numpy(dtype=np.int64) > int(start_run_index)
    if stop_run_index is not None:
        mask &= intervals["run_start"].to_numpy(dtype=np.int64) < int(stop_run_index)
    return intervals.loc[mask].sort_values(["run_start", "row_start"]).reset_index(drop=True)


def _iter_csv_sparse_interval_chunks(
    *,
    path: Path,
    intervals: pd.DataFrame,
    value_column: str,
    max_rows_per_chunk: int | None,
):
    batch_rows = _bounded_sparse_read_batch_rows(max_rows_per_chunk=max_rows_per_chunk)
    for frame in _iter_csv_interval_frames(
        path=path,
        intervals=intervals,
        chunksize=batch_rows,
    ):
        yield _sparse_rows_from_frame(
            frame=frame,
            value_column=value_column,
        )


def _iter_parquet_sparse_interval_chunks(
    *,
    path: Path,
    output_format: str,
    intervals: pd.DataFrame,
    value_column: str,
    max_rows_per_chunk: int | None,
):
    batch_rows = _bounded_sparse_read_batch_rows(max_rows_per_chunk=max_rows_per_chunk)
    for batch in _iter_parquet_interval_batches(
        path=path,
        intervals=intervals,
        batch_size=batch_rows,
    ):
        yield SparseRunRows(
            run_index=batch.column("run_index")
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False),
            public_row_id=batch.column("public_row_id")
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False),
            values=batch.column(value_column)
            .to_numpy(zero_copy_only=False)
            .astype(np.float64, copy=False),
            value_column=value_column,
        )


def sparse_run_row_read_batch_rows() -> int:
    """Return the dynamic row budget for sparse run row scans."""
    numeric_row_bytes = sparse_run_row_numeric_bytes_per_row()
    row_bytes = numeric_row_bytes * len(
        ("reader_columns", "sparse_rows", "pending_rows")
    ) + np.dtype(np.bool_).itemsize * len(("ready_mask", "pending_mask", "range_mask"))
    return memory_bounded_rows(
        bytes_per_row=row_bytes,
    )


def sparse_run_rows_per_run_window(
    *,
    path: Path,
    output_format: str,
    batch_size: int,
) -> int:
    """Return the row cap for reading one sparse run window plus its boundary run."""
    return max(
        1,
        sparse_run_rows_per_run(path=path, output_format=output_format) * (int(batch_size) + 1),
    )


def sparse_run_rows_per_run(
    *,
    path: Path,
    output_format: str,
) -> int:
    """Return the largest emitted sparse row count for one completed run."""
    intervals = read_run_interval_index(path=path, output_format=output_format)
    if intervals.empty:
        return 1
    run_counts = intervals["run_stop"].to_numpy(dtype=np.int64) - intervals["run_start"].to_numpy(
        dtype=np.int64
    )
    row_counts = intervals["row_count"].to_numpy(dtype=np.int64)
    return max(1, int(np.ceil(np.max(row_counts / np.maximum(run_counts, 1)))))


def _bounded_sparse_read_batch_rows(*, max_rows_per_chunk: int | None) -> int:
    rows = sparse_run_row_read_batch_rows()
    if max_rows_per_chunk is None:
        return rows
    return min(rows, max(1, int(max_rows_per_chunk)))


def _iter_csv_interval_frames(
    *,
    path: Path,
    intervals: pd.DataFrame,
    chunksize: int,
    usecols: list[str] | None = None,
):
    for fragment_path in _interval_fragment_paths(
        path=path,
        intervals=intervals,
        output_format="csv_compact",
    ):
        with csv_run_fragment_input(path=fragment_path) as source:
            yield from pd.read_csv(
                source,
                usecols=usecols,
                chunksize=int(chunksize),
                float_precision="round_trip",
            )


def _iter_parquet_interval_batches(
    *,
    path: Path,
    intervals: pd.DataFrame,
    batch_size: int,
    columns: list[str] | None = None,
):
    for fragment_path in _interval_fragment_paths(
        path=path,
        intervals=intervals,
        output_format="parquet",
    ):
        parquet = pq.ParquetFile(fragment_path)
        yield from parquet.iter_batches(batch_size=int(batch_size), columns=columns)


def _interval_fragment_paths(
    *,
    path: Path,
    intervals: pd.DataFrame,
    output_format: str,
):
    del output_format
    for fragment in intervals["fragment"].astype(str):
        yield Path(path) / fragment


def iter_sparse_run_row_windows(
    *,
    chunks: Iterator[SparseRunRows],
    start_run_index: int,
    stop_run_index: int,
    batch_size: int,
    empty_rows: SparseRunRows,
) -> Iterator[tuple[np.ndarray, SparseRunRows]]:
    """Yield ordered run windows from one selected run row iterator."""
    current = int(start_run_index)
    stop_at = int(stop_run_index)
    pending = empty_rows
    while current < stop_at:
        window_stop = min(stop_at, current + int(batch_size))
        pending, rows = _collect_sparse_rows_for_window(
            pending=pending,
            chunks=chunks,
            start=current,
            stop=window_stop,
            empty_rows=empty_rows,
        )
        yield np.arange(current, window_stop, dtype=np.int64), rows
        current = window_stop


def _compact_chunk_in_range(
    *,
    run_index: np.ndarray,
    values: np.ndarray,
    start_run_index: int,
    stop_run_index: int | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    mask = run_index >= int(start_run_index)
    if stop_run_index is not None:
        mask &= run_index < int(stop_run_index)
    if not np.any(mask):
        return None
    return run_index[mask], values[mask]


def _sparse_rows_in_range(
    *,
    rows: SparseRunRows,
    start_run_index: int,
    stop_run_index: int | None,
) -> SparseRunRows | None:
    mask = rows.run_index >= int(start_run_index)
    if stop_run_index is not None:
        mask &= rows.run_index < int(stop_run_index)
    if not np.any(mask):
        return None
    return SparseRunRows(
        run_index=rows.run_index[mask],
        public_row_id=rows.public_row_id[mask],
        values=rows.values[mask],
        value_column=rows.value_column,
    )


def _sparse_rows_from_frame(
    *,
    frame: pd.DataFrame,
    value_column: str,
) -> SparseRunRows:
    return SparseRunRows(
        run_index=frame["run_index"].to_numpy(dtype=np.int64),
        public_row_id=frame["public_row_id"].to_numpy(dtype=np.int64),
        values=frame[value_column].to_numpy(dtype=np.float64),
        value_column=value_column,
    )


def _collect_sparse_rows_for_window(
    *,
    pending: SparseRunRows,
    chunks: Iterator[SparseRunRows],
    start: int,
    stop: int,
    empty_rows: SparseRunRows,
) -> tuple[SparseRunRows, SparseRunRows]:
    """Split ordered run rows into the current run range and later rows."""
    pieces: list[SparseRunRows] = []
    current = pending
    while True:
        inside_mask = (current.run_index >= int(start)) & (current.run_index < int(stop))
        after_mask = current.run_index >= int(stop)
        inside = _mask_sparse_rows(rows=current, mask=inside_mask)
        after = _mask_sparse_rows(rows=current, mask=after_mask)
        if inside.run_index.size:
            pieces.append(inside)
        if after.run_index.size:
            return after, _concat_sparse_rows(pieces=pieces, empty_rows=empty_rows)
        try:
            current = next(chunks)
        except StopIteration:
            return empty_rows, _concat_sparse_rows(pieces=pieces, empty_rows=empty_rows)


def _mask_sparse_rows(*, rows: SparseRunRows, mask: np.ndarray) -> SparseRunRows:
    return SparseRunRows(
        run_index=rows.run_index[mask],
        public_row_id=rows.public_row_id[mask],
        values=rows.values[mask],
        value_column=rows.value_column,
    )


def _concat_sparse_rows(*, pieces: list[SparseRunRows], empty_rows: SparseRunRows) -> SparseRunRows:
    non_empty = [piece for piece in pieces if piece.run_index.size]
    if not non_empty:
        return empty_rows
    return SparseRunRows(
        run_index=np.concatenate([piece.run_index for piece in non_empty]),
        public_row_id=np.concatenate([piece.public_row_id for piece in non_empty]),
        values=np.concatenate([piece.values for piece in non_empty]),
        value_column=non_empty[0].value_column,
    )

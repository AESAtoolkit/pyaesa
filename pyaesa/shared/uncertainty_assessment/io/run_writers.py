"""Writers for generated uncertainty Monte Carlo run artifacts."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.runtime.memory import memory_bounded_rows
from pyaesa.shared.uncertainty_assessment.io.csv_fragments import (
    compact_csv_text_bytes,
    csv_render_working_arrays,
    sparse_csv_text_bytes,
    sparse_csv_text_bytes_from_bounds,
    write_csv_run_fragment_table,
)
from pyaesa.shared.uncertainty_assessment.io.formats import is_csv_compact_output
from pyaesa.shared.uncertainty_assessment.io.run_artifacts import RunIntervalWriterState


@dataclass(frozen=True)
class SparseRunRows:
    """Sparse selected uncertainty run values keyed by public row id."""

    run_index: np.ndarray
    public_row_id: np.ndarray
    values: np.ndarray
    value_column: str


class CompactRunMatrixWriter:
    """Write one compact uncertainty run matrix.

    CSV compact and Parquet outputs are dataset directories containing
    immutable ``part-*`` fragments plus a mandatory interval index sidecar.
    """

    def __init__(
        self,
        *,
        path: Path,
        output_format: str,
        append_existing: bool = False,
        memory_budget_bytes: int | None = None,
    ) -> None:
        self._state = RunIntervalWriterState.create(
            path=path,
            output_format=output_format,
            append_existing=append_existing,
        )
        self._memory_budget_bytes = memory_budget_bytes

    def __enter__(self) -> "CompactRunMatrixWriter":
        """Return this writer for context manager use."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Flush the interval index when leaving a context manager."""
        self.close()

    def write_batch(
        self,
        *,
        run_indices,
        values,
        batch_index: int,
    ) -> Path:
        """Write one compact run value batch as one or more fragments."""
        run_index = np.asarray(run_indices, dtype=np.int64)
        matrix = np.asarray(values, dtype=np.float64)
        rows_per_fragment = _compact_write_fragment_rows(
            output_format=self._state.output_format,
            column_count=matrix.shape[1],
            run_index=run_index,
            memory_budget_bytes=self._memory_budget_bytes,
        )
        for start, stop in _row_slices(row_count=len(run_index), rows_per_slice=rows_per_fragment):
            self._write_fragment(
                run_index=run_index[start:stop],
                matrix=matrix[start:stop, :],
                batch_index=batch_index,
            )
        return self._state.path

    def close(self) -> None:
        """Write pending interval metadata."""
        self._state.close()

    def _write_fragment(
        self,
        *,
        run_index: np.ndarray,
        matrix: np.ndarray,
        batch_index: int,
    ) -> None:
        """Write one compact fragment and record its run interval."""
        table = compact_run_matrix_arrow_table(run_index=run_index, matrix=matrix)
        _write_run_table_fragment(
            state=self._state,
            table=table,
            batch_index=batch_index,
            run_index=run_index,
            row_count=table.num_rows,
        )


class SparseRunRowsWriter:
    """Write sparse selected uncertainty run rows.

    CSV compact and Parquet outputs are dataset directories containing
    immutable ``part-*`` fragments plus a mandatory interval index sidecar.
    """

    def __init__(
        self,
        *,
        path: Path,
        output_format: str,
        append_existing: bool = False,
        memory_budget_bytes: int | None = None,
    ) -> None:
        self._state = RunIntervalWriterState.create(
            path=path,
            output_format=output_format,
            append_existing=append_existing,
        )
        self._memory_budget_bytes = memory_budget_bytes

    def __enter__(self) -> "SparseRunRowsWriter":
        """Return this writer for context manager use."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Flush the interval index when leaving a context manager."""
        self.close()

    def write_batch(self, *, rows: SparseRunRows, batch_index: int) -> Path:
        """Write one sparse selected row batch as one or more fragments."""
        run_index = np.asarray(rows.run_index, dtype=np.int64)
        public_row_id = np.asarray(rows.public_row_id, dtype=np.int64)
        values = np.asarray(rows.values, dtype=np.float64)
        if len(values) == 0:
            self._write_empty_batch(rows=rows)
            return self._state.path
        rows_per_fragment = _sparse_write_fragment_rows(
            output_format=self._state.output_format,
            run_index=run_index,
            public_row_id=public_row_id,
            memory_budget_bytes=self._memory_budget_bytes,
        )
        for start, stop in _row_slices(row_count=len(values), rows_per_slice=rows_per_fragment):
            fragment_rows = SparseRunRows(
                run_index=run_index[start:stop],
                public_row_id=public_row_id[start:stop],
                values=values[start:stop],
                value_column=rows.value_column,
            )
            self._write_fragment(rows=fragment_rows, batch_index=batch_index)
        return self._state.path

    def close(self) -> None:
        """Write pending interval metadata."""
        self._state.close()

    def _write_empty_batch(self, *, rows: SparseRunRows) -> None:
        """Create an empty sparse artifact when no selected rows are emitted."""
        self._state.prepare()
        if self._state.next_fragment_index == 0:
            _write_run_table_fragment(
                state=self._state,
                table=sparse_render_rows_arrow_table(rows=rows),
                batch_index=0,
                run_index=rows.run_index,
                row_count=0,
            )
        self._state.dirty = True

    def _write_fragment(self, *, rows: SparseRunRows, batch_index: int) -> None:
        """Write one sparse fragment and record its run interval."""
        table = sparse_render_rows_arrow_table(rows=rows)
        _write_run_table_fragment(
            state=self._state,
            table=table,
            batch_index=batch_index,
            run_index=rows.run_index,
            row_count=len(rows.values),
        )


def compact_run_matrix_arrow_table(*, run_index: np.ndarray, matrix: np.ndarray) -> pa.Table:
    """Return an Arrow table for a dense run by public row matrix."""
    arrays = [pa.array(run_index, type=pa.int64())]
    arrays.extend(pa.array(matrix[:, index], type=pa.float64()) for index in range(matrix.shape[1]))
    names = ["run_index", *(str(index) for index in range(matrix.shape[1]))]
    return pa.table(arrays, names=names)


def sparse_render_rows_arrow_table(*, rows: SparseRunRows) -> pa.Table:
    """Return an Arrow table for sparse selected run rows."""
    return pa.table(
        [
            pa.array(np.asarray(rows.run_index, dtype=np.int64), type=pa.int64()),
            pa.array(np.asarray(rows.public_row_id, dtype=np.int64), type=pa.int64()),
            pa.array(np.asarray(rows.values, dtype=np.float64), type=pa.float64()),
        ],
        names=["run_index", "public_row_id", rows.value_column],
    )


def sparse_run_row_numeric_bytes_per_row() -> int:
    """Return numeric bytes stored by one sparse selected row."""
    return (
        np.dtype(np.int64).itemsize * len(("run_index", "public_row_id"))
        + np.dtype(np.float64).itemsize
    )


def sparse_run_row_csv_render_working_bytes_per_row(
    *,
    max_run_index: int,
    max_public_row_id: int,
) -> int:
    """Return sparse CSV render bytes per row for memory planning."""
    row_bytes = sparse_run_row_numeric_bytes_per_row()
    row_bytes += sparse_csv_text_bytes_from_bounds(
        max_run_index=max_run_index,
        max_public_row_id=max_public_row_id,
    )
    return row_bytes * csv_render_working_arrays()


def _write_run_table_fragment(
    *,
    state: RunIntervalWriterState,
    table: pa.Table,
    batch_index: int,
    run_index: np.ndarray,
    row_count: int,
) -> None:
    """Write one dataset fragment and record its interval sidecar row."""
    state.prepare()
    fragment = state.next_fragment()
    fragment_path = ensure_file_parent(state.path / fragment)
    if is_csv_compact_output(state.output_format):
        write_csv_run_fragment_table(path=fragment_path, table=table)
    else:
        pq.write_table(table, fragment_path)
    state.record_interval(
        batch_index=batch_index,
        run_index=run_index,
        row_count=row_count,
        fragment=fragment,
    )


def _compact_write_fragment_rows(
    *,
    output_format: str,
    column_count: int,
    run_index: np.ndarray,
    memory_budget_bytes: int | None,
) -> int:
    """Return the compact fragment row cap for the active memory budget."""
    row_bytes = _compact_run_matrix_numeric_bytes_per_row(column_count=column_count)
    working_arrays = 1
    if is_csv_compact_output(output_format):
        row_bytes += compact_csv_text_bytes(column_count=column_count, run_index=run_index)
        working_arrays = csv_render_working_arrays()
    return memory_bounded_rows(
        bytes_per_row=row_bytes,
        working_arrays=working_arrays,
        memory_budget_bytes=memory_budget_bytes,
    )


def _sparse_write_fragment_rows(
    *,
    output_format: str,
    run_index: np.ndarray,
    public_row_id: np.ndarray,
    memory_budget_bytes: int | None,
) -> int:
    """Return the sparse fragment row cap for the active memory budget."""
    row_bytes = sparse_run_row_numeric_bytes_per_row()
    working_arrays = 1
    if is_csv_compact_output(output_format):
        row_bytes += sparse_csv_text_bytes(run_index=run_index, public_row_id=public_row_id)
        working_arrays = csv_render_working_arrays()
    return memory_bounded_rows(
        bytes_per_row=row_bytes,
        working_arrays=working_arrays,
        memory_budget_bytes=memory_budget_bytes,
    )


def _compact_run_matrix_numeric_bytes_per_row(*, column_count: int) -> int:
    """Return numeric bytes stored by one compact matrix row."""
    return np.dtype(np.int64).itemsize + np.dtype(np.float64).itemsize * int(column_count)


def _row_slices(*, row_count: int, rows_per_slice: int) -> Iterator[tuple[int, int]]:
    """Yield contiguous row slices for fragment writing."""
    rows = int(row_count)
    step = max(1, int(rows_per_slice))
    for start in range(0, rows, step):
        yield start, min(start + step, rows)

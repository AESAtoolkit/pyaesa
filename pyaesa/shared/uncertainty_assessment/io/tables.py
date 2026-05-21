"""Uncertainty tables and public Monte Carlo run artifacts."""

from dataclasses import dataclass
from pathlib import Path
import shutil
from types import TracebackType
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent, write_via_atomic_temp
from pyaesa.shared.uncertainty_assessment.io.formats import (
    is_csv_compact_output,
    normalize_uncertainty_output_format,
    suffix_for_uncertainty_output,
)

UNCERTAINTY_CSV_FLOAT_FORMAT = "%.12g"
SPARSE_CSV_WRITE_CHUNK_ROWS = 1_000_000
RUN_INTERVAL_INDEX_STEM_SUFFIX = ".run_intervals"
RUN_INTERVAL_COLUMNS = (
    "batch_index",
    "run_start",
    "run_stop",
    "row_start",
    "row_count",
    "fragment",
)


@dataclass(frozen=True)
class SparseRunRows:
    """Sparse selected uncertainty run values keyed by public row id."""

    run_index: np.ndarray
    public_row_id: np.ndarray
    values: np.ndarray
    value_column: str


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
        return pd.read_csv(table_path, dtype=cast(Any, csv_dtypes))
    return pd.read_parquet(table_path)


def uncertainty_table_columns(*, path: Path, output_format: str) -> list[str]:
    """Return columns for one persisted uncertainty table."""
    table_path = Path(path)
    if is_csv_compact_output(output_format):
        return list(pd.read_csv(table_path, nrows=0).columns)
    if table_path.is_dir():
        table_path = sorted(table_path.glob("part-*.parquet"))[0]
    return list(pq.ParquetFile(table_path).schema_arrow.names)


def run_interval_index_path(*, path: Path, output_format: str) -> Path:
    """Return the public run interval index path for a run artifact."""
    table_path = Path(path)
    suffix = suffix_for_uncertainty_output(output_format)
    return table_path.with_name(f"{table_path.stem}{RUN_INTERVAL_INDEX_STEM_SUFFIX}{suffix}")


def public_run_artifact_contract(*, path: Path, output_format: str) -> dict[str, str]:
    """Return metadata for one public Monte Carlo run artifact."""
    table_path = Path(path)
    if is_csv_compact_output(output_format):
        return {
            "artifact_kind": "csv_compact_file",
            "path": str(table_path),
            "interval_index_path": str(
                run_interval_index_path(path=table_path, output_format=output_format)
            ),
            "interval_index_kind": "csv_file",
        }
    return {
        "artifact_kind": "parquet_dataset_directory",
        "path": str(table_path),
        "fragment_pattern": "part-*.parquet",
        "interval_index_path": str(
            run_interval_index_path(path=table_path, output_format=output_format)
        ),
        "interval_index_kind": "parquet_file",
    }


def public_run_artifact_readme_lines(*, run_name: str) -> list[str]:
    """Return README lines for the shared public run artifact contract."""
    return [
        f"- {run_name}: public Monte Carlo run values.",
        "  For csv_compact output this is one CSV file.",
        "  For Parquet output this path is a dataset directory containing",
        "  part-*.parquet fragments.",
        f"- {run_name}.run_intervals.<suffix>: interval index for run windows",
        "  and row ranges.",
        "  The manifest records the exact interval index path.",
    ]


def read_run_interval_index(*, path: Path, output_format: str) -> pd.DataFrame:
    """Read the public run interval index for a run artifact."""
    index_path = run_interval_index_path(path=path, output_format=output_format)
    if is_csv_compact_output(output_format):
        frame = pd.read_csv(index_path)
    else:
        frame = pd.read_parquet(index_path)
    return _normalize_run_interval_index(frame=frame, path=index_path)


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


@dataclass
class _RunIntervalWriterState:
    path: Path
    output_format: str
    append_existing: bool
    intervals: list[dict[str, Any]]
    next_row_offset: int
    next_fragment_index: int
    prepared: bool = False
    dirty: bool = False

    @classmethod
    def create(
        cls,
        *,
        path: Path,
        output_format: str,
        append_existing: bool,
    ) -> "_RunIntervalWriterState":
        table_path = Path(path)
        fmt = normalize_uncertainty_output_format(output_format)
        intervals = _existing_run_intervals(
            path=table_path,
            output_format=fmt,
            append_existing=bool(append_existing),
        )
        return cls(
            path=table_path,
            output_format=fmt,
            append_existing=bool(append_existing),
            intervals=intervals,
            next_row_offset=_next_interval_row_offset(intervals=intervals),
            next_fragment_index=len(intervals),
        )

    def close(self) -> None:
        if self.dirty:
            _write_run_interval_index(
                path=self.path,
                output_format=self.output_format,
                intervals=self.intervals,
            )
            self.dirty = False

    def prepare(self) -> None:
        if self.prepared:
            return
        _prepare_run_artifact(
            path=self.path,
            output_format=self.output_format,
            append_existing=self.append_existing,
        )
        self.prepared = True

    def next_fragment(self) -> str:
        fragment = _parquet_fragment_name(index=self.next_fragment_index)
        self.next_fragment_index += 1
        return fragment

    def record_interval(
        self,
        *,
        batch_index: int,
        run_index: np.ndarray,
        row_count: int,
        fragment: str,
    ) -> None:
        if row_count == 0 or len(run_index) == 0:
            return
        self.intervals.append(
            {
                "batch_index": int(batch_index),
                "run_start": int(run_index[0]),
                "run_stop": int(run_index[-1]) + 1,
                "row_start": int(self.next_row_offset),
                "row_count": int(row_count),
                "fragment": str(fragment),
            }
        )
        self.next_row_offset += int(row_count)
        self.dirty = True


class CompactRunMatrixWriter:
    """Write one compact uncertainty run matrix.

    CSV compact output is one CSV file. Parquet output is a dataset directory
    containing immutable ``part-*.parquet`` fragments plus a mandatory interval
    index sidecar.
    """

    def __init__(
        self,
        *,
        path: Path,
        output_format: str,
        append_existing: bool = False,
    ) -> None:
        self._state = _RunIntervalWriterState.create(
            path=path,
            output_format=output_format,
            append_existing=append_existing,
        )

    def __enter__(self) -> "CompactRunMatrixWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def write_batch(
        self,
        *,
        run_indices,
        values,
        batch_index: int,
    ) -> Path:
        """Write one run by public row numeric matrix batch."""
        run_index = np.asarray(run_indices, dtype=np.int64)
        matrix = np.asarray(values, dtype=np.float64)
        if is_csv_compact_output(self._state.output_format):
            self._write_csv_batch(
                run_index=run_index,
                matrix=matrix,
                batch_index=batch_index,
            )
        else:
            table = compact_run_matrix_arrow_table(run_index=run_index, matrix=matrix)
            self._write_parquet_batch(table=table, batch_index=batch_index)
        return self._state.path

    def close(self) -> None:
        """Close transient matrix writer resources."""
        self._state.close()

    def _write_csv_batch(
        self,
        *,
        run_index: np.ndarray,
        matrix: np.ndarray,
        batch_index: int,
    ) -> None:
        path = ensure_file_parent(self._state.path)
        self._state.prepare()
        append_to_existing = path.exists()
        mode = "a" if append_to_existing else "w"
        with path.open(mode, encoding="utf-8", newline="") as handle:
            if not append_to_existing and batch_index == 0:
                handle.write(_compact_matrix_header(column_count=matrix.shape[1]))
            out = np.column_stack((run_index, matrix))
            np.savetxt(
                handle,
                out,
                delimiter=",",
                fmt=["%d", *([UNCERTAINTY_CSV_FLOAT_FORMAT] * matrix.shape[1])],
            )
        self._state.record_interval(
            batch_index=batch_index,
            run_index=run_index,
            row_count=len(run_index),
            fragment="",
        )

    def _write_parquet_batch(self, *, table: pa.Table, batch_index: int) -> None:
        self._state.prepare()
        fragment = self._state.next_fragment()
        pq.write_table(table, self._state.path / fragment)
        self._state.record_interval(
            batch_index=batch_index,
            run_index=table.column("run_index").to_numpy(zero_copy_only=False),
            row_count=table.num_rows,
            fragment=fragment,
        )


class SparseRunRowsWriter:
    """Write sparse selected uncertainty run rows.

    CSV compact output is one CSV file. Parquet output is a dataset directory
    containing immutable ``part-*.parquet`` fragments plus a mandatory interval
    index sidecar.
    """

    def __init__(
        self,
        *,
        path: Path,
        output_format: str,
        append_existing: bool = False,
    ) -> None:
        self._state = _RunIntervalWriterState.create(
            path=path,
            output_format=output_format,
            append_existing=append_existing,
        )

    def __enter__(self) -> "SparseRunRowsWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def write_batch(self, *, rows: SparseRunRows, batch_index: int) -> Path:
        """Write one sparse selected run row batch."""
        if is_csv_compact_output(self._state.output_format):
            self._write_csv_batch(rows=rows, batch_index=batch_index)
        else:
            self._write_parquet_batch(
                table=sparse_render_rows_arrow_table(rows=rows),
                batch_index=batch_index,
            )
        return self._state.path

    def close(self) -> None:
        """Close transient sparse writer resources."""
        self._state.close()

    def _write_csv_batch(self, *, rows: SparseRunRows, batch_index: int) -> None:
        path = ensure_file_parent(self._state.path)
        self._state.prepare()
        append_to_existing = path.exists()
        # "wb" creates the binary CSV; "ab" appends later Arrow CSV batches.
        mode = "ab" if append_to_existing else "wb"
        with path.open(mode) as handle:
            if not append_to_existing and batch_index == 0:
                handle.write(f"run_index,public_row_id,{rows.value_column}\n".encode("utf-8"))
            run_index = np.asarray(rows.run_index, dtype=np.int64)
            public_row_id = np.asarray(rows.public_row_id, dtype=np.int64)
            values = np.asarray(rows.values, dtype=np.float64)
            # Inter-method batches can contain millions of selected rows.
            # Chunking keeps the temporary three column numeric block bounded.
            for start in range(0, len(values), SPARSE_CSV_WRITE_CHUNK_ROWS):
                stop = min(start + SPARSE_CSV_WRITE_CHUNK_ROWS, len(values))
                table = pa.table(
                    {
                        "run_index": pa.array(run_index[start:stop]),
                        "public_row_id": pa.array(public_row_id[start:stop]),
                        rows.value_column: pa.array(values[start:stop]),
                    }
                )
                pacsv.write_csv(
                    table,
                    handle,
                    write_options=pacsv.WriteOptions(include_header=False),
                )
        if len(rows.values) == 0:
            self._state.dirty = True
        self._state.record_interval(
            batch_index=batch_index,
            run_index=rows.run_index,
            row_count=len(rows.values),
            fragment="",
        )

    def _write_parquet_batch(self, *, table: pa.Table, batch_index: int) -> None:
        self._state.prepare()
        if table.num_rows == 0:
            if self._state.next_fragment_index == 0:
                pq.write_table(table, self._state.path / self._state.next_fragment())
            self._state.dirty = True
            return
        fragment = self._state.next_fragment()
        pq.write_table(table, self._state.path / fragment)
        self._state.record_interval(
            batch_index=batch_index,
            run_index=table.column("run_index").to_numpy(zero_copy_only=False),
            row_count=table.num_rows,
            fragment=fragment,
        )


def compact_run_matrix_arrow_table(*, run_index: np.ndarray, matrix: np.ndarray) -> pa.Table:
    """Return one Arrow table for a compact numeric run matrix."""
    arrays = [pa.array(run_index, type=pa.int64())]
    arrays.extend(pa.array(matrix[:, index], type=pa.float64()) for index in range(matrix.shape[1]))
    names = ["run_index", *(str(index) for index in range(matrix.shape[1]))]
    return pa.table(arrays, names=names)


def sparse_render_rows_arrow_table(*, rows: SparseRunRows) -> pa.Table:
    """Return one Arrow table for sparse selected run rows."""
    return pa.table(
        [
            pa.array(np.asarray(rows.run_index, dtype=np.int64), type=pa.int64()),
            pa.array(np.asarray(rows.public_row_id, dtype=np.int64), type=pa.int64()),
            pa.array(np.asarray(rows.values, dtype=np.float64), type=pa.float64()),
        ],
        names=["run_index", "public_row_id", rows.value_column],
    )


def _compact_matrix_header(*, column_count: int) -> str:
    columns = ["run_index", *(str(index) for index in range(column_count))]
    return ",".join(columns) + "\n"


def _existing_run_intervals(
    *,
    path: Path,
    output_format: str,
    append_existing: bool,
) -> list[dict[str, Any]]:
    if not append_existing or not path.exists():
        return []
    frame = read_run_interval_index(path=path, output_format=output_format)
    return [
        {
            "batch_index": int(record["batch_index"]),
            "run_start": int(record["run_start"]),
            "run_stop": int(record["run_stop"]),
            "row_start": int(record["row_start"]),
            "row_count": int(record["row_count"]),
            "fragment": str(record["fragment"]),
        }
        for record in cast(list[dict[str, Any]], frame.to_dict("records"))
    ]


def _next_interval_row_offset(*, intervals: list[dict[str, Any]]) -> int:
    if not intervals:
        return 0
    last = intervals[-1]
    return int(last["row_start"]) + int(last["row_count"])


def _prepare_run_artifact(*, path: Path, output_format: str, append_existing: bool) -> None:
    ensure_file_parent(path)
    if append_existing:
        if path.exists() and not is_csv_compact_output(output_format) and not path.is_dir():
            raise ValueError(
                f"Parquet run artifact '{path}' must be a fragment directory for append."
            )
        if not is_csv_compact_output(output_format):
            path.mkdir(parents=True, exist_ok=True)
        return
    index_path = run_interval_index_path(path=path, output_format=output_format)
    if index_path.exists():
        index_path.unlink()
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    if not is_csv_compact_output(output_format):
        path.mkdir(parents=True, exist_ok=True)


def _write_run_interval_index(
    *,
    path: Path,
    output_format: str,
    intervals: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame.from_records(intervals, columns=RUN_INTERVAL_COLUMNS)
    index_path = run_interval_index_path(path=path, output_format=output_format)
    if is_csv_compact_output(output_format):
        write_via_atomic_temp(
            index_path,
            writer=lambda tmp_path: frame.to_csv(tmp_path, index=False),
        )
        return
    write_via_atomic_temp(
        index_path,
        writer=lambda tmp_path: frame.to_parquet(tmp_path, index=False),
    )


def _normalize_run_interval_index(*, frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = [column for column in RUN_INTERVAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Run interval index '{path}' is missing columns: {missing}.")
    out = frame.loc[:, list(RUN_INTERVAL_COLUMNS)].copy()
    for column in ("batch_index", "run_start", "run_stop", "row_start", "row_count"):
        numeric = pd.Series(pd.to_numeric(out[column], errors="raise"), index=out.index)
        out[column] = numeric.astype("int64")
    out["fragment"] = out["fragment"].fillna("").astype(str)
    return out


def _parquet_fragment_name(*, index: int) -> str:
    return f"part-{int(index):08d}.parquet"

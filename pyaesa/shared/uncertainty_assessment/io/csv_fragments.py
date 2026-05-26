"""CSV fragment streams and render byte estimates for uncertainty run artifacts."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

CSV_COMPACT_RUN_FRAGMENT_COMPRESSION = "zstd"
CSV_COMPACT_RUN_FRAGMENT_SUFFIX = ".csv.zst"


def write_csv_run_fragment_table(*, path: Path, table: pa.Table) -> None:
    """Write one compressed CSV fragment from an Arrow table."""
    with pa.CompressedOutputStream(str(path), CSV_COMPACT_RUN_FRAGMENT_COMPRESSION) as handle:
        pacsv.write_csv(
            table,
            handle,
            write_options=pacsv.WriteOptions(include_header=True),
        )


@contextmanager
def csv_run_fragment_input(*, path: Path) -> Iterator[Any]:
    """Yield a readable stream for one generated CSV fragment."""
    fragment_path = Path(path)
    if str(fragment_path).endswith(CSV_COMPACT_RUN_FRAGMENT_SUFFIX):
        with pa.OSFile(str(fragment_path), "rb") as raw:
            with pa.CompressedInputStream(raw, CSV_COMPACT_RUN_FRAGMENT_COMPRESSION) as source:
                yield source
        return
    with fragment_path.open("rb") as source:
        yield source


def compact_csv_text_bytes(*, column_count: int, run_index: np.ndarray) -> int:
    """Return a per row CSV text byte estimate for compact run matrices."""
    return (
        integer_text_bytes(max_abs_value=max_abs_int(values=run_index))
        + csv_float_text_bytes() * int(column_count)
        + len(",") * int(column_count)
        + len("\n")
    )


def sparse_csv_text_bytes(*, run_index: np.ndarray, public_row_id: np.ndarray) -> int:
    """Return a per row CSV text byte estimate for sparse selected rows."""
    return sparse_csv_text_bytes_from_bounds(
        max_run_index=max_abs_int(values=run_index),
        max_public_row_id=max_abs_int(values=public_row_id),
    )


def sparse_csv_text_bytes_from_bounds(*, max_run_index: int, max_public_row_id: int) -> int:
    """Return sparse CSV text bytes from integer value bounds."""
    return (
        integer_text_bytes(max_abs_value=max_run_index)
        + integer_text_bytes(max_abs_value=max_public_row_id)
        + csv_float_text_bytes()
        + len(",,\n")
    )


def csv_render_working_arrays() -> int:
    """Return the Arrow CSV render working array multiplier."""
    return len(
        (
            "arrow_numeric_buffers",
            "formatted_value_buffers",
            "encoded_output_buffers",
            "record_batch_writer_buffers",
            "filesystem_transfer_buffers",
            "arrow_allocator_scratch",
        )
    )


def max_abs_int(*, values: np.ndarray) -> int:
    """Return the maximum absolute integer value in an array."""
    if values.size == 0:
        return 0
    return int(np.max(np.abs(values)))


def integer_text_bytes(*, max_abs_value: int) -> int:
    """Return decimal text bytes needed for one integer magnitude."""
    return len(str(int(max_abs_value)))


def csv_float_text_bytes() -> int:
    """Return the widest Python float text representation used in CSV sizing."""
    sample_values = (
        0.0,
        -0.0,
        np.finfo(np.float64).max,
        -np.finfo(np.float64).max,
        np.finfo(np.float64).tiny,
        -np.finfo(np.float64).tiny,
        np.nan,
        np.inf,
        -np.inf,
    )
    return max(len(str(float(value))) for value in sample_values)

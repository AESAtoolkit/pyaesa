"""Sparse summary replay for public uncertainty run artifacts."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from pyaesa.shared.runtime.memory import memory_bounded_rows
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    sparse_group_run_means,
    sparse_public_row_group_membership_index,
    sparse_rows_to_overlapping_group_values,
)
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import iter_sparse_run_rows
from pyaesa.shared.uncertainty_assessment.io.summary_kernels import (
    observed_group_summary_arrays,
)

SPARSE_SUMMARY_BUCKET_RECORD_DTYPE = np.dtype([("group", np.int64), ("value", np.float64)])


def sparse_grouped_summary_scan(
    *,
    identity_frame: pd.DataFrame,
    runs_path: Path,
    output_format: str,
    run_count: int,
    public_row_groups: tuple[tuple[str, ...], ...],
    include_frequency: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build summary statistics for sparse public run artifacts."""
    with tempfile.TemporaryDirectory(prefix="pyaesa_sparse_summary_") as temp_root:
        bucket_paths, bucket_group_width = _write_sparse_group_value_buckets(
            path=runs_path,
            output_format=output_format,
            public_row_groups=public_row_groups,
            run_count=run_count,
            temp_root=Path(temp_root),
        )
        stats, frequency = observed_group_summary_arrays(
            group_count=len(public_row_groups),
            value_groups=_iter_sparse_group_bucket_values(
                bucket_paths=bucket_paths,
                bucket_group_width=bucket_group_width,
            ),
            frequency_threshold=1.0 if include_frequency else None,
        )
    summary = identity_frame.reset_index(drop=True).copy()
    for statistic, data in stats.items():
        summary[statistic] = data
    return summary, frequency


def _write_sparse_group_value_buckets(
    *,
    path: Path,
    output_format: str,
    public_row_groups: tuple[tuple[str, ...], ...],
    run_count: int,
    temp_root: Path,
) -> tuple[list[Path], int]:
    """Write temporary observed group mean buckets for a sparse summary scan."""
    group_index = sparse_public_row_group_membership_index(public_row_groups=public_row_groups)
    group_count = len(public_row_groups)
    bucket_group_width = _sparse_summary_bucket_group_width(
        run_count=run_count,
        group_count=group_count,
    )
    source_rows_per_chunk = _sparse_summary_source_rows_per_chunk(
        public_row_group_index=group_index,
    )
    bucket_paths = [
        temp_root / f"sparse_summary_bucket_{index:05d}.bin"
        for index in range(_ceil_div(group_count, bucket_group_width))
    ]
    for rows in iter_sparse_run_rows(
        path=path,
        output_format=output_format,
        max_rows_per_chunk=source_rows_per_chunk,
    ):
        run_indices = np.arange(
            int(rows.run_index[0]),
            int(rows.run_index[-1]) + 1,
            dtype=np.int64,
        )
        row_runs, row_groups, values = sparse_rows_to_overlapping_group_values(
            sparse_rows=rows,
            run_indices=run_indices,
            public_row_group_index=group_index,
        )
        groups, means = sparse_group_run_means(
            row_runs=row_runs,
            row_groups=row_groups,
            values=values,
            group_count=group_count,
        )
        _append_sparse_group_bucket_records(
            bucket_paths=bucket_paths,
            bucket_group_width=bucket_group_width,
            groups=groups,
            values=means,
        )
    return bucket_paths, bucket_group_width


def _sparse_summary_bucket_group_width(*, run_count: int, group_count: int) -> int:
    """Return the number of public groups stored in one temporary bucket."""
    max_records = memory_bounded_rows(
        bytes_per_row=_sparse_summary_bucket_working_bytes_per_record(),
    )
    return max(1, min(int(group_count), int(max_records) // max(1, int(run_count))))


def _sparse_summary_bucket_working_bytes_per_record() -> int:
    """Return the per record memory estimate for reading and sorting buckets."""
    integer_bytes = np.dtype(np.int64).itemsize * len(
        (
            "record_group",
            "sort_order",
            "sort_workspace",
            "sorted_group",
            "unique_group",
            "unique_start",
            "unique_count",
        )
    )
    float_bytes = np.dtype(np.float64).itemsize * len(
        ("record_value", "group_value", "observed_value", "quantile_workspace")
    )
    bool_bytes = np.dtype(np.bool_).itemsize * len(("observed_mask",))
    return integer_bytes + float_bytes + bool_bytes


def _sparse_summary_source_rows_per_chunk(*, public_row_group_index: np.ndarray) -> int:
    """Return the sparse source row chunk size for group expansion."""
    _public_row_ids, membership_counts = np.unique(
        public_row_group_index[:, 0],
        return_counts=True,
    )
    return memory_bounded_rows(
        bytes_per_row=_sparse_summary_source_row_working_bytes(
            max_memberships_per_row=int(np.max(membership_counts)),
        ),
    )


def _sparse_summary_source_row_working_bytes(*, max_memberships_per_row: int) -> int:
    """Return the per source row memory estimate for sparse group expansion."""
    integer_bytes = np.dtype(np.int64).itemsize
    float_bytes = np.dtype(np.float64).itemsize
    bool_bytes = np.dtype(np.bool_).itemsize
    source_bytes = integer_bytes * len(
        (
            "arrow_run_index",
            "arrow_public_row_id",
            "run_index",
            "public_row_id",
            "work_run_index",
            "work_public_row_id",
            "range_run_index",
            "range_public_row_id",
            "pending_run_index",
            "pending_public_row_id",
            "membership_start",
            "membership_count",
            "membership_count_cumsum",
            "membership_count_offset",
            "source_index",
            "source_cumsum",
        )
    ) + float_bytes * len(
        (
            "arrow_value",
            "source_value",
            "work_value",
            "range_value",
            "pending_value",
        )
    )
    source_bytes += bool_bytes * len(("ready_mask", "pending_mask", "range_mask"))
    expanded_bytes = integer_bytes * len(
        (
            "source_position",
            "source_position_input",
            "membership_start_repeat",
            "membership_offset",
            "membership_offset_input",
            "membership_arange",
            "membership_add",
            "membership_add_left",
            "membership_add_right",
            "membership_subtract",
            "membership_position",
            "row_value_index",
            "row_run_index",
            "row_group",
            "row_run",
            "flat_group",
            "flat_group_sort_input",
            "sort_order",
            "sort_workspace",
            "argsort_workspace",
            "sorted_group",
            "append_sort_order",
            "append_sorted_group",
            "unique_group",
            "unique_start",
            "unique_count",
            "reduceat_start",
            "reduceat_count",
        )
    ) + float_bytes * len(
        (
            "expanded_value",
            "expanded_value_input",
            "row_value",
            "sorted_value",
            "reduceat_sum",
            "reduceat_divisor",
            "group_mean",
            "group_mean_observed",
            "append_sorted_value",
            "append_value_copy",
        )
    )
    return source_bytes + int(max_memberships_per_row) * (
        expanded_bytes + bool_bytes * len(("observed_group",))
    )


def _append_sparse_group_bucket_records(
    *,
    bucket_paths: list[Path],
    bucket_group_width: int,
    groups: np.ndarray,
    values: np.ndarray,
) -> None:
    """Append observed group means to their temporary bucket files."""
    bucket_ids = groups // int(bucket_group_width)
    for bucket_id in np.unique(bucket_ids):
        mask = bucket_ids == int(bucket_id)
        records = np.empty(np.count_nonzero(mask), dtype=SPARSE_SUMMARY_BUCKET_RECORD_DTYPE)
        records["group"] = groups[mask]
        records["value"] = values[mask]
        with bucket_paths[int(bucket_id)].open("ab") as handle:
            records.tofile(handle)


def _iter_sparse_group_bucket_values(
    *,
    bucket_paths: list[Path],
    bucket_group_width: int,
):
    """Yield observed values grouped by public summary row from bucket files."""
    del bucket_group_width
    for path in bucket_paths:
        if not path.exists():
            continue
        records = np.fromfile(path, dtype=SPARSE_SUMMARY_BUCKET_RECORD_DTYPE)
        order = np.argsort(records["group"])
        groups = records["group"][order]
        unique_groups, starts, counts = np.unique(groups, return_index=True, return_counts=True)
        for group, start, count in zip(unique_groups, starts, counts, strict=True):
            value_positions = order[start : start + count]
            yield (
                int(group),
                records["value"][value_positions].astype(
                    np.float64,
                    copy=True,
                ),
            )


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return integer ceiling division."""
    return -(-int(numerator) // int(denominator))

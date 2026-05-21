"""Exact uncertainty summaries from public run artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd

from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    collapse_sparse_rows_to_overlapping_summary_groups,
    sparse_public_row_group_membership_index,
)
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import (
    iter_compact_run_matrix_columns,
    iter_sparse_run_row_windows,
    iter_sparse_run_rows,
)
from pyaesa.shared.uncertainty_assessment.io.summary_kernels import (
    SUMMARY_MAX_NUMERIC_CELLS_PER_BLOCK,
    SUMMARY_STATISTICS,
    assign_summary_columns,
    column_block_width,
    collapse_grouped_run_values,
    group_block_stop,
    public_row_groups_are_identity_ordered,
    summary_scan_max_numeric_cells,
)
from pyaesa.shared.uncertainty_assessment.io.tables import SparseRunRows


def exact_summary_from_public_runs(
    *,
    identity_frame: pd.DataFrame,
    runs_path: Path,
    output_format: str,
    run_count: int,
    public_row_groups: tuple[tuple[str, ...], ...] | None = None,
    sparse: bool = False,
) -> pd.DataFrame:
    """Build exact summary statistics from public compact or sparse run artifacts."""
    summary, _frequency = _summary_scan_from_public_runs(
        identity_frame=identity_frame,
        runs_path=runs_path,
        output_format=output_format,
        run_count=run_count,
        public_row_groups=public_row_groups,
        sparse=sparse,
        include_frequency=False,
    )
    return summary


def exact_summary_and_frequency_from_public_runs(
    *,
    identity_frame: pd.DataFrame,
    runs_path: Path,
    output_format: str,
    run_count: int,
    public_row_groups: tuple[tuple[str, ...], ...] | None = None,
    sparse: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build exact summaries and frequency means from one public artifact scan."""
    summary, frequency = _summary_scan_from_public_runs(
        identity_frame=identity_frame,
        runs_path=runs_path,
        output_format=output_format,
        run_count=run_count,
        public_row_groups=public_row_groups,
        sparse=sparse,
        include_frequency=True,
    )
    return summary, frequency


def _summary_scan_from_public_runs(
    *,
    identity_frame: pd.DataFrame,
    runs_path: Path,
    output_format: str,
    run_count: int,
    public_row_groups: tuple[tuple[str, ...], ...] | None,
    sparse: bool,
    include_frequency: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    groups = public_row_groups
    if sparse:
        if groups is None:
            groups = tuple((str(index),) for index in range(len(identity_frame)))
        return _sparse_grouped_summary_scan(
            identity_frame=identity_frame,
            runs_path=runs_path,
            output_format=output_format,
            run_count=run_count,
            public_row_groups=groups,
            include_frequency=include_frequency,
        )
    if groups is not None and public_row_groups_are_identity_ordered(public_row_groups=groups):
        groups = None
    if groups is not None:
        return _compact_grouped_summary_scan(
            identity_frame=identity_frame,
            runs_path=runs_path,
            output_format=output_format,
            run_count=run_count,
            public_row_groups=groups,
            include_frequency=include_frequency,
        )
    column_positions = np.arange(len(identity_frame), dtype=np.int64)
    return _compact_column_summary_scan(
        identity_frame=identity_frame,
        runs_path=runs_path,
        output_format=output_format,
        run_count=run_count,
        column_positions=column_positions,
        include_frequency=include_frequency,
    )


def _compact_column_summary_scan(
    *,
    identity_frame: pd.DataFrame,
    runs_path: Path,
    output_format: str,
    run_count: int,
    column_positions: np.ndarray,
    include_frequency: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    pieces: list[pd.DataFrame] = []
    means: list[np.ndarray] = []
    max_cells = summary_scan_max_numeric_cells(output_format=output_format)
    block_columns = column_block_width(
        run_count=run_count,
        row_count=len(column_positions),
        max_numeric_cells_per_block=max_cells,
    )
    for start in range(0, len(column_positions), block_columns):
        stop = min(start + block_columns, len(column_positions))
        values = _read_compact_columns(
            path=runs_path,
            output_format=output_format,
            run_count=run_count,
            column_positions=column_positions[start:stop],
        )
        if include_frequency:
            means.append(_frequency_mean(values=values))
        summary = identity_frame.iloc[start:stop].reset_index(drop=True).copy()
        assign_summary_columns(summary=summary, values=values)
        pieces.append(summary)
    return _summary_frame(pieces=pieces, identity_frame=identity_frame), _frequency_array(
        means=means,
        include_frequency=include_frequency,
    )


def _compact_grouped_summary_scan(
    *,
    identity_frame: pd.DataFrame,
    runs_path: Path,
    output_format: str,
    run_count: int,
    public_row_groups: tuple[tuple[str, ...], ...],
    include_frequency: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[pd.DataFrame] = []
    means: list[np.ndarray] = []
    start = 0
    while start < len(public_row_groups):
        stop = group_block_stop(
            groups=public_row_groups,
            start=start,
            run_count=run_count,
            max_numeric_cells_per_block=summary_scan_max_numeric_cells(output_format=output_format),
        )
        block_groups = public_row_groups[start:stop]
        columns = sorted({column for group in block_groups for column in group}, key=int)
        values = collapse_grouped_run_values(
            values=_read_compact_columns(
                path=runs_path,
                output_format=output_format,
                run_count=run_count,
                column_positions=np.array([int(column) for column in columns], dtype=np.int64),
            ),
            columns=columns,
            public_row_groups=block_groups,
        )
        if include_frequency:
            means.append(_frequency_mean(values=values))
        summary = identity_frame.iloc[start:stop].reset_index(drop=True).copy()
        assign_summary_columns(summary=summary, values=values)
        rows.append(summary)
        start = stop
    return _summary_frame(pieces=rows, identity_frame=identity_frame), _frequency_array(
        means=means,
        include_frequency=include_frequency,
    )


def _sparse_grouped_summary_scan(
    *,
    identity_frame: pd.DataFrame,
    runs_path: Path,
    output_format: str,
    run_count: int,
    public_row_groups: tuple[tuple[str, ...], ...],
    include_frequency: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    values = _read_sparse_group_values(
        path=runs_path,
        output_format=output_format,
        run_count=run_count,
        public_row_groups=public_row_groups,
    )
    frequency = (
        _frequency_mean(values=values) if include_frequency else np.empty(0, dtype=np.float64)
    )
    summary = identity_frame.reset_index(drop=True).copy()
    assign_summary_columns(summary=summary, values=values)
    return _summary_frame(pieces=[summary], identity_frame=identity_frame), _frequency_array(
        means=[frequency],
        include_frequency=include_frequency,
    )


def _summary_frame(*, pieces: list[pd.DataFrame], identity_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(pieces, ignore_index=True).loc[
        :, [*identity_frame.columns, *SUMMARY_STATISTICS]
    ]


def _frequency_array(*, means: list[np.ndarray], include_frequency: bool) -> np.ndarray:
    if not include_frequency:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(means)


def _read_compact_columns(
    *,
    path: Path,
    output_format: str,
    run_count: int,
    column_positions: np.ndarray,
) -> np.ndarray:
    names = [str(int(position)) for position in column_positions]
    values = np.empty((int(run_count), len(names)), dtype=np.float64)
    cursor = 0
    for _run_indices, block in iter_compact_run_matrix_columns(
        path=path,
        output_format=output_format,
        column_names=names,
    ):
        values[cursor : cursor + len(block), :] = block
        cursor += len(block)
    return _validated_run_values(values=values, cursor=cursor, run_count=run_count)


def _read_sparse_group_values(
    *,
    path: Path,
    output_format: str,
    run_count: int,
    public_row_groups: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    group_index = sparse_public_row_group_membership_index(public_row_groups=public_row_groups)
    group_count = len(public_row_groups)
    batch_size = max(1, SUMMARY_MAX_NUMERIC_CELLS_PER_BLOCK // max(1, group_count))
    empty_rows = SparseRunRows(
        run_index=np.empty(0, dtype=np.int64),
        public_row_id=np.empty(0, dtype=np.int64),
        values=np.empty(0, dtype=np.float64),
        value_column="value",
    )
    pieces: list[np.ndarray] = []
    windows = iter_sparse_run_row_windows(
        chunks=iter_sparse_run_rows(path=path, output_format=output_format),
        start_run_index=0,
        stop_run_index=run_count,
        batch_size=batch_size,
        empty_rows=empty_rows,
    )
    for run_indices, rows in windows:
        pieces.append(
            collapse_sparse_rows_to_overlapping_summary_groups(
                sparse_rows=rows,
                run_indices=run_indices,
                public_row_groups=public_row_groups,
                public_row_group_index=group_index,
            )
        )
    return np.vstack(pieces) if pieces else np.empty((0, group_count), dtype=np.float64)


def _validated_run_values(*, values: np.ndarray, cursor: int, run_count: int) -> np.ndarray:
    if int(cursor) != int(run_count):
        raise ValueError("Public run artifact contains fewer rows than the completed run count.")
    return values


def _frequency_mean(*, values: np.ndarray) -> np.ndarray:
    observed = ~np.isnan(values)
    counts = observed.sum(axis=0)
    return np.divide(
        np.logical_and(observed, values <= 1.0).sum(axis=0),
        counts,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=counts > 0,
    )

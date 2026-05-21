"""Shared public row grouping for uncertainty summaries."""

import numpy as np
import pandas as pd

from pyaesa.shared.uncertainty_assessment.io.tables import SparseRunRows


def identity_groups_from_excluded_columns(
    *,
    identity: pd.DataFrame,
    excluded_columns: set[str],
) -> tuple[pd.DataFrame, tuple[tuple[str, ...], ...]]:
    """Return summary identity rows and their backing public row id groups."""
    excluded = {column for column in excluded_columns if column in identity.columns}
    if not excluded:
        return identity, tuple((str(value),) for value in identity["public_row_id"].tolist())
    columns = [
        column
        for column in identity.columns
        if column not in excluded and column != "public_row_id"
    ]
    indexed = identity.loc[:, ["public_row_id", *columns]].copy()
    grouped = indexed.groupby(columns, dropna=False, sort=False)["public_row_id"].agg(tuple)
    summary_identity = grouped.index.to_frame(index=False)
    public_row_groups = tuple(tuple(str(value) for value in values) for values in grouped.tolist())
    return summary_identity.reset_index(drop=True), public_row_groups


def sparse_public_row_group_index(
    *,
    public_row_groups: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    """Return public row id to summary group positions for sparse run rows."""
    max_public_row_id = max(
        int(public_row_id) for group in public_row_groups for public_row_id in group
    )
    index = np.empty(max_public_row_id + 1, dtype=np.int64)
    for group_index, group in enumerate(public_row_groups):
        index[[int(public_row_id) for public_row_id in group]] = group_index
    return index


def sparse_public_row_group_membership_index(
    *,
    public_row_groups: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    """Return public row id to all summary group memberships for sparse rows."""
    group_sizes = np.fromiter((len(group) for group in public_row_groups), dtype=np.int64)
    public_row_id = np.fromiter(
        (int(public_row_id) for group in public_row_groups for public_row_id in group),
        dtype=np.int64,
    )
    group_index = np.repeat(np.arange(len(public_row_groups), dtype=np.int64), group_sizes)
    order = np.lexsort((group_index, public_row_id))
    return np.column_stack((public_row_id[order], group_index[order]))


def collapse_values_to_summary_groups(
    *,
    values: np.ndarray,
    public_row_groups: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    """Collapse dense public row values to summary group columns."""
    stable_ordered_groups = all(
        len(group) == 1 and int(group[0]) == index for index, group in enumerate(public_row_groups)
    )
    if stable_ordered_groups:
        return values
    out = np.empty((values.shape[0], len(public_row_groups)), dtype=np.float64)
    for index, group in enumerate(public_row_groups):
        positions = np.array([int(public_row_id) for public_row_id in group], dtype=np.int64)
        block = values[:, positions]
        counts = np.sum(~np.isnan(block), axis=1)
        sums = np.nansum(block, axis=1)
        out[:, index] = np.divide(
            sums,
            counts,
            out=np.full(values.shape[0], np.nan, dtype=np.float64),
            where=counts > 0,
        )
    return out


def collapse_sparse_rows_to_summary_groups(
    *,
    sparse_rows: SparseRunRows,
    run_indices: np.ndarray,
    public_row_groups: tuple[tuple[str, ...], ...],
    public_row_group_index: np.ndarray,
) -> np.ndarray:
    """Collapse sparse public row values to summary group columns."""
    row_runs = run_positions_in_window(
        run_indices=run_indices,
        row_run_index=sparse_rows.run_index,
    )
    row_groups = public_row_group_index[sparse_rows.public_row_id]
    return _collapse_sparse_run_group_values(
        row_runs=row_runs,
        row_groups=row_groups,
        values=sparse_rows.values,
        run_count=len(run_indices),
        group_count=len(public_row_groups),
    )


def collapse_sparse_rows_to_overlapping_summary_groups(
    *,
    sparse_rows: SparseRunRows,
    run_indices: np.ndarray,
    public_row_groups: tuple[tuple[str, ...], ...],
    public_row_group_index: np.ndarray,
) -> np.ndarray:
    """Collapse sparse rows when one public row belongs to several summary groups."""
    source_positions, row_groups = _sparse_membership_positions(
        public_row_id=sparse_rows.public_row_id,
        public_row_group_index=public_row_group_index,
    )
    row_runs = run_positions_in_window(
        run_indices=run_indices,
        row_run_index=sparse_rows.run_index[source_positions],
    )
    return _collapse_sparse_run_group_values(
        row_runs=row_runs,
        row_groups=row_groups,
        values=sparse_rows.values[source_positions],
        run_count=len(run_indices),
        group_count=len(public_row_groups),
    )


def _collapse_sparse_run_group_values(
    *,
    row_runs: np.ndarray,
    row_groups: np.ndarray,
    values: np.ndarray,
    run_count: int,
    group_count: int,
) -> np.ndarray:
    """Average sparse values by run position and summary group."""
    shape = (int(run_count), int(group_count))
    flat = row_runs * int(group_count) + row_groups
    sums = np.bincount(
        flat,
        weights=values,
        minlength=shape[0] * shape[1],
    ).reshape(shape)
    counts = np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)
    return np.divide(
        sums,
        counts,
        out=np.full(shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _sparse_membership_positions(
    *,
    public_row_id: np.ndarray,
    public_row_group_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand sparse row positions to every overlapping summary group."""
    row_membership_ids = public_row_group_index[:, 0]
    row_membership_groups = public_row_group_index[:, 1]
    group_ids, group_starts, counts = np.unique(
        row_membership_ids,
        return_index=True,
        return_counts=True,
    )
    max_sparse_id = int(public_row_id.max()) if public_row_id.size else 0
    index_size = max(int(row_membership_ids[-1]), max_sparse_id) + 1
    starts_by_id = np.zeros(index_size, dtype=np.int64)
    counts_by_id = np.zeros(index_size, dtype=np.int64)
    starts_by_id[group_ids] = group_starts
    counts_by_id[group_ids] = counts
    membership_starts = starts_by_id[public_row_id]
    membership_counts = counts_by_id[public_row_id]
    source_positions = np.repeat(
        np.arange(len(public_row_id), dtype=np.int64),
        membership_counts,
    )
    membership_offsets = np.repeat(
        np.cumsum(membership_counts, dtype=np.int64) - membership_counts,
        membership_counts,
    )
    membership_positions = (
        np.repeat(membership_starts, membership_counts)
        + np.arange(int(membership_counts.sum()), dtype=np.int64)
        - membership_offsets
    )
    return source_positions, row_membership_groups[membership_positions]


def run_positions_in_window(*, run_indices: np.ndarray, row_run_index: np.ndarray) -> np.ndarray:
    """Return row positions for sparse run ids inside the current run window."""
    if run_indices.size and int(run_indices[-1]) == int(run_indices[0]) + len(run_indices) - 1:
        return np.asarray(row_run_index, dtype=np.int64) - int(run_indices[0])
    return np.searchsorted(run_indices, row_run_index)

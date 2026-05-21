"""Shared selected run value readers for uncertainty figures."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import (
    iter_compact_run_matrix_columns,
    iter_sparse_run_rows,
)

RUN_INDEX_ARRAY_COLUMN = "__run_indices"


def collect_selected_compact_run_values(
    *,
    path: Path,
    output_format: str,
    public_row_ids: Iterable[int],
    stop_run_index: int | None = None,
) -> dict[int, np.ndarray]:
    """Collect selected compact run matrix columns by public row id."""
    selected = sorted({int(value) for value in public_row_ids})
    chunks: dict[int, list[np.ndarray]] = {public_id: [] for public_id in selected}
    if not selected:
        return {}
    columns = [str(public_id) for public_id in selected]
    for _run_indices, values in iter_compact_run_matrix_columns(
        path=path,
        output_format=output_format,
        column_names=columns,
        stop_run_index=stop_run_index,
    ):
        _append_compact_chunk(chunks=chunks, values=values)
    return _concatenated_chunks(chunks)


def collect_selected_sparse_run_values(
    *,
    path: Path,
    output_format: str,
    public_row_ids: Iterable[int],
    stop_run_index: int | None = None,
) -> dict[int, np.ndarray]:
    """Collect selected sparse run values by public row id."""
    return {
        public_id: values
        for public_id, (_run_indices, values) in collect_selected_sparse_run_indexed_values(
            path=path,
            output_format=output_format,
            public_row_ids=public_row_ids,
            stop_run_index=stop_run_index,
        ).items()
    }


def collect_selected_sparse_run_indexed_values(
    *,
    path: Path,
    output_format: str,
    public_row_ids: Iterable[int],
    stop_run_index: int | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Collect selected sparse run indices and values by public row id."""
    selected = sorted({int(value) for value in public_row_ids})
    run_chunks: dict[int, list[np.ndarray]] = {public_id: [] for public_id in selected}
    value_chunks: dict[int, list[np.ndarray]] = {public_id: [] for public_id in selected}
    if not selected:
        return {}
    for sparse_rows in iter_sparse_run_rows(
        path=path,
        output_format=output_format,
        stop_run_index=stop_run_index,
    ):
        mask = np.isin(sparse_rows.public_row_id, selected)
        if not bool(np.any(mask)):
            continue
        ids = sparse_rows.public_row_id[mask].astype(np.int64, copy=False)
        runs = sparse_rows.run_index[mask].astype(np.int64, copy=False)
        values = sparse_rows.values[mask].astype(np.float64, copy=False)
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        sorted_runs = runs[order]
        sorted_values = values[order]
        boundaries = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [len(sorted_ids)]))
        for start, stop in zip(starts, stops, strict=True):
            public_id = int(sorted_ids[int(start)])
            run_chunks[public_id].append(sorted_runs[int(start) : int(stop)])
            value_chunks[public_id].append(sorted_values[int(start) : int(stop)])
    return {
        public_id: (
            np.concatenate(run_chunks[public_id])
            if run_chunks[public_id]
            else np.empty(0, dtype=np.int64),
            np.concatenate(value_chunks[public_id])
            if value_chunks[public_id]
            else np.empty(0, dtype=np.float64),
        )
        for public_id in selected
    }


def sum_values_by_run_index(
    *,
    run_indices: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return run indices and summed values after grouping by run index."""
    order = np.argsort(run_indices, kind="stable")
    ordered_runs = run_indices[order]
    ordered_values = values[order]
    boundaries = np.flatnonzero(ordered_runs[1:] != ordered_runs[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    return ordered_runs[starts], np.add.reduceat(ordered_values, starts)


def _append_compact_chunk(*, chunks: dict[int, list[np.ndarray]], values: np.ndarray) -> None:
    for position, public_id in enumerate(chunks):
        chunks[public_id].append(values[:, position].astype(np.float64, copy=False))


def _concatenated_chunks(chunks: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
    return {
        public_id: np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        for public_id, parts in chunks.items()
    }

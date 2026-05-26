"""Sparse row expansion helpers for aCC uncertainty evaluation."""

from dataclasses import dataclass
from typing import cast

import numpy as np

from pyaesa.acc.uncertainty.runtime.models import ACCBranchPlan
from pyaesa.shared.uncertainty_assessment.io.run_writers import SparseRunRows


@dataclass(frozen=True)
class SparseBranchExpansion:
    """Sorted sparse expansion map from aSoCC public rows to ACC public rows."""

    asocc_positions: np.ndarray
    acc_public_row_id: np.ndarray
    static_cc_values: np.ndarray | None
    cc_positions: np.ndarray | None
    dynamic_cc_factors: np.ndarray | None


@dataclass(frozen=True)
class CCSparseBranchExpansion:
    """Sorted expansion map from AR6 CC public rows to ACC public rows."""

    cc_positions: np.ndarray
    asocc_positions: np.ndarray
    acc_public_row_id: np.ndarray
    dynamic_cc_factors: np.ndarray


def sparse_branch_expansions(
    *,
    branch_plans: tuple[ACCBranchPlan, ...],
    cc_type: str | None = None,
) -> tuple[SparseBranchExpansion, ...]:
    """Return sorted expansion maps for sparse aSoCC rows."""
    expansions: list[SparseBranchExpansion] = []
    offset = 0
    for branch in branch_plans:
        if cc_type is not None and branch.cc_type != cc_type:
            offset += len(branch.identity)
            continue
        order = np.argsort(branch.asocc_positions, kind="mergesort")
        branch_positions = np.arange(len(branch.asocc_positions), dtype=np.int64)[order]
        expansions.append(
            SparseBranchExpansion(
                asocc_positions=branch.asocc_positions[order],
                acc_public_row_id=offset + branch_positions,
                static_cc_values=(
                    None
                    if branch.static_cc_values is None
                    else branch.static_cc_values[branch_positions]
                ),
                cc_positions=(
                    None if branch.cc_positions is None else branch.cc_positions[branch_positions]
                ),
                dynamic_cc_factors=(
                    None
                    if branch.dynamic_cc_factors is None
                    else branch.dynamic_cc_factors[branch_positions]
                ),
            )
        )
        offset += len(branch.identity)
    return tuple(expansions)


def cc_sparse_branch_expansions(
    *,
    branch_plans: tuple[ACCBranchPlan, ...],
) -> tuple[CCSparseBranchExpansion, ...]:
    """Return sorted expansion maps for sparse dynamic AR6 CC rows."""
    expansions: list[CCSparseBranchExpansion] = []
    offset = 0
    for branch in branch_plans:
        if branch.cc_type != "static":
            cc_positions = cast(np.ndarray, branch.cc_positions)
            order = np.argsort(cc_positions, kind="mergesort")
            branch_positions = np.arange(len(cc_positions), dtype=np.int64)[order]
            expansions.append(
                CCSparseBranchExpansion(
                    cc_positions=cc_positions[order],
                    asocc_positions=branch.asocc_positions[branch_positions],
                    acc_public_row_id=offset + branch_positions,
                    dynamic_cc_factors=cast(np.ndarray, branch.dynamic_cc_factors)[
                        branch_positions
                    ],
                )
            )
        offset += len(branch.identity)
    return tuple(expansions)


def selected_asocc_expansion_positions(
    *,
    asocc_rows: SparseRunRows,
    expansion: SparseBranchExpansion,
) -> tuple[np.ndarray, np.ndarray]:
    """Return branch positions selected by sparse aSoCC rows."""
    left = np.searchsorted(expansion.asocc_positions, asocc_rows.public_row_id, side="left")
    right = np.searchsorted(expansion.asocc_positions, asocc_rows.public_row_id, side="right")
    return _matched_expansion_positions(left=left, right=right)


def selected_cc_expansion_positions(
    *,
    cc_rows: SparseRunRows,
    expansion: CCSparseBranchExpansion,
) -> tuple[np.ndarray, np.ndarray]:
    """Return branch positions selected by sparse AR6 CC rows."""
    left = np.searchsorted(expansion.cc_positions, cc_rows.public_row_id, side="left")
    right = np.searchsorted(expansion.cc_positions, cc_rows.public_row_id, side="right")
    return _matched_expansion_positions(left=left, right=right)


def collect_sparse_rows_for_range(
    *,
    pending: SparseRunRows,
    chunks,
    start: int,
    stop: int,
) -> tuple[SparseRunRows, SparseRunRows]:
    """Collect sparse source rows that belong to a contiguous run interval."""
    pieces: list[SparseRunRows] = []
    current = pending
    while True:
        inside, after = _split_sparse_rows(rows=current, start=start, stop=stop)
        if inside.run_index.size:
            pieces.append(inside)
        if after.run_index.size:
            return after, concat_cc_sparse_rows(pieces=pieces)
        try:
            current = next(chunks)
        except StopIteration:
            return empty_cc_sparse_rows(), concat_cc_sparse_rows(pieces=pieces)


def sparse_rows_from_blocks(
    *,
    run_blocks: list[np.ndarray],
    row_blocks: list[np.ndarray],
    value_blocks: list[np.ndarray],
) -> SparseRunRows:
    """Return sorted finite ACC sparse rows from numeric blocks."""
    if not run_blocks:
        return empty_acc_sparse_rows()
    run_index = np.concatenate(run_blocks).astype(np.int64, copy=False)
    public_row_id = np.concatenate(row_blocks).astype(np.int64, copy=False)
    values = np.concatenate(value_blocks).astype(np.float64, copy=False)
    finite = np.isfinite(values)
    order = np.lexsort((public_row_id[finite], run_index[finite]))
    return SparseRunRows(
        run_index=run_index[finite][order],
        public_row_id=public_row_id[finite][order],
        values=values[finite][order],
        value_column="acc",
    )


def concat_acc_sparse_rows(*, pieces: list[SparseRunRows]) -> SparseRunRows:
    """Return sorted finite ACC sparse rows from already evaluated pieces."""
    non_empty = [piece for piece in pieces if piece.run_index.size]
    if not non_empty:
        return empty_acc_sparse_rows()
    return sparse_rows_from_blocks(
        run_blocks=[piece.run_index for piece in non_empty],
        row_blocks=[piece.public_row_id for piece in non_empty],
        value_blocks=[piece.values for piece in non_empty],
    )


def concat_cc_sparse_rows(*, pieces: list[SparseRunRows]) -> SparseRunRows:
    """Return sorted sparse AR6 CC source rows."""
    non_empty = [piece for piece in pieces if piece.run_index.size]
    if not non_empty:
        return empty_cc_sparse_rows()
    run_index = np.concatenate([piece.run_index for piece in non_empty])
    public_row_id = np.concatenate([piece.public_row_id for piece in non_empty])
    values = np.concatenate([piece.values for piece in non_empty])
    order = np.lexsort((public_row_id, run_index))
    return SparseRunRows(
        run_index=run_index[order],
        public_row_id=public_row_id[order],
        values=values[order],
        value_column="cc",
    )


def empty_acc_sparse_rows() -> SparseRunRows:
    """Return an empty ACC sparse row batch."""
    return SparseRunRows(
        run_index=np.empty(0, dtype=np.int64),
        public_row_id=np.empty(0, dtype=np.int64),
        values=np.empty(0, dtype=np.float64),
        value_column="acc",
    )


def empty_cc_sparse_rows() -> SparseRunRows:
    """Return an empty AR6 CC sparse row batch."""
    return SparseRunRows(
        run_index=np.empty(0, dtype=np.int64),
        public_row_id=np.empty(0, dtype=np.int64),
        values=np.empty(0, dtype=np.float64),
        value_column="cc",
    )


def asocc_public_row_count_from_expansions(
    *,
    expansions: tuple[CCSparseBranchExpansion, ...],
) -> int:
    """Return the aSoCC public row count covered by dynamic sparse expansions."""
    return 1 + max(
        int(position) for expansion in expansions for position in expansion.asocc_positions.tolist()
    )


def _matched_expansion_positions(
    *,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts = right - left
    selected = np.flatnonzero(counts > 0)
    if selected.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    repeated_source = np.repeat(selected, counts[selected])
    repeated_left = np.repeat(left[selected], counts[selected])
    group_starts = np.repeat(np.cumsum(counts[selected]) - counts[selected], counts[selected])
    match_positions = repeated_left + np.arange(len(repeated_source), dtype=np.int64) - group_starts
    return match_positions, repeated_source


def _split_sparse_rows(
    *,
    rows: SparseRunRows,
    start: int,
    stop: int,
) -> tuple[SparseRunRows, SparseRunRows]:
    inside = (rows.run_index >= int(start)) & (rows.run_index < int(stop))
    after = rows.run_index >= int(stop)
    return _mask_sparse_rows(rows=rows, mask=inside), _mask_sparse_rows(rows=rows, mask=after)


def _mask_sparse_rows(*, rows: SparseRunRows, mask: np.ndarray) -> SparseRunRows:
    return SparseRunRows(
        run_index=rows.run_index[mask],
        public_row_id=rows.public_row_id[mask],
        values=rows.values[mask],
        value_column=rows.value_column,
    )

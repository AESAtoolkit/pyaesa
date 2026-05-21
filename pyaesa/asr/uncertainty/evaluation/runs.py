"""ASR uncertainty yearly run evaluation."""

import numpy as np

from pyaesa.acc.uncertainty.io.artifacts import acc_run_paths_from_manifest
from pyaesa.asr.uncertainty.evaluation.cumulative import (
    evaluate_asr_cumulative_sparse_matrix,
    evaluate_asr_cumulative_value_matrix,
)
from pyaesa.asr.uncertainty.runtime.models import ASRUncertaintyPlan
from pyaesa.asr.uncertainty.sources.lca_inputs import lca_values_for_runs
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    run_positions_in_window,
)
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import (
    iter_compact_run_matrix,
    iter_sparse_run_rows,
    iter_sparse_run_row_windows,
)
from pyaesa.shared.uncertainty_assessment.io.tables import SparseRunRows


def iter_asr_compact_render_product_batches(
    *,
    plan: ASRUncertaintyPlan,
    output_format: str,
    start_run_index: int = 0,
    stop_run_index: int | None = None,
):
    """Yield yearly ASR values and dynamic cumulative ASR values when owned."""
    acc_paths = acc_run_paths_from_manifest(manifest=plan.acc_manifest)
    for run_indices, acc_values in iter_compact_run_matrix(
        path=acc_paths.public_runs,
        output_format=output_format,
        column_count=_acc_public_row_count(plan=plan),
        start_run_index=start_run_index,
        stop_run_index=stop_run_index,
    ):
        lca_values = lca_values_for_runs(lca_input=plan.lca_input, run_indices=run_indices)
        values = evaluate_asr_value_matrix(
            acc_values=acc_values,
            lca_values=lca_values,
            acc_positions=plan.acc_positions,
            lca_positions=plan.lca_positions,
            lca_unit_factors=plan.lca_unit_factors,
        )
        cumulative = (
            evaluate_asr_cumulative_value_matrix(
                acc_values=acc_values,
                lca_values=lca_values,
                plan=plan,
            )
            if plan.has_cumulative_outputs
            else None
        )
        yield run_indices, values, cumulative


def iter_asr_sparse_render_product_batches(
    *,
    plan: ASRUncertaintyPlan,
    output_format: str,
    start_run_index: int,
    stop_run_index: int,
    batch_size: int,
):
    """Yield ASR values for selected run rows without rereading ACC rows."""
    acc_paths = acc_run_paths_from_manifest(manifest=plan.acc_manifest)
    chunks = iter_sparse_run_rows(
        path=acc_paths.public_runs,
        output_format=output_format,
        start_run_index=start_run_index,
        stop_run_index=stop_run_index,
    )
    for run_indices, acc_rows in iter_sparse_run_row_windows(
        chunks=chunks,
        start_run_index=start_run_index,
        stop_run_index=stop_run_index,
        batch_size=batch_size,
        empty_rows=_empty_sparse_acc_rows(),
    ):
        lca_values = lca_values_for_runs(lca_input=plan.lca_input, run_indices=run_indices)
        rows = evaluate_asr_sparse_rows(
            acc_rows=acc_rows,
            run_indices=run_indices,
            lca_values=lca_values,
            plan=plan,
        )
        cumulative = (
            evaluate_asr_cumulative_sparse_matrix(
                acc_rows=acc_rows,
                run_indices=run_indices,
                lca_values=lca_values,
                plan=plan,
            )
            if plan.has_cumulative_outputs
            else None
        )
        yield run_indices, rows, cumulative


def evaluate_asr_value_matrix(
    *,
    acc_values: np.ndarray,
    lca_values: np.ndarray,
    acc_positions: np.ndarray,
    lca_positions: np.ndarray,
    lca_unit_factors: np.ndarray,
) -> np.ndarray:
    """Evaluate ASR values for one dense run or Sobol batch."""
    numerator = lca_values[:, lca_positions] * lca_unit_factors[None, :]
    denominator = acc_values[:, acc_positions]
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator != 0,
    )


def evaluate_asr_sparse_rows(
    *,
    acc_rows: SparseRunRows,
    run_indices: np.ndarray,
    lca_values: np.ndarray,
    plan: ASRUncertaintyPlan,
) -> SparseRunRows:
    """Evaluate ASR for selected sparse ACC rows."""
    starts = np.searchsorted(plan.acc_positions_sorted, acc_rows.public_row_id, side="left")
    stops = np.searchsorted(plan.acc_positions_sorted, acc_rows.public_row_id, side="right")
    counts = stops - starts
    selected = counts > 0
    starts = starts[selected]
    counts = counts[selected]
    source_positions = np.flatnonzero(selected).astype(np.int64, copy=False)
    repeated_sources = np.repeat(source_positions, counts)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    expansion_positions = np.repeat(starts, counts) + np.arange(int(counts.sum())) - offsets
    public_row_id = plan.acc_position_order[expansion_positions].astype(np.int64, copy=False)
    run_index = acc_rows.run_index[repeated_sources].astype(np.int64, copy=False)
    run_positions = run_positions_in_window(
        run_indices=run_indices,
        row_run_index=run_index,
    ).astype(np.int64, copy=False)
    numerator = lca_values[run_positions, plan.lca_positions[public_row_id]]
    numerator *= plan.lca_unit_factors[public_row_id]
    denominator = acc_rows.values[repeated_sources]
    values = np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan, dtype=np.float64),
        where=denominator != 0,
    )
    # Upstream sparse ACC chunks are already grouped by run.
    return SparseRunRows(
        run_index=run_index,
        public_row_id=public_row_id,
        values=values,
        value_column="asr",
    )


def _empty_sparse_acc_rows() -> SparseRunRows:
    return SparseRunRows(
        run_index=np.empty(0, dtype=np.int64),
        public_row_id=np.empty(0, dtype=np.int64),
        values=np.empty(0, dtype=np.float64),
        value_column="acc",
    )


def _acc_public_row_count(*, plan: ASRUncertaintyPlan) -> int:
    return int(plan.acc_positions.max()) + 1

"""ASR uncertainty cumulative period identity and value evaluation."""

import numpy as np
import pandas as pd

from pyaesa.asr.uncertainty.evaluation.scenario_groups import (
    scenario_identity_groups_from_excluded_columns,
)
from pyaesa.asr.uncertainty.runtime.models import ASRUncertaintyPlan
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    run_positions_in_window,
)
from pyaesa.shared.uncertainty_assessment.io.tables import SparseRunRows


def cumulative_period_identity_groups(
    *,
    identity: pd.DataFrame,
    excluded_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, tuple[tuple[str, ...], ...]]:
    """Return yearless cumulative ASR period identities and backing yearly row ids."""
    return scenario_identity_groups_from_excluded_columns(
        identity=identity,
        excluded_columns={*(excluded_columns or ()), "year"},
    )


def evaluate_asr_cumulative_value_matrix(
    *,
    acc_values: np.ndarray,
    lca_values: np.ndarray,
    plan: ASRUncertaintyPlan,
) -> np.ndarray:
    """Evaluate cumulative ASR from cumulative LCA and ACC run components."""
    numerator = lca_values[:, plan.lca_positions] * plan.lca_unit_factors[None, :]
    denominator = acc_values[:, plan.acc_positions]
    return _cumulative_asr_from_components(
        numerator=numerator,
        denominator=denominator,
        plan=plan,
    )


def evaluate_asr_cumulative_sparse_matrix(
    *,
    acc_rows: SparseRunRows,
    run_indices: np.ndarray,
    lca_values: np.ndarray,
    plan: ASRUncertaintyPlan,
) -> np.ndarray:
    """Evaluate compact cumulative ASR matrix from selected sparse ACC rows."""
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
    return _cumulative_asr_from_sparse_components(
        run_positions=run_positions,
        public_row_id=public_row_id,
        numerator=numerator,
        denominator=denominator,
        run_count=len(run_indices),
        plan=plan,
    )


def _cumulative_asr_from_components(
    *,
    numerator: np.ndarray,
    denominator: np.ndarray,
    plan: ASRUncertaintyPlan,
) -> np.ndarray:
    run_positions = np.repeat(
        np.arange(numerator.shape[0], dtype=np.int64),
        len(plan.cumulative_member_public_row_id),
    )
    group_positions = np.tile(plan.cumulative_member_group_id, numerator.shape[0])
    nums = np.zeros((numerator.shape[0], len(plan.cumulative_public_row_groups)), dtype=np.float64)
    dens = np.zeros((numerator.shape[0], len(plan.cumulative_public_row_groups)), dtype=np.float64)
    public_ids = plan.cumulative_member_public_row_id
    np.add.at(nums, (run_positions, group_positions), numerator[:, public_ids].reshape(-1))
    np.add.at(dens, (run_positions, group_positions), denominator[:, public_ids].reshape(-1))
    return np.divide(
        nums,
        dens,
        out=np.full(nums.shape, np.nan, dtype=np.float64),
        where=dens != 0.0,
    )


def _cumulative_asr_from_sparse_components(
    *,
    run_positions: np.ndarray,
    public_row_id: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    run_count: int,
    plan: ASRUncertaintyPlan,
) -> np.ndarray:
    starts = np.searchsorted(
        plan.cumulative_member_public_row_id_sorted,
        public_row_id,
        side="left",
    )
    stops = np.searchsorted(
        plan.cumulative_member_public_row_id_sorted,
        public_row_id,
        side="right",
    )
    counts = stops - starts
    selected = counts > 0
    starts = starts[selected]
    counts = counts[selected]
    selected_positions = np.flatnonzero(selected).astype(np.int64, copy=False)
    repeated_positions = np.repeat(selected_positions, counts)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    membership_positions = np.repeat(starts, counts) + np.arange(int(counts.sum())) - offsets
    groups = plan.cumulative_member_group_id_sorted[membership_positions]
    nums = np.zeros((run_count, len(plan.cumulative_public_row_groups)), dtype=np.float64)
    dens = np.zeros((run_count, len(plan.cumulative_public_row_groups)), dtype=np.float64)
    np.add.at(nums, (run_positions[repeated_positions], groups), numerator[repeated_positions])
    np.add.at(dens, (run_positions[repeated_positions], groups), denominator[repeated_positions])
    return np.divide(
        nums,
        dens,
        out=np.full(nums.shape, np.nan, dtype=np.float64),
        where=dens != 0.0,
    )

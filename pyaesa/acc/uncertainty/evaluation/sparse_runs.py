"""aCC sparse selected run row evaluation."""

from typing import cast

import numpy as np

from pyaesa.acc.uncertainty.evaluation.run_inputs import (
    asocc_layout,
    asocc_paths,
    asocc_public_row_count,
    dynamic_cc_layout,
    dynamic_cc_paths,
    fixed_cc_values_for_runs,
    iter_asocc_values,
)
from pyaesa.acc.uncertainty.evaluation.sparse_rows import (
    CCSparseBranchExpansion,
    SparseBranchExpansion,
    asocc_public_row_count_from_expansions,
    cc_sparse_branch_expansions,
    collect_sparse_rows_for_range,
    concat_acc_sparse_rows,
    empty_cc_sparse_rows,
    selected_asocc_expansion_positions,
    selected_cc_expansion_positions,
    sparse_branch_expansions,
    sparse_rows_from_blocks,
)
from pyaesa.acc.uncertainty.runtime.models import ACCBranchPlan, ACCUncertaintyPlan
from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    run_positions_in_window,
)
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import iter_sparse_run_rows
from pyaesa.shared.uncertainty_assessment.io.tables import SparseRunRows


def iter_acc_sparse_run_batches(
    *,
    plan: ACCUncertaintyPlan,
    output_format: str,
    start_run_index: int = 0,
    stop_run_index: int | None = None,
):
    """Yield sparse ACC run row batches for selected upstream source rows."""
    source_layout = asocc_layout(asocc_input=plan.asocc_input)
    paths = asocc_paths(asocc_input=plan.asocc_input) if source_layout != "fixed_values" else None
    cc_layout = dynamic_cc_layout(dynamic_cc_input=plan.dynamic_cc_input)
    if source_layout == "sparse_selected_rows" and cc_layout == "sparse_selected_rows":
        yield from _iter_sparse_asocc_sparse_cc(
            plan=plan,
            asocc_paths=cast(AsoccUncertaintyRunPaths, paths),
            output_format=output_format,
            start_run_index=start_run_index,
            stop_run_index=stop_run_index,
        )
        return
    if source_layout == "sparse_selected_rows":
        yield from _iter_sparse_asocc_fixed_cc(
            plan=plan,
            asocc_paths=cast(AsoccUncertaintyRunPaths, paths),
            output_format=output_format,
            start_run_index=start_run_index,
            stop_run_index=stop_run_index,
        )
        return
    yield from _iter_compact_asocc_sparse_cc(
        plan=plan,
        output_format=output_format,
        start_run_index=start_run_index,
        stop_run_index=stop_run_index,
    )


def evaluate_acc_sparse_rows(
    *,
    asocc_rows: SparseRunRows,
    run_indices: np.ndarray,
    expansions: tuple[SparseBranchExpansion, ...],
    cc_values: np.ndarray | None,
) -> SparseRunRows:
    """Expand selected aSoCC sparse rows into selected ACC sparse rows."""
    source_run_positions = run_positions_in_window(
        run_indices=run_indices,
        row_run_index=asocc_rows.run_index,
    ).astype(
        np.int64,
        copy=False,
    )
    run_blocks: list[np.ndarray] = []
    row_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []
    for expansion in expansions:
        match_positions, repeated_source = selected_asocc_expansion_positions(
            asocc_rows=asocc_rows,
            expansion=expansion,
        )
        if match_positions.size == 0:
            continue
        if expansion.static_cc_values is not None:
            multiplier = expansion.static_cc_values[match_positions]
        else:
            multiplier = cast(np.ndarray, cc_values)[
                source_run_positions[repeated_source],
                cast(np.ndarray, expansion.cc_positions)[match_positions],
            ]
            multiplier = (
                multiplier * cast(np.ndarray, expansion.dynamic_cc_factors)[match_positions]
            )
        run_blocks.append(asocc_rows.run_index[repeated_source])
        row_blocks.append(expansion.acc_public_row_id[match_positions])
        value_blocks.append(asocc_rows.values[repeated_source] * multiplier)
    return sparse_rows_from_blocks(
        run_blocks=run_blocks,
        row_blocks=row_blocks,
        value_blocks=value_blocks,
    )


def evaluate_acc_sparse_source_rows(
    *,
    asocc_rows: SparseRunRows,
    cc_rows: SparseRunRows,
    expansions: tuple[CCSparseBranchExpansion, ...],
) -> SparseRunRows:
    """Evaluate dynamic ACC rows from sparse aSoCC and sparse AR6 CC runs."""
    asocc_key_base = asocc_public_row_count_from_expansions(expansions=expansions)
    asocc_keys = asocc_rows.run_index * asocc_key_base + asocc_rows.public_row_id
    order = np.argsort(asocc_keys, kind="mergesort")
    sorted_keys = asocc_keys[order]
    run_blocks: list[np.ndarray] = []
    row_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []
    for expansion in expansions:
        match_positions, repeated_source = selected_cc_expansion_positions(
            cc_rows=cc_rows,
            expansion=expansion,
        )
        if match_positions.size == 0:
            continue
        candidate_keys = (
            cc_rows.run_index[repeated_source] * asocc_key_base
            + expansion.asocc_positions[match_positions]
        )
        found = np.searchsorted(sorted_keys, candidate_keys)
        in_bounds = found < len(sorted_keys)
        valid = np.zeros(len(candidate_keys), dtype=bool)
        valid[in_bounds] = sorted_keys[found[in_bounds]] == candidate_keys[in_bounds]
        if not np.any(valid):
            continue
        source_positions = repeated_source[valid]
        asocc_positions = order[found[valid]]
        branch_positions = match_positions[valid]
        run_blocks.append(cc_rows.run_index[source_positions])
        row_blocks.append(expansion.acc_public_row_id[branch_positions])
        value_blocks.append(
            asocc_rows.values[asocc_positions]
            * cc_rows.values[source_positions]
            * expansion.dynamic_cc_factors[branch_positions]
        )
    return sparse_rows_from_blocks(
        run_blocks=run_blocks,
        row_blocks=row_blocks,
        value_blocks=value_blocks,
    )


def _iter_sparse_asocc_fixed_cc(
    *,
    plan: ACCUncertaintyPlan,
    asocc_paths: AsoccUncertaintyRunPaths,
    output_format: str,
    start_run_index: int,
    stop_run_index: int | None,
):
    expansions = sparse_branch_expansions(branch_plans=plan.branch_plans)
    for asocc_rows in iter_sparse_run_rows(
        path=asocc_paths.public_runs,
        output_format=output_format,
        start_run_index=start_run_index,
        stop_run_index=stop_run_index,
    ):
        run_indices = np.unique(asocc_rows.run_index)
        yield (
            run_indices,
            evaluate_acc_sparse_rows(
                asocc_rows=asocc_rows,
                run_indices=run_indices,
                expansions=expansions,
                cc_values=fixed_cc_values_for_runs(
                    run_indices=run_indices,
                    deterministic_cc_values=plan.deterministic_cc_values,
                ),
            ),
        )


def _iter_sparse_asocc_sparse_cc(
    *,
    plan: ACCUncertaintyPlan,
    asocc_paths: AsoccUncertaintyRunPaths,
    output_format: str,
    start_run_index: int,
    stop_run_index: int | None,
):
    cc_chunks = iter_sparse_run_rows(
        path=dynamic_cc_paths(dynamic_cc_input=plan.dynamic_cc_input).public_runs,
        output_format=output_format,
        start_run_index=start_run_index,
        stop_run_index=stop_run_index,
    )
    pending = empty_cc_sparse_rows()
    dynamic_expansions = cc_sparse_branch_expansions(branch_plans=plan.branch_plans)
    static_expansions = sparse_branch_expansions(
        branch_plans=plan.branch_plans,
        cc_type="static",
    )
    for asocc_rows in iter_sparse_run_rows(
        path=asocc_paths.public_runs,
        output_format=output_format,
        start_run_index=start_run_index,
        stop_run_index=stop_run_index,
    ):
        run_indices = np.unique(asocc_rows.run_index)
        pending, cc_rows = collect_sparse_rows_for_range(
            pending=pending,
            chunks=cc_chunks,
            start=int(run_indices[0]),
            stop=int(run_indices[-1]) + 1,
        )
        yield (
            run_indices,
            concat_acc_sparse_rows(
                pieces=[
                    evaluate_acc_sparse_rows(
                        asocc_rows=asocc_rows,
                        run_indices=run_indices,
                        expansions=static_expansions,
                        cc_values=None,
                    ),
                    evaluate_acc_sparse_source_rows(
                        asocc_rows=asocc_rows,
                        cc_rows=cc_rows,
                        expansions=dynamic_expansions,
                    ),
                ]
            ),
        )


def _iter_compact_asocc_sparse_cc(
    *,
    plan: ACCUncertaintyPlan,
    output_format: str,
    start_run_index: int,
    stop_run_index: int | None,
):
    cc_chunks = iter_sparse_run_rows(
        path=dynamic_cc_paths(dynamic_cc_input=plan.dynamic_cc_input).public_runs,
        output_format=output_format,
        start_run_index=start_run_index,
        stop_run_index=stop_run_index,
    )
    pending = empty_cc_sparse_rows()
    expansions = cc_sparse_branch_expansions(branch_plans=plan.branch_plans)
    for run_indices, asocc_values in iter_asocc_values(
        asocc_input=plan.asocc_input,
        output_format=output_format,
        public_row_count=asocc_public_row_count(plan=plan),
        start_run_index=start_run_index,
        stop_run_index=cast(int, stop_run_index),
    ):
        pending, cc_rows = collect_sparse_rows_for_range(
            pending=pending,
            chunks=cc_chunks,
            start=int(run_indices[0]),
            stop=int(run_indices[-1]) + 1,
        )
        yield (
            run_indices,
            concat_acc_sparse_rows(
                pieces=[
                    _evaluate_compact_static_rows(
                        run_indices=run_indices,
                        branch_plans=plan.branch_plans,
                        asocc_values=asocc_values,
                    ),
                    _evaluate_compact_asocc_sparse_cc_rows(
                        cc_rows=cc_rows,
                        run_indices=run_indices,
                        asocc_values=asocc_values,
                        expansions=expansions,
                    ),
                ]
            ),
        )


def _evaluate_compact_static_rows(
    *,
    run_indices: np.ndarray,
    branch_plans: tuple[ACCBranchPlan, ...],
    asocc_values: np.ndarray,
) -> SparseRunRows:
    run_blocks: list[np.ndarray] = []
    row_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []
    offset = 0
    for branch in branch_plans:
        if branch.cc_type == "static":
            row_ids = offset + np.arange(len(branch.asocc_positions), dtype=np.int64)
            values = (
                asocc_values[:, branch.asocc_positions]
                * cast(np.ndarray, branch.static_cc_values)[None, :]
            )
            run_blocks.append(np.repeat(run_indices, len(row_ids)))
            row_blocks.append(np.tile(row_ids, len(run_indices)))
            value_blocks.append(values.reshape(-1))
        offset += len(branch.identity)
    return sparse_rows_from_blocks(
        run_blocks=run_blocks,
        row_blocks=row_blocks,
        value_blocks=value_blocks,
    )


def _evaluate_compact_asocc_sparse_cc_rows(
    *,
    cc_rows: SparseRunRows,
    run_indices: np.ndarray,
    asocc_values: np.ndarray,
    expansions: tuple[CCSparseBranchExpansion, ...],
) -> SparseRunRows:
    source_run_positions = run_positions_in_window(
        run_indices=run_indices,
        row_run_index=cc_rows.run_index,
    ).astype(
        np.int64,
        copy=False,
    )
    run_blocks: list[np.ndarray] = []
    row_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []
    for expansion in expansions:
        match_positions, repeated_source = selected_cc_expansion_positions(
            cc_rows=cc_rows,
            expansion=expansion,
        )
        if match_positions.size == 0:
            continue
        asocc = asocc_values[
            source_run_positions[repeated_source],
            expansion.asocc_positions[match_positions],
        ]
        run_blocks.append(cc_rows.run_index[repeated_source])
        row_blocks.append(expansion.acc_public_row_id[match_positions])
        value_blocks.append(
            asocc * cc_rows.values[repeated_source] * expansion.dynamic_cc_factors[match_positions]
        )
    return sparse_rows_from_blocks(
        run_blocks=run_blocks,
        row_blocks=row_blocks,
        value_blocks=value_blocks,
    )

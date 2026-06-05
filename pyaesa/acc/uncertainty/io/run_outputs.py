"""aCC Monte Carlo run materialization."""

from typing import Any

import numpy as np

from pyaesa.acc.uncertainty.runtime.models import ACCUncertaintyPlan, ACCUncertaintyRunPaths
from pyaesa.acc.uncertainty.evaluation.runs import (
    iter_acc_run_batches,
)
from pyaesa.acc.uncertainty.evaluation.sparse_runs import (
    iter_acc_sparse_run_batches,
)
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    collapse_values_to_summary_groups,
    sparse_public_row_group_membership_index,
    sparse_rows_to_overlapping_group_values,
)
from pyaesa.shared.uncertainty_assessment.io.public_summary import exact_summary_from_public_runs
from pyaesa.shared.uncertainty_assessment.io.downstream_run_outputs import (
    DownstreamRunOutputState,
    DownstreamRunOutputPaths,
    DownstreamRunOutputPlan,
    append_downstream_run_outputs,
    close_downstream_run_output_state,
    new_downstream_run_output_state,
    write_downstream_run_outputs,
)
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import (
    iter_compact_run_matrix,
    iter_sparse_run_rows,
    iter_sparse_run_row_windows,
    sparse_run_rows_per_run_window,
)
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest
from pyaesa.shared.uncertainty_assessment.io.run_writers import (
    CompactRunMatrixWriter,
    SparseRunRows,
)
from pyaesa.shared.uncertainty_assessment.io.tables import write_uncertainty_table


def write_acc_run_outputs(
    *,
    paths: ACCUncertaintyRunPaths,
    plan: ACCUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    show_progress: bool = True,
) -> tuple[int, dict[str, Any] | None]:
    """Write ACC run values, summary statistics, and convergence status."""
    completed_runs, convergence = write_downstream_run_outputs(
        paths=_downstream_paths(paths=paths),
        plan=_downstream_plan(plan=plan),
        runtime=runtime,
        show_progress=show_progress,
    )
    write_acc_cumulative_outputs(
        paths=paths,
        plan=plan,
        run_count=completed_runs,
        output_format=runtime.output_format,
        batch_size=runtime.batch_size,
    )
    return completed_runs, convergence


def new_acc_run_output_state(
    *,
    paths: ACCUncertaintyRunPaths,
    completed_runs: int = 0,
) -> DownstreamRunOutputState:
    """Create append state for one aCC run output run."""
    return new_downstream_run_output_state(
        paths=_downstream_paths(paths=paths),
        completed_runs=completed_runs,
    )


def close_acc_run_output_state(*, state: DownstreamRunOutputState) -> None:
    """Release append state for one aCC run output run."""
    close_downstream_run_output_state(state=state)


def append_acc_run_outputs(
    *,
    paths: ACCUncertaintyRunPaths,
    plan: ACCUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    state: DownstreamRunOutputState,
    target_runs: int,
    final_checkpoint: bool,
    show_progress: bool = True,
) -> tuple[DownstreamRunOutputState, dict[str, Any] | None]:
    """Append one aCC run interval and update summaries."""
    next_state, convergence = append_downstream_run_outputs(
        paths=_downstream_paths(paths=paths),
        plan=_downstream_plan(plan=plan),
        runtime=runtime,
        state=state,
        target_runs=target_runs,
        final_checkpoint=final_checkpoint,
        show_progress=show_progress,
    )
    if final_checkpoint or convergence is not None:
        write_acc_cumulative_outputs(
            paths=paths,
            plan=plan,
            run_count=next_state.completed_runs,
            output_format=runtime.output_format,
            batch_size=runtime.batch_size,
        )
    return next_state, convergence


def write_acc_cumulative_outputs(
    *,
    paths: ACCUncertaintyRunPaths,
    plan: ACCUncertaintyPlan,
    run_count: int,
    output_format: str,
    batch_size: int | None,
) -> None:
    """Write dynamic cumulative aCC run and summary artifacts from yearly runs."""
    if not plan.has_cumulative_outputs or int(run_count) <= 0:
        return
    if plan.acc_run_layout == "sparse_selected_rows":
        _write_sparse_cumulative_runs(
            paths=paths,
            plan=plan,
            run_count=run_count,
            output_format=output_format,
            batch_size=batch_size,
        )
    else:
        _write_compact_cumulative_runs(
            paths=paths,
            plan=plan,
            run_count=run_count,
            output_format=output_format,
            batch_size=batch_size,
        )
    summary = exact_summary_from_public_runs(
        identity_frame=plan.cumulative_summary_identity,
        runs_path=paths.cumulative_runs,
        output_format=output_format,
        run_count=run_count,
        public_row_groups=plan.cumulative_summary_public_row_groups,
        sparse=False,
    )
    write_uncertainty_table(
        path=paths.cumulative_summary_stats_runs,
        frame=summary,
        output_format=output_format,
    )


def _write_compact_cumulative_runs(
    *,
    paths: ACCUncertaintyRunPaths,
    plan: ACCUncertaintyPlan,
    run_count: int,
    output_format: str,
    batch_size: int | None,
) -> None:
    with CompactRunMatrixWriter(
        path=paths.cumulative_runs,
        output_format=output_format,
        append_existing=False,
    ) as writer:
        for batch_index, (run_indices, values) in enumerate(
            iter_compact_run_matrix(
                path=paths.public_runs,
                output_format=output_format,
                column_count=len(plan.identity),
                stop_run_index=run_count,
                max_rows_per_chunk=batch_size,
            )
        ):
            writer.write_batch(
                run_indices=run_indices,
                values=_cumulative_acc_values(
                    values=values,
                    public_row_groups=plan.cumulative_public_row_groups,
                ),
                batch_index=batch_index,
            )


def _write_sparse_cumulative_runs(
    *,
    paths: ACCUncertaintyRunPaths,
    plan: ACCUncertaintyPlan,
    run_count: int,
    output_format: str,
    batch_size: int | None,
) -> None:
    window_size = max(1, int(batch_size or run_count))
    group_index = sparse_public_row_group_membership_index(
        public_row_groups=plan.cumulative_public_row_groups,
    )
    chunks = iter_sparse_run_rows(
        path=paths.public_runs,
        output_format=output_format,
        stop_run_index=run_count,
        max_rows_per_chunk=sparse_run_rows_per_run_window(
            path=paths.public_runs,
            output_format=output_format,
            batch_size=window_size,
        ),
    )
    with CompactRunMatrixWriter(
        path=paths.cumulative_runs,
        output_format=output_format,
        append_existing=False,
    ) as writer:
        for batch_index, (run_indices, rows) in enumerate(
            iter_sparse_run_row_windows(
                chunks=chunks,
                start_run_index=0,
                stop_run_index=run_count,
                batch_size=window_size,
                empty_rows=_empty_sparse_acc_rows(),
            )
        ):
            writer.write_batch(
                run_indices=run_indices,
                values=_cumulative_acc_sparse_values(
                    rows=rows,
                    run_indices=run_indices,
                    public_row_group_index=group_index,
                    group_count=len(plan.cumulative_public_row_groups),
                ),
                batch_index=batch_index,
            )


def _downstream_paths(*, paths: ACCUncertaintyRunPaths) -> DownstreamRunOutputPaths:
    return DownstreamRunOutputPaths(
        run_root=paths.run_root,
        public_runs=paths.public_runs,
        summary_stats_runs=paths.summary_stats_runs,
    )


def _cumulative_acc_values(
    *,
    values: np.ndarray,
    public_row_groups: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    out = np.empty((values.shape[0], len(public_row_groups)), dtype=np.float64)
    for index, group in enumerate(public_row_groups):
        positions = np.asarray([int(public_row_id) for public_row_id in group], dtype=np.int64)
        out[:, index] = np.sum(values[:, positions], axis=1)
    return out


def _cumulative_acc_sparse_values(
    *,
    rows: SparseRunRows,
    run_indices: np.ndarray,
    public_row_group_index: np.ndarray,
    group_count: int,
) -> np.ndarray:
    out = np.full((len(run_indices), group_count), np.nan, dtype=np.float64)
    row_runs, row_groups, values = sparse_rows_to_overlapping_group_values(
        sparse_rows=rows,
        run_indices=run_indices,
        public_row_group_index=public_row_group_index,
    )
    sums = np.zeros_like(out)
    counts = np.zeros(out.shape, dtype=np.int64)
    np.add.at(sums, (row_runs, row_groups), values)
    np.add.at(counts, (row_runs, row_groups), 1)
    out[counts > 0] = sums[counts > 0]
    return out


def _downstream_plan(*, plan: ACCUncertaintyPlan) -> DownstreamRunOutputPlan:
    return DownstreamRunOutputPlan(
        run_layout=plan.acc_run_layout,
        summary_identity=plan.summary_identity,
        public_row_count=len(plan.identity),
        compact_batches=lambda output_format, start, stop, batch_size: iter_acc_run_batches(
            plan=plan,
            output_format=output_format,
            start_run_index=start,
            stop_run_index=stop,
            batch_size=batch_size,
        ),
        sparse_batches=lambda output_format, start, stop, batch_size: iter_acc_sparse_run_batches(
            plan=plan,
            output_format=output_format,
            start_run_index=start,
            stop_run_index=stop,
            batch_size=batch_size,
        ),
        collapse_compact=lambda values: collapse_values_to_summary_groups(
            values=values,
            public_row_groups=plan.summary_public_row_groups,
        ),
        sparse_public_row_group_membership_index=lambda: sparse_public_row_group_membership_index(
            public_row_groups=plan.summary_public_row_groups
        ),
        empty_sparse_rows=_empty_sparse_acc_rows,
        summary_public_row_groups=plan.summary_public_row_groups,
    )


def _empty_sparse_acc_rows() -> SparseRunRows:
    return SparseRunRows(
        run_index=np.empty(0, dtype=np.int64),
        public_row_id=np.empty(0, dtype=np.int64),
        values=np.empty(0, dtype=np.float64),
        value_column="acc",
    )

"""ASR Monte Carlo run materialization."""

from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from pyaesa.asr.uncertainty.runtime.models import ASRUncertaintyPlan, ASRUncertaintyRunPaths
from pyaesa.asr.uncertainty.evaluation.runs import (
    iter_asr_compact_render_product_batches,
    iter_asr_sparse_render_product_batches,
)
from pyaesa.asr.uncertainty.evaluation.summary import (
    ASR_CUMULATIVE_VALUE_METRIC,
    ASR_SUMMARY_SCOPE_COLUMN,
    ASR_SUMMARY_SCOPE_INTER_METHOD,
    ASR_SUMMARY_METRIC_COLUMN,
    ASR_VALUE_METRIC,
    asr_sparse_public_row_group_membership_index,
    collapse_asr_cumulative_values_to_summary,
    collapse_asr_values_to_summary,
    write_asr_summary_table,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.convergence import (
    ConvergenceCheckpointCursor,
    MeanConvergenceAccumulator,
    iter_sparse_group_mean_updates,
    mean_convergence_payload_for_targets,
)
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    sparse_rows_to_overlapping_group_values,
)
from pyaesa.shared.uncertainty_assessment.io.downstream_run_outputs import (
    DownstreamRunOutputState,
    DownstreamRunOutputPaths,
    close_downstream_run_output_state,
    final_downstream_convergence_payload,
    new_downstream_run_output_state,
)
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest
from pyaesa.shared.uncertainty_assessment.io.run_writers import (
    CompactRunMatrixWriter,
    SparseRunRows,
    SparseRunRowsWriter,
)
from pyaesa.shared.runtime.reporting.run_progress import (
    monte_carlo_completion_is_persistent,
    monte_carlo_run_drawing_label,
    monte_carlo_run_progress,
    monte_carlo_run_progress_label,
)
from pyaesa.shared.runtime.reporting.status import StatusSink


@dataclass
class ASRMetricConvergenceState:
    """Streaming convergence state for one ASR value and fNT target set."""

    value: MeanConvergenceAccumulator
    frequency: MeanConvergenceAccumulator
    positions: np.ndarray


@dataclass(frozen=True)
class ASRRunOutputState:
    """Append state for yearly and cumulative ASR run artifacts."""

    yearly: DownstreamRunOutputState
    cumulative: DownstreamRunOutputState | None
    yearly_convergence: ASRMetricConvergenceState
    cumulative_convergence: ASRMetricConvergenceState | None = None

    @property
    def completed_runs(self) -> int:
        """Return the completed yearly ASR run count."""
        return int(self.yearly.completed_runs)


def write_asr_run_outputs(
    *,
    paths: ASRUncertaintyRunPaths,
    plan: ASRUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    show_progress: bool = True,
    status: StatusSink | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Write ASR run values, summary statistics, and convergence status."""
    state = new_asr_run_output_state(paths=paths, plan=plan)
    try:
        state, convergence = append_asr_run_outputs(
            paths=paths,
            plan=plan,
            runtime=runtime,
            state=state,
            target_runs=runtime.n_runs,
            final_checkpoint=True,
            show_progress=show_progress,
            status=status,
        )
    finally:
        close_asr_run_output_state(state=state)
    return state.completed_runs, convergence


def new_asr_run_output_state(
    *,
    paths: ASRUncertaintyRunPaths,
    plan: ASRUncertaintyPlan,
) -> ASRRunOutputState:
    """Create append state for one ASR run output run."""
    return ASRRunOutputState(
        yearly=new_downstream_run_output_state(paths=_downstream_paths(paths=paths)),
        cumulative=(
            new_downstream_run_output_state(paths=_cumulative_downstream_paths(paths=paths))
            if plan.has_cumulative_outputs
            else None
        ),
        yearly_convergence=_metric_convergence_state(identity=plan.summary_identity),
        cumulative_convergence=(
            _metric_convergence_state(identity=plan.cumulative_summary_identity)
            if plan.has_cumulative_outputs
            else None
        ),
    )


def close_asr_run_output_state(*, state: ASRRunOutputState) -> None:
    """Release append state for one ASR run output run."""
    close_downstream_run_output_state(state=state.yearly)
    if state.cumulative is not None:
        close_downstream_run_output_state(state=state.cumulative)


def append_asr_run_outputs(
    *,
    paths: ASRUncertaintyRunPaths,
    plan: ASRUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    state: ASRRunOutputState,
    target_runs: int,
    final_checkpoint: bool,
    show_progress: bool = True,
    status: StatusSink | None = None,
) -> tuple[ASRRunOutputState, dict[str, Any] | None]:
    """Append one ASR run interval and update summaries."""
    if plan.asr_run_layout == "sparse_selected_rows":
        return _append_sparse_asr_run_outputs(
            paths=paths,
            plan=plan,
            runtime=runtime,
            state=state,
            target_runs=target_runs,
            final_checkpoint=final_checkpoint,
            show_progress=show_progress,
            status=status,
        )
    return _append_compact_asr_run_outputs(
        paths=paths,
        plan=plan,
        runtime=runtime,
        state=state,
        target_runs=target_runs,
        final_checkpoint=final_checkpoint,
        show_progress=show_progress,
        status=status,
    )


def _append_compact_asr_run_outputs(
    *,
    paths: ASRUncertaintyRunPaths,
    plan: ASRUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    state: ASRRunOutputState,
    target_runs: int,
    final_checkpoint: bool,
    show_progress: bool,
    status: StatusSink | None,
) -> tuple[ASRRunOutputState, dict[str, Any] | None]:
    completed = int(state.yearly.completed_runs)
    convergence = None
    checkpoints = ConvergenceCheckpointCursor.from_runtime(runtime=runtime)
    checkpoints.advance_to_completed(completed_runs=completed)
    progress = monte_carlo_run_progress(
        source=f"uncertainty_{runtime.family}",
        enabled=show_progress,
        status=status,
    )
    yearly_batch_index = int(state.yearly.batch_index)
    cumulative_state = state.cumulative
    cumulative_batch_index = 0 if cumulative_state is None else int(cumulative_state.batch_index)
    try:
        with ExitStack() as stack:
            yearly_writer = stack.enter_context(
                CompactRunMatrixWriter(
                    path=paths.public_runs,
                    output_format=runtime.output_format,
                    append_existing=completed > 0,
                )
            )
            cumulative_writer = (
                stack.enter_context(
                    CompactRunMatrixWriter(
                        path=paths.cumulative_runs,
                        output_format=runtime.output_format,
                        append_existing=int(cumulative_state.completed_runs) > 0,
                    )
                )
                if cumulative_state is not None
                else None
            )
            for (
                source_run_indices,
                values,
                cumulative_values,
            ) in iter_asr_compact_render_product_batches(
                plan=plan,
                output_format=runtime.output_format,
                start_run_index=completed,
                stop_run_index=int(target_runs),
                batch_size=runtime.batch_size,
            ):
                for start in range(0, len(source_run_indices), runtime.batch_size):
                    stop = min(start + runtime.batch_size, len(source_run_indices))
                    run_indices = source_run_indices[start:stop]
                    progress.begin(
                        label=monte_carlo_run_drawing_label(
                            start=int(run_indices[0]),
                            stop=int(run_indices[-1]) + 1,
                            max_runs=runtime.n_runs,
                            mode=runtime.mode,
                        )
                    )
                    yearly_writer.write_batch(
                        run_indices=run_indices,
                        values=values[start:stop],
                        batch_index=yearly_batch_index,
                    )
                    if cumulative_writer is not None and cumulative_values is not None:
                        cumulative_writer.write_batch(
                            run_indices=run_indices,
                            values=cumulative_values[start:stop],
                            batch_index=cumulative_batch_index,
                        )
                        cumulative_batch_index += 1
                    yearly_batch_index += 1
                    completed_run_count = int(run_indices[-1]) + 1
                    completed, convergence = _append_compact_summaries_and_check(
                        state=state,
                        values=values[start:stop],
                        cumulative_values=(
                            cumulative_values[start:stop]
                            if cumulative_state is not None and cumulative_values is not None
                            else None
                        ),
                        run_indices=run_indices,
                        plan=plan,
                        runtime=runtime,
                        check_convergence=checkpoints.reached(completed_runs=completed_run_count),
                    )
                    checkpoints.mark_checked(completed_runs=completed_run_count)
                    progress.complete(
                        label=_run_progress_label(
                            completed=completed,
                            target=runtime.n_runs,
                            mode=runtime.mode,
                        ),
                        persistent=monte_carlo_completion_is_persistent(
                            completed=completed,
                            max_runs=runtime.n_runs,
                            mode=runtime.mode,
                        ),
                    )
                    if convergence is not None:
                        break
                if convergence is not None:
                    break
    finally:
        progress.finish()
    return _finish_append(
        paths=paths,
        plan=plan,
        runtime=runtime,
        state=state,
        completed=completed,
        yearly_batch_index=yearly_batch_index,
        cumulative_batch_index=cumulative_batch_index,
        convergence=convergence,
        final_checkpoint=final_checkpoint,
    )


def _append_sparse_asr_run_outputs(
    *,
    paths: ASRUncertaintyRunPaths,
    plan: ASRUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    state: ASRRunOutputState,
    target_runs: int,
    final_checkpoint: bool,
    show_progress: bool,
    status: StatusSink | None,
) -> tuple[ASRRunOutputState, dict[str, Any] | None]:
    completed = int(state.yearly.completed_runs)
    convergence = None
    checkpoints = ConvergenceCheckpointCursor.from_runtime(runtime=runtime)
    checkpoints.advance_to_completed(completed_runs=completed)
    public_row_group_index = asr_sparse_public_row_group_membership_index(plan=plan)
    progress = monte_carlo_run_progress(
        source=f"uncertainty_{runtime.family}",
        enabled=show_progress,
        status=status,
    )
    yearly_batch_index = int(state.yearly.batch_index)
    cumulative_state = state.cumulative
    cumulative_batch_index = 0 if cumulative_state is None else int(cumulative_state.batch_index)
    try:
        with ExitStack() as stack:
            yearly_writer = stack.enter_context(
                SparseRunRowsWriter(
                    path=paths.public_runs,
                    output_format=runtime.output_format,
                    append_existing=completed > 0,
                )
            )
            cumulative_writer = (
                stack.enter_context(
                    CompactRunMatrixWriter(
                        path=paths.cumulative_runs,
                        output_format=runtime.output_format,
                        append_existing=int(cumulative_state.completed_runs) > 0,
                    )
                )
                if cumulative_state is not None
                else None
            )
            for run_indices, rows, cumulative_values in iter_asr_sparse_render_product_batches(
                plan=plan,
                output_format=runtime.output_format,
                start_run_index=completed,
                stop_run_index=int(target_runs),
                batch_size=runtime.batch_size,
            ):
                progress.begin(
                    label=monte_carlo_run_drawing_label(
                        start=int(run_indices[0]),
                        stop=int(run_indices[-1]) + 1,
                        max_runs=runtime.n_runs,
                        mode=runtime.mode,
                    )
                )
                yearly_writer.write_batch(rows=rows, batch_index=yearly_batch_index)
                if cumulative_writer is not None and cumulative_values is not None:
                    cumulative_writer.write_batch(
                        run_indices=run_indices,
                        values=cumulative_values,
                        batch_index=cumulative_batch_index,
                    )
                    cumulative_batch_index += 1
                yearly_batch_index += 1
                completed_run_count = int(run_indices[-1]) + 1
                completed, convergence = _append_sparse_summaries_and_check(
                    state=state,
                    rows=rows,
                    cumulative_values=cumulative_values,
                    run_indices=run_indices,
                    public_row_group_index=public_row_group_index,
                    plan=plan,
                    runtime=runtime,
                    check_convergence=checkpoints.reached(completed_runs=completed_run_count),
                )
                checkpoints.mark_checked(completed_runs=completed_run_count)
                progress.complete(
                    label=_run_progress_label(
                        completed=completed,
                        target=runtime.n_runs,
                        mode=runtime.mode,
                    ),
                    persistent=monte_carlo_completion_is_persistent(
                        completed=completed,
                        max_runs=runtime.n_runs,
                        mode=runtime.mode,
                    ),
                )
                if convergence is not None:
                    break
    finally:
        progress.finish()
    return _finish_append(
        paths=paths,
        plan=plan,
        runtime=runtime,
        state=state,
        completed=completed,
        yearly_batch_index=yearly_batch_index,
        cumulative_batch_index=cumulative_batch_index,
        convergence=convergence,
        final_checkpoint=final_checkpoint,
    )


def _finish_append(
    *,
    paths: ASRUncertaintyRunPaths,
    plan: ASRUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    state: ASRRunOutputState,
    completed: int,
    yearly_batch_index: int,
    cumulative_batch_index: int,
    convergence: dict[str, Any] | None,
    final_checkpoint: bool,
) -> tuple[ASRRunOutputState, dict[str, Any] | None]:
    convergence = final_downstream_convergence_payload(
        convergence=convergence,
        completed_runs=completed,
        runtime=runtime,
        final_checkpoint=final_checkpoint,
    )
    if final_checkpoint or convergence is not None:
        _write_summaries(paths=paths, plan=plan, runtime=runtime, run_count=completed, state=state)
    return (
        ASRRunOutputState(
            yearly=replace(
                state.yearly,
                completed_runs=completed,
                batch_index=yearly_batch_index,
            ),
            cumulative=replace(
                state.cumulative,
                completed_runs=completed,
                batch_index=cumulative_batch_index,
            )
            if state.cumulative is not None
            else None,
            yearly_convergence=state.yearly_convergence,
            cumulative_convergence=state.cumulative_convergence,
        ),
        convergence,
    )


def _downstream_paths(*, paths: ASRUncertaintyRunPaths) -> DownstreamRunOutputPaths:
    return DownstreamRunOutputPaths(
        run_root=paths.run_root,
        public_runs=paths.public_runs,
        summary_stats_runs=paths.summary_stats_runs,
    )


def _cumulative_downstream_paths(*, paths: ASRUncertaintyRunPaths) -> DownstreamRunOutputPaths:
    return DownstreamRunOutputPaths(
        run_root=paths.run_root,
        public_runs=paths.cumulative_runs,
        summary_stats_runs=paths.cumulative_summary_stats_runs,
    )


def _append_compact_summaries_and_check(
    *,
    state: ASRRunOutputState,
    values: np.ndarray,
    cumulative_values: np.ndarray | None,
    run_indices: np.ndarray,
    plan: ASRUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    check_convergence: bool,
) -> tuple[int, dict[str, Any] | None]:
    return _append_summaries_and_check(
        state=state,
        yearly_summary_values=collapse_asr_values_to_summary(values=values, plan=plan),
        cumulative_summary_values=(
            collapse_asr_cumulative_values_to_summary(values=cumulative_values, plan=plan)
            if cumulative_values is not None
            else None
        ),
        run_indices=run_indices,
        runtime=runtime,
        check_convergence=check_convergence,
    )


def _append_sparse_summaries_and_check(
    *,
    state: ASRRunOutputState,
    rows: SparseRunRows,
    cumulative_values: np.ndarray | None,
    run_indices: np.ndarray,
    public_row_group_index: np.ndarray,
    plan: ASRUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    check_convergence: bool,
) -> tuple[int, dict[str, Any] | None]:
    _update_sparse_metric_convergence_state(
        state=state.yearly_convergence,
        sparse_rows=rows,
        run_indices=run_indices,
        public_row_group_index=public_row_group_index,
    )
    cumulative_summary_values = (
        collapse_asr_cumulative_values_to_summary(values=cumulative_values, plan=plan)
        if cumulative_values is not None
        else None
    )
    if state.cumulative_convergence is not None and cumulative_summary_values is not None:
        _update_metric_convergence_state(
            state=state.cumulative_convergence,
            summary_values=cumulative_summary_values,
        )
    completed = int(run_indices[-1]) + 1
    convergence = mean_convergence_payload_for_targets(
        targets=_asr_convergence_targets(state=state),
        completed_runs=completed,
        runtime=runtime,
        check_convergence=check_convergence,
    )
    return completed, convergence


def _append_summaries_and_check(
    *,
    state: ASRRunOutputState,
    yearly_summary_values: np.ndarray,
    cumulative_summary_values: np.ndarray | None,
    run_indices: np.ndarray,
    runtime: UncertaintyRuntimeRequest,
    check_convergence: bool,
) -> tuple[int, dict[str, Any] | None]:
    _update_metric_convergence_state(
        state=state.yearly_convergence,
        summary_values=yearly_summary_values,
    )
    if state.cumulative_convergence is not None and cumulative_summary_values is not None:
        _update_metric_convergence_state(
            state=state.cumulative_convergence,
            summary_values=cumulative_summary_values,
        )
    completed = int(run_indices[-1]) + 1
    convergence = mean_convergence_payload_for_targets(
        targets=_asr_convergence_targets(state=state),
        completed_runs=completed,
        runtime=runtime,
        check_convergence=check_convergence,
    )
    return completed, convergence


def _metric_convergence_state(*, identity: Any) -> ASRMetricConvergenceState:
    metric = identity[ASR_SUMMARY_METRIC_COLUMN].astype(str)
    value_rows = identity.loc[metric.eq(_value_metric_for_identity(identity=identity))]
    if ASR_SUMMARY_SCOPE_COLUMN in value_rows.columns:
        scope = value_rows[ASR_SUMMARY_SCOPE_COLUMN].astype(str)
        inter_positions = np.flatnonzero(scope.eq(ASR_SUMMARY_SCOPE_INTER_METHOD).to_numpy())
        positions = (
            inter_positions if inter_positions.size else np.arange(len(value_rows), dtype=np.int64)
        )
    else:
        positions = np.arange(len(value_rows), dtype=np.int64)
    return ASRMetricConvergenceState(
        value=MeanConvergenceAccumulator.empty(row_count=len(positions)),
        frequency=MeanConvergenceAccumulator.empty(row_count=len(positions)),
        positions=positions,
    )


def _value_metric_for_identity(*, identity: Any) -> str:
    metric = identity[ASR_SUMMARY_METRIC_COLUMN].astype(str)
    if bool(metric.eq(ASR_CUMULATIVE_VALUE_METRIC).any()):
        return ASR_CUMULATIVE_VALUE_METRIC
    return ASR_VALUE_METRIC


def _update_metric_convergence_state(
    *,
    state: ASRMetricConvergenceState,
    summary_values: np.ndarray,
) -> None:
    values = summary_values[:, state.positions]
    observed = ~np.isnan(values)
    frequency = np.where(observed, values <= 1.0, np.nan).astype(np.float64)
    state.value.update(values=values)
    state.frequency.update(values=frequency)


def _update_sparse_metric_convergence_state(
    *,
    state: ASRMetricConvergenceState,
    sparse_rows: SparseRunRows,
    run_indices: np.ndarray,
    public_row_group_index: np.ndarray,
) -> None:
    """Accumulate sparse ASR value and frequency means for convergence checks."""
    row_runs, row_groups, values = sparse_rows_to_overlapping_group_values(
        sparse_rows=sparse_rows,
        run_indices=run_indices,
        public_row_group_index=public_row_group_index,
    )
    row_runs, row_groups, values = _select_sparse_summary_positions(
        row_runs=row_runs,
        row_groups=row_groups,
        values=values,
        positions=state.positions,
    )
    for groups, means in iter_sparse_group_mean_updates(
        row_runs=row_runs,
        row_groups=row_groups,
        values=values,
        group_count=len(state.positions),
    ):
        state.value.accumulate_group_observations(groups=groups, values=means)
        state.frequency.accumulate_group_observations(
            groups=groups,
            values=np.where(means <= 1.0, 1.0, 0.0),
        )


def _select_sparse_summary_positions(
    *,
    row_runs: np.ndarray,
    row_groups: np.ndarray,
    values: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter sparse summary groups to the ASR metric positions tracked by state."""
    if row_groups.size == 0 or positions.size == 0:
        empty_int = np.empty(0, dtype=np.int64)
        return empty_int, empty_int, np.empty(0, dtype=np.float64)
    lookup = np.full(int(positions[-1]) + 1, -1, dtype=np.int64)
    lookup[positions] = np.arange(len(positions), dtype=np.int64)
    in_range = row_groups <= int(positions[-1])
    selected = np.zeros(row_groups.shape, dtype=bool)
    selected[in_range] = lookup[row_groups[in_range]] >= 0
    return row_runs[selected], lookup[row_groups[selected]], values[selected]


def _asr_convergence_targets(
    *,
    state: ASRRunOutputState,
) -> tuple[MeanConvergenceAccumulator, ...]:
    """Return ASR value and frequency convergence targets in checkpoint order."""
    targets: list[MeanConvergenceAccumulator] = []
    for item in (state.yearly_convergence, state.cumulative_convergence):
        if item is None:
            continue
        targets.extend((item.value, item.frequency))
    return tuple(targets)


def _write_summaries(
    *,
    paths: ASRUncertaintyRunPaths,
    plan: ASRUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    run_count: int,
    state: ASRRunOutputState,
) -> None:
    write_asr_summary_table(
        path=paths.summary_stats_runs,
        summary_identity=plan.summary_identity,
        runs_path=paths.public_runs,
        run_count=run_count,
        output_format=runtime.output_format,
        public_row_groups=plan.summary_public_row_groups,
        sparse=plan.asr_run_layout == "sparse_selected_rows",
    )
    if state.cumulative is not None:
        write_asr_summary_table(
            path=paths.cumulative_summary_stats_runs,
            summary_identity=plan.cumulative_summary_identity,
            runs_path=paths.cumulative_runs,
            run_count=run_count,
            output_format=runtime.output_format,
            public_row_groups=plan.cumulative_summary_public_row_groups,
            sparse=False,
        )


def _run_progress_label(*, completed: int, target: int, mode: str) -> str:
    return monte_carlo_run_progress_label(completed=completed, max_runs=target, mode=mode)

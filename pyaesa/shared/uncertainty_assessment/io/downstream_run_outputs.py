"""Shared downstream Monte Carlo run output writer."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.shared.runtime.reporting.run_progress import (
    RunProgressPrinter,
    monte_carlo_run_drawing_label,
    monte_carlo_run_progress,
    monte_carlo_run_progress_label,
)
from pyaesa.shared.uncertainty_assessment.io.public_summary import (
    exact_summary_from_public_runs,
)
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import (
    iter_compact_run_matrix,
    iter_sparse_run_rows,
    iter_sparse_run_row_windows,
)
from pyaesa.shared.uncertainty_assessment.io.tables import (
    CompactRunMatrixWriter,
    SparseRunRows,
    SparseRunRowsWriter,
    write_uncertainty_table,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.convergence import (
    ConvergenceCheckpointCursor,
    MeanConvergenceAccumulator,
    ordered_mean_convergence_reached,
)
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest


@dataclass(frozen=True)
class DownstreamRunOutputPaths:
    """Common output paths for a downstream uncertainty run table."""

    run_root: Path
    public_runs: Path
    summary_stats_runs: Path


@dataclass(frozen=True)
class DownstreamRunOutputPlan:
    """Family supplied callbacks for downstream run output materialization."""

    run_layout: str
    summary_identity: pd.DataFrame
    public_row_count: int
    compact_batches: Callable[[str, int, int], Iterator[tuple[np.ndarray, np.ndarray]]]
    sparse_batches: Callable[[str, int, int], Iterator[tuple[np.ndarray, SparseRunRows]]]
    collapse_compact: Callable[[np.ndarray], np.ndarray]
    collapse_sparse: Callable[[SparseRunRows, np.ndarray, np.ndarray], np.ndarray]
    sparse_public_row_group_index: Callable[[], np.ndarray]
    empty_sparse_rows: Callable[[], SparseRunRows]
    summary_public_row_groups: tuple[tuple[str, ...], ...] | None = None


@dataclass(frozen=True)
class DownstreamRunOutputState:
    """Append state for one downstream run output run."""

    completed_runs: int = 0
    batch_index: int = 0
    convergence_state: MeanConvergenceAccumulator | None = None


def write_downstream_run_outputs(
    *,
    paths: DownstreamRunOutputPaths,
    plan: DownstreamRunOutputPlan,
    runtime: UncertaintyRuntimeRequest,
    show_progress: bool = True,
    progress: RunProgressPrinter | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Write downstream run values, exact summaries, and convergence status."""
    state = new_downstream_run_output_state(paths=paths)
    try:
        state, convergence = append_downstream_run_outputs(
            paths=paths,
            plan=plan,
            runtime=runtime,
            state=state,
            target_runs=runtime.n_runs,
            final_checkpoint=True,
            show_progress=show_progress,
            progress=progress,
        )
    finally:
        close_downstream_run_output_state(state=state)
    return state.completed_runs, convergence


def new_downstream_run_output_state(
    *,
    paths: DownstreamRunOutputPaths,
    completed_runs: int = 0,
) -> DownstreamRunOutputState:
    """Create a run local downstream run append state."""
    del paths
    return DownstreamRunOutputState(completed_runs=int(completed_runs))


def close_downstream_run_output_state(*, state: DownstreamRunOutputState) -> None:
    """Release downstream append state resources."""
    del state


def append_downstream_run_outputs(
    *,
    paths: DownstreamRunOutputPaths,
    plan: DownstreamRunOutputPlan,
    runtime: UncertaintyRuntimeRequest,
    state: DownstreamRunOutputState,
    target_runs: int,
    final_checkpoint: bool,
    show_progress: bool = True,
    progress: RunProgressPrinter | None = None,
    progress_mode: str | None = None,
    progress_max_runs: int | None = None,
    progress_component: bool = False,
) -> tuple[DownstreamRunOutputState, dict[str, Any] | None]:
    """Append one downstream run interval and update exact summaries."""
    state = _initialize_downstream_convergence_state(
        paths=paths,
        plan=plan,
        runtime=runtime,
        state=state,
    )
    if int(target_runs) <= int(state.completed_runs):
        return _finalize_existing_downstream_outputs(
            paths=paths,
            plan=plan,
            runtime=runtime,
            state=state,
            final_checkpoint=final_checkpoint,
        )
    if plan.run_layout == "sparse_selected_rows":
        return _append_sparse_run_outputs(
            paths=paths,
            plan=plan,
            runtime=runtime,
            state=state,
            target_runs=target_runs,
            final_checkpoint=final_checkpoint,
            show_progress=show_progress,
            progress=progress,
            progress_mode=progress_mode,
            progress_max_runs=progress_max_runs,
            progress_component=progress_component,
        )
    return _append_compact_run_outputs(
        paths=paths,
        plan=plan,
        runtime=runtime,
        state=state,
        target_runs=target_runs,
        final_checkpoint=final_checkpoint,
        show_progress=show_progress,
        progress=progress,
        progress_mode=progress_mode,
        progress_max_runs=progress_max_runs,
        progress_component=progress_component,
    )


def _finalize_existing_downstream_outputs(
    *,
    paths: DownstreamRunOutputPaths,
    plan: DownstreamRunOutputPlan,
    runtime: UncertaintyRuntimeRequest,
    state: DownstreamRunOutputState,
    final_checkpoint: bool,
) -> tuple[DownstreamRunOutputState, dict[str, Any] | None]:
    convergence = final_downstream_convergence_payload(
        convergence=None,
        completed_runs=state.completed_runs,
        runtime=runtime,
        final_checkpoint=final_checkpoint,
    )
    if final_checkpoint:
        write_downstream_summary(
            paths=paths,
            plan=plan,
            run_count=state.completed_runs,
            output_format=runtime.output_format,
            sparse=plan.run_layout == "sparse_selected_rows",
        )
    return state, convergence


def _append_compact_run_outputs(
    *,
    paths: DownstreamRunOutputPaths,
    plan: DownstreamRunOutputPlan,
    runtime: UncertaintyRuntimeRequest,
    state: DownstreamRunOutputState,
    target_runs: int,
    final_checkpoint: bool,
    show_progress: bool,
    progress: RunProgressPrinter | None,
    progress_mode: str | None,
    progress_max_runs: int | None,
    progress_component: bool,
) -> tuple[DownstreamRunOutputState, dict[str, Any] | None]:
    completed = int(state.completed_runs)
    batch_index = int(state.batch_index)
    convergence = None
    checkpoints = ConvergenceCheckpointCursor.from_runtime(runtime=runtime)
    checkpoints.advance_to_completed(completed_runs=completed)
    own_progress = progress is None
    if own_progress:
        progress = monte_carlo_run_progress(
            source=f"uncertainty_{runtime.family}",
            enabled=show_progress,
        )
    progress_target = runtime.n_runs if progress_max_runs is None else progress_max_runs
    progress_label_mode = runtime.mode if progress_mode is None else progress_mode
    try:
        with CompactRunMatrixWriter(
            path=paths.public_runs,
            output_format=runtime.output_format,
            append_existing=completed > 0,
        ) as writer:
            for source_run_indices, source_values in plan.compact_batches(
                runtime.output_format,
                completed,
                int(target_runs),
            ):
                for start in range(0, len(source_run_indices), runtime.batch_size):
                    stop = min(start + runtime.batch_size, len(source_run_indices))
                    run_indices = source_run_indices[start:stop]
                    values = source_values[start:stop]
                    progress.begin(
                        label=monte_carlo_run_drawing_label(
                            start=int(run_indices[0]),
                            stop=int(run_indices[-1]) + 1,
                            max_runs=progress_target,
                            mode=progress_label_mode,
                            component=progress_component,
                        )
                    )
                    writer.write_batch(
                        run_indices=run_indices,
                        values=values,
                        batch_index=batch_index,
                    )
                    batch_index += 1
                    completed_run_count = int(run_indices[-1]) + 1
                    check_convergence = checkpoints.reached(completed_runs=completed_run_count)
                    completed, convergence = append_downstream_summary_values(
                        convergence_state=state.convergence_state,
                        summary_values=plan.collapse_compact(values),
                        run_indices=run_indices,
                        runtime=runtime,
                        check_convergence=check_convergence,
                    )
                    if check_convergence:
                        checkpoints.mark_checked(completed_runs=completed_run_count)
                    progress.complete(
                        label=monte_carlo_run_progress_label(
                            completed=completed,
                            max_runs=progress_target,
                            mode=progress_label_mode,
                            component=progress_component,
                        ),
                        persistent=str(progress_label_mode) == "fixed",
                    )
                    if convergence is not None:
                        break
                if convergence is not None:
                    break
        convergence = final_downstream_convergence_payload(
            convergence=convergence,
            completed_runs=completed,
            runtime=runtime,
            final_checkpoint=final_checkpoint,
        )
        if final_checkpoint or convergence is not None:
            write_downstream_summary(
                paths=paths,
                plan=plan,
                run_count=completed,
                output_format=runtime.output_format,
                sparse=False,
            )
    finally:
        if own_progress:
            progress.finish()
    return replace(state, completed_runs=completed, batch_index=batch_index), convergence


def _append_sparse_run_outputs(
    *,
    paths: DownstreamRunOutputPaths,
    plan: DownstreamRunOutputPlan,
    runtime: UncertaintyRuntimeRequest,
    state: DownstreamRunOutputState,
    target_runs: int,
    final_checkpoint: bool,
    show_progress: bool,
    progress: RunProgressPrinter | None,
    progress_mode: str | None,
    progress_max_runs: int | None,
    progress_component: bool,
) -> tuple[DownstreamRunOutputState, dict[str, Any] | None]:
    completed = int(state.completed_runs)
    batch_index = int(state.batch_index)
    convergence = None
    checkpoints = ConvergenceCheckpointCursor.from_runtime(runtime=runtime)
    checkpoints.advance_to_completed(completed_runs=completed)
    public_row_group_index = plan.sparse_public_row_group_index()
    own_progress = progress is None
    if own_progress:
        progress = monte_carlo_run_progress(
            source=f"uncertainty_{runtime.family}",
            enabled=show_progress,
        )
    progress_target = runtime.n_runs if progress_max_runs is None else progress_max_runs
    progress_label_mode = runtime.mode if progress_mode is None else progress_mode
    try:
        with SparseRunRowsWriter(
            path=paths.public_runs,
            output_format=runtime.output_format,
            append_existing=completed > 0,
        ) as writer:
            source_chunks = (
                rows
                for _run_indices, rows in plan.sparse_batches(
                    runtime.output_format,
                    completed,
                    int(target_runs),
                )
            )
            for run_indices, rows in iter_sparse_run_row_windows(
                chunks=source_chunks,
                start_run_index=completed,
                stop_run_index=int(target_runs),
                batch_size=runtime.batch_size,
                empty_rows=plan.empty_sparse_rows(),
            ):
                progress.begin(
                    label=monte_carlo_run_drawing_label(
                        start=int(run_indices[0]),
                        stop=int(run_indices[-1]) + 1,
                        max_runs=progress_target,
                        mode=progress_label_mode,
                        component=progress_component,
                    )
                )
                completed_run_count = int(run_indices[-1]) + 1
                writer.write_batch(rows=rows, batch_index=batch_index)
                check_convergence = checkpoints.reached(completed_runs=completed_run_count)
                completed, convergence = append_downstream_summary_values(
                    convergence_state=state.convergence_state,
                    summary_values=plan.collapse_sparse(
                        rows,
                        run_indices,
                        public_row_group_index,
                    ),
                    run_indices=run_indices,
                    runtime=runtime,
                    check_convergence=check_convergence,
                )
                if check_convergence:
                    checkpoints.mark_checked(completed_runs=completed_run_count)
                progress.complete(
                    label=monte_carlo_run_progress_label(
                        completed=completed,
                        max_runs=progress_target,
                        mode=progress_label_mode,
                        component=progress_component,
                    ),
                    persistent=str(progress_label_mode) == "fixed",
                )
                batch_index += 1
                if convergence is not None:
                    break
        convergence = final_downstream_convergence_payload(
            convergence=convergence,
            completed_runs=completed,
            runtime=runtime,
            final_checkpoint=final_checkpoint,
        )
        if final_checkpoint or convergence is not None:
            write_downstream_summary(
                paths=paths,
                plan=plan,
                run_count=completed,
                output_format=runtime.output_format,
                sparse=True,
            )
    finally:
        if own_progress:
            progress.finish()
    return replace(state, completed_runs=completed, batch_index=batch_index), convergence


def _initialize_downstream_convergence_state(
    *,
    paths: DownstreamRunOutputPaths,
    plan: DownstreamRunOutputPlan,
    runtime: UncertaintyRuntimeRequest,
    state: DownstreamRunOutputState,
) -> DownstreamRunOutputState:
    if runtime.mode != "convergence" or state.convergence_state is not None:
        return state
    convergence_state = MeanConvergenceAccumulator.empty(row_count=len(plan.summary_identity))
    if state.completed_runs > 0:
        if plan.run_layout == "sparse_selected_rows":
            _replay_sparse_convergence_state(
                paths=paths,
                plan=plan,
                runtime=runtime,
                completed_runs=state.completed_runs,
                convergence_state=convergence_state,
            )
        else:
            _replay_compact_convergence_state(
                paths=paths,
                plan=plan,
                runtime=runtime,
                completed_runs=state.completed_runs,
                convergence_state=convergence_state,
            )
        convergence_state.record_baseline(completed_runs=state.completed_runs)
    return replace(state, convergence_state=convergence_state)


def _replay_compact_convergence_state(
    *,
    paths: DownstreamRunOutputPaths,
    plan: DownstreamRunOutputPlan,
    runtime: UncertaintyRuntimeRequest,
    completed_runs: int,
    convergence_state: MeanConvergenceAccumulator,
) -> None:
    for _run_indices, values in iter_compact_run_matrix(
        path=paths.public_runs,
        output_format=runtime.output_format,
        column_count=plan.public_row_count,
        stop_run_index=int(completed_runs),
    ):
        convergence_state.update(values=plan.collapse_compact(values))


def _replay_sparse_convergence_state(
    *,
    paths: DownstreamRunOutputPaths,
    plan: DownstreamRunOutputPlan,
    runtime: UncertaintyRuntimeRequest,
    completed_runs: int,
    convergence_state: MeanConvergenceAccumulator,
) -> None:
    public_row_group_index = plan.sparse_public_row_group_index()
    windows = iter_sparse_run_row_windows(
        chunks=iter_sparse_run_rows(
            path=paths.public_runs,
            output_format=runtime.output_format,
            stop_run_index=int(completed_runs),
        ),
        start_run_index=0,
        stop_run_index=int(completed_runs),
        batch_size=runtime.batch_size,
        empty_rows=plan.empty_sparse_rows(),
    )
    for run_indices, rows in windows:
        convergence_state.update(
            values=plan.collapse_sparse(rows, run_indices, public_row_group_index)
        )


def append_downstream_summary_values(
    *,
    convergence_state: MeanConvergenceAccumulator | None,
    summary_values: np.ndarray,
    run_indices: np.ndarray,
    runtime: UncertaintyRuntimeRequest,
    check_convergence: bool = True,
) -> tuple[int, dict[str, Any] | None]:
    """Append summary values and return convergence status when requested."""
    if convergence_state is not None:
        convergence_state.update(values=summary_values)
    completed = int(run_indices[-1]) + 1
    if not check_convergence or runtime.mode != "convergence":
        return completed, None
    target = cast(MeanConvergenceAccumulator, convergence_state)
    reached = ordered_mean_convergence_reached(
        targets=(target,),
        completed_runs=completed,
        rtol=runtime.rtol,
    )
    convergence = (
        downstream_convergence_payload(
            reached=True,
            completed_runs=completed,
            runtime=runtime,
        )
        if reached
        else None
    )
    return completed, convergence


def final_downstream_convergence_payload(
    *,
    convergence: dict[str, Any] | None,
    completed_runs: int,
    runtime: UncertaintyRuntimeRequest,
    final_checkpoint: bool,
) -> dict[str, Any] | None:
    """Return the final convergence payload for a downstream run output."""
    if runtime.mode != "convergence" or convergence is not None:
        return convergence
    if not final_checkpoint:
        return None
    return downstream_convergence_payload(
        reached=False,
        completed_runs=completed_runs,
        runtime=runtime,
    )


def write_downstream_summary(
    *,
    paths: DownstreamRunOutputPaths,
    plan: DownstreamRunOutputPlan,
    run_count: int,
    output_format: str,
    sparse: bool,
) -> None:
    """Write exact summary statistics from downstream public run artifacts."""
    summary = exact_summary_from_public_runs(
        identity_frame=plan.summary_identity,
        runs_path=paths.public_runs,
        output_format=output_format,
        run_count=run_count,
        public_row_groups=plan.summary_public_row_groups,
        sparse=sparse,
    )
    write_uncertainty_table(
        path=paths.summary_stats_runs,
        frame=summary,
        output_format=output_format,
    )


def downstream_convergence_payload(
    *,
    reached: bool,
    completed_runs: int,
    runtime: UncertaintyRuntimeRequest,
) -> dict[str, Any]:
    """Return the common Monte Carlo convergence payload."""
    return {
        "reached": bool(reached),
        "completed_runs": int(completed_runs),
        "max_runs": int(runtime.max_runs),
        "rtol": float(runtime.rtol),
        "stable_runs": int(runtime.stable_runs),
        "statistics": list(runtime.convergence_statistics),
    }

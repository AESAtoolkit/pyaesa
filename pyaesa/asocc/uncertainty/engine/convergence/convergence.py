"""Convergence mode execution for aSoCC Monte Carlo runs."""

from dataclasses import dataclass
from typing import Any
from typing import cast

import numpy as np

from pyaesa.asocc.uncertainty.inputs.external_rows import ExternalAsoccRowsPlan
from pyaesa.asocc.uncertainty.inputs.external_rows import ExternalAsoccRunInventoryExhausted
from pyaesa.asocc.uncertainty.sources.inter_method import InterMethodPlan
from pyaesa.asocc.uncertainty.sources.inter_mrio import InterMrioPlan
from pyaesa.asocc.uncertainty.sources.lcia import LCIAPlan
from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.asocc.uncertainty.sources.projection import ProjectionPlan
from pyaesa.asocc.uncertainty.engine.monte_carlo.sampling import sample_compact_batch
from pyaesa.asocc.uncertainty.engine.convergence.state import (
    initial_convergence_state,
    remaining_run_plan,
)
from pyaesa.asocc.uncertainty.engine.inter_method.execution import InterMethodExecutionPlan
from pyaesa.asocc.uncertainty.engine.inter_method.sampling import (
    external_run_offsets_for_start,
    sample_sparse_inter_method_batch,
)
from pyaesa.asocc.uncertainty.engine.evaluation.summary_identity import summary_identity_groups
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    collapse_values_to_summary_groups,
    collapse_sparse_rows_to_overlapping_summary_groups,
    sparse_public_row_group_membership_index,
)
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import (
    iter_compact_run_matrix,
    iter_sparse_run_rows,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.convergence import (
    ConvergenceCheckpointCursor,
    MeanConvergenceAccumulator,
    ordered_mean_convergence_reached,
)
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan
from pyaesa.shared.uncertainty_assessment.io.tables import (
    CompactRunMatrixWriter,
    SparseRunRowsWriter,
    write_uncertainty_table,
)
from pyaesa.shared.runtime.reporting.run_progress import (
    RunProgressPrinter,
    monte_carlo_run_drawing_label,
    monte_carlo_run_progress_label,
)


@dataclass(frozen=True)
class MonteCarloRunResult:
    """Completed run count and optional convergence status for one run."""

    completed_runs: int
    summary_run_count: int
    public_runs_sparse: bool = False
    convergence: dict[str, Any] | None = None


def write_convergence_batches(
    *,
    paths: AsoccUncertaintyRunPaths,
    loaded,
    inter_mrio_plan: InterMrioPlan | None,
    lcia_plan: LCIAPlan | None,
    projection_plan: ProjectionPlan | None,
    runtime,
    sources: SourceActivationPlan,
    external_plan: ExternalAsoccRowsPlan,
    run_seed: int,
    start_run_index: int,
    append_existing: bool,
    progress: RunProgressPrinter,
) -> MonteCarloRunResult:
    """Write convergence mode batches until stability or max runs."""
    state = initial_convergence_state(
        paths=paths,
        output_format=runtime.output_format,
        sources=sources,
        append_existing=append_existing,
        sparse=False,
        completed_runs=start_run_index,
    )
    identity_written = state.identity_written
    row_count = state.row_count
    public_row_groups = state.public_row_groups
    completed_runs = state.completed_runs
    reached = False
    stop_reason: str | None = None
    checkpoints = ConvergenceCheckpointCursor.from_runtime(runtime=runtime)
    checkpoints.advance_to_completed(completed_runs=completed_runs)
    accumulator = MeanConvergenceAccumulator.empty(row_count=row_count)
    if append_existing and row_count:
        _replay_existing_compact_to_accumulator(
            accumulator=accumulator,
            paths=paths,
            output_format=runtime.output_format,
            public_row_groups=public_row_groups,
            public_column_count=len(cast(Any, state.identity)),
        )
        accumulator.record_baseline(completed_runs=completed_runs)
    with CompactRunMatrixWriter(
        path=paths.public_runs,
        output_format=runtime.output_format,
        append_existing=append_existing,
    ) as matrix_writer:
        plan = remaining_run_plan(
            n_runs=runtime.n_runs,
            batch_size=runtime.batch_size,
            run_seed=run_seed,
            start_run_index=start_run_index,
        )
        batches = () if plan is None else plan.batches()
        for batch in batches:
            progress.begin(
                label=monte_carlo_run_drawing_label(
                    start=batch.start_run_index,
                    stop=batch.stop_run_index,
                    max_runs=runtime.n_runs,
                    mode="convergence",
                )
            )
            try:
                identity, run_indices, values = sample_compact_batch(
                    loaded=loaded,
                    inter_mrio_plan=inter_mrio_plan,
                    lcia_plan=lcia_plan,
                    projection_plan=projection_plan,
                    batch=batch,
                    sources=sources,
                    external_plan=external_plan,
                )
            except ExternalAsoccRunInventoryExhausted as exc:
                stop_reason = str(exc)
                break
            if not identity_written:
                write_uncertainty_table(
                    path=paths.public_row_identity,
                    frame=identity,
                    output_format=runtime.output_format,
                )
                _summary_identity, public_row_groups = summary_identity_groups(
                    identity=identity,
                    sources=sources,
                )
                row_count = len(public_row_groups)
                identity_written = True
            convergence_values = collapse_values_to_summary_groups(
                values=values,
                public_row_groups=public_row_groups,
            )
            if accumulator.sums.size == 0:
                accumulator = MeanConvergenceAccumulator.empty(row_count=len(public_row_groups))
            matrix_writer.write_batch(
                run_indices=run_indices,
                values=values,
                batch_index=batch.batch_index,
            )
            accumulator.update(values=convergence_values)
            completed_runs = int(batch.stop_run_index)
            progress.complete(
                label=monte_carlo_run_progress_label(
                    completed=completed_runs,
                    max_runs=runtime.n_runs,
                    mode="convergence",
                ),
                persistent=False,
            )
            if checkpoints.reached(completed_runs=completed_runs):
                if ordered_mean_convergence_reached(
                    targets=(accumulator,),
                    completed_runs=completed_runs,
                    rtol=runtime.rtol,
                ):
                    reached = True
                    break
                checkpoints.mark_checked(completed_runs=completed_runs)
    convergence = {
        "reached": reached,
        "statistics": ["mean"],
        "rtol": runtime.rtol,
        "stable_runs": runtime.stable_runs,
        "stable_run_count": accumulator.stable_run_count,
        "last_check_runs": accumulator.last_check_runs,
    }
    if stop_reason is not None:
        convergence["reason"] = stop_reason
    return MonteCarloRunResult(
        completed_runs=completed_runs,
        convergence=convergence,
        summary_run_count=completed_runs,
    )


def write_sparse_inter_method_convergence_batches(
    *,
    paths: AsoccUncertaintyRunPaths,
    loaded,
    inter_method_plan: InterMethodPlan,
    inter_method_execution_plan: InterMethodExecutionPlan,
    inter_mrio_plan: InterMrioPlan | None,
    runtime,
    sources: SourceActivationPlan,
    run_seed: int,
    start_run_index: int,
    append_existing: bool,
    progress: RunProgressPrinter,
) -> MonteCarloRunResult:
    """Write inter-method convergence batches as sparse selected rows."""
    state = initial_convergence_state(
        paths=paths,
        output_format=runtime.output_format,
        sources=sources,
        append_existing=append_existing,
        sparse=True,
        completed_runs=start_run_index,
    )
    identity = state.identity
    public_row_groups = state.public_row_groups
    row_count = state.row_count
    completed_runs = state.completed_runs
    reached = False
    stop_reason: str | None = None
    checkpoints = ConvergenceCheckpointCursor.from_runtime(runtime=runtime)
    checkpoints.advance_to_completed(completed_runs=completed_runs)
    identity_written = state.identity_written
    public_row_group_index = state.public_row_group_index
    accumulator = MeanConvergenceAccumulator.empty(row_count=row_count)
    if append_existing and row_count:
        _replay_existing_sparse_to_accumulator(
            accumulator=accumulator,
            paths=paths,
            output_format=runtime.output_format,
            public_row_groups=public_row_groups,
            public_row_group_index=cast(np.ndarray, public_row_group_index),
        )
        accumulator.record_baseline(completed_runs=completed_runs)
    external_render_offsets = external_run_offsets_for_start(
        inter_method_plan=inter_method_plan,
        start_run_index=start_run_index,
        external_labels=tuple(
            source.selection.asocc_method_label
            for branch in inter_method_execution_plan.branches
            for source in branch.external_plan.monte_carlo_sources
        ),
    )
    with SparseRunRowsWriter(
        path=paths.public_runs,
        output_format=runtime.output_format,
        append_existing=append_existing,
    ) as row_writer:
        plan = remaining_run_plan(
            n_runs=runtime.n_runs,
            batch_size=runtime.batch_size,
            run_seed=run_seed,
            start_run_index=start_run_index,
        )
        batches = () if plan is None else plan.batches()
        for batch in batches:
            progress.begin(
                label=monte_carlo_run_drawing_label(
                    start=batch.start_run_index,
                    stop=batch.stop_run_index,
                    max_runs=runtime.n_runs,
                    mode="convergence",
                )
            )
            try:
                sparse_batch = sample_sparse_inter_method_batch(
                    loaded=loaded,
                    inter_method_plan=inter_method_plan,
                    execution_plan=inter_method_execution_plan,
                    inter_mrio_plan=inter_mrio_plan,
                    batch=batch,
                    sources=sources,
                    identity=identity,
                    external_render_offsets=external_render_offsets,
                )
            except ExternalAsoccRunInventoryExhausted as exc:
                stop_reason = str(exc)
                break
            for label, count in sparse_batch.external_run_counts.items():
                external_render_offsets[label] = external_render_offsets.get(label, 0) + count
            if not identity_written:
                identity = sparse_batch.identity
                write_uncertainty_table(
                    path=paths.public_row_identity,
                    frame=identity,
                    output_format=runtime.output_format,
                )
                _summary_identity, public_row_groups = summary_identity_groups(
                    identity=identity,
                    sources=sources,
                )
                public_row_group_index = sparse_public_row_group_membership_index(
                    public_row_groups=public_row_groups
                )
                row_count = len(public_row_groups)
                accumulator = MeanConvergenceAccumulator.empty(row_count=row_count)
                identity_written = True
            convergence_values = collapse_sparse_rows_to_overlapping_summary_groups(
                sparse_rows=sparse_batch.sparse_rows,
                run_indices=sparse_batch.run_indices,
                public_row_groups=public_row_groups,
                public_row_group_index=cast(np.ndarray, public_row_group_index),
            )
            row_writer.write_batch(rows=sparse_batch.sparse_rows, batch_index=batch.batch_index)
            accumulator.update(values=convergence_values)
            completed_runs = int(batch.stop_run_index)
            progress.complete(
                label=monte_carlo_run_progress_label(
                    completed=completed_runs,
                    max_runs=runtime.n_runs,
                    mode="convergence",
                ),
                persistent=False,
            )
            if checkpoints.reached(completed_runs=completed_runs):
                if ordered_mean_convergence_reached(
                    targets=(accumulator,),
                    completed_runs=completed_runs,
                    rtol=runtime.rtol,
                ):
                    reached = True
                    break
                checkpoints.mark_checked(completed_runs=completed_runs)
    return MonteCarloRunResult(
        completed_runs=completed_runs,
        convergence={
            "reached": reached,
            "statistics": ["mean"],
            "rtol": runtime.rtol,
            "stable_runs": runtime.stable_runs,
            "stable_run_count": accumulator.stable_run_count,
            "last_check_runs": accumulator.last_check_runs,
            **({} if stop_reason is None else {"reason": stop_reason}),
        },
        summary_run_count=completed_runs,
        public_runs_sparse=True,
    )


def _replay_existing_compact_to_accumulator(
    *,
    accumulator: MeanConvergenceAccumulator,
    paths: AsoccUncertaintyRunPaths,
    output_format: str,
    public_row_groups: tuple[tuple[str, ...], ...],
    public_column_count: int,
) -> None:
    for _run_indices, values in iter_compact_run_matrix(
        path=paths.public_runs,
        output_format=output_format,
        column_count=public_column_count,
    ):
        accumulator.update(
            values=collapse_values_to_summary_groups(
                values=values,
                public_row_groups=public_row_groups,
            )
        )


def _replay_existing_sparse_to_accumulator(
    *,
    accumulator: MeanConvergenceAccumulator,
    paths: AsoccUncertaintyRunPaths,
    output_format: str,
    public_row_groups: tuple[tuple[str, ...], ...],
    public_row_group_index: np.ndarray,
) -> None:
    for rows in iter_sparse_run_rows(path=paths.public_runs, output_format=output_format):
        first_run = int(rows.run_index[0])
        last_run = int(rows.run_index[-1])
        accumulator.update(
            values=collapse_sparse_rows_to_overlapping_summary_groups(
                sparse_rows=rows,
                run_indices=np.arange(first_run, last_run + 1, dtype=np.int64),
                public_row_groups=public_row_groups,
                public_row_group_index=public_row_group_index,
            )
        )

"""IO-LCA Monte Carlo run materialization."""

from collections.abc import Iterator
from typing import Any

import numpy as np

from pyaesa.io_lca.uncertainty.evaluation.sampling import sample_io_lca_lcia_matrix
from pyaesa.io_lca.uncertainty.runtime.models import (
    IOLCAUncertaintyPlan,
    IOLCAUncertaintyRunPaths,
)
from pyaesa.shared.uncertainty_assessment.io.downstream_run_outputs import (
    DownstreamRunOutputState,
    DownstreamRunOutputPaths,
    DownstreamRunOutputPlan,
    append_downstream_run_outputs,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import (
    fixed_run_plan,
    run_seed_from_run_id,
)
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest
from pyaesa.shared.uncertainty_assessment.io.tables import SparseRunRows
from pyaesa.shared.runtime.reporting.run_progress import RunProgressPrinter


def append_io_lca_run_outputs(
    *,
    paths: IOLCAUncertaintyRunPaths,
    plan: IOLCAUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    seed_run_id: str,
    state: DownstreamRunOutputState,
    target_runs: int,
    final_checkpoint: bool,
    show_progress: bool,
    progress: RunProgressPrinter | None = None,
    progress_mode: str | None = None,
    progress_max_runs: int | None = None,
    progress_component: bool = False,
) -> tuple[DownstreamRunOutputState, dict[str, Any] | None]:
    """Append IO-LCA run values and optionally write final summaries."""
    return append_downstream_run_outputs(
        paths=_downstream_paths(paths=paths),
        plan=_downstream_plan(plan=plan, runtime=runtime, seed_run_id=seed_run_id),
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


def _downstream_paths(*, paths: IOLCAUncertaintyRunPaths) -> DownstreamRunOutputPaths:
    return DownstreamRunOutputPaths(
        run_root=paths.run_root,
        public_runs=paths.public_runs,
        summary_stats_runs=paths.summary_stats_runs,
    )


def _downstream_plan(
    *,
    plan: IOLCAUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    seed_run_id: str,
) -> DownstreamRunOutputPlan:
    return DownstreamRunOutputPlan(
        run_layout="compact_run_matrix",
        summary_identity=plan.identity,
        public_row_count=len(plan.identity),
        compact_batches=lambda _output_format, start, stop: _sample_batches(
            plan=plan,
            runtime=runtime,
            seed_run_id=seed_run_id,
            start=start,
            stop=stop,
        ),
        sparse_batches=lambda _output_format, _start, _stop: iter(()),
        collapse_compact=lambda values: values,
        collapse_sparse=lambda _rows, _run_indices, _public_row_group_index: np.empty(
            (0, 0),
            dtype=np.float64,
        ),
        sparse_public_row_group_index=lambda: np.empty(0, dtype=np.int64),
        empty_sparse_rows=lambda: SparseRunRows(
            run_index=np.empty(0, dtype=np.int64),
            public_row_id=np.empty(0, dtype=np.int64),
            values=np.empty(0, dtype=np.float64),
            value_column="lca",
        ),
    )


def _sample_batches(
    *,
    plan: IOLCAUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    seed_run_id: str,
    start: int,
    stop: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    run_plan = fixed_run_plan(
        n_runs=int(stop) - int(start),
        batch_size=runtime.batch_size,
        seed=run_seed_from_run_id(run_id=seed_run_id),
        start_run_index=int(start),
    )
    for batch in run_plan.batches():
        yield batch.run_indices(), sample_io_lca_lcia_matrix(plan=plan, batch=batch)

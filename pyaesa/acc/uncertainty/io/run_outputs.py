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
)
from pyaesa.shared.uncertainty_assessment.io.downstream_run_outputs import (
    DownstreamRunOutputState,
    DownstreamRunOutputPaths,
    DownstreamRunOutputPlan,
    append_downstream_run_outputs,
    close_downstream_run_output_state,
    new_downstream_run_output_state,
    write_downstream_run_outputs,
)
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest
from pyaesa.shared.uncertainty_assessment.io.run_writers import SparseRunRows


def write_acc_run_outputs(
    *,
    paths: ACCUncertaintyRunPaths,
    plan: ACCUncertaintyPlan,
    runtime: UncertaintyRuntimeRequest,
    show_progress: bool = True,
) -> tuple[int, dict[str, Any] | None]:
    """Write ACC run values, summary statistics, and convergence status."""
    return write_downstream_run_outputs(
        paths=_downstream_paths(paths=paths),
        plan=_downstream_plan(plan=plan),
        runtime=runtime,
        show_progress=show_progress,
    )


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
    return append_downstream_run_outputs(
        paths=_downstream_paths(paths=paths),
        plan=_downstream_plan(plan=plan),
        runtime=runtime,
        state=state,
        target_runs=target_runs,
        final_checkpoint=final_checkpoint,
        show_progress=show_progress,
    )


def _downstream_paths(*, paths: ACCUncertaintyRunPaths) -> DownstreamRunOutputPaths:
    return DownstreamRunOutputPaths(
        run_root=paths.run_root,
        public_runs=paths.public_runs,
        summary_stats_runs=paths.summary_stats_runs,
    )


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

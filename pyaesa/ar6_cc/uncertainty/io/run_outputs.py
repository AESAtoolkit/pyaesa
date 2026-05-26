"""AR6 CC Monte Carlo run materialization."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyaesa.ar6_cc.uncertainty.evaluation.sampling import sample_ar6_cc_sparse_rows
from pyaesa.ar6_cc.uncertainty.io.period_dispatch import (
    budget_identity_and_segments,
    budget_matrix_from_full_sparse_rows,
    remap_sparse_rows,
    trajectory_segment,
)
from pyaesa.ar6_cc.uncertainty.runtime.models import (
    AR6CCUncertaintyPlan,
    AR6CCUncertaintyRunPaths,
)
from pyaesa.shared.runtime.reporting.run_progress import (
    RunProgressPrinter,
    monte_carlo_run_drawing_label,
    monte_carlo_run_progress_label,
)
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    run_positions_in_window,
)
from pyaesa.shared.uncertainty_assessment.io.public_summary import (
    exact_summary_from_public_runs,
)
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import (
    iter_compact_run_matrix,
    iter_sparse_run_rows,
)
from pyaesa.shared.uncertainty_assessment.io.run_writers import (
    CompactRunMatrixWriter,
    SparseRunRows,
    SparseRunRowsWriter,
)
from pyaesa.shared.uncertainty_assessment.io.tables import write_uncertainty_table
from pyaesa.shared.uncertainty_assessment.monte_carlo.convergence import (
    ConvergenceCheckpointCursor,
    MeanConvergenceAccumulator,
    mean_convergence_payload,
    mean_convergence_payload_for_targets,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import (
    run_seed_from_run_id,
    fixed_run_plan,
)
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest


def write_ar6_cc_study_post_outputs(
    *,
    paths: AR6CCUncertaintyRunPaths,
    plan: AR6CCUncertaintyPlan,
    study_years: list[int],
    post_study_years: list[int],
    runtime: UncertaintyRuntimeRequest,
    seed_run_id: str | None = None,
    start_run_index: int = 0,
    progress: RunProgressPrinter,
    progress_mode: str | None = None,
    progress_max_runs: int | None = None,
    progress_component: bool = False,
    states: dict[str, dict[str, Any]] | None = None,
    final_checkpoint: bool = True,
) -> tuple[int, dict[str, Any] | None, bool, dict[str, dict[str, Any]] | None]:
    """Write study, post study, and budget artifacts from one sampled path stream."""
    seed_id = paths.run_root.name if seed_run_id is None else seed_run_id
    study = trajectory_segment(
        plan=plan,
        years=study_years,
        category_uncertainty=bool(plan.source_parameters["category_uncertainty"]),
    )
    post = (
        None
        if not post_study_years
        else trajectory_segment(
            plan=plan,
            years=post_study_years,
            category_uncertainty=bool(plan.source_parameters["category_uncertainty"]),
        )
    )
    budget_identity, budget_segments = budget_identity_and_segments(
        study=study,
        post=post,
        category_uncertainty=bool(plan.source_parameters["category_uncertainty"]),
    )
    write_uncertainty_table(
        path=paths.public_row_identity,
        frame=study["identity"],
        output_format=runtime.output_format,
    )
    if post is not None:
        write_uncertainty_table(
            path=paths.post_study_public_row_identity,
            frame=post["identity"],
            output_format=runtime.output_format,
        )
    write_uncertainty_table(
        path=paths.budget_row_identity,
        frame=budget_identity,
        output_format=runtime.output_format,
    )
    completed, convergence, states = _write_dispatched_runs(
        paths=paths,
        plan=plan,
        study=study,
        post=post,
        budget_identity=budget_identity,
        budget_segments=budget_segments,
        runtime=runtime,
        seed_run_id=seed_id,
        start_run_index=int(start_run_index),
        progress=progress,
        progress_mode=progress_mode,
        progress_max_runs=progress_max_runs,
        progress_component=progress_component,
        states=states,
        final_checkpoint=final_checkpoint,
    )
    return completed, convergence, post is not None, states


def _write_dispatched_runs(
    *,
    paths: AR6CCUncertaintyRunPaths,
    plan: AR6CCUncertaintyPlan,
    study: dict[str, Any],
    post: dict[str, Any] | None,
    budget_identity: pd.DataFrame,
    budget_segments: tuple[dict[str, Any], ...],
    runtime: UncertaintyRuntimeRequest,
    seed_run_id: str,
    start_run_index: int,
    progress: RunProgressPrinter,
    progress_mode: str | None,
    progress_max_runs: int | None,
    progress_component: bool,
    states: dict[str, dict[str, Any]] | None,
    final_checkpoint: bool,
) -> tuple[int, dict[str, Any] | None, dict[str, dict[str, Any]] | None]:
    has_live_states = states is not None
    states = states or _open_summary_states(
        paths.run_root,
        study=study,
        post=post,
        budget=budget_identity,
    )
    completed = int(start_run_index)
    if completed and not has_live_states:
        _prime_existing_ar6_cc_states(
            paths=paths,
            states=states,
            study=study,
            post=post,
            budget_identity=budget_identity,
            runtime=runtime,
            completed=completed,
        )
        _record_state_baselines(states=states, completed_runs=completed)
    convergence = None
    checkpoints = ConvergenceCheckpointCursor.from_runtime(runtime=runtime)
    checkpoints.advance_to_completed(completed_runs=completed)
    progress_target = runtime.n_runs if progress_max_runs is None else progress_max_runs
    progress_label_mode = runtime.mode if progress_mode is None else progress_mode
    with (
        SparseRunRowsWriter(
            path=paths.public_runs,
            output_format=runtime.output_format,
            append_existing=completed > 0,
        ) as study_writer,
        CompactRunMatrixWriter(
            path=paths.budget_runs,
            output_format=runtime.output_format,
            append_existing=completed > 0,
        ) as budget_writer,
    ):
        post_writer = (
            None
            if post is None
            else SparseRunRowsWriter(
                path=paths.post_study_public_runs,
                output_format=runtime.output_format,
                append_existing=completed > 0,
            )
        )
        try:
            batches = (
                iter(())
                if completed >= runtime.n_runs
                else fixed_run_plan(
                    n_runs=runtime.n_runs - completed,
                    batch_size=runtime.batch_size,
                    seed=run_seed_from_run_id(run_id=seed_run_id),
                    start_run_index=completed,
                ).batches()
            )
            for batch in batches:
                run_indices = batch.run_indices()
                progress.begin(
                    label=monte_carlo_run_drawing_label(
                        start=int(run_indices[0]),
                        stop=int(run_indices[-1]) + 1,
                        max_runs=progress_target,
                        mode=progress_label_mode,
                        component=progress_component,
                    )
                )
                full_rows = sample_ar6_cc_sparse_rows(plan=plan, batch=batch)
                study_rows = remap_sparse_rows(
                    rows=full_rows,
                    full_to_segment=study["full_to_segment"],
                )
                study_writer.write_batch(rows=study_rows, batch_index=batch.batch_index)
                completed = _append_segment_summary(
                    state=states["study"],
                    rows=study_rows,
                    run_indices=run_indices,
                    public_row_group_index=study["summary_group_index"],
                )
                if post is not None and post_writer is not None:
                    post_rows = remap_sparse_rows(
                        rows=full_rows,
                        full_to_segment=post["full_to_segment"],
                    )
                    post_writer.write_batch(rows=post_rows, batch_index=batch.batch_index)
                    _append_segment_summary(
                        state=states["post"],
                        rows=post_rows,
                        run_indices=run_indices,
                        public_row_group_index=post["summary_group_index"],
                    )
                budget_values = budget_matrix_from_full_sparse_rows(
                    rows=full_rows,
                    run_indices=run_indices,
                    segments=budget_segments,
                )
                budget_writer.write_batch(
                    run_indices=run_indices,
                    values=budget_values,
                    batch_index=batch.batch_index,
                )
                _append_compact_summary(state=states["budget"], values=budget_values)
                if checkpoints.reached(completed_runs=completed):
                    convergence = _checkpoint_convergence(
                        states=states,
                        completed_runs=completed,
                        runtime=runtime,
                    )
                    checkpoints.mark_checked(completed_runs=completed)
                progress.complete(
                    label=monte_carlo_run_progress_label(
                        completed=completed,
                        max_runs=progress_target,
                        mode=progress_label_mode,
                        component=progress_component,
                    ),
                    persistent=(
                        str(progress_label_mode) == "fixed"
                        and (completed >= runtime.n_runs or convergence is not None)
                    ),
                )
                if convergence is not None:
                    break
        finally:
            if post_writer is not None:
                post_writer.close()
    convergence = _final_convergence_payload(
        convergence=convergence,
        completed_runs=completed,
        runtime=runtime,
    )
    if final_checkpoint:
        _write_summaries(
            paths=paths,
            states=states,
            completed=completed,
            post=post is not None,
            output_format=runtime.output_format,
        )
    return completed, convergence, None if final_checkpoint else states


def _append_segment_summary(
    *,
    state: dict[str, Any],
    rows: SparseRunRows,
    run_indices: np.ndarray,
    public_row_group_index: np.ndarray,
) -> int:
    """Accumulate sparse trajectory means for convergence without a dense matrix."""
    accumulator: MeanConvergenceAccumulator = state["accumulator"]
    row_runs = run_positions_in_window(
        run_indices=run_indices,
        row_run_index=rows.run_index,
    )
    row_groups = public_row_group_index[rows.public_row_id]
    accumulator.accumulate_sparse_group_means(
        row_runs=row_runs,
        row_groups=row_groups,
        values=rows.values,
    )
    return int(run_indices[-1]) + 1


def _append_compact_summary(*, state: dict[str, Any], values: np.ndarray) -> None:
    accumulator: MeanConvergenceAccumulator = state["accumulator"]
    accumulator.update(values=values)


def _checkpoint_convergence(
    *,
    states: dict[str, dict[str, Any]],
    completed_runs: int,
    runtime: UncertaintyRuntimeRequest,
) -> dict[str, Any] | None:
    if runtime.mode != "convergence":
        return None
    targets: list[MeanConvergenceAccumulator] = [states["study"]["accumulator"]]
    if "post" in states:
        targets.append(states["post"]["accumulator"])
    targets.append(states["budget"]["accumulator"])
    return mean_convergence_payload_for_targets(
        targets=tuple(targets),
        completed_runs=completed_runs,
        runtime=runtime,
        check_convergence=True,
    )


def _record_state_baselines(
    *,
    states: dict[str, dict[str, Any]],
    completed_runs: int,
) -> None:
    for state in states.values():
        accumulator: MeanConvergenceAccumulator = state["accumulator"]
        accumulator.record_baseline(completed_runs=completed_runs)


def _prime_existing_ar6_cc_states(
    *,
    paths: AR6CCUncertaintyRunPaths,
    states: dict[str, dict[str, Any]],
    study: dict[str, Any],
    post: dict[str, Any] | None,
    budget_identity: pd.DataFrame,
    runtime: UncertaintyRuntimeRequest,
    completed: int,
) -> None:
    _prime_sparse_segment_cache(
        path=paths.public_runs,
        output_format=runtime.output_format,
        completed=completed,
        state=states["study"],
        public_row_group_index=study["summary_group_index"],
    )
    if post is not None:
        _prime_sparse_segment_cache(
            path=paths.post_study_public_runs,
            output_format=runtime.output_format,
            completed=completed,
            state=states["post"],
            public_row_group_index=post["summary_group_index"],
        )
    for run_indices, values in iter_compact_run_matrix(
        path=paths.budget_runs,
        output_format=runtime.output_format,
        column_count=len(budget_identity),
        stop_run_index=completed,
    ):
        _append_compact_summary(state=states["budget"], values=values)


def _prime_sparse_segment_cache(
    *,
    path: Path,
    output_format: str,
    completed: int,
    state: dict[str, Any],
    public_row_group_index: np.ndarray,
) -> None:
    for rows in iter_sparse_run_rows(
        path=path,
        output_format=output_format,
        stop_run_index=completed,
    ):
        first_run = int(rows.run_index[0])
        last_run = int(rows.run_index[-1])
        _append_segment_summary(
            state=state,
            rows=rows,
            run_indices=np.arange(first_run, last_run + 1, dtype=np.int64),
            public_row_group_index=public_row_group_index,
        )


def _open_summary_states(
    run_root: Path,
    *,
    study: dict[str, Any],
    post: dict[str, Any] | None,
    budget: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    del run_root
    states = {
        "study": _sparse_summary_state(segment=study),
        "budget": {
            "identity": budget,
            "accumulator": MeanConvergenceAccumulator.empty(row_count=len(budget)),
            "sparse": False,
        },
    }
    if post is not None:
        states["post"] = _sparse_summary_state(segment=post)
    return states


def _sparse_summary_state(*, segment: dict[str, Any]) -> dict[str, Any]:
    identity = segment["summary_identity"]
    return {
        "identity": identity,
        "public_row_groups": segment["public_row_groups"],
        "accumulator": MeanConvergenceAccumulator.empty(row_count=len(identity)),
        "sparse": True,
    }


def _write_summaries(
    *,
    paths: AR6CCUncertaintyRunPaths,
    states: dict[str, dict[str, Any]],
    completed: int,
    post: bool,
    output_format: str,
) -> None:
    _write_summary(
        summary_path=paths.summary_stats_runs,
        runs_path=paths.public_runs,
        state=states["study"],
        completed=completed,
        output_format=output_format,
    )
    if post:
        _write_summary(
            summary_path=paths.post_study_summary_stats_runs,
            runs_path=paths.post_study_public_runs,
            state=states["post"],
            completed=completed,
            output_format=output_format,
        )
    _write_summary(
        summary_path=paths.budget_summary_stats_runs,
        runs_path=paths.budget_runs,
        state=states["budget"],
        completed=completed,
        output_format=output_format,
    )


def _write_summary(
    *,
    summary_path: Path,
    runs_path: Path,
    state: dict[str, Any],
    completed: int,
    output_format: str,
) -> None:
    summary = exact_summary_from_public_runs(
        identity_frame=state["identity"],
        runs_path=runs_path,
        output_format=output_format,
        run_count=completed,
        public_row_groups=state.get("public_row_groups"),
        sparse=bool(state["sparse"]),
    )
    write_uncertainty_table(path=summary_path, frame=summary, output_format=output_format)


def _final_convergence_payload(
    *,
    convergence: dict[str, Any] | None,
    completed_runs: int,
    runtime: UncertaintyRuntimeRequest,
) -> dict[str, Any] | None:
    if runtime.mode != "convergence" or convergence is not None:
        return convergence
    return mean_convergence_payload(
        reached=False,
        completed_runs=completed_runs,
        runtime=runtime,
    )

"""Fixed run batch execution for aSoCC Monte Carlo runs."""

from typing import cast

from pyaesa.asocc.uncertainty.engine.convergence.convergence import MonteCarloRunResult
from pyaesa.asocc.uncertainty.inputs.external_rows import ExternalAsoccRowsPlan
from pyaesa.asocc.uncertainty.sources.inter_method import InterMethodPlan
from pyaesa.asocc.uncertainty.sources.inter_mrio import InterMrioPlan
from pyaesa.asocc.uncertainty.sources.lcia import LCIAPlan
from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.asocc.uncertainty.sources.projection import ProjectionPlan
from pyaesa.asocc.uncertainty.engine.monte_carlo.sampling import sample_compact_batch
from pyaesa.asocc.uncertainty.engine.inter_method.execution import InterMethodExecutionPlan
from pyaesa.asocc.uncertainty.engine.inter_method.sampling import (
    external_run_offsets_for_start,
    sample_sparse_inter_method_batch,
)
from pyaesa.asocc.uncertainty.schema.public_rows import ASOCC_UNCERTAINTY_CSV_DTYPES
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import fixed_run_plan
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan
from pyaesa.shared.uncertainty_assessment.io.tables import (
    CompactRunMatrixWriter,
    SparseRunRowsWriter,
    read_uncertainty_table,
    write_uncertainty_table,
)
from pyaesa.shared.runtime.reporting.run_progress import (
    RunProgressPrinter,
    monte_carlo_run_progress,
    monte_carlo_run_drawing_label,
    monte_carlo_run_progress_label,
)


def write_fixed_batches(
    *,
    paths: AsoccUncertaintyRunPaths,
    loaded,
    inter_method_plan: InterMethodPlan | None,
    inter_method_execution_plan: InterMethodExecutionPlan | None,
    inter_mrio_plan: InterMrioPlan | None,
    lcia_plan: LCIAPlan | None,
    projection_plan: ProjectionPlan | None,
    runtime,
    sources: SourceActivationPlan,
    external_plan: ExternalAsoccRowsPlan,
    start_run_index: int,
    append_existing: bool,
    run_seed: int,
    show_progress: bool = True,
    progress: RunProgressPrinter | None = None,
    progress_mode: str = "fixed",
    progress_max_runs: int | None = None,
    progress_component: bool = False,
) -> MonteCarloRunResult:
    """Write fixed mode compact run batches."""
    plan = fixed_run_plan(
        n_runs=runtime.n_runs - start_run_index,
        batch_size=runtime.batch_size,
        seed=run_seed,
        start_run_index=start_run_index,
    )
    identity_written = append_existing
    if inter_method_plan is not None:
        return _write_sparse_inter_method_fixed_batches(
            paths=paths,
            loaded=loaded,
            inter_method_plan=inter_method_plan,
            inter_method_execution_plan=cast(InterMethodExecutionPlan, inter_method_execution_plan),
            inter_mrio_plan=inter_mrio_plan,
            runtime=runtime,
            sources=sources,
            start_run_index=start_run_index,
            append_existing=append_existing,
            run_seed=run_seed,
            show_progress=show_progress,
            progress=progress,
            progress_mode=progress_mode,
            progress_max_runs=progress_max_runs,
            progress_component=progress_component,
        )
    own_progress = progress is None
    if own_progress:
        progress = monte_carlo_run_progress(source="uncertainty_asocc", enabled=show_progress)
    try:
        with CompactRunMatrixWriter(
            path=paths.public_runs,
            output_format=runtime.output_format,
            append_existing=append_existing,
        ) as matrix_writer:
            for batch in plan.batches():
                progress.begin(
                    label=monte_carlo_run_drawing_label(
                        start=batch.start_run_index,
                        stop=batch.stop_run_index,
                        max_runs=runtime.n_runs if progress_max_runs is None else progress_max_runs,
                        mode=progress_mode,
                        component=progress_component,
                    )
                )
                identity, run_indices, values = sample_compact_batch(
                    loaded=loaded,
                    inter_mrio_plan=inter_mrio_plan,
                    lcia_plan=lcia_plan,
                    projection_plan=projection_plan,
                    batch=batch,
                    sources=sources,
                    external_plan=external_plan,
                )
                if not identity_written:
                    write_uncertainty_table(
                        path=paths.public_row_identity,
                        frame=identity,
                        output_format=runtime.output_format,
                    )
                    identity_written = True
                matrix_writer.write_batch(
                    run_indices=run_indices,
                    values=values,
                    batch_index=batch.batch_index,
                )
                progress.complete(
                    label=monte_carlo_run_progress_label(
                        completed=batch.stop_run_index,
                        max_runs=runtime.n_runs if progress_max_runs is None else progress_max_runs,
                        mode=progress_mode,
                        component=progress_component,
                    ),
                    persistent=str(progress_mode) == "fixed",
                )
    finally:
        if own_progress:
            progress.finish()
    return MonteCarloRunResult(
        completed_runs=runtime.n_runs,
        summary_run_count=runtime.n_runs,
    )


def _write_sparse_inter_method_fixed_batches(
    *,
    paths: AsoccUncertaintyRunPaths,
    loaded,
    inter_method_plan: InterMethodPlan,
    inter_method_execution_plan: InterMethodExecutionPlan,
    inter_mrio_plan: InterMrioPlan | None,
    runtime,
    sources: SourceActivationPlan,
    start_run_index: int,
    append_existing: bool,
    run_seed: int,
    show_progress: bool,
    progress: RunProgressPrinter | None,
    progress_mode: str,
    progress_max_runs: int | None,
    progress_component: bool,
) -> MonteCarloRunResult:
    plan = fixed_run_plan(
        n_runs=runtime.n_runs - start_run_index,
        batch_size=runtime.batch_size,
        seed=run_seed,
        start_run_index=start_run_index,
    )
    identity = None
    identity_written = append_existing
    if append_existing:
        identity = read_uncertainty_table(
            path=paths.public_row_identity,
            output_format=runtime.output_format,
            csv_dtypes=ASOCC_UNCERTAINTY_CSV_DTYPES,
        )
    external_render_offsets = external_run_offsets_for_start(
        inter_method_plan=inter_method_plan,
        start_run_index=start_run_index,
        external_labels=tuple(
            source.selection.asocc_method_label
            for branch in inter_method_execution_plan.branches
            for source in branch.external_plan.monte_carlo_sources
        ),
    )
    own_progress = progress is None
    if own_progress:
        progress = monte_carlo_run_progress(source="uncertainty_asocc", enabled=show_progress)
    try:
        with SparseRunRowsWriter(
            path=paths.public_runs,
            output_format=runtime.output_format,
            append_existing=append_existing,
        ) as row_writer:
            for batch in plan.batches():
                progress.begin(
                    label=monte_carlo_run_drawing_label(
                        start=batch.start_run_index,
                        stop=batch.stop_run_index,
                        max_runs=runtime.n_runs if progress_max_runs is None else progress_max_runs,
                        mode=progress_mode,
                        component=progress_component,
                    )
                )
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
                for label, count in sparse_batch.external_run_counts.items():
                    external_render_offsets[label] = external_render_offsets.get(label, 0) + count
                identity = sparse_batch.identity
                if not identity_written:
                    write_uncertainty_table(
                        path=paths.public_row_identity,
                        frame=identity,
                        output_format=runtime.output_format,
                    )
                    identity_written = True
                row_writer.write_batch(
                    rows=sparse_batch.sparse_rows,
                    batch_index=batch.batch_index,
                )
                progress.complete(
                    label=monte_carlo_run_progress_label(
                        completed=batch.stop_run_index,
                        max_runs=runtime.n_runs if progress_max_runs is None else progress_max_runs,
                        mode=progress_mode,
                        component=progress_component,
                    ),
                    persistent=str(progress_mode) == "fixed",
                )
    finally:
        if own_progress:
            progress.finish()
    return MonteCarloRunResult(
        completed_runs=runtime.n_runs,
        summary_run_count=runtime.n_runs,
        public_runs_sparse=True,
    )

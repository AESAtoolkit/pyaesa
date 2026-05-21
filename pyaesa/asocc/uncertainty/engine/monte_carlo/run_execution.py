"""aSoCC uncertainty Monte Carlo run execution selection."""

from typing import Any, cast

from pyaesa.asocc.uncertainty.engine.convergence.convergence import (
    MonteCarloRunResult,
    write_convergence_batches,
    write_sparse_inter_method_convergence_batches,
)
from pyaesa.asocc.uncertainty.engine.monte_carlo.fixed_batches import write_fixed_batches
from pyaesa.asocc.uncertainty.engine.inter_method.execution import InterMethodExecutionPlan
from pyaesa.shared.runtime.reporting.run_progress import RunProgressPrinter


def write_monte_carlo_run_outputs(
    *,
    paths: Any,
    loaded: Any,
    inter_method_plan: Any,
    inter_method_execution_plan: InterMethodExecutionPlan | None,
    inter_mrio_plan: Any,
    lcia_plan: Any,
    projection_plan: Any,
    runtime: Any,
    sources: Any,
    external_plan: Any,
    append_run: Any,
    run_seed: int,
    progress: RunProgressPrinter,
    show_progress: bool = True,
    progress_mode: str = "fixed",
    progress_max_runs: int | None = None,
    progress_component: bool = False,
    previous_result: MonteCarloRunResult | None = None,
) -> MonteCarloRunResult:
    """Write or reuse the Monte Carlo run artifacts for one aSoCC run."""
    start_run_index = (
        previous_result.completed_runs
        if previous_result is not None
        else 0
        if append_run is None
        else append_run.manifest.completed_runs
    )
    if previous_result is not None and start_run_index >= runtime.n_runs:
        return previous_result
    if append_run is not None and start_run_index >= runtime.n_runs:
        return MonteCarloRunResult(
            completed_runs=start_run_index,
            summary_run_count=start_run_index,
            public_runs_sparse=inter_method_plan is not None,
            convergence=append_run.manifest.convergence,
        )
    append_existing = start_run_index > 0
    if runtime.mode == "convergence":
        if inter_method_plan is not None:
            return write_sparse_inter_method_convergence_batches(
                paths=paths,
                loaded=loaded,
                inter_method_plan=inter_method_plan,
                inter_method_execution_plan=cast(
                    InterMethodExecutionPlan,
                    inter_method_execution_plan,
                ),
                inter_mrio_plan=inter_mrio_plan,
                runtime=runtime,
                sources=sources,
                run_seed=run_seed,
                start_run_index=start_run_index,
                append_existing=append_existing,
                progress=progress,
            )
        return write_convergence_batches(
            paths=paths,
            loaded=loaded,
            inter_mrio_plan=inter_mrio_plan,
            lcia_plan=lcia_plan,
            projection_plan=projection_plan,
            runtime=runtime,
            sources=sources,
            external_plan=external_plan,
            run_seed=run_seed,
            start_run_index=start_run_index,
            append_existing=append_existing,
            progress=progress,
        )
    return write_fixed_batches(
        paths=paths,
        loaded=loaded,
        inter_method_plan=inter_method_plan,
        inter_method_execution_plan=inter_method_execution_plan,
        inter_mrio_plan=inter_mrio_plan,
        lcia_plan=lcia_plan,
        projection_plan=projection_plan,
        runtime=runtime,
        sources=sources,
        external_plan=external_plan,
        start_run_index=start_run_index,
        append_existing=append_existing,
        run_seed=run_seed,
        show_progress=show_progress,
        progress=progress,
        progress_mode=progress_mode,
        progress_max_runs=progress_max_runs,
        progress_component=progress_component,
    )

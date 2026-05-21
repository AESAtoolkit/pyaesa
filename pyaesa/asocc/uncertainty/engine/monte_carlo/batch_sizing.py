"""aSoCC uncertainty batch row count planning."""

from typing import Any

from pyaesa.asocc.uncertainty.engine.inter_method.execution import InterMethodExecutionPlan
from pyaesa.asocc.uncertainty.engine.inter_method.identity import (
    public_row_ids_for_branch,
)
from pyaesa.asocc.uncertainty.inputs.external_rows import ExternalAsoccRowsPlan
from pyaesa.asocc.uncertainty.sources.lcia import LCIAPlan


def batch_row_count(
    *,
    loaded: Any,
    inter_method_execution_plan: InterMethodExecutionPlan | None,
    inter_mrio_plan: Any,
    lcia_plan: LCIAPlan | None,
    projection_plan: Any,
    sources: Any,
    external_plan: ExternalAsoccRowsPlan,
) -> int:
    """Return the memory planning row width for one aSoCC run batch."""
    if inter_method_execution_plan is not None:
        return max(
            len(
                public_row_ids_for_branch(
                    row_universe=inter_method_execution_plan.row_universe,
                    label=branch.label,
                )
            )
            for branch in inter_method_execution_plan.branches
        )
    external_row_count = _external_monte_carlo_row_count(plan=external_plan)
    if lcia_plan is not None:
        return _lcia_plan_row_count(plan=lcia_plan) + external_row_count
    if projection_plan is not None:
        return (
            len(projection_plan.passthrough_rows)
            + len(projection_plan.sampled_rows)
            + external_row_count
        )
    return len(loaded.rows) + external_row_count


def _lcia_plan_row_count(*, plan: LCIAPlan) -> int:
    return (
        len(plan.passthrough_rows)
        + len(plan.direct_rows)
        + sum(len(route.final_rows) for route in plan.combined_routes)
    )


def _external_monte_carlo_row_count(*, plan: ExternalAsoccRowsPlan) -> int:
    return sum(len(source.run_matrix.template) for source in plan.monte_carlo_sources)

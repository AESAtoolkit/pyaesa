"""Inter-method execution plans for aSoCC uncertainty."""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from pyaesa.asocc.uncertainty.inputs.external_rows import ExternalAsoccRowsPlan
from pyaesa.asocc.uncertainty.engine.inter_method.identity import (
    InterMethodRowUniverse,
    build_inter_method_row_universe,
    execution_external_plan_from_branches,
)
from pyaesa.asocc.uncertainty.sources.inter_method import (
    InterMethodPlan,
    inter_method_row_labels,
)
from pyaesa.asocc.uncertainty.sources.lcia import (
    LCIAPlan,
    lcia_sample_block,
)
from pyaesa.asocc.uncertainty.sources.projection import (
    PROJECTION_SOURCE,
    ProjectionPlan,
    build_projection_plan,
)
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan


@dataclass(frozen=True)
class InterMethodBranchExecution:
    """Prebuilt selected branch source plans for one inter-method leaf."""

    label: str
    loaded: Any
    external_plan: ExternalAsoccRowsPlan
    lcia_plan: LCIAPlan | None
    projection_plan: ProjectionPlan | None


@dataclass(frozen=True)
class InterMethodExecutionPlan:
    """Run scoped execution plans for selected inter-method leaves."""

    branches: tuple[InterMethodBranchExecution, ...]
    lcia_plan: LCIAPlan | None
    projection_plan: ProjectionPlan | None
    row_universe: InterMethodRowUniverse


def build_inter_method_execution_plan(
    *,
    loaded,
    inter_method_plan: InterMethodPlan,
    sources: SourceActivationPlan,
    external_plan: ExternalAsoccRowsPlan,
    lcia_plan: LCIAPlan | None,
    projection_plan: ProjectionPlan | None,
) -> InterMethodExecutionPlan:
    """Build reusable branch plans for one inter-method run."""
    row_labels = inter_method_row_labels(rows=loaded.rows)
    branches: list[InterMethodBranchExecution] = []
    for label in inter_method_plan.candidate_labels:
        branch_external_plan = external_plan_for_inter_method_label(
            plan=external_plan,
            label=label,
        )
        branch_loaded = _loaded_for_inter_method_label(
            loaded=loaded,
            label=label,
            row_labels=row_labels,
        )
        external_branch = bool(branch_external_plan.method_labels)
        # External methods carry user supplied aSoCC values. Active inner
        # sources that belong to native aSoCC methods are skipped for external
        # leaves; reference year uncertainty still samples exposed
        # reference_year candidates.
        branch_lcia = (
            None
            if external_branch
            else _lcia_plan_for_inter_method_label(plan=lcia_plan, label=label)
        )
        branch_projection = (
            build_projection_plan(loaded=branch_loaded)
            if sources.is_active(PROJECTION_SOURCE)
            and not branch_loaded.rows.empty
            and not external_branch
            else None
        )
        branches.append(
            InterMethodBranchExecution(
                label=label,
                loaded=branch_loaded,
                external_plan=branch_external_plan,
                lcia_plan=branch_lcia,
                projection_plan=branch_projection,
            )
        )
    branch_tuple = tuple(branches)
    row_external_plan = execution_external_plan_from_branches(branches=branch_tuple)
    return InterMethodExecutionPlan(
        branches=branch_tuple,
        lcia_plan=lcia_plan,
        projection_plan=projection_plan,
        row_universe=build_inter_method_row_universe(
            loaded=loaded,
            sources=sources,
            external_plan=row_external_plan,
            lcia_plan=lcia_plan,
            projection_plan=projection_plan,
            branches=branch_tuple,
        ),
    )


def external_plan_for_inter_method_label(
    *,
    plan: ExternalAsoccRowsPlan,
    label: str,
) -> ExternalAsoccRowsPlan:
    """Return the external source slice for one inter-method label."""
    monte_carlo_sources = tuple(
        source
        for source in plan.monte_carlo_sources
        if source.selection.asocc_method_label == str(label)
    )
    deterministic_sources = tuple(
        source
        for source in plan.deterministic_sources
        if source.selection.asocc_method_label == str(label)
    )
    has_external_source = bool(monte_carlo_sources or deterministic_sources)
    method_labels = (
        (str(label),) if str(label) in set(plan.method_labels) or has_external_source else ()
    )
    return ExternalAsoccRowsPlan(
        method_labels=method_labels,
        deterministic_sources=deterministic_sources,
        monte_carlo_sources=monte_carlo_sources,
    )


def _lcia_plan_for_inter_method_label(*, plan: LCIAPlan | None, label: str) -> LCIAPlan | None:
    if plan is None:
        return None
    direct_rows = _lcia_rows_for_label(rows=plan.direct_rows, label=label)
    routes = tuple(route for route in plan.combined_routes if route.method_label == str(label))
    if direct_rows.empty and not routes:
        return None
    passthrough_rows = _lcia_rows_for_label(rows=plan.passthrough_rows, label=label)
    return LCIAPlan(
        public_columns=plan.public_columns,
        passthrough_rows=passthrough_rows,
        direct_rows=direct_rows,
        direct_block=lcia_sample_block(template=direct_rows) if not direct_rows.empty else None,
        combined_routes=routes,
        source_method_rows=(),
    )


def _lcia_rows_for_label(*, rows: pd.DataFrame, label: str) -> pd.DataFrame:
    return rows.loc[inter_method_row_labels(rows=rows) == str(label)].reset_index(drop=True)


def _loaded_for_inter_method_label(*, loaded, label: str, row_labels: np.ndarray):
    rows = loaded.rows.loc[row_labels == str(label)].reset_index(drop=True)
    return replace(loaded, rows=rows)

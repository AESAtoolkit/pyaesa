"""Inter-method identity helpers for sparse aSoCC uncertainty batches."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pyaesa.asocc.uncertainty.inputs.external_rows import (
    ExternalAsoccRowsPlan,
    append_external_monte_carlo_template,
)
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import ASOCC_VALUE_COLUMN
from pyaesa.asocc.uncertainty.schema.public_rows import (
    align_asocc_lcia_public_axis,
    finalize_asocc_public_row_identity,
    lcia_public_axis,
)
from pyaesa.asocc.uncertainty.sources.lcia import LCIAPlan, lcia_public_row_template
from pyaesa.asocc.uncertainty.sources.names import REFERENCE_YEAR_SOURCE
from pyaesa.asocc.uncertainty.sources.projection import (
    ProjectionPlan,
    collapse_projection_public_template,
    projection_public_row_template,
)
from pyaesa.asocc.uncertainty.sources.reference_year import (
    collapse_reference_year_public_template,
)
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan


@dataclass(frozen=True)
class InterMethodBranchRowIds:
    """Public row ids selected by one inter-method branch."""

    label: str
    public_row_ids: np.ndarray


@dataclass(frozen=True)
class InterMethodRowUniverse:
    """Complete sparse inter-method public identity and branch row mapping."""

    identity: pd.DataFrame
    public_lcia_axis: pd.DataFrame
    branch_row_ids: tuple[InterMethodBranchRowIds, ...]


def execution_external_plan_from_branches(*, branches: tuple[Any, ...]) -> ExternalAsoccRowsPlan:
    """Return the union external aSoCC plan represented by inter-method branches."""
    return ExternalAsoccRowsPlan(
        method_labels=tuple(
            sorted({label for branch in branches for label in branch.external_plan.method_labels})
        ),
        monte_carlo_sources=tuple(
            source for branch in branches for source in branch.external_plan.monte_carlo_sources
        ),
        deterministic_sources=tuple(
            source for branch in branches for source in branch.external_plan.deterministic_sources
        ),
    )


def build_inter_method_row_universe(
    *,
    loaded,
    sources: SourceActivationPlan,
    external_plan: ExternalAsoccRowsPlan,
    lcia_plan: LCIAPlan | None,
    projection_plan: ProjectionPlan | None,
    branches: tuple[Any, ...],
) -> InterMethodRowUniverse:
    """Return the complete public identity and branch public row ids."""
    identity = public_identity_for_sampling_plan(
        loaded=loaded,
        sources=sources,
        external_plan=external_plan,
        lcia_plan=lcia_plan,
        projection_plan=projection_plan,
        reference_axis=None,
    )
    public_axis = lcia_public_axis(frame=identity)
    branch_ids = tuple(
        InterMethodBranchRowIds(
            label=str(branch.label),
            public_row_ids=public_row_ids(
                public_identity=identity,
                branch_identity=public_identity_for_sampling_plan(
                    loaded=branch.loaded,
                    sources=sources,
                    external_plan=branch.external_plan,
                    lcia_plan=branch.lcia_plan,
                    projection_plan=branch.projection_plan,
                    reference_axis=public_axis,
                ),
            ),
        )
        for branch in branches
    )
    return InterMethodRowUniverse(
        identity=identity,
        public_lcia_axis=public_axis,
        branch_row_ids=branch_ids,
    )


def public_identity_for_sampling_plan(
    *,
    loaded,
    sources: SourceActivationPlan,
    external_plan: ExternalAsoccRowsPlan,
    lcia_plan: LCIAPlan | None,
    projection_plan: ProjectionPlan | None,
    reference_axis: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return public identity from source templates without drawing a sample."""
    template = _sampling_public_template(
        loaded=loaded,
        lcia_plan=lcia_plan,
        projection_plan=projection_plan,
    )
    template = append_external_monte_carlo_template(template=template, plan=external_plan)
    template, _values = align_asocc_lcia_public_axis(
        frame=template,
        values=np.empty((0, len(template)), dtype=np.float64),
        reference_axis=reference_axis,
    )
    if sources.is_active(REFERENCE_YEAR_SOURCE):
        template = collapse_reference_year_public_template(template=template)
    return finalize_asocc_public_row_identity(
        frame=template,
        value_column=ASOCC_VALUE_COLUMN,
    )


def public_row_ids_for_branch(
    *,
    row_universe: InterMethodRowUniverse,
    label: str,
) -> np.ndarray:
    """Return the public row ids selected by one inter-method branch."""
    return next(
        branch.public_row_ids
        for branch in row_universe.branch_row_ids
        if branch.label == str(label)
    )


def _sampling_public_template(
    *,
    loaded,
    lcia_plan: LCIAPlan | None,
    projection_plan: ProjectionPlan | None,
) -> pd.DataFrame:
    if lcia_plan is not None:
        template = lcia_public_row_template(plan=lcia_plan)
    elif projection_plan is not None:
        template = projection_public_row_template(plan=projection_plan)
    else:
        template = loaded.rows
    if projection_plan is not None:
        return collapse_projection_public_template(template=template)
    return template


def public_row_ids(
    *,
    public_identity: pd.DataFrame,
    branch_identity: pd.DataFrame,
) -> np.ndarray:
    """Map one branch identity into positions of the complete public identity."""
    identity_columns = [column for column in branch_identity.columns if column != "public_row_id"]
    merged = branch_identity.loc[:, identity_columns].merge(
        public_identity.loc[:, ["public_row_id", *identity_columns]],
        how="left",
        on=identity_columns,
        sort=False,
    )
    return merged["public_row_id"].to_numpy(dtype=np.int64)

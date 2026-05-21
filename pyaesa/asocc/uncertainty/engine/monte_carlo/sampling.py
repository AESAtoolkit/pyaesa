"""Compact batch sampling for aSoCC uncertainty values."""

from typing import Any, cast

import numpy as np

from pyaesa.asocc.uncertainty.engine.evaluation.source_unit_intervals import (
    SourceUnitIntervalSamples,
)
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import ASOCC_VALUE_COLUMN
from pyaesa.asocc.uncertainty.inputs.external_rows import (
    EXTERNAL_ASOCC_RUN_SOURCE,
    ExternalAsoccRowsPlan,
    append_external_monte_carlo_matrix,
)
from pyaesa.asocc.uncertainty.lcia_support.sampling import LCIASharedUMatrix
from pyaesa.asocc.uncertainty.schema.public_rows import (
    align_asocc_lcia_public_axis,
    finalize_asocc_public_row_identity,
)
from pyaesa.asocc.uncertainty.sources.inter_mrio import (
    INTER_MRIO_SOURCE,
    InterMrioPlan,
    apply_inter_mrio_uncertainty_to_matrix,
)
from pyaesa.asocc.uncertainty.sources.lcia import (
    LCIA_SOURCE,
    LCIAPlan,
    lcia_public_row_template,
    sample_lcia_public_value_matrix,
)
from pyaesa.asocc.uncertainty.sources.projection import (
    PROJECTION_SOURCE,
    ProjectionPlan,
    apply_projection_uncertainty_to_matrix,
    projection_indices_for_l2_reuse_years,
    projection_public_row_template,
    projection_value_matrix_for_indices,
    sample_projection_indices,
    sample_projection_l2_reuse_years,
)
from pyaesa.asocc.uncertainty.sources.reference_year import (
    REFERENCE_YEAR_SOURCE,
    apply_reference_year_uncertainty_to_matrix,
)
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan


def sample_compact_batch(
    *,
    loaded,
    inter_mrio_plan: InterMrioPlan | None,
    lcia_plan: LCIAPlan | None,
    projection_plan: ProjectionPlan | None,
    batch,
    sources: SourceActivationPlan,
    external_plan: ExternalAsoccRowsPlan,
    projection_selection: np.ndarray | None = None,
    lcia_shared_u: LCIASharedUMatrix | None = None,
    source_units: SourceUnitIntervalSamples | None = None,
    external_run_indices_by_label: dict[str, np.ndarray] | None = None,
    lcia_public_axis: Any = None,
) -> tuple[Any, Any, Any]:
    """Return public identity, run indices, and compact value matrix for one batch."""
    if (
        projection_plan is not None
        and projection_selection is None
        and source_units is not None
        and source_units.values_for(PROJECTION_SOURCE) is not None
    ):
        projection_selection = projection_indices_for_l2_reuse_years(
            plan=projection_plan,
            l2_reuse_years=sample_projection_l2_reuse_years(
                plan=projection_plan,
                batch=batch,
                unit_values=source_units.values_for(PROJECTION_SOURCE),
            ),
        )
    elif projection_plan is not None and projection_selection is None:
        projection_selection = sample_projection_indices(plan=projection_plan, batch=batch)
    projection_selection = projection_selection if projection_plan is not None else None
    if lcia_plan is not None:
        template = lcia_public_row_template(plan=lcia_plan)
        values = sample_lcia_public_value_matrix(
            plan=lcia_plan,
            batch=batch,
            shared_u=lcia_shared_u,
            unit_values=None if source_units is None else source_units.values_for(LCIA_SOURCE),
        )
        if projection_plan is not None:
            template, values = apply_projection_uncertainty_to_matrix(
                template=template,
                values=values,
                plan=projection_plan,
                batch=batch,
                selected_indices=cast(np.ndarray, projection_selection),
            )
    elif projection_plan is not None:
        template = projection_public_row_template(plan=projection_plan)
        values = projection_value_matrix_for_indices(
            plan=projection_plan,
            batch=batch,
            selected_indices=cast(np.ndarray, projection_selection),
        )
    else:
        template = loaded.rows
        values = _stable_value_matrix(loaded=loaded, batch=batch)
    if sources.is_active(INTER_MRIO_SOURCE) and inter_mrio_plan is not None:
        template, values = apply_inter_mrio_uncertainty_to_matrix(
            template=template,
            values=values,
            plan=inter_mrio_plan,
            batch=batch,
            projection_selection=projection_selection,
            unit_values=None
            if source_units is None
            else source_units.values_for(INTER_MRIO_SOURCE),
        )
    template, values = append_external_monte_carlo_matrix(
        template=template,
        values=values,
        plan=external_plan,
        batch=batch,
        unit_values=None
        if source_units is None
        else source_units.values_for(EXTERNAL_ASOCC_RUN_SOURCE),
        external_run_indices_by_label=external_run_indices_by_label,
    )
    template, values = align_asocc_lcia_public_axis(
        frame=template,
        values=values,
        reference_axis=lcia_public_axis,
    )
    if sources.is_active(REFERENCE_YEAR_SOURCE):
        template, values = apply_reference_year_uncertainty_to_matrix(
            template=template,
            values=values,
            batch=batch,
            unit_values=None
            if source_units is None
            else source_units.values_for(REFERENCE_YEAR_SOURCE),
        )
    identity = finalize_asocc_public_row_identity(
        frame=template,
        value_column=ASOCC_VALUE_COLUMN,
    )
    return identity, batch.run_indices(), values


def _stable_value_matrix(*, loaded, batch) -> Any:
    values = loaded.rows[ASOCC_VALUE_COLUMN].to_numpy(dtype="float64")
    return np.broadcast_to(values, (batch.n_runs, len(values)))

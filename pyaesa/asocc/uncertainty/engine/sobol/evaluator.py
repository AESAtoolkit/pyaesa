"""Reusable aSoCC Sobol source unit evaluator."""

from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.asocc.uncertainty.engine.inter_method.execution import (
    InterMethodExecutionPlan,
    build_inter_method_execution_plan,
)
from pyaesa.asocc.uncertainty.engine.inter_method.sampling import (
    sample_inter_method_summary_matrix_batch,
)
from pyaesa.asocc.uncertainty.engine.reuse.prerequisites import (
    prepare_asocc_deterministic_prerequisite,
)
from pyaesa.asocc.uncertainty.engine.monte_carlo.sampling import sample_compact_batch
from pyaesa.asocc.uncertainty.engine.sobol.scope import (
    inter_mrio_plan_for_sobol_years,
    loaded_for_sobol_years,
    selected_sobol_years,
)
from pyaesa.asocc.uncertainty.engine.evaluation.source_unit_intervals import (
    SourceUnitIntervalSamples,
)
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import (
    load_final_deterministic_asocc_rows,
    resolve_final_deterministic_asocc_row_scope,
    validate_single_l2_reuse_year_per_identity,
    validate_single_reference_year_per_identity,
)
from pyaesa.asocc.uncertainty.inputs.external_rows import (
    EXTERNAL_ASOCC_RUN_SOURCE,
    external_plan_for_years,
    resolve_external_asocc_rows,
)
from pyaesa.asocc.uncertainty.sources.activation import (
    deactivate_inter_mrio_without_targets,
    deactivate_sources_without_row_targets,
)
from pyaesa.asocc.uncertainty.sources.inter_method import (
    INTER_METHOD_SOURCE,
    InterMethodPlan,
    build_inter_method_plan,
)
from pyaesa.asocc.uncertainty.sources.inter_mrio import (
    INTER_MRIO_SOURCE,
    InterMrioPlan,
    build_inter_mrio_plan,
)
from pyaesa.asocc.uncertainty.sources.lcia import (
    LCIA_SOURCE,
    LCIASupportRowCache,
    LCIAPlan,
    build_lcia_plan,
)
from pyaesa.asocc.uncertainty.sources.projection import (
    PROJECTION_SOURCE,
    ProjectionPlan,
    build_projection_plan,
)
from pyaesa.asocc.uncertainty.sources.reference_year import (
    REFERENCE_YEAR_SOURCE,
    admissible_reference_year_rows,
)
from pyaesa.asocc.uncertainty.sources.names import DEFAULT_ASOCC_UNCERTAINTY_SOURCES
from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch
from pyaesa.shared.uncertainty_assessment.sobol.plan import SobolPlan
from pyaesa.shared.uncertainty_assessment.sobol.design import SobolEvaluationChunk
from pyaesa.shared.uncertainty_assessment.sobol.runner import EvaluatedSobolChunk
from pyaesa.shared.uncertainty_assessment.request.sources import (
    SourceActivationPlan,
    build_source_activation_plan,
)

ASOCC_SOBOL_EVALUATOR_SOURCES: tuple[str, ...] = (
    INTER_METHOD_SOURCE,
    INTER_MRIO_SOURCE,
    LCIA_SOURCE,
    PROJECTION_SOURCE,
    REFERENCE_YEAR_SOURCE,
)


@dataclass(frozen=True)
class AsoccSobolEvaluationContext:
    """Prepared aSoCC Sobol context for reuse by downstream families."""

    loaded: Any
    source_names: tuple[str, ...]
    inter_method_plan: InterMethodPlan | None
    inter_method_execution_plan: InterMethodExecutionPlan | None
    inter_mrio_plan: InterMrioPlan | None
    lcia_plan: LCIAPlan | None
    projection_plan: ProjectionPlan | None
    sources: SourceActivationPlan
    external_plan: Any
    selected_years: tuple[int, ...]
    requested_ssp_scenarios: tuple[str, ...]


def asocc_sobol_source_names(*, sources: SourceActivationPlan, external_plan) -> tuple[str, ...]:
    """Return active aSoCC Sobol source dimensions."""
    names = list(sources.names)
    if external_plan.monte_carlo_sources:
        names.append(EXTERNAL_ASOCC_RUN_SOURCE)
    return tuple(names)


def build_asocc_sobol_evaluation_context(
    *,
    loaded,
    inter_mrio_plan: InterMrioPlan | None,
    sources: SourceActivationPlan,
    external_plan,
    sobol_plan: SobolPlan,
    selected_years: tuple[int, ...],
) -> AsoccSobolEvaluationContext:
    """Build the canonical aSoCC Sobol source unit evaluator context."""
    del sobol_plan
    sobol_loaded = loaded_for_sobol_years(loaded=loaded, selected_years=selected_years)
    sobol_external_plan = external_plan_for_years(plan=external_plan, years=selected_years)
    sources = deactivate_sources_without_row_targets(
        loaded=sobol_loaded,
        sources=sources,
        external_plan=sobol_external_plan,
    )
    sources, inter_mrio_plan = deactivate_inter_mrio_without_targets(
        sources=sources,
        plan=inter_mrio_plan,
    )
    source_names = asocc_sobol_source_names(
        sources=sources,
        external_plan=sobol_external_plan,
    )
    inter_method_plan = (
        build_inter_method_plan(
            loaded=sobol_loaded,
            parameters=sources.parameters_for(INTER_METHOD_SOURCE),
            external_plan=sobol_external_plan,
        )
        if sources.is_active(INTER_METHOD_SOURCE)
        else None
    )
    lcia_support_cache = LCIASupportRowCache()
    lcia_plan = (
        build_lcia_plan(
            loaded=sobol_loaded,
            parameters=sources.parameters_for(LCIA_SOURCE),
            support_cache=lcia_support_cache,
            include_source_methods=False,
            external_method_labels=sobol_external_plan.method_labels,
        )
        if sources.is_active(LCIA_SOURCE)
        else None
    )
    projection_plan = (
        build_projection_plan(
            loaded=sobol_loaded,
            external_method_labels=sobol_external_plan.method_labels,
        )
        if sources.is_active(PROJECTION_SOURCE)
        else None
    )
    sobol_inter_mrio_plan = (
        inter_mrio_plan_for_sobol_years(
            plan=cast(InterMrioPlan, inter_mrio_plan),
            selected_years=selected_years,
            projection_active=sources.is_active(PROJECTION_SOURCE),
        )
        if sources.is_active(INTER_MRIO_SOURCE)
        else None
    )
    inter_method_execution_plan = (
        build_inter_method_execution_plan(
            loaded=sobol_loaded,
            inter_method_plan=inter_method_plan,
            sources=sources,
            external_plan=sobol_external_plan,
            lcia_plan=lcia_plan,
            projection_plan=projection_plan,
        )
        if inter_method_plan is not None
        else None
    )
    return AsoccSobolEvaluationContext(
        loaded=sobol_loaded,
        source_names=source_names,
        inter_method_plan=inter_method_plan,
        inter_method_execution_plan=inter_method_execution_plan,
        inter_mrio_plan=sobol_inter_mrio_plan,
        lcia_plan=lcia_plan,
        projection_plan=projection_plan,
        sources=sources,
        external_plan=sobol_external_plan,
        selected_years=selected_years,
        requested_ssp_scenarios=tuple(
            normalize_ssp_tokens(sobol_loaded.base_asocc_args.get("ssp_scenario"))
        ),
    )


def build_asocc_sobol_evaluation_context_from_request(
    *,
    base_asocc_args: dict[str, Any],
    uncertainty_config: dict[str, Any],
    external_method: dict[str, Any] | None,
    sobol_plan: SobolPlan,
) -> AsoccSobolEvaluationContext:
    """Build an aSoCC Sobol evaluator from the public uncertainty request shape."""
    config = dict(uncertainty_config)
    config.pop("mc_parameters", None)
    sources = build_source_activation_plan(
        uncertainty_config=config,
        allowed_sources=ASOCC_SOBOL_EVALUATOR_SOURCES,
        default_sources=DEFAULT_ASOCC_UNCERTAINTY_SOURCES,
    )
    prerequisite = prepare_asocc_deterministic_prerequisite(
        base_asocc_args=base_asocc_args,
        refresh=False,
        reference_year_uncertainty_active=sources.is_active(REFERENCE_YEAR_SOURCE),
    )
    row_scope = resolve_final_deterministic_asocc_row_scope(prerequisite=prerequisite)
    loaded = load_final_deterministic_asocc_rows(
        prerequisite=prerequisite,
        row_scope=row_scope,
    )
    if sources.is_active(REFERENCE_YEAR_SOURCE):
        loaded = replace(loaded, rows=admissible_reference_year_rows(frame=loaded.rows))
    selected_years = selected_sobol_years(
        plan=sobol_plan,
        requested_years=tuple(int(year) for year in loaded.requested_years),
    )
    loaded = loaded_for_sobol_years(loaded=loaded, selected_years=selected_years)
    loaded, external_plan = resolve_external_asocc_rows(
        loaded=loaded,
        external_method=external_method,
        required_runs=None,
    )
    sources = deactivate_sources_without_row_targets(
        loaded=loaded,
        sources=sources,
        external_plan=external_plan,
    )
    sampled_identity_columns = (
        ("l1_l2_method", "l1_method", "l2_method") if sources.is_active(INTER_METHOD_SOURCE) else ()
    )
    if not sources.is_active(PROJECTION_SOURCE):
        validate_single_l2_reuse_year_per_identity(
            rows=loaded.rows,
            sampled_identity_columns=sampled_identity_columns,
        )
    if sources.is_active(INTER_METHOD_SOURCE) and not sources.is_active(REFERENCE_YEAR_SOURCE):
        validate_single_reference_year_per_identity(
            rows=loaded.rows,
            sampled_identity_columns=sampled_identity_columns,
        )
    inter_mrio_plan = (
        build_inter_mrio_plan(
            loaded=loaded,
            parameters=sources.parameters_for(INTER_MRIO_SOURCE),
            projection_active=sources.is_active(PROJECTION_SOURCE),
            reference_year_uncertainty_active=sources.is_active(REFERENCE_YEAR_SOURCE),
            external_method_labels=external_plan.method_labels,
        )
        if sources.is_active(INTER_MRIO_SOURCE)
        else None
    )
    return build_asocc_sobol_evaluation_context(
        loaded=loaded,
        inter_mrio_plan=inter_mrio_plan,
        sources=sources,
        external_plan=external_plan,
        sobol_plan=sobol_plan,
        selected_years=selected_years,
    )


def evaluate_asocc_sobol_chunk(
    *,
    context: AsoccSobolEvaluationContext,
    chunk: SobolEvaluationChunk,
) -> EvaluatedSobolChunk:
    """Evaluate one Saltelli chunk through the aSoCC source owners."""
    units = np.vstack((chunk.a, chunk.b, *chunk.ab))
    identity, values = evaluate_asocc_sobol_units(context=context, units=units)
    a_stop = chunk.a.shape[0]
    b_stop = a_stop + chunk.b.shape[0]
    ab_values = []
    start = b_stop
    for block in chunk.ab:
        stop = start + block.shape[0]
        ab_values.append(values[start:stop])
        start = stop
    return EvaluatedSobolChunk(
        identity=identity,
        a_values=values[:a_stop],
        b_values=values[a_stop:b_stop],
        ab_values=tuple(ab_values),
    )


def evaluate_asocc_sobol_units(
    *,
    context: AsoccSobolEvaluationContext,
    units: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Evaluate unit interval source values into aSoCC public row values."""
    batch = RunBatch(
        batch_index=0,
        start_run_index=0,
        stop_run_index=units.shape[0],
        rng_seed=0,
        run_index_values=tuple(range(units.shape[0])),
    )
    source_units = SourceUnitIntervalSamples(
        values_by_source={
            source: units[:, index].astype(np.float64, copy=False)
            for index, source in enumerate(context.source_names)
        }
    )
    if context.inter_method_plan is not None:
        summary_identity, values = sample_inter_method_summary_matrix_batch(
            loaded=context.loaded,
            inter_method_plan=context.inter_method_plan,
            execution_plan=cast(
                InterMethodExecutionPlan,
                context.inter_method_execution_plan,
            ),
            inter_mrio_plan=context.inter_mrio_plan,
            batch=batch,
            sources=context.sources,
            source_units=source_units,
        )
        return summary_identity, values
    identity, _run_indices, values = sample_compact_batch(
        loaded=context.loaded,
        inter_mrio_plan=context.inter_mrio_plan,
        lcia_plan=context.lcia_plan,
        projection_plan=context.projection_plan,
        batch=batch,
        sources=context.sources,
        external_plan=context.external_plan,
        source_units=source_units,
    )
    return identity, values

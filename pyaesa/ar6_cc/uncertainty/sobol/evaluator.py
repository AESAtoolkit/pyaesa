"""AR6 CC source unit evaluator used by downstream ACC Sobol."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyaesa.ar6_cc.uncertainty.evaluation.summary_identity import (
    ar6_cc_summary_identity_groups,
)
from pyaesa.ar6_cc.uncertainty.runtime.models import AR6CCSamplingGroup, AR6CCUncertaintyPlan
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    collapse_values_to_summary_groups,
)


@dataclass(frozen=True)
class AR6CCSobolEvaluationContext:
    """Prepared AR6 CC Sobol evaluator context."""

    plan: AR6CCUncertaintyPlan


def build_ar6_cc_sobol_evaluation_context(
    *,
    plan: AR6CCUncertaintyPlan,
) -> AR6CCSobolEvaluationContext:
    """Build the canonical AR6 CC source unit evaluator context."""
    return AR6CCSobolEvaluationContext(plan=plan)


def evaluate_ar6_cc_sobol_units(
    *,
    context: AR6CCSobolEvaluationContext,
    units: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Evaluate unit interval source values into AR6 CC public row values."""
    unit_values = np.asarray(units[:, 0], dtype=np.float64)
    plan = context.plan
    values = np.empty((unit_values.shape[0], len(plan.group_identity)), dtype=np.float64)
    if bool(plan.source_parameters["category_uncertainty"]):
        values.fill(np.nan)
        _fill_category_uncertainty_values(values=values, plan=plan, unit_values=unit_values)
    else:
        for group in plan.groups:
            selected = _sample_group_positions(
                plan=plan,
                group=group,
                unit_values=unit_values,
            )
            values[:, group.output_start : group.output_stop] = plan.trajectory_values[selected, :]
    identity, public_row_groups = ar6_cc_summary_identity_groups(
        identity=plan.group_identity,
        category_uncertainty=bool(plan.source_parameters["category_uncertainty"]),
    )
    return identity, collapse_values_to_summary_groups(
        values=values,
        public_row_groups=public_row_groups,
    )


def _fill_category_uncertainty_values(
    *,
    values: np.ndarray,
    plan: AR6CCUncertaintyPlan,
    unit_values: np.ndarray,
) -> None:
    for pool in plan.category_pools:
        local = _unit_positions(unit_values=unit_values, count=len(pool.group_indices))
        nested_u = np.remainder(unit_values * len(pool.group_indices), 1.0)
        for local_position, group_index in enumerate(pool.group_indices):
            run_positions = np.flatnonzero(local == local_position)
            if run_positions.size == 0:
                continue
            group = plan.groups[group_index]
            selected = _sample_group_positions(
                plan=plan,
                group=group,
                unit_values=nested_u[run_positions],
            )
            values[run_positions, group.output_start : group.output_stop] = plan.trajectory_values[
                selected, :
            ]


def _sample_group_positions(
    *,
    plan: AR6CCUncertaintyPlan,
    group: AR6CCSamplingGroup,
    unit_values: np.ndarray,
) -> np.ndarray:
    if str(plan.source_parameters["sampling_method"]) == "srs":
        positions = _unit_positions(
            unit_values=unit_values,
            count=len(group.candidate_positions),
        )
        return group.candidate_positions[positions]
    weights = _model_stratified_weights(plan=plan, group=group)
    return group.candidate_positions[_weighted_positions(unit_values=unit_values, weights=weights)]


def _model_stratified_weights(
    *,
    plan: AR6CCUncertaintyPlan,
    group: AR6CCSamplingGroup,
) -> np.ndarray:
    weights = np.empty(len(group.candidate_positions), dtype=np.float64)
    model_count = len(group.model_candidate_positions)
    candidate_lookup = {
        int(candidate_position): index
        for index, candidate_position in enumerate(group.candidate_positions)
    }
    for model_positions in group.model_candidate_positions:
        count = len(model_positions)
        for candidate_position in model_positions:
            weights[candidate_lookup[int(candidate_position)]] = 1.0 / (model_count * count)
    return weights


def _unit_positions(*, unit_values: np.ndarray, count: int) -> np.ndarray:
    clipped = np.clip(unit_values, 0.0, np.nextafter(1.0, 0.0))
    return np.floor(clipped * count).astype(np.int64)


def _weighted_positions(*, unit_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(weights / weights.sum())
    clipped = np.clip(unit_values, 0.0, np.nextafter(1.0, 0.0))
    return np.searchsorted(cumulative, clipped, side="right").astype(np.int64)

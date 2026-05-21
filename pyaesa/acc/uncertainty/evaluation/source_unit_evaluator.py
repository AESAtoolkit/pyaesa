"""aCC source unit evaluator for ACC and downstream ASR Sobol analysis."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.acc.uncertainty.sources.dynamic_cc import (
    deterministic_dynamic_cc_input,
    dynamic_cc_base_args,
)
from pyaesa.acc.uncertainty.sources.source_keys import ar6_cc_source_name, asocc_source_name
from pyaesa.acc.uncertainty.runtime.models import ACCBranchPlan
from pyaesa.acc.uncertainty.evaluation.branches import (
    build_acc_branch_plans,
    combined_acc_identity,
)
from pyaesa.acc.uncertainty.evaluation.runs import (
    evaluate_acc_value_matrix,
)
from pyaesa.ar6_cc.uncertainty.runtime.prerequisites import (
    load_deterministic_ar6_cc_rows,
    prepare_ar6_cc_deterministic_prerequisite,
)
from pyaesa.ar6_cc.deterministic.io.tables import filter_to_denominator_cc_rows
from pyaesa.ar6_cc.uncertainty.request.normalization import (
    AR6_DYNAMIC_CC_SOURCE,
    normalize_ar6_cc_uncertainty_request,
)
from pyaesa.ar6_cc.uncertainty.evaluation.sampling import build_ar6_cc_sampling_plan
from pyaesa.ar6_cc.uncertainty.sobol.evaluator import (
    AR6CCSobolEvaluationContext,
    build_ar6_cc_sobol_evaluation_context,
    evaluate_ar6_cc_sobol_units,
)
from pyaesa.asocc.uncertainty.engine.sobol.evaluator import (
    AsoccSobolEvaluationContext,
    build_asocc_sobol_evaluation_context_from_request,
    evaluate_asocc_sobol_units,
)
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter
from pyaesa.shared.uncertainty_assessment.sobol.plan import SobolPlan


@dataclass(frozen=True)
class ACCSobolEvaluationContext:
    """Prepared aCC Sobol evaluator context."""

    identity: pd.DataFrame
    branch_plans: tuple[ACCBranchPlan, ...]
    source_names: tuple[str, ...]
    asocc_context: AsoccSobolEvaluationContext
    cc_context: AR6CCSobolEvaluationContext | None
    cc_selected_positions: np.ndarray | None
    deterministic_cc_values: np.ndarray | None


def build_acc_sobol_evaluation_context(
    *,
    branches: list[dict[str, Any]],
    base_asocc_args: dict[str, Any],
    asocc_uncertainty_config: dict[str, Any],
    external_method: dict[str, Any] | None,
    dynamic_cc_config: dict[str, Any] | None,
    full_years: int | list[int] | range,
    sobol_plan: SobolPlan,
) -> ACCSobolEvaluationContext:
    """Build a reusable aCC Sobol evaluator context."""
    asocc_context = build_asocc_sobol_evaluation_context_from_request(
        base_asocc_args=base_asocc_args,
        uncertainty_config=asocc_uncertainty_config,
        external_method=external_method,
        sobol_plan=sobol_plan,
    )
    asocc_identity, _ = evaluate_asocc_sobol_units(
        context=asocc_context,
        units=np.zeros((1, len(asocc_context.source_names)), dtype=np.float64),
    )
    cc_context = _dynamic_cc_context(
        branches=branches,
        dynamic_cc_config=dynamic_cc_config,
        full_years=full_years,
    )
    cc_identity = None
    cc_selected_positions = None
    deterministic_cc_values = None
    source_names = tuple(asocc_source_name(name) for name in asocc_context.source_names)
    if cc_context is not None:
        cc_identity_full, _ = evaluate_ar6_cc_sobol_units(
            context=cc_context,
            units=np.zeros((1, 1), dtype=np.float64),
        )
        cc_year = np.asarray(cc_identity_full["year"], dtype=np.int64)
        selected = np.isin(cc_year, np.asarray(asocc_context.selected_years, dtype=np.int64))
        cc_selected_positions = np.flatnonzero(selected).astype(np.int64, copy=False)
        cc_identity = cc_identity_full.loc[selected].reset_index(drop=True)
        source_names = (*source_names, ar6_cc_source_name(AR6_DYNAMIC_CC_SOURCE))
    elif any(branch["cc_type"] == "dynamic_ar6" for branch in branches):
        cc_input = deterministic_dynamic_cc_input(
            branch=[branch for branch in branches if branch["cc_type"] == "dynamic_ar6"][0],
            years=full_years,
        )
        cc_identity_full = cast(pd.DataFrame, cc_input.identity)
        cc_year = np.asarray(cc_identity_full["year"], dtype=np.int64)
        selected = np.isin(cc_year, np.asarray(asocc_context.selected_years, dtype=np.int64))
        cc_selected_positions = np.flatnonzero(selected).astype(np.int64, copy=False)
        cc_identity = cc_identity_full.loc[selected].reset_index(drop=True)
        cc_values = cast(np.ndarray, cc_input.deterministic_values)
        deterministic_cc_values = cc_values[cc_selected_positions]
    branch_plans = build_acc_branch_plans(
        asocc_identity=asocc_identity,
        cc_identity=cc_identity,
        branches=branches,
    )
    return ACCSobolEvaluationContext(
        identity=combined_acc_identity(branch_plans=branch_plans),
        branch_plans=branch_plans,
        source_names=source_names,
        asocc_context=asocc_context,
        cc_context=cc_context,
        cc_selected_positions=cc_selected_positions,
        deterministic_cc_values=deterministic_cc_values,
    )


def evaluate_acc_sobol_units(
    *,
    context: ACCSobolEvaluationContext,
    units: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Evaluate unit interval source values into aCC public row values."""
    asocc_count = len(context.asocc_context.source_names)
    _, asocc_values = evaluate_asocc_sobol_units(
        context=context.asocc_context,
        units=units[:, :asocc_count],
    )
    cc_values = None
    if context.cc_context is not None:
        _, cc_values_full = evaluate_ar6_cc_sobol_units(
            context=context.cc_context,
            units=units[:, asocc_count : asocc_count + 1],
        )
        cc_values = cc_values_full[:, context.cc_selected_positions]
    elif context.deterministic_cc_values is not None:
        cc_values = np.broadcast_to(
            context.deterministic_cc_values,
            (units.shape[0], len(context.deterministic_cc_values)),
        )
    return (
        context.identity,
        evaluate_acc_value_matrix(
            branch_plans=context.branch_plans,
            asocc_values=asocc_values,
            cc_values=cc_values,
        ),
    )


def _dynamic_cc_context(
    *,
    branches: list[dict[str, Any]],
    dynamic_cc_config: dict[str, Any] | None,
    full_years: int | list[int] | range,
) -> AR6CCSobolEvaluationContext | None:
    dynamic_branches = [branch for branch in branches if branch["cc_type"] == "dynamic_ar6"]
    if not dynamic_branches or dynamic_cc_config is None:
        return None
    branch = dynamic_branches[0]
    request = normalize_ar6_cc_uncertainty_request(
        base_ar6_cc_args=dynamic_cc_base_args(branch=branch, years=full_years),
        source_parameters=dynamic_cc_config,
    )
    prerequisite = prepare_ar6_cc_deterministic_prerequisite(
        request=request,
        refresh=False,
        status=NullPhasePrinter(),
    )
    deterministic_rows = load_deterministic_ar6_cc_rows(request=request, scope=prerequisite)
    deterministic_rows = filter_to_denominator_cc_rows(deterministic_rows)
    return build_ar6_cc_sobol_evaluation_context(
        plan=build_ar6_cc_sampling_plan(
            request=request,
            deterministic_rows=deterministic_rows,
        )
    )

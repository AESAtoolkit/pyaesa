"""aCC source unit evaluator for ACC and downstream ASR Sobol analysis."""

from dataclasses import dataclass, replace
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
from pyaesa.acc.uncertainty.evaluation.summary import acc_summary_excluded_columns
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
    asocc_sobol_base_chunk_rows,
    build_asocc_sobol_evaluation_context_from_request,
    evaluate_asocc_sobol_units,
)
from pyaesa.shared.acc_asr_common.branches.expand import has_dynamic_ar6_branch
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter
from pyaesa.shared.runtime.scenario.columns import ASOCC_TIME_ROUTE_PUBLIC_COLUMN
from pyaesa.shared.uncertainty_assessment.evaluation.scenario_groups import (
    scenario_identity_groups_from_excluded_columns,
)
from pyaesa.shared.uncertainty_assessment.sobol.plan import (
    SobolPlan,
    selected_sobol_output_years,
    studied_output_years,
)


@dataclass(frozen=True)
class ACCSobolEvaluationContext:
    """Prepared aCC Sobol evaluator context."""

    identity: pd.DataFrame
    target_identity: pd.DataFrame
    target_public_row_groups: tuple[tuple[str, ...], ...]
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
    full_year_tokens = studied_output_years(full_years)
    target_years = selected_sobol_output_years(
        plan=sobol_plan,
        available_years=full_year_tokens,
    )
    source_plan = (
        replace(sobol_plan, sobol_years=full_year_tokens)
        if has_dynamic_ar6_branch(branches=branches)
        else sobol_plan
    )
    asocc_context = build_asocc_sobol_evaluation_context_from_request(
        base_asocc_args=base_asocc_args,
        uncertainty_config=asocc_uncertainty_config,
        external_method=external_method,
        sobol_plan=source_plan,
    )
    asocc_identity, _ = evaluate_asocc_sobol_units(
        context=asocc_context,
        units=np.zeros((1, len(asocc_context.source_names)), dtype=np.float64),
    )
    asocc_identity = _value_matrix_identity(identity=asocc_identity)
    selected_years = list(asocc_context.selected_years)
    cc_context = _dynamic_cc_context(
        branches=branches,
        dynamic_cc_config=dynamic_cc_config,
        full_years=selected_years,
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
        cc_identity = _value_matrix_identity(identity=cc_identity_full.loc[selected])
        source_names = (*source_names, ar6_cc_source_name(AR6_DYNAMIC_CC_SOURCE))
    elif has_dynamic_ar6_branch(branches=branches):
        cc_input = deterministic_dynamic_cc_input(
            branch=[branch for branch in branches if branch["cc_type"] == "dynamic_ar6"][0],
            years=selected_years,
        )
        cc_identity_full = cast(pd.DataFrame, cc_input.identity)
        cc_year = np.asarray(cc_identity_full["year"], dtype=np.int64)
        selected = np.isin(cc_year, np.asarray(asocc_context.selected_years, dtype=np.int64))
        cc_selected_positions = np.flatnonzero(selected).astype(np.int64, copy=False)
        cc_identity = _value_matrix_identity(identity=cc_identity_full.loc[selected])
        cc_values = cast(np.ndarray, cc_input.deterministic_values)
        deterministic_cc_values = cc_values[cc_selected_positions]
    branch_plans = build_acc_branch_plans(
        asocc_identity=asocc_identity,
        cc_identity=cc_identity,
        branches=branches,
    )
    identity = combined_acc_identity(branch_plans=branch_plans)
    target_identity, target_public_row_groups = _acc_sobol_target_identity_groups(
        identity=identity,
        target_years=target_years,
        active_sources=source_names,
        dynamic_category_uncertainty_active=_dynamic_category_uncertainty_active(
            context=cc_context
        ),
    )
    return ACCSobolEvaluationContext(
        identity=identity,
        target_identity=target_identity,
        target_public_row_groups=target_public_row_groups,
        branch_plans=branch_plans,
        source_names=source_names,
        asocc_context=asocc_context,
        cc_context=cc_context,
        cc_selected_positions=cc_selected_positions,
        deterministic_cc_values=deterministic_cc_values,
    )


def _value_matrix_identity(*, identity: pd.DataFrame) -> pd.DataFrame:
    out = identity.drop(columns=["public_row_id"], errors="ignore").reset_index(drop=True)
    out["public_row_id"] = np.arange(len(out), dtype=np.int64)
    return out


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


def evaluate_acc_sobol_target_units(
    *,
    context: ACCSobolEvaluationContext,
    units: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Evaluate Sobol units into the public aCC target selected for Sobol."""
    _identity, values = evaluate_acc_sobol_units(context=context, units=units)
    return context.target_identity, _sum_values_to_groups(
        values=values,
        public_row_groups=context.target_public_row_groups,
    )


def acc_sobol_base_chunk_rows(*, context: ACCSobolEvaluationContext) -> int:
    """Return Sobol base rows bounded by nested aSoCC source evaluation."""
    return asocc_sobol_base_chunk_rows(
        context=context.asocc_context,
        dimension_count=len(context.source_names),
    )


def _dynamic_category_uncertainty_active(
    *,
    context: AR6CCSobolEvaluationContext | None,
) -> bool:
    """Return whether dynamic AR6 CC category uncertainty is active."""
    if context is None:
        return False
    return bool(context.plan.source_parameters["category_uncertainty"])


def _acc_sobol_target_identity_groups(
    *,
    identity: pd.DataFrame,
    target_years: tuple[int, ...],
    active_sources: tuple[str, ...],
    dynamic_category_uncertainty_active: bool,
) -> tuple[pd.DataFrame, tuple[tuple[str, ...], ...]]:
    """Return yearly static and cumulative dynamic aCC Sobol target groups."""
    dynamic = identity["cc_type"].astype("string").fillna("").eq("dynamic_ar6")
    frames: list[pd.DataFrame] = []
    groups: list[tuple[str, ...]] = []
    static = identity.loc[~dynamic].copy()
    if "year" in static.columns:
        static_year = pd.Series(
            pd.to_numeric(static["year"], errors="raise"),
            index=static.index,
        ).astype(int)
        static = static.loc[static_year.isin(target_years)]
    if not static.empty:
        frames.append(static.drop(columns=["public_row_id"], errors="ignore"))
        groups.extend((str(public_id),) for public_id in static["public_row_id"].tolist())
    dynamic_identity = identity.loc[dynamic].copy()
    if not dynamic_identity.empty:
        excluded = acc_summary_excluded_columns(
            active_sources=active_sources,
            dynamic_category_uncertainty_active=dynamic_category_uncertainty_active,
        )
        excluded.add("year")
        excluded.add(ASOCC_TIME_ROUTE_PUBLIC_COLUMN)
        excluded.difference_update({"l1_l2_method", "l1_method", "l2_method"})
        cumulative_identity, cumulative_groups = scenario_identity_groups_from_excluded_columns(
            identity=dynamic_identity,
            excluded_columns=excluded,
        )
        frames.append(cumulative_identity.drop(columns=["public_row_id"], errors="ignore"))
        groups.extend(cumulative_groups)
    target_identity = pd.concat(frames, ignore_index=True, sort=False)
    target_identity.insert(0, "public_row_id", np.arange(len(target_identity), dtype=np.int64))
    return target_identity, tuple(groups)


def _sum_values_to_groups(
    *,
    values: np.ndarray,
    public_row_groups: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    """Sum value matrix columns to the requested public row groups."""
    stable_ordered_groups = all(
        len(group) == 1 and int(group[0]) == index for index, group in enumerate(public_row_groups)
    )
    if stable_ordered_groups:
        return values
    out = np.empty((values.shape[0], len(public_row_groups)), dtype=np.float64)
    for index, group in enumerate(public_row_groups):
        positions = np.array([int(public_row_id) for public_row_id in group], dtype=np.int64)
        out[:, index] = values[:, positions].sum(axis=1)
    return out


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

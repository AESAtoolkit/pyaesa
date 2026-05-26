"""Scenario-level yearly processing."""

from dataclasses import replace

import pandas as pd

from ...method_scope import _unique_l2_methods_in_scope
from ....methods.registry.registry import REGISTRY
from ...projection.payload.cache import get_projected_payload
from ..l1.l1_compute import _compute_l1_for_year
from ..l1.l1_types import _L1RunContext
from ..l2.l2_compute import _compute_l2_for_year
from ..l2.l2_contracts import require_compute_inputs
from ..l2.l2_slicing import _slice_l2_inputs_for_compute
from ..l2.l2_types import _L2ComputeInputs, _L2RunContext
from ..enacting_metric.enacting_metric_base import (
    record_base_enacting_metrics,
)
from ..enacting_metric.enacting_metric_lcia_percap import (
    record_lcia_percap_enacting_metrics,
)
from ..enacting_metric.enacting_metric_pr import (
    record_pr_enacting_metrics,
)
from .year_inputs import (
    _MrioPayload,
    _ScenarioRunContext,
    _load_scenario_population_gdp,
)
from .scenario_routing import (
    flush_pr_hr_rp1_zero_fallback_notices,
    is_scenario_dependent_l1,
    is_scenario_dependent_l2_projection,
)


def _needs_regression_projection_payload(*, context, year: int) -> bool:
    """Return whether future year projection requires synthetic MRIO enacting metrics."""
    projection_context = context.projection_context
    if projection_context is None or not projection_context.enabled:
        return False
    if not projection_context.is_future_year(int(year)):
        return False
    method_names = _unique_l2_methods_in_scope(
        selected_l2_one_step=context.selected_l2_one_step,
        combined=context.combined,
    )
    return any(
        projection_context.route_for_l2_method(name) == "regression" for name in method_names
    )


def _select_scenario_methods(*, run_ctx: _ScenarioRunContext, process_invariant_methods: bool):
    """Resolve scenario effective L1/L2 method lists."""
    if process_invariant_methods:
        return (
            list(run_ctx.context.selected_l1),
            list(run_ctx.context.selected_l2_one_step),
            list(run_ctx.context.combined),
        )

    selected_l2_one_step_effective = [
        name
        for name in run_ctx.context.selected_l2_one_step
        if is_scenario_dependent_l2_projection(
            context=run_ctx.context,
            year=run_ctx.year,
            l2_method=name,
        )
    ]
    combined_effective = [
        pair
        for pair in run_ctx.context.combined
        if (
            is_scenario_dependent_l1(pair[1])
            or is_scenario_dependent_l2_projection(
                context=run_ctx.context,
                year=run_ctx.year,
                l2_method=pair[0],
            )
        )
    ]
    selected_l1_effective = [
        name for name in run_ctx.context.selected_l1 if is_scenario_dependent_l1(name)
    ]
    return selected_l1_effective, selected_l2_one_step_effective, combined_effective


def _l1_key_matches_method(*, key: str, l1_method: str) -> bool:
    """Return whether one L1 per year cache key belongs to a given method."""
    return (
        key == l1_method or key.startswith(f"{l1_method}_") or key.startswith(f"{l1_method}__for__")
    )


def _cache_invariant_l1_results_for_year(
    *,
    run_ctx: _ScenarioRunContext,
    l1_results_year: dict[str, pd.DataFrame],
) -> None:
    """Persist invariant L1 per year results from the primary scenario run."""
    invariant_methods = [
        name for name in run_ctx.context.selected_l1 if not is_scenario_dependent_l1(name)
    ]
    if not invariant_methods:
        run_ctx.state.l1_year_invariant_cache[int(run_ctx.year)] = {}
        return
    cached: dict[str, pd.DataFrame] = {}
    for key, frame in l1_results_year.items():
        if any(
            _l1_key_matches_method(key=str(key), l1_method=l1_method)
            for l1_method in invariant_methods
        ):
            cached[str(key)] = frame.copy()
    run_ctx.state.l1_year_invariant_cache[int(run_ctx.year)] = cached


def _merge_cached_invariant_l1_results(
    *,
    run_ctx: _ScenarioRunContext,
    l1_results_year: dict[str, pd.DataFrame],
) -> None:
    """Merge invariant L1 year results computed by primary scenario branch."""
    cached = run_ctx.state.l1_year_invariant_cache.get(int(run_ctx.year), {})
    for key, frame in cached.items():
        l1_results_year.setdefault(str(key), frame.copy())


def _build_empty_l2_inputs() -> _L2ComputeInputs:
    """Return empty L2 inputs for AR/projection reuse branches without MRIO enacting metrics."""
    return _L2ComputeInputs(
        fd_rf=pd.Series(dtype=float),
        gva_rp=pd.Series(dtype=float),
        fd_rp_sp_rf=pd.DataFrame(),
        fd_rp_sp=pd.Series(dtype=float),
        fd_rf_sp=pd.Series(dtype=float),
        gva_rp_sp=pd.Series(dtype=float),
        x_to_rc=pd.DataFrame(),
        kappa=pd.DataFrame(),
        omega_reg=pd.DataFrame(),
    )


def _process_scenario_for_year(
    *,
    run_ctx: _ScenarioRunContext,
    lcia_by_method: dict[str, dict] | None,
    lcia_by_method_original: dict[str, dict] | None,
    lcia_effective_year_by_method: dict[str, int] | None,
    lcia_effective_year_by_method_original: dict[str, int] | None,
    reg_agg_map: dict[str, str],
    mrio_payload: _MrioPayload | None,
    l2_inputs_sliced: _L2ComputeInputs | None,
    process_invariant_methods: bool,
) -> None:
    """Process one scenario branch for a studied year."""
    scenario_inputs = _load_scenario_population_gdp(run_ctx=run_ctx)
    if bool(getattr(run_ctx.context, "intermediate_outputs", True)):
        record_pr_enacting_metrics(
            context=run_ctx.context,
            state=run_ctx.state,
            year=run_ctx.year,
            ssp_scenario=run_ctx.ssp_scenario,
            reg_agg_map=reg_agg_map,
            pop_iso=scenario_inputs.pop_iso,
            gdp_iso=scenario_inputs.gdp_iso,
            iso_to_mrio=scenario_inputs.iso_to_mrio,
        )
    selected_l1, selected_l2_one_step, combined = _select_scenario_methods(
        run_ctx=run_ctx,
        process_invariant_methods=process_invariant_methods,
    )
    scenario_context = replace(
        run_ctx.context,
        selected_l1=selected_l1,
        selected_l2_one_step=selected_l2_one_step,
        combined=combined,
    )

    l1_run = _L1RunContext(
        context=scenario_context,
        state=run_ctx.state,
        year=run_ctx.year,
        ssp_scenario=run_ctx.ssp_scenario,
        pop_series=scenario_inputs.pop_series,
        pop_series_original=scenario_inputs.pop_series_original,
        pr_pop=scenario_inputs.pop_iso,
        pr_gdp=scenario_inputs.gdp_iso,
        pr_to_mrio=scenario_inputs.iso_to_mrio,
        l1_results_year={},
    )
    l1_results_year = _compute_l1_for_year(
        run=l1_run,
        lcia_by_method=lcia_by_method,
        lcia_by_method_original=lcia_by_method_original,
    )
    if process_invariant_methods:
        _cache_invariant_l1_results_for_year(
            run_ctx=run_ctx,
            l1_results_year=l1_results_year,
        )
    else:
        _merge_cached_invariant_l1_results(
            run_ctx=run_ctx,
            l1_results_year=l1_results_year,
        )
    if not process_invariant_methods:
        scenario_l1_results = run_ctx.state.l1_results_by_ssp_scenario.get(run_ctx.ssp_scenario, {})
        for output_spec in list(scenario_l1_results):
            if not output_spec.scenario_dependent:
                scenario_l1_results.pop(output_spec, None)
    use_original_l1_post = (
        scenario_context.use_original_l1_post_domain and scenario_context.l1_reg_aggreg == "post"
    )
    lcia_percap_source = (
        lcia_by_method_original
        if use_original_l1_post and lcia_by_method_original is not None
        else lcia_by_method
    )
    pop_percap_source = (
        scenario_inputs.pop_series_original
        if use_original_l1_post and scenario_inputs.pop_series_original is not None
        else scenario_inputs.pop_series
    )
    if bool(getattr(scenario_context, "intermediate_outputs", True)):
        record_lcia_percap_enacting_metrics(
            context=scenario_context,
            state=run_ctx.state,
            year=run_ctx.year,
            ssp_scenario=run_ctx.ssp_scenario,
            lcia_by_method=lcia_percap_source,
            pop_series=pop_percap_source,
            use_original_domain=use_original_l1_post,
            lcia_effective_year_by_method=(
                lcia_effective_year_by_method_original
                if use_original_l1_post
                else lcia_effective_year_by_method
            ),
        )
    flush_pr_hr_rp1_zero_fallback_notices(
        context=scenario_context,
        state=run_ctx.state,
    )
    active_payload = mrio_payload
    active_inputs = l2_inputs_sliced
    projection_context = scenario_context.projection_context
    if (
        active_payload is None
        and projection_context is not None
        and projection_context.enabled
        and projection_context.is_future_year(run_ctx.year)
        and _needs_regression_projection_payload(
            context=scenario_context,
            year=run_ctx.year,
        )
    ):
        active_payload = get_projected_payload(
            context=scenario_context,
            state=run_ctx.state,
            year=run_ctx.year,
            ssp_scenario=run_ctx.ssp_scenario,
            gdp_series=scenario_inputs.gdp_series,
        )
        active_inputs = _slice_l2_inputs_for_compute(
            context=scenario_context,
            inputs=active_payload.l2_inputs,
        )
    if active_payload is not None and bool(getattr(scenario_context, "intermediate_outputs", True)):
        wb_year_columns = {str(col) for col in scenario_context.wb_df.columns if str(col).isdigit()}
        wb_backed_year = str(int(run_ctx.year)) in wb_year_columns
        should_record_base_enacting_metric = process_invariant_methods or not wb_backed_year
        if should_record_base_enacting_metric:
            record_base_enacting_metrics(
                context=scenario_context,
                state=run_ctx.state,
                year=run_ctx.year,
                ssp_scenario=run_ctx.ssp_scenario,
                enacting_metric_l1=active_payload.enacting_metric_l1,
                enacting_metric_l2=active_payload.enacting_metric_l2,
                utility=active_payload.utility,
            )
    if active_payload is None:
        ar_one_step = [
            name
            for name in scenario_context.selected_l2_one_step
            if REGISTRY.method_is_ar(name, level="L2", fu_code=scenario_context.fu_code)
        ]
        ar_combined = [
            (l2_name, l1_name)
            for l2_name, l1_name in scenario_context.combined
            if REGISTRY.method_is_ar(
                l2_name,
                level="L2",
                fu_code=scenario_context.fu_code,
            )
        ]
        projection_reuse_one_step = [
            name
            for name in scenario_context.selected_l2_one_step
            if (
                projection_context is not None
                and projection_context.enabled
                and projection_context.is_future_year(run_ctx.year)
                and projection_context.route_for_l2_method(name) == "historical_reuse"
            )
        ]
        projection_reuse_combined = [
            (l2_name, l1_name)
            for l2_name, l1_name in scenario_context.combined
            if (
                projection_context is not None
                and projection_context.enabled
                and projection_context.is_future_year(run_ctx.year)
                and projection_context.route_for_l2_method(l2_name) == "historical_reuse"
            )
        ]
        selected_one_step = [*ar_one_step, *projection_reuse_one_step]
        selected_combined = [*ar_combined, *projection_reuse_combined]
        if not selected_one_step and not selected_combined:
            return
        partial_context = replace(
            scenario_context,
            selected_l2_one_step=selected_one_step,
            combined=selected_combined,
        )
        l2_run = _L2RunContext(
            context=partial_context,
            state=run_ctx.state,
            year=run_ctx.year,
            ssp_scenario=run_ctx.ssp_scenario,
            lcia_by_method=lcia_by_method,
            l1_results_year=l1_results_year,
            inputs=_build_empty_l2_inputs(),
        )
        _compute_l2_for_year(run=l2_run)
        return
    l2_run = _L2RunContext(
        context=scenario_context,
        state=run_ctx.state,
        year=run_ctx.year,
        ssp_scenario=run_ctx.ssp_scenario,
        lcia_by_method=lcia_by_method,
        l1_results_year=l1_results_year,
        inputs=require_compute_inputs(
            inputs=active_inputs,
            where=(
                f"year={run_ctx.year}, fu_code='{scenario_context.fu_code}', "
                f"selected_l2={scenario_context.selected_l2_one_step}, "
                f"combined={scenario_context.combined}"
            ),
        ),
    )
    _compute_l2_for_year(run=l2_run)

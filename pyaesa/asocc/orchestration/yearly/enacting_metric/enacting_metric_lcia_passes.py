"""Pass orchestration for enacting metric LCIA-derived metrics."""

import pandas as pd

from ....io.metadata import EnactingMetricKey, RunContext, RunState
from ....methods.equations.pr_hr_ecap_cum import build_parent_cumulative_per_cap
from ..l1.l1_pr_hr_setup import _prepare_pr_hr_parent_cumulative_runtime
from .enacting_metric_lcia_routing import (
    _PerCapMetricPair,
    _resolve_pr_hr_cumulative_metric_contract,
)
from .enacting_metric_lcia_selection import (
    _iter_direct_percap_scopes,
    _iter_pr_hr_cumulative_scopes,
    _resolve_pr_hr_base_inputs,
)
from .enacting_metric_lcia_shaping import (
    _shape_lcia_percap_series,
    _shape_pr_hr_cumulative_series,
)
from .enacting_metric_common import _record_enacting_metric_input


def _record_direct_lcia_percap_pass(
    *,
    context: RunContext,
    state: RunState,
    year: int,
    scenario_key: str | None,
    lcia_by_method: dict[str, dict] | None,
    pop_series: pd.Series,
    pairs: list[_PerCapMetricPair],
    use_original_domain: bool,
    lcia_effective_year_by_method: dict[str, int] | None,
) -> None:
    """Run the direct LCIA per-cap enacting metric recording pass."""
    for scope in _iter_direct_percap_scopes(
        year=int(year),
        lcia_by_method=lcia_by_method,
        lcia_effective_year_by_method=lcia_effective_year_by_method,
    ):
        for pair in pairs:
            if pair.source_metric not in scope.lcia_data:
                continue
            series = _shape_lcia_percap_series(
                lcia_frame=scope.lcia_data[pair.source_metric],
                population=pop_series,
                output_metric=pair.output_metric,
                region_label=pair.region_label,
                use_original_domain=bool(use_original_domain),
                source_key=context.source,
                group_version=context.group_version_reg,
            )
            key = EnactingMetricKey(
                metric=pair.output_metric,
                lcia_method=scope.lcia_method,
                ssp_scenario=scenario_key,
            )
            _record_enacting_metric_input(
                context=context,
                state=state,
                key=key,
                year=scope.effective_year,
                series=series,
                level="level_1",
            )


def _record_pr_hr_cumulative_pass(
    *,
    context: RunContext,
    state: RunState,
    year: int,
    ssp_scenario: str | None,
    scenario_key: str | None,
    lcia_by_method: dict[str, dict] | None,
    cumulative_kinds: set[str],
    use_original_domain: bool,
    lcia_effective_year_by_method: dict[str, int] | None,
) -> None:
    """Run the PR-HR cumulative enacting metric recording pass."""
    lcia_store, population_by_year, lcia_methods_in_scope = _resolve_pr_hr_base_inputs(
        context=context,
        state=state,
        ssp_scenario=ssp_scenario,
        use_original_domain=bool(use_original_domain),
        lcia_by_method=lcia_by_method,
    )
    for scope in _iter_pr_hr_cumulative_scopes(
        state=state,
        year=int(year),
        lcia_kinds=cumulative_kinds,
        lcia_store=lcia_store,
        lcia_methods_in_scope=lcia_methods_in_scope,
        lcia_effective_year_by_method=lcia_effective_year_by_method,
    ):
        try:
            pr_hr_runtime = _prepare_pr_hr_parent_cumulative_runtime(
                context=context,
                state=state,
                lcia_method=scope.lcia_method,
                lcia_kind=scope.lcia_kind,
                ssp_scenario=ssp_scenario,
                population_by_year=population_by_year,
                impact_year=scope.effective_year,
                use_original_domain=bool(use_original_domain),
                include_wb_history_for_scenario=True,
            )
            parent_cum = build_parent_cumulative_per_cap(
                impact_year=scope.effective_year,
                population_by_year=pr_hr_runtime.population_by_year,
                lcia_reg_by_year=scope.lcia_reg_by_year,
                rps_df=scope.rps_df,
                impact_parent_map=scope.impact_parent_map,
                available_years=scope.available_years,
                parent_cum_cache=pr_hr_runtime.parent_cum_cache,
                fallback_callback=pr_hr_runtime.fallback_callback,
            )
        except ValueError:
            continue
        if not parent_cum:
            continue
        cumulative_contract = _resolve_pr_hr_cumulative_metric_contract(lcia_kind=scope.lcia_kind)
        cumulative_series = _shape_pr_hr_cumulative_series(
            parent_cum=parent_cum,
            contract=cumulative_contract,
            use_original_domain=bool(use_original_domain),
            source_key=context.source,
            group_version=context.group_version_reg,
        )
        _record_enacting_metric_input(
            context=context,
            state=state,
            key=EnactingMetricKey(
                metric=cumulative_contract.output_metric,
                lcia_method=scope.lcia_method,
                ssp_scenario=scenario_key,
            ),
            year=scope.effective_year,
            series=cumulative_series,
            level="level_1",
        )

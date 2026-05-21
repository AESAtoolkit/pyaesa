"""PR-HR setup for yearly orchestration."""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ....io.metadata import RunContext, RunState
from ..shared.scenario_routing import build_pr_hr_rp1_zero_fallback_recorder
from .l1_population_inputs import _load_population_for_year
from .l1_types import _L1RunContext, _LciaMethodInputs


@dataclass(frozen=True)
class _PreparedPrHrRuntime:
    population_by_year: dict[int, pd.Series]
    parent_cum_cache: dict[int, dict[str, pd.Series]]
    fallback_callback: Callable[[list[str], int, int], None] | None


def _ensure_pr_hr_wb_population_history(
    *,
    context: RunContext,
    state: RunState,
    impact_year: int,
    use_original_domain: bool,
) -> dict[int, pd.Series]:
    """Return WB population history required by PR-HR responsibility windows."""
    base_history = (
        state.pr_post_pop_series_by_ssp_scenario
        if use_original_domain
        else state.pop_series_by_ssp_scenario
    )
    wb_history = base_history.setdefault(None, {})
    group_version_reg = None if use_original_domain else context.group_version_reg
    for hist_year in context.historical_years:
        year_key = int(hist_year)
        if year_key > int(impact_year) or year_key in wb_history:
            continue
        wb_history[year_key] = _load_population_for_year(
            context=context,
            year=year_key,
            ssp_scenario=None,
            group_version_reg=group_version_reg,
        )
    return wb_history


def _prepare_pr_hr_parent_cumulative_runtime(
    *,
    context: RunContext,
    state: RunState,
    lcia_method: str,
    lcia_kind: str,
    ssp_scenario: str | None,
    population_by_year: dict[int, pd.Series],
    impact_year: int,
    use_original_domain: bool,
    include_wb_history_for_scenario: bool,
) -> _PreparedPrHrRuntime:
    """Resolve shared PR-HR population history, cache bucket, and recorder."""
    wb_history = _ensure_pr_hr_wb_population_history(
        context=context,
        state=state,
        impact_year=int(impact_year),
        use_original_domain=bool(use_original_domain),
    )
    if include_wb_history_for_scenario:
        # PR-HR cumulative windows need WB history plus SSP years for future population.
        merged_history = dict(wb_history)
        merged_history.update(population_by_year)
        population_by_year = merged_history

    cache_scenario = None if str(impact_year) in context.wb_df.columns else ssp_scenario
    pr_hr_cache_key = (
        str(lcia_method),
        str(lcia_kind),
        bool(use_original_domain),
        cache_scenario,
    )
    parent_cum_cache = state.pr_hr_parent_cum_cache.setdefault(
        pr_hr_cache_key,
        {},
    )
    fallback_callback = build_pr_hr_rp1_zero_fallback_recorder(
        state=state,
        l1_method=str(lcia_method),
        lcia_kind=str(lcia_kind),
        use_original_domain=bool(use_original_domain),
        ssp_scenario=cache_scenario,
    )
    return _PreparedPrHrRuntime(
        population_by_year=population_by_year,
        parent_cum_cache=parent_cum_cache,
        fallback_callback=fallback_callback,
    )


def _prepare_pr_hr_standard_inputs(
    *,
    run: _L1RunContext,
    lcia_inputs: _LciaMethodInputs,
    population_by_year: dict[int, pd.Series],
    use_original_domain: bool,
) -> _PreparedPrHrRuntime:
    """Resolve PR-HR history, cache bucket, and fallback recorder for one L1 run."""
    return _prepare_pr_hr_parent_cumulative_runtime(
        context=run.context,
        state=run.state,
        lcia_method=str(lcia_inputs.lcia_method),
        lcia_kind=str(lcia_inputs.lcia_kind),
        ssp_scenario=run.ssp_scenario,
        population_by_year=population_by_year,
        impact_year=(
            int(lcia_inputs.impact_year) if lcia_inputs.impact_year is not None else int(run.year)
        ),
        use_original_domain=bool(use_original_domain),
        include_wb_history_for_scenario=True,
    )

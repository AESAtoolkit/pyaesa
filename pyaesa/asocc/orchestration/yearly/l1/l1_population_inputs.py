"""Population input resolution for L1 yearly orchestration."""

import pandas as pd

from pyaesa.download.pop_gdp.contracts import POP_SSP_INDICATOR, POP_WB_INDICATOR

from ....data.load_pop_gdp import _get_series_for_year
from .l1_types import _L1RunContext


def _load_population_for_year(
    *,
    context,
    year: int,
    ssp_scenario: str | None,
    group_version_reg: str | None,
) -> pd.Series:
    """Load one L1 population series from processed pop/gdp tables."""
    year_col = str(int(year))
    use_ssp = year_col not in context.wb_df.columns
    if context.l1_only_no_mrio:
        pop_df = context.ssp_df_raw if use_ssp else context.wb_df_raw
    else:
        pop_df = context.ssp_df if use_ssp else context.wb_df
    pop_var = POP_SSP_INDICATOR if use_ssp else POP_WB_INDICATOR
    region_override = "iso3_code" if context.l1_only_no_mrio else None
    return _get_series_for_year(
        df=pop_df,
        variable=pop_var,
        year=year,
        source_key=context.source,
        group_version=group_version_reg,
        ssp_scenario=ssp_scenario if use_ssp else None,
        region_col_override=region_override,
    )


def _resolve_l1_population_inputs(
    *,
    run: _L1RunContext,
    use_original_domain: bool,
) -> tuple[pd.Series, dict[int, pd.Series]]:
    """Return current-year L1 population inputs and history cache."""
    pop_series = (
        run.pop_series_original
        if use_original_domain and run.pop_series_original is not None
        else run.pop_series
    )
    pop_by_year = (
        run.state.pr_post_pop_series_by_ssp_scenario[run.ssp_scenario]
        if use_original_domain
        else run.state.pop_series_by_ssp_scenario[run.ssp_scenario]
    )
    if use_original_domain and run.context.group_version_reg:
        post_cache = run.state.pr_post_pop_series_by_ssp_scenario.setdefault(run.ssp_scenario, {})
        for hist_year in run.context.historical_years:
            if hist_year not in post_cache:
                post_cache[hist_year] = _load_population_for_year(
                    context=run.context,
                    year=hist_year,
                    ssp_scenario=run.ssp_scenario,
                    group_version_reg=None,
                )
        pop_series = post_cache[run.year]
        pop_by_year = post_cache
    return pop_series, pop_by_year


def _load_ar_reference_population(
    *,
    run: _L1RunContext,
    ref_year: int,
    use_original_domain: bool,
) -> pd.Series:
    """Load one AR reference population series through the L1 cache contract."""
    ref_pop_store = (
        run.state.pr_post_pop_series_by_ssp_scenario
        if use_original_domain
        else run.state.pop_series_by_ssp_scenario
    )
    ref_pop_cache = ref_pop_store.setdefault(None, {})
    pop_ref = ref_pop_cache.get(ref_year)
    if pop_ref is None:
        pop_ref = _load_population_for_year(
            context=run.context,
            year=ref_year,
            ssp_scenario=None,
            group_version_reg=(None if use_original_domain else run.context.group_version_reg),
        )
        ref_pop_cache[ref_year] = pop_ref
    return pop_ref

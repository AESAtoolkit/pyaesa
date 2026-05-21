"""AR orchestration for L1/L2 yearly execution."""

import pandas as pd

from ..data.reference_payloads import load_reference_lcia_reg_for_domain
from .ar_cache import _ensure_ar_l2_cached, _project_cached_baseline_for_year
from .compute_l1 import compute_l1_method, resolve_l1_region_label
from .equations.ar_result_indexing import (
    _add_reference_level,
)
from .equations.ar_nan_outputs import (
    _nan_like_ar_l1,
    _nan_like_ar_l2,
)


def _shared_ar_cache(
    cache_by_scenario: dict[str | None, dict[tuple, pd.DataFrame]],
) -> dict[tuple, pd.DataFrame]:
    """Return scenario agnostic AR cache bucket."""
    return cache_by_scenario.setdefault(None, {})


def _compute_ar_l2_result(
    *,
    context,
    state,
    cache_key: tuple,
    l2_method: str,
    year: int,
    ref_year: int,
    lcia_data,
    l1_weights,
) -> pd.DataFrame | None:
    """Compute AR L2 output for one (year, reference year) pair.

    Returns:
        Wide frame with one studied year column and ``reference_year`` index level,
        or ``None`` when no LCIA payload is available.
    """
    lcia_key = str(cache_key[2])
    cache = _shared_ar_cache(state.ar_l2_cache_by_ssp_scenario)
    if year < ref_year:
        # Before reference year, AR outputs are defined as NaN placeholders.
        if lcia_data is None:
            return None
        state.empty_ref_years.setdefault(ref_year, []).append(year)
        result = _nan_like_ar_l2(
            l2_method=l2_method,
            fu_code=context.fu_code,
            lcia=lcia_data,
            year=year,
            pre_weighting=False,
        )
        return _add_reference_level(
            result,
            ref_year,
            index_cache=state.output_index_level_cache,
        )

    def _compute_ref_baseline() -> pd.DataFrame:
        # Baseline is always computed at ref_year and then projected to target year.
        return _ensure_ar_l2_cached(
            context=context,
            state=state,
            ssp_scenario=None,
            cache_key=cache_key,
            l2_method=l2_method,
            ref_year=ref_year,
            lcia_key=lcia_key,
            l1_weights=l1_weights,
            pre_weighting=False,
            force_recompute=True,
        )

    result = _project_cached_baseline_for_year(
        cache=cache,
        cache_key=cache_key,
        year=year,
        compute_baseline=_compute_ref_baseline,
        force_recompute_at_ref=(year == ref_year),
    )
    return _add_reference_level(
        result,
        ref_year,
        index_cache=state.output_index_level_cache,
    )


def _compute_ar_l2_preweight(
    *,
    context,
    state,
    cache_key: tuple,
    l2_method: str,
    year: int,
    ref_year: int,
    lcia_data,
) -> pd.DataFrame | None:
    """Compute AR L2 pre weight matrix for a specific year/reference pair."""
    lcia_key = str(cache_key[2])
    cache = _shared_ar_cache(state.ar_l2_cache_by_ssp_scenario)
    if year < ref_year:
        if lcia_data is None:
            return None
        return _nan_like_ar_l2(
            l2_method=l2_method,
            fu_code=context.fu_code,
            lcia=lcia_data,
            year=year,
            pre_weighting=True,
        )

    def _compute_ref_baseline() -> pd.DataFrame:
        # Preweight baseline is cached separately from final weighted outputs.
        return _ensure_ar_l2_cached(
            context=context,
            state=state,
            ssp_scenario=None,
            cache_key=cache_key,
            l2_method=l2_method,
            ref_year=ref_year,
            lcia_key=lcia_key,
            l1_weights=None,
            pre_weighting=True,
            force_recompute=True,
        )

    return _project_cached_baseline_for_year(
        cache=cache,
        cache_key=cache_key,
        year=year,
        compute_baseline=_compute_ref_baseline,
        force_recompute_at_ref=(year == ref_year),
    )


def _compute_ar_l1_result(
    *,
    context,
    state,
    ssp_scenario: str | None,
    cache_key: tuple,
    l1_method: str,
    year: int,
    ref_year: int,
    lcia_method: str,
    lcia_kind: str,
    lcia_reg: pd.DataFrame,
    lcia_reg_by_year,
    rps_df,
    impact_parent_map,
    pop_series: pd.Series,
    pop_ref: pd.Series | None,
    pr_pop: pd.Series | None,
    pr_gdp: pd.Series | None,
    pr_to_mrio: pd.Series | None,
    region_label_override: str | None,
    use_original_domain: bool,
) -> pd.DataFrame:
    """Compute AR L1 output for one (year, reference year) pair."""
    cache = _shared_ar_cache(state.ar_l1_cache_by_ssp_scenario)
    if year < ref_year:
        state.empty_ref_years.setdefault(ref_year, []).append(year)
        region_label = (
            str(region_label_override)
            if region_label_override is not None
            else resolve_l1_region_label(
                l1_method=l1_method,
                fu_code=context.fu_code,
            )
        )
        return _nan_like_ar_l1(
            lcia_reg,
            year,
            region_label=region_label,
            index_cache=state.output_index_level_cache,
        )

    if "cap" in l1_method:
        # AR(Ecap) uses ref year LCIA+population directly for each target year.
        lcia_reg_ref = load_reference_lcia_reg_for_domain(
            context=context,
            state=state,
            ref_year=ref_year,
            lcia_method=lcia_method,
            lcia_kind=lcia_kind,
            use_original_domain=bool(use_original_domain),
        )
        return compute_l1_method(
            l1_method=l1_method,
            fu_code=context.fu_code,
            year=year,
            population=pop_series,
            population_by_year=state.pop_series_by_ssp_scenario[ssp_scenario],
            population_ref=pop_ref,
            pr_pop=pr_pop,
            pr_gdp=pr_gdp,
            pr_to_mrio=pr_to_mrio,
            lcia_reg=lcia_reg_ref,
            lcia_reg_by_year=lcia_reg_by_year,
            rps_df=rps_df,
            impact_parent_map=impact_parent_map,
            available_years=context.historical_years,
            reference_year=ref_year,
            source_key=context.source,
            group_version_reg=context.group_version_reg,
            l1_reg_aggreg=context.l1_reg_aggreg,
            region_label_override=region_label_override,
            index_cache=state.output_index_level_cache,
        )

    def _compute_ref_baseline() -> pd.DataFrame:
        # Non cap AR L1 is projected from a cached baseline computed at ref_year.
        lcia_reg_ref = load_reference_lcia_reg_for_domain(
            context=context,
            state=state,
            ref_year=ref_year,
            lcia_method=lcia_method,
            lcia_kind=lcia_kind,
            use_original_domain=bool(use_original_domain),
        )
        return compute_l1_method(
            l1_method=l1_method,
            fu_code=context.fu_code,
            year=ref_year,
            population=pop_series,
            population_by_year=state.pop_series_by_ssp_scenario[ssp_scenario],
            population_ref=pop_ref,
            pr_pop=pr_pop,
            pr_gdp=pr_gdp,
            pr_to_mrio=pr_to_mrio,
            lcia_reg=lcia_reg_ref,
            lcia_reg_by_year=lcia_reg_by_year,
            rps_df=rps_df,
            impact_parent_map=impact_parent_map,
            available_years=context.historical_years,
            reference_year=ref_year,
            source_key=context.source,
            group_version_reg=context.group_version_reg,
            l1_reg_aggreg=context.l1_reg_aggreg,
            region_label_override=region_label_override,
        )

    return _project_cached_baseline_for_year(
        cache=cache,
        cache_key=cache_key,
        year=year,
        compute_baseline=_compute_ref_baseline,
        force_recompute_at_ref=(year == ref_year),
    )

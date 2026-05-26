"""Year/scenario/projection planning for setup orchestration."""

from dataclasses import dataclass
from dataclasses import replace

import pandas as pd

from ....data.load_pop_gdp import _resolve_ssp_scenarios
from ...projection.config.config import (
    required_projection_years,
    resolve_projection_context,
)
from ...projection.config.types import ProjectionContext
from pyaesa.asocc.orchestration.setup.pipeline.builders import _validate_bundle_for_selection
from pyaesa.asocc.orchestration.setup.loading.loading import (
    _resolve_reference_years,
    _resolve_years,
    _resolve_years_iso3,
    _validate_pop_gdp_year_coverage,
)
from pyaesa.asocc.orchestration.setup.request.types import (
    PrepareContextRequest,
    _AggregationBundle,
    _SelectionBundle,
    _YearBundle,
)


@dataclass(frozen=True)
class _YearPlan:
    """Resolved year/scenario/projection planning payload."""

    year_bundle: _YearBundle
    reference_years: list[int] | None
    requested_years: list[int]
    ssp_scenario_options_requested: list[str | None]
    projection_context: ProjectionContext
    compute_years: list[int]


def _resolve_year_plan(
    *,
    request: PrepareContextRequest,
    source: str,
    source_is_iso3: bool,
    aggregation: _AggregationBundle,
    selection: _SelectionBundle,
    lcia_methods: list[str] | None,
    fu_code_norm: str,
    use_original_l1_post_domain: bool,
    wb_df: pd.DataFrame,
    ssp_df: pd.DataFrame,
    wb_df_raw: pd.DataFrame,
    ssp_df_raw: pd.DataFrame,
    years_override: list[int] | None = None,
) -> _YearPlan:
    """Resolve studied/historical years, scenarios, references and projection config."""
    requested_years_input = request.years
    # Keep public studied years separate from any internal projection expansion
    # needed to compute those requested years.
    compute_years_input = years_override if years_override is not None else requested_years_input
    if source_is_iso3:
        requested_year_bundle = _resolve_years_iso3(
            years=requested_years_input,
            wb_df=wb_df_raw,
            ssp_df=ssp_df_raw,
        )
        year_bundle = (
            requested_year_bundle
            if years_override is None
            else _resolve_years_iso3(
                years=compute_years_input,
                wb_df=wb_df_raw,
                ssp_df=ssp_df_raw,
            )
        )
    else:
        requested_year_bundle = _resolve_years(
            years=requested_years_input,
            source=source,
            agg_version=request.agg_version,
            agg_reg=aggregation.apply_agg_reg,
            agg_sec=aggregation.apply_agg_sec,
            historical_year_cap=request.historical_year_cap,
        )
        year_bundle = (
            requested_year_bundle
            if years_override is None
            else _resolve_years(
                years=compute_years_input,
                source=source,
                agg_version=request.agg_version,
                agg_reg=aggregation.apply_agg_reg,
                agg_sec=aggregation.apply_agg_sec,
                historical_year_cap=request.historical_year_cap,
            )
        )
        _validate_pop_gdp_year_coverage(
            years=year_bundle.resolved_years,
            wb_df=wb_df,
            ssp_df=ssp_df,
        )
    _validate_bundle_for_selection(
        source=source,
        agg_version=request.agg_version,
        agg_reg=aggregation.apply_agg_reg,
        agg_sec=aggregation.apply_agg_sec,
        selection=selection,
        lcia_methods=lcia_methods,
        historical_years=year_bundle.historical_years,
        fu_code=fu_code_norm,
        use_original_l1_post_domain=use_original_l1_post_domain,
    )
    reference_years = _resolve_reference_years(
        reference_years=request.reference_years,
        historical_years=year_bundle.historical_years,
        source=source,
        agg_version=request.agg_version,
        agg_reg=aggregation.apply_agg_reg,
        agg_sec=aggregation.apply_agg_sec,
    )
    requested_years = list(requested_year_bundle.resolved_years)
    ssp_scenario_options_requested = _resolve_ssp_scenarios(
        resolved_years=year_bundle.resolved_years,
        wb_df=wb_df,
        ssp_df=ssp_df,
        ssp_scenario=request.ssp_scenario,
    )
    projection_context = resolve_projection_context(
        source=source,
        fu_code=fu_code_norm,
        resolved_years=year_bundle.resolved_years,
        historical_years=year_bundle.historical_years,
        selected_l2_one_step=selection.selected_l2_one_step,
        combined=selection.combined,
        projection_mode=request.projection_mode,
        reg_window=request.reg_window,
        l2_reuse_years=request.l2_reuse_years,
    )
    return _YearPlan(
        year_bundle=year_bundle,
        reference_years=reference_years,
        requested_years=requested_years,
        ssp_scenario_options_requested=ssp_scenario_options_requested,
        projection_context=projection_context,
        compute_years=list(year_bundle.resolved_years),
    )


def _expand_year_plan_for_projection(
    *,
    plan: _YearPlan,
    request: PrepareContextRequest,
    source: str,
    source_is_iso3: bool,
    aggregation: _AggregationBundle,
    selection: _SelectionBundle,
    lcia_methods: list[str] | None,
    fu_code_norm: str,
    use_original_l1_post_domain: bool,
    wb_df: pd.DataFrame,
    ssp_df: pd.DataFrame,
    wb_df_raw: pd.DataFrame,
    ssp_df_raw: pd.DataFrame,
) -> tuple[_YearPlan, list[int]]:
    """Expand plan years to include any years required by projection internals."""
    required_years = required_projection_years(projection_context=plan.projection_context)
    initial_years = sorted(set(int(y) for y in plan.compute_years))
    expanded_years = sorted(set(initial_years) | set(required_years))
    if expanded_years == initial_years:
        return plan, []
    added_years = sorted(set(expanded_years) - set(initial_years))
    expanded = _resolve_year_plan(
        request=request,
        source=source,
        source_is_iso3=source_is_iso3,
        aggregation=aggregation,
        selection=selection,
        lcia_methods=lcia_methods,
        fu_code_norm=fu_code_norm,
        use_original_l1_post_domain=use_original_l1_post_domain,
        wb_df=wb_df,
        ssp_df=ssp_df,
        wb_df_raw=wb_df_raw,
        ssp_df_raw=ssp_df_raw,
        years_override=expanded_years,
    )
    return replace(plan, compute_years=list(expanded.year_bundle.resolved_years)), added_years

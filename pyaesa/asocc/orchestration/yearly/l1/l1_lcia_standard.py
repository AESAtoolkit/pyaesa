"""Standard (non AR) LCIA compute for L1 orchestration."""

from ....methods.compute_l1 import compute_l1_method
from ....methods.registry.registry import REGISTRY
from .l1_population_inputs import _resolve_l1_population_inputs
from .l1_pr_hr_setup import _prepare_pr_hr_standard_inputs
from .l1_store import _store_l1_frame
from .l1_types import _L1RunContext, _L1StorePayload, _LciaMethodInputs


def _compute_l1_standard_lcia_method(
    *,
    run: _L1RunContext,
    l1_method: str,
    lcia_inputs: _LciaMethodInputs,
    region_label_override: str | None,
) -> None:
    """Compute one non AR L1 method for one LCIA payload."""
    family = REGISTRY.method_family(l1_method, level="L1")
    use_original_domain = run.context.use_original_l1_post_domain
    pop_series, pop_by_year = _resolve_l1_population_inputs(
        run=run,
        use_original_domain=bool(use_original_domain),
    )
    available_lcia_years = (
        sorted(int(y) for y in lcia_inputs.lcia_reg_by_year.keys())
        if isinstance(lcia_inputs.lcia_reg_by_year, dict)
        else run.context.historical_years
    )
    pr_hr_parent_cum_cache = None
    pr_hr_fallback_callback = None
    if family == "PR_HR":
        pr_hr_inputs = _prepare_pr_hr_standard_inputs(
            run=run,
            lcia_inputs=lcia_inputs,
            population_by_year=pop_by_year,
            use_original_domain=bool(use_original_domain),
        )
        pop_by_year = pr_hr_inputs.population_by_year
        pr_hr_parent_cum_cache = pr_hr_inputs.parent_cum_cache
        pr_hr_fallback_callback = pr_hr_inputs.fallback_callback
    result = compute_l1_method(
        l1_method=l1_method,
        fu_code=run.context.fu_code,
        year=run.year,
        population=pop_series,
        population_by_year=pop_by_year,
        population_ref=None,
        pr_pop=run.pr_pop,
        pr_gdp=run.pr_gdp,
        pr_to_mrio=run.pr_to_mrio,
        lcia_reg=lcia_inputs.lcia_reg,
        lcia_reg_by_year=lcia_inputs.lcia_reg_by_year,
        rps_df=lcia_inputs.rps_df,
        impact_parent_map=lcia_inputs.impact_parent_map,
        available_years=available_lcia_years,
        reference_year=(run.context.reference_years[0] if run.context.reference_years else None),
        impact_year=lcia_inputs.impact_year,
        pr_hr_parent_cum_cache=pr_hr_parent_cum_cache,
        pr_hr_fallback_callback=pr_hr_fallback_callback,
        source_key=run.context.source,
        agg_version_reg=run.context.agg_version_reg,
        l1_reg_aggreg=run.context.l1_reg_aggreg,
        region_label_override=region_label_override,
    )
    _store_l1_frame(
        run=run,
        payload=_L1StorePayload(
            resolved_name=lcia_inputs.resolved_name,
            lcia_method=lcia_inputs.lcia_method,
            frame=result,
            year_key=f"{lcia_inputs.resolved_name}_{lcia_inputs.lcia_method}",
        ),
    )

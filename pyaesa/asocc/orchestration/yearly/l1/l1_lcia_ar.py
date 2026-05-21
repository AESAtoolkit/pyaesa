"""AR family LCIA compute for L1 orchestration."""

from ....methods.equations.ar_result_indexing import _add_reference_level
from ....methods.registry.registry import REGISTRY
from ....methods.run_ar import _compute_ar_l1_result
from .l1_lcia_reference_policy import (
    emit_ar_no_reference_years_notice,
    resolve_reference_years_for_ar,
)
from .l1_population_inputs import _load_ar_reference_population
from .l1_store import _store_l1_frame
from .l1_types import _L1RunContext, _L1StorePayload, _LciaMethodInputs


def _compute_l1_ar_lcia_method(
    *,
    run: _L1RunContext,
    l1_method: str,
    lcia_inputs: _LciaMethodInputs,
    region_label_override: str | None,
) -> None:
    """Compute AR L1 results across reference years for one LCIA method."""
    needs_pop_ref = REGISTRY.method_family(l1_method, level="L1") == "AR_ECAP"
    use_original_domain = run.context.use_original_l1_post_domain and REGISTRY.method_family(
        l1_method, level="L1"
    ) in {"PR_HR", "AR_ECAP"}
    pop_series = (
        run.pop_series_original
        if use_original_domain and run.pop_series_original is not None
        else run.pop_series
    )

    refs = resolve_reference_years_for_ar(
        run=run,
        lcia_inputs=lcia_inputs,
        use_original_domain=use_original_domain,
    )
    if not refs:
        emit_ar_no_reference_years_notice(
            run=run,
            l1_method=l1_method,
            lcia_inputs=lcia_inputs,
        )
        return

    for ref_year in refs:
        cache_key = (lcia_inputs.resolved_name, lcia_inputs.lcia_method, ref_year)
        invariant_cache_key = (
            int(run.year),
            str(l1_method),
            str(lcia_inputs.resolved_name),
            str(lcia_inputs.lcia_method),
            str(lcia_inputs.lcia_kind),
            int(ref_year),
            str(region_label_override) if region_label_override is not None else None,
            bool(use_original_domain),
            str(run.context.group_version_reg),
            str(run.context.l1_reg_aggreg),
        )
        if (
            REGISTRY.method_family(l1_method, level="L1") == "AR_E"
            and run.ssp_scenario is not None
            and invariant_cache_key in run.state.l1_invariant_cache
        ):
            cached_frame, cached_value = run.state.l1_invariant_cache[invariant_cache_key]
            _store_l1_frame(
                run=run,
                payload=_L1StorePayload(
                    resolved_name=lcia_inputs.resolved_name,
                    lcia_method=lcia_inputs.lcia_method,
                    frame=cached_frame,
                    year_key=(
                        f"{lcia_inputs.resolved_name}_{lcia_inputs.lcia_method}_ref_{ref_year}"
                    ),
                    value_frame=cached_value,
                ),
            )
            continue
        pop_ref = None
        if needs_pop_ref and ref_year is not None:
            pop_ref = _load_ar_reference_population(
                run=run,
                ref_year=ref_year,
                use_original_domain=bool(use_original_domain),
            )
        result = _compute_ar_l1_result(
            context=run.context,
            state=run.state,
            ssp_scenario=run.ssp_scenario,
            cache_key=cache_key,
            l1_method=l1_method,
            year=run.year,
            ref_year=ref_year,
            lcia_method=lcia_inputs.lcia_method,
            lcia_kind=lcia_inputs.lcia_kind,
            lcia_reg=lcia_inputs.lcia_reg,
            lcia_reg_by_year=lcia_inputs.lcia_reg_by_year,
            rps_df=lcia_inputs.rps_df,
            impact_parent_map=lcia_inputs.impact_parent_map,
            pop_series=pop_series,
            pop_ref=pop_ref,
            pr_pop=run.pr_pop,
            pr_gdp=run.pr_gdp,
            pr_to_mrio=run.pr_to_mrio,
            region_label_override=region_label_override,
            use_original_domain=bool(use_original_domain),
        )
        result_out = _add_reference_level(
            result,
            ref_year,
            index_cache=run.state.output_index_level_cache,
        )
        if REGISTRY.method_family(l1_method, level="L1") == "AR_E":
            run.state.l1_invariant_cache[invariant_cache_key] = (
                result_out,
                result,
            )
        _store_l1_frame(
            run=run,
            payload=_L1StorePayload(
                resolved_name=lcia_inputs.resolved_name,
                lcia_method=lcia_inputs.lcia_method,
                frame=result_out,
                year_key=(f"{lcia_inputs.resolved_name}_{lcia_inputs.lcia_method}_ref_{ref_year}"),
                value_frame=result,
            ),
        )

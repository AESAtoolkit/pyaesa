"""Context/state builders for setup orchestration."""

from ....io.metadata import RunContext, RunState
from pyaesa.asocc.orchestration.setup.validation.history_checks import (
    _validate_history_since_baseline,
)
from pyaesa.asocc.orchestration.setup.validation.lcia_checks import (
    _validate_grouped_lcia_ready,
    _validate_original_lcia_ready,
)
from pyaesa.asocc.orchestration.setup.request.types import (
    _GroupingBundle,
    _SelectionBundle,
    _YearBundle,
)


def _build_context(
    *,
    common: dict,
    requested_years: list[int],
    persisted_years: list[int],
    compute_years: list[int],
    year_bundle: _YearBundle,
    reference_years: list[int] | None,
    ssp_scenario_options: list[str | None],
    metadata_completed_years: list[int] | None = None,
    metadata_prior_outputs: list[str] | None = None,
) -> RunContext:
    """Build a RunContext object from resolved setup inputs."""
    grouping: _GroupingBundle = common["grouping"]
    selection: _SelectionBundle = common["selection"]
    return RunContext(
        project_name=common["project_name"],
        source=common["source"],
        fu_code=common["fu_code"],
        group_version=common["group_version"],
        group_version_reg=grouping.group_version_reg,
        group_reg=grouping.apply_group_reg,
        group_sec=grouping.apply_group_sec,
        lcia_method=common["lcia_method"],
        years_input=common["years"],
        reference_years_input=common["reference_years"],
        ssp_scenario=common["ssp_scenario"],
        is_exio=common["is_exio"],
        l1_lcia_kind=common["l1_lcia_kind"],
        lcia_methods=common["lcia_methods"],
        selected_l1=selection.selected_l1,
        combined=selection.combined,
        selected_l2_one_step=selection.selected_l2_one_step,
        required_indices=selection.required_indices,
        filters=common["filters"],
        studied_indices_tag=common["studied_indices_tag"],
        proj_base=common["proj_base"],
        logger=common["logger"],
        requested_years=requested_years,
        resolved_years=year_bundle.resolved_years,
        persisted_years=persisted_years,
        compute_years=compute_years,
        historical_years=year_bundle.historical_years,
        reference_years=reference_years,
        ssp_scenario_options=ssp_scenario_options,
        run_signature=common["run_signature"],
        needs_lcia=selection.needs_lcia_flag,
        repo_root=common["repo_root"],
        wb_df=common["wb_df"],
        ssp_df=common["ssp_df"],
        wb_df_raw=common["wb_df_raw"],
        ssp_df_raw=common["ssp_df_raw"],
        selected_methods=selection.selected_methods,
        l1_kinds_needed=selection.l1_kinds_needed,
        l1_only_no_mrio=selection.l1_only_no_mrio,
        l1_reg_aggreg=common["l1_reg_aggreg"],
        use_original_l1_post_domain=common["use_original_l1_post_domain"],
        variant_tag=common["variant_tag"],
        aggreg_indices=common["aggreg_indices"],
        output_format=common["output_format"],
        intermediate_outputs=common["intermediate_outputs"],
        output_source_label=common["output_source_label"],
        projection_context=common["projection_context"],
        ssp_scenario_options_by_year=common["ssp_scenario_options_by_year"],
        metadata_completed_years=metadata_completed_years,
        metadata_prior_outputs=metadata_prior_outputs,
    )


def _build_context_common(
    *,
    project_name: str,
    source: str,
    fu_code: str,
    group_version: str | None,
    grouping: _GroupingBundle,
    lcia_method: str | list[str] | None,
    years: int | list[int] | range | None,
    reference_years: int | list[int] | range | None,
    ssp_scenario: str | list[str] | None,
    is_exio: bool,
    l1_lcia_kind: str,
    lcia_methods: list[str] | None,
    selection: _SelectionBundle,
    filters: dict[str, list[str] | None],
    studied_indices_tag: str,
    proj_base,
    logger,
    run_signature: dict,
    repo_root,
    wb_df,
    ssp_df,
    wb_df_raw,
    ssp_df_raw,
    l1_reg_aggreg: str,
    use_original_l1_post_domain: bool,
    variant_tag: str | None,
    aggreg_indices: bool,
    output_format: str,
    intermediate_outputs: bool,
    output_source_label: str | None = None,
    projection_context=None,
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None = None,
) -> dict:
    """Build immutable common context payload used by context builders."""
    return {
        "project_name": project_name,
        "source": source,
        "fu_code": fu_code,
        "group_version": group_version,
        "grouping": grouping,
        "lcia_method": lcia_method,
        "years": years,
        "reference_years": reference_years,
        "ssp_scenario": ssp_scenario,
        "is_exio": is_exio,
        "l1_lcia_kind": l1_lcia_kind,
        "lcia_methods": lcia_methods,
        "selection": selection,
        "filters": filters,
        "studied_indices_tag": studied_indices_tag,
        "proj_base": proj_base,
        "logger": logger,
        "run_signature": run_signature,
        "repo_root": repo_root,
        "wb_df": wb_df,
        "ssp_df": ssp_df,
        "wb_df_raw": wb_df_raw,
        "ssp_df_raw": ssp_df_raw,
        "l1_reg_aggreg": l1_reg_aggreg,
        "use_original_l1_post_domain": use_original_l1_post_domain,
        "variant_tag": variant_tag,
        "aggreg_indices": aggreg_indices,
        "output_format": output_format,
        "intermediate_outputs": intermediate_outputs,
        "output_source_label": output_source_label,
        "projection_context": projection_context,
        "ssp_scenario_options_by_year": ssp_scenario_options_by_year,
    }


def _build_initial_state(*, ssp_scenario_options: list[str | None]) -> RunState:
    """Build initialized mutable run state."""
    state = RunState()
    state.l1_results_by_ssp_scenario = {ssp_scenario: {} for ssp_scenario in ssp_scenario_options}
    state.l2_results_by_ssp_scenario = {ssp_scenario: {} for ssp_scenario in ssp_scenario_options}
    state.pre_weighting_written_by_ssp_scenario = {
        ssp_scenario: set() for ssp_scenario in ssp_scenario_options
    }
    state.pop_series_by_ssp_scenario = {ssp_scenario: {} for ssp_scenario in ssp_scenario_options}
    state.pr_post_pop_series_by_ssp_scenario = {
        ssp_scenario: {} for ssp_scenario in ssp_scenario_options
    }
    state.gdp_series_by_ssp_scenario = {ssp_scenario: {} for ssp_scenario in ssp_scenario_options}
    state.ar_l1_cache_by_ssp_scenario = {ssp_scenario: {} for ssp_scenario in ssp_scenario_options}
    state.ar_l2_cache_by_ssp_scenario = {ssp_scenario: {} for ssp_scenario in ssp_scenario_options}
    state.preweight_cache_by_ssp_scenario = {
        ssp_scenario: {} for ssp_scenario in ssp_scenario_options
    }
    state.lcia_timeseries = {}
    state.lcia_timeseries_original = {}
    return state


def _validate_bundle_for_selection(
    *,
    source: str,
    group_version: str | None,
    group_reg: bool,
    group_sec: bool,
    selection: _SelectionBundle,
    lcia_methods: list[str] | None,
    historical_years: list[int],
    fu_code: str,
    use_original_l1_post_domain: bool,
) -> None:
    """Run historical/LCIA validations for one resolved year bundle."""
    if selection.needs_lcia_flag:
        _validate_grouped_lcia_ready(
            source=source,
            years=historical_years,
            lcia_methods=lcia_methods,
            group_version=group_version,
            group_reg=group_reg,
            group_sec=group_sec,
        )
    _validate_history_since_baseline(
        source=source,
        group_version=group_version,
        group_reg=group_reg,
        group_sec=group_sec,
        historical_years=historical_years,
        selection=selection,
        fu_code=fu_code,
    )
    if use_original_l1_post_domain:
        _validate_original_lcia_ready(
            source=source,
            years=historical_years,
            lcia_methods=lcia_methods,
        )

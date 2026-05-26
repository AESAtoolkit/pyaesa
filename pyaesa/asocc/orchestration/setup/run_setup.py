"""Allocation setup orchestration."""

import shutil
from dataclasses import replace

from pyaesa.asocc.orchestration.setup.loading.loading import (
    _load_source_tables,
    _validate_region_filter_labels,
    _validate_sector_filter_labels,
)
from pyaesa.asocc.orchestration.setup.pipeline.builders import (
    _build_context,
    _build_context_common,
    _build_initial_state,
)
from pyaesa.asocc.orchestration.setup.request.scenarios import (
    build_scenario_plan_by_year,
    scenario_state_options_from_plan,
)
from pyaesa.asocc.orchestration.setup.request.selection import (
    _build_selection_bundle,
    _l1_methods_in_scope,
    _prune_lcia_methods_without_lcia_input,
    _resolve_filters,
    _resolve_aggregation,
    _resolve_selection_bundle,
    _restrict_selection_for_iso3_mode,
    _uses_l1_post_original_domain,
    _validate_td_grouped_output,
)
from pyaesa.asocc.orchestration.setup.request.types import PrepareContextRequest, _YearBundle
from pyaesa.asocc.orchestration.setup.request.year_plan import (
    _expand_year_plan_for_projection,
    _resolve_year_plan,
)
from pyaesa.asocc.orchestration.setup.reuse.completed_run_policy import apply_completed_run_policy
from pyaesa.asocc.orchestration.setup.validation.lcia_checks import _validate_lcia_requirements
from pyaesa.asocc.runtime.scope.branch_resolution import outputs_project_root
from pyaesa.workspace_initialisation.workspace import (
    get_default_repo_root,
)
from pyaesa.shared.runtime.reporting.values import format_values

from ...data.enacting_metric_units import load_enacting_metric_units_from_metadata
from ...data.source_schema import is_exio_source, is_iso3_source
from ...io.logging import close_loggers_for_scope, get_logger
from ...io.metadata import RunContext, RunState
from ...methods.lcia_inputs import (
    initialize_pr_hr_timeseries,
    normalize_lcia_methods,
)
from ...methods.registry.registry import normalize_fu_code
from ...runtime.methods.labels import l1_l2_method_label
from ...runtime.paths.deterministic import (
    _get_allocate_refresh_scope_root,
    _get_allocate_summary_log_path,
)
from ...runtime.paths.family_roots import (
    effective_agg_flags_for_source,
    effective_agg_version_for_source,
)
from ...runtime.request.scope import AsoccScope
from ...runtime.selection.normalize import normalize_l1_reg_mode_required
from ..projection.config.config import resolve_projection_context


def _prune_selection_for_append_scope(
    *,
    selection,
    append_scope,
    fu_code: str,
    l1_lcia_kind: str,
):
    """Return selection limited to missing method selectors for selector append."""
    if append_scope is None or append_scope.years or append_scope.selected_methods is None:
        return selection
    selected_labels = append_scope.selected_methods
    selected_l2_in_l1 = set(selected_labels.get("l2_in_l1", []))
    combined = [
        pair
        for pair in selection.combined
        if l1_l2_method_label(l1_method=pair[1], l2_method=pair[0]) in selected_l2_in_l1
    ]
    selected_l1 = sorted({*selected_labels.get("l1", []), *(l1_name for _, l1_name in combined)})
    selected_one_step_labels = selected_labels.get("l2_vs_global", [])
    selected_l2_one_step = [
        name for name in selection.selected_l2_one_step if name in selected_one_step_labels
    ]
    return _build_selection_bundle(
        fu_code=fu_code,
        selected_l1=selected_l1,
        combined=combined,
        selected_l2_one_step=selected_l2_one_step,
        l1_lcia_kind=l1_lcia_kind,
    )


def _prune_optional_axis_for_append(
    *,
    current,
    append_scope,
    field_name: str,
):
    """Return selector missing axis values for append, otherwise the current axis."""
    if append_scope is None or append_scope.years:
        return current
    value = getattr(append_scope, field_name)
    return value if value is not None else current


def _prepare_context(
    *,
    request: PrepareContextRequest,
) -> tuple[RunContext, RunState, bool]:
    """Prepare allocation context/state and run signature."""
    # 1) Normalize FU/aggregation/method selection from API request.
    if request.source is None:
        raise ValueError(
            "source cannot be None. Provide an explicit source "
            "(for example 'oecd_v2025', 'exiobase_3102_pxp', or 'iso3')."
        )
    source = str(request.source).strip()
    if not source:
        raise ValueError(
            "source cannot be empty. Provide an explicit source "
            "(for example 'oecd_v2025', 'exiobase_3102_pxp', or 'iso3')."
        )
    output_source = request.output_source
    l1_reg_aggreg_mode = normalize_l1_reg_mode_required(request.l1_reg_aggreg)
    source_is_iso3 = is_iso3_source(source)
    fu_code_norm = normalize_fu_code(request.fu_code)
    _validate_td_grouped_output(
        fu_code=fu_code_norm,
        group_indices=request.group_indices,
    )

    aggregation = _resolve_aggregation(
        agg_reg=request.agg_reg,
        agg_sec=request.agg_sec,
        agg_version=request.agg_version,
    )
    published_agg_version = effective_agg_version_for_source(
        source=output_source,
        agg_version=request.agg_version,
    )
    published_agg_reg, published_agg_sec = effective_agg_flags_for_source(
        source=output_source,
        agg_reg=aggregation.apply_agg_reg,
        agg_sec=aggregation.apply_agg_sec,
    )
    if source_is_iso3:
        if aggregation.apply_agg_reg or aggregation.apply_agg_sec or request.agg_version:
            raise ValueError(
                "source='iso3' does not support aggregation controls (agg_reg/agg_sec/agg_version)."
            )
        if request.lcia_method is not None:
            raise ValueError("source='iso3' does not support lcia_method.")
        if request.reference_years is not None:
            raise ValueError("source='iso3' does not support reference_years.")
    is_exio = is_exio_source(source)
    l1_lcia_kind = "PBA" if fu_code_norm == "L1.b" else "CBA_FD"
    lcia_methods = normalize_lcia_methods(request.lcia_method)

    selection = _resolve_selection_bundle(
        fu_code=fu_code_norm,
        l_1=request.l_1,
        l_2_combined_with_l_1=request.l_2_combined_with_l_1,
        l_2_one_step=request.l_2_one_step,
        l1_lcia_kind=l1_lcia_kind,
    )
    if source_is_iso3:
        selection = _restrict_selection_for_iso3_mode(
            fu_code=fu_code_norm,
            selection=selection,
        )
    selection, dropped_for_missing_lcia = _prune_lcia_methods_without_lcia_input(
        fu_code=fu_code_norm,
        lcia_methods=lcia_methods,
        selection=selection,
    )
    if not selection.selected_l1 and not selection.combined and not selection.selected_l2_one_step:
        raise ValueError(
            "No allocation methods remain for this run after applying source "
            f"constraints (source={source})."
        )

    # 2) Validate filter contract and LCIA prerequisites for selected methods.
    filters, studied_indices_tag = _resolve_filters(
        required_indices=selection.required_indices,
        r_p=request.r_p,
        s_p=request.s_p,
        r_c=request.r_c,
        r_f=request.r_f,
    )
    _validate_lcia_requirements(
        source=source,
        is_exio=is_exio,
        needs_lcia_flag=selection.needs_lcia_flag,
        lcia_methods=lcia_methods,
    )
    use_original_l1_post_domain = _uses_l1_post_original_domain(
        selection=selection,
        aggregation=aggregation,
        l1_reg_aggreg=l1_reg_aggreg_mode,
    )
    wb_df, ssp_df, wb_df_raw, ssp_df_raw = _load_source_tables(
        source=source,
    )
    _validate_region_filter_labels(
        source=source,
        agg_version=request.agg_version,
        agg_reg=aggregation.apply_agg_reg,
        filters=filters,
        wb_df=wb_df,
        ssp_df=ssp_df,
    )
    _validate_sector_filter_labels(
        source=source,
        agg_version=request.agg_version,
        filters=filters,
    )
    proj_base = outputs_project_root(project_name=request.project_name)
    scope_root = _get_allocate_refresh_scope_root(
        proj_base=proj_base,
        source=output_source,
        agg_version=published_agg_version,
    )
    log_path = _get_allocate_summary_log_path(
        proj_base,
        source=output_source,
        agg_version=published_agg_version,
    )
    if request.refresh:
        close_loggers_for_scope(scope_root)
        if scope_root.exists():
            shutil.rmtree(scope_root)
        proj_base = outputs_project_root(project_name=request.project_name)
        scope_root = _get_allocate_refresh_scope_root(
            proj_base=proj_base,
            source=output_source,
            agg_version=published_agg_version,
        )
        log_path = _get_allocate_summary_log_path(
            proj_base,
            source=output_source,
            agg_version=published_agg_version,
        )
    # 3) Resolve years/history/reference year against available MRIO metadata.
    year_plan = _resolve_year_plan(
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
    )
    year_bundle = year_plan.year_bundle
    persisted_years = list(year_plan.requested_years)
    reference_years = year_plan.reference_years
    requested_years = year_plan.requested_years
    ssp_scenario_options_requested = year_plan.ssp_scenario_options_requested
    projection_context = year_plan.projection_context
    scope = AsoccScope(
        base_allocate_args={
            "source": output_source,
            "agg_version": published_agg_version,
            "agg_reg": published_agg_reg,
            "agg_sec": published_agg_sec,
            "fu_code": fu_code_norm,
            "lcia_method": lcia_methods,
            "ssp_scenario": request.ssp_scenario,
            "reference_years": request.reference_years,
            "l1_reg_aggreg": l1_reg_aggreg_mode,
            "group_indices": request.group_indices,
            "projection_mode": projection_context.mode,
            "reg_window": projection_context.reg_window,
            "l2_reuse_years": list(projection_context.l2_reuse_years)
            if projection_context.l2_reuse_years
            else None,
        },
        selected_l1=list(selection.selected_l1),
        combined=list(selection.combined),
        selected_l2_one_step=list(selection.selected_l2_one_step),
        selected_methods=selection.selected_methods,
        filters=filters,
        studied_indices_tag=studied_indices_tag,
    )
    run_signature = scope.compute_signature(
        years=requested_years,
        output_format=request.output_format,
        intermediate_outputs=request.intermediate_outputs,
        historical_year_cap=request.historical_year_cap,
        variant_tag=request.variant_tag,
    )
    ssp_scenario_options_by_year = build_scenario_plan_by_year(
        years=requested_years,
        wb_df=wb_df,
        ssp_scenarios=ssp_scenario_options_requested,
    )
    ssp_scenario_options = scenario_state_options_from_plan(
        scenario_plan_by_year=ssp_scenario_options_by_year
    )

    # 4) Completed run policy: exact scoped metadata can skip the branch.
    (
        year_bundle,
        reference_years,
        ssp_scenario_options,
        is_complete,
        metadata_completed_years,
        metadata_prior_outputs,
        append_compute_scope,
    ) = apply_completed_run_policy(
        refresh=request.refresh,
        proj_base=proj_base,
        run_signature=run_signature,
        year_bundle=year_bundle,
        reference_years=reference_years,
        requested_years=requested_years,
        ssp_scenario_options=ssp_scenario_options,
        output_source=output_source,
    )
    selection = _prune_selection_for_append_scope(
        selection=selection,
        append_scope=append_compute_scope,
        fu_code=fu_code_norm,
        l1_lcia_kind=l1_lcia_kind,
    )
    lcia_methods = _prune_optional_axis_for_append(
        current=lcia_methods,
        append_scope=append_compute_scope,
        field_name="lcia_methods",
    )
    reference_years = _prune_optional_axis_for_append(
        current=reference_years,
        append_scope=append_compute_scope,
        field_name="reference_years_input",
    )
    compute_l2_reuse_years = _prune_optional_axis_for_append(
        current=request.l2_reuse_years,
        append_scope=append_compute_scope,
        field_name="l2_reuse_years",
    )
    ssp_scenario_options_requested = _prune_optional_axis_for_append(
        current=ssp_scenario_options_requested,
        append_scope=append_compute_scope,
        field_name="ssp_scenario_input",
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
        l2_reuse_years=compute_l2_reuse_years,
    )
    compute_year_plan, added_years = _expand_year_plan_for_projection(
        plan=replace(
            year_plan,
            year_bundle=year_bundle,
            reference_years=reference_years,
            projection_context=projection_context,
            compute_years=list(year_bundle.resolved_years),
        ),
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
    )
    compute_years = list(compute_year_plan.compute_years)
    ssp_scenario_options_by_year = build_scenario_plan_by_year(
        years=compute_years,
        wb_df=wb_df,
        ssp_scenarios=ssp_scenario_options_requested,
    )
    ssp_scenario_options = scenario_state_options_from_plan(
        scenario_plan_by_year=ssp_scenario_options_by_year
    )
    logger = get_logger(log_path)
    repo_root = get_default_repo_root()
    context_common = _build_context_common(
        project_name=request.project_name,
        source=source,
        fu_code=fu_code_norm,
        agg_version=request.agg_version,
        aggregation=aggregation,
        lcia_method=lcia_methods,
        years=request.years,
        reference_years=request.reference_years,
        ssp_scenario=request.ssp_scenario,
        is_exio=is_exio,
        l1_lcia_kind=l1_lcia_kind,
        lcia_methods=lcia_methods,
        selection=selection,
        filters=filters,
        studied_indices_tag=studied_indices_tag,
        proj_base=proj_base,
        logger=logger,
        run_signature=run_signature,
        repo_root=repo_root,
        wb_df=wb_df,
        ssp_df=ssp_df,
        wb_df_raw=wb_df_raw,
        ssp_df_raw=ssp_df_raw,
        l1_reg_aggreg=l1_reg_aggreg_mode,
        use_original_l1_post_domain=use_original_l1_post_domain,
        variant_tag=request.variant_tag,
        group_indices=request.group_indices,
        output_format=request.output_format,
        intermediate_outputs=request.intermediate_outputs,
        output_source_label=request.output_source_label,
        projection_context=projection_context,
        ssp_scenario_options_by_year=ssp_scenario_options_by_year,
    )

    def _build_context_for_bundle(
        *,
        current_year_bundle: _YearBundle,
        current_reference_years: list[int] | None,
        current_ssp_scenario_options: list[str | None],
    ) -> RunContext:
        return _build_context(
            common=context_common,
            requested_years=requested_years,
            persisted_years=persisted_years,
            compute_years=compute_years,
            year_bundle=current_year_bundle,
            reference_years=current_reference_years,
            ssp_scenario_options=current_ssp_scenario_options,
            metadata_completed_years=metadata_completed_years,
            metadata_prior_outputs=metadata_prior_outputs,
        )

    if is_complete:
        context = _build_context_for_bundle(
            current_year_bundle=year_bundle,
            current_reference_years=reference_years,
            current_ssp_scenario_options=ssp_scenario_options,
        )
        return context, RunState(), True

    state = _build_initial_state(ssp_scenario_options=ssp_scenario_options)
    if not source_is_iso3 and not selection.l1_only_no_mrio:
        (
            state.mrio_default_monetary_unit,
            state.mrio_units,
            lcia_units_by_method,
        ) = load_enacting_metric_units_from_metadata(
            source=source,
            matrix_version=request.agg_version,
            years=year_bundle.historical_years,
        )
        for lcia_method_name in lcia_methods or []:
            unit_map = lcia_units_by_method.get(lcia_method_name)
            if unit_map is None:
                raise ValueError(
                    "Missing LCIA unit metadata for method "
                    f"'{lcia_method_name}'. Re-run process_mrio for this domain."
                )
            state.lcia_units[lcia_method_name] = unit_map
    startup_notices: list[tuple[str, str]] = []
    if dropped_for_missing_lcia:
        drop_msg = (
            "Dropped LCIA-dependent methods because lcia_method was not "
            f"provided: {format_values(dropped_for_missing_lcia)}."
        )
        logger.warning(drop_msg)
        startup_notices.append(("WARNING", drop_msg))
    setattr(state, "startup_notices", startup_notices)
    # 5) Prime historical LCIA responsibility windows before yearly processing.
    initialize_pr_hr_timeseries(
        source=source,
        state=state,
        lcia_methods=lcia_methods,
        selected_l1=sorted(_l1_methods_in_scope(selection)),
        store="grouped",
    )
    if use_original_l1_post_domain:
        initialize_pr_hr_timeseries(
            source=source,
            state=state,
            lcia_methods=lcia_methods,
            selected_l1=sorted(_l1_methods_in_scope(selection)),
            store="original",
        )
    context = _build_context_for_bundle(
        current_year_bundle=year_bundle,
        current_reference_years=reference_years,
        current_ssp_scenario_options=ssp_scenario_options,
    )
    return context, state, False

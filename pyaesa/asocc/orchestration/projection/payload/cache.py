"""Projected/reused MRIO payload builders for future UT years."""

from typing import cast

import pandas as pd

from ....methods.registry.registry import REGISTRY
from ...method_scope import _unique_l2_methods_in_scope
from ...yearly.shared.year_inputs import _MrioPayload, build_l2_compute_inputs
from ..config.types import ProjectionContext
from .basis import regression_basis
from .payload_builders_l2 import project_fd_payload, project_gva_payload
from .builders_x_to_rc import project_x_to_rc_payload


def get_projected_payload(
    *,
    context,
    state,
    year: int,
    ssp_scenario: str | None,
    gdp_series: pd.Series,
) -> _MrioPayload:
    """Return projected MRIO payload for one future year/scenario."""
    cache_key = (int(year), ssp_scenario)
    cached = state.projection_payload_cache.get(cache_key)
    if isinstance(cached, _MrioPayload):
        return cached
    projection_context = cast(ProjectionContext, context.projection_context)

    routed_methods = [
        name
        for name in _unique_l2_methods_in_scope(
            selected_l2_one_step=context.selected_l2_one_step,
            combined=context.combined,
        )
        if projection_context.route_for_l2_method(name) == "regression"
    ]
    routed_one_step_methods = [
        name
        for name in context.selected_l2_one_step
        if projection_context.route_for_l2_method(name) == "regression"
    ]
    families = {
        REGISTRY.method_family(name, level="L2", fu_code=context.fu_code) for name in routed_methods
    }
    one_step_families = {
        REGISTRY.method_family(name, level="L2", fu_code=context.fu_code)
        for name in routed_one_step_methods
    }
    needs_fd = "UT_FD" in families
    needs_td = "UT_TD" in families
    needs_gva = "UT_GVA" in families
    needs_global_fd_total = ("UT_FD" in one_step_families) or ("UT_TD" in one_step_families)
    needs_global_gva_total = "UT_GVA" in one_step_families
    fit_start, fit_end = projection_context.reg_window or (
        projection_context.max_historical_year,
        projection_context.max_historical_year,
    )
    historical_years = list(range(int(fit_start), int(fit_end) + 1))
    basis = regression_basis(
        context=context,
        state=state,
        historical_years=historical_years,
        fit_end=int(fit_end),
        needs_fd_total=bool(needs_fd or needs_td),
        needs_fd_detail=bool(needs_fd),
        needs_gva=bool(needs_gva),
        needs_x_to_rc=bool(needs_td),
    )

    projected_fd_rf = pd.Series(dtype=float)
    projected_fd_rp_sp_rf = pd.DataFrame()
    projected_fd_rp_sp = pd.Series(dtype=float)
    projected_fd_rf_sp = pd.Series(dtype=float)
    if needs_fd or needs_td:
        (
            projected_fd_rf,
            projected_fd_rp_sp_rf,
            projected_fd_rp_sp,
            projected_fd_rf_sp,
        ) = project_fd_payload(
            context=context,
            state=state,
            basis=basis,
            historical_years=historical_years,
            target_year=int(year),
            future_years=list(projection_context.future_years),
            gdp_target=gdp_series,
            needs_ut_fd=bool(needs_fd),
            needs_global_fd_total=bool(needs_global_fd_total),
        )

    projected_gva_rp = pd.Series(dtype=float)
    projected_gva_rp_sp = pd.Series(dtype=float)
    if needs_gva:
        projected_gva_rp, projected_gva_rp_sp = project_gva_payload(
            context=context,
            state=state,
            basis=basis,
            historical_years=historical_years,
            target_year=int(year),
            future_years=list(projection_context.future_years),
            gdp_target=gdp_series,
            needs_global_gva_total=bool(needs_global_gva_total),
        )

    projected_x_to_rc = pd.DataFrame()
    if needs_td:
        projected_x_to_rc = project_x_to_rc_payload(
            context=context,
            state=state,
            basis=basis,
            historical_years=historical_years,
            target_year=int(year),
            future_years=list(projection_context.future_years),
            gdp_target=gdp_series,
        )

    # Reassemble a payload object with the yearly MRIO layout expected by
    # downstream L2 compute entrypoints.
    # This payload includes only regression routed UT MRIO enacting metrics; routes handled
    # through historical reuse load their own inputs in L2 compute.
    enacting_metric_l1 = {
        "fd_rf": projected_fd_rf,
        "gva_rp": projected_gva_rp,
    }
    enacting_metric_l2 = {
        "fd_rp_sp_rf": projected_fd_rp_sp_rf,
        "fd_rp_sp": projected_fd_rp_sp,
        "fd_rf_sp": projected_fd_rf_sp,
        "gva_rp_sp": projected_gva_rp_sp,
    }
    utility = {
        "x_to_rc": projected_x_to_rc,
    }
    payload = _MrioPayload(
        enacting_metric_l1=enacting_metric_l1,
        enacting_metric_l2=enacting_metric_l2,
        utility=utility,
        l2_inputs=build_l2_compute_inputs(
            enacting_metric_l1=enacting_metric_l1,
            enacting_metric_l2=enacting_metric_l2,
            utility=utility,
        ),
    )
    state.projection_payload_cache[cache_key] = payload
    return payload

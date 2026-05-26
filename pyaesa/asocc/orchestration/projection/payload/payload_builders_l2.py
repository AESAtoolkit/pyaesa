"""FU specific regression payload builders for projected UT inputs."""

from dataclasses import dataclass

import pandas as pd

from ..regression.filtering import (
    selected_values_for_level,
    selected_values_for_levels,
)
from ..regression.level_ols_gdp_scaled_projection import project_series_from_gdp
from ..regression.share_logit_time_projection import project_share_from_time_logit
from .basis import (
    RegressionBasis,
    coerce_index_like,
    safe_share,
)
from .common import (
    frame_payload as _frame,
)
from .common import (
    reorder_series_levels_payload as _reorder_levels,
)
from .common import (
    series_payload as _series,
)
from .common import (
    stack_series_payload as _stack_series,
)


@dataclass(frozen=True)
class _FDProjectionPlan:
    """Equation plan for UT(FD) MRIO enacting metrics by functional unit."""

    needs_sector_share_by_rf: bool
    needs_producer_share_by_rf_sp: bool


def _fd_projection_plan_for_fu(*, fu_code: str) -> _FDProjectionPlan:
    """Return minimal UT(FD) projection equations required for a FU."""
    by_fu = {
        "L2.a.a": _FDProjectionPlan(
            needs_sector_share_by_rf=True,
            needs_producer_share_by_rf_sp=True,
        ),
        "L2.b.a": _FDProjectionPlan(
            needs_sector_share_by_rf=True,
            needs_producer_share_by_rf_sp=True,
        ),
        "L2.c.a": _FDProjectionPlan(
            needs_sector_share_by_rf=True,
            needs_producer_share_by_rf_sp=False,
        ),
    }
    return by_fu[str(fu_code)]


def project_fd_payload(
    *,
    context,
    state,
    basis: RegressionBasis,
    historical_years: list[int],
    target_year: int,
    future_years: list[int],
    gdp_target: pd.Series,
    needs_ut_fd: bool,
    needs_global_fd_total: bool,
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Project FD based MRIO enacting metrics used by UT(FD)/UT(TD) routes."""
    setattr(state, "runtime_proj_base", context.proj_base)
    setattr(
        state,
        "runtime_output_source",
        context.output_source,
    )
    setattr(state, "runtime_agg_version", context.agg_version)
    setattr(state, "runtime_agg_reg", context.agg_reg)
    setattr(state, "runtime_group_indices", context.group_indices)
    setattr(state, "runtime_l1_reg_aggreg", context.l1_reg_aggreg)
    # 1) Regress total final demand by receiving region.
    fd_rf_history = {
        int(year): _series(
            basis.payload_by_year[int(year)].enacting_metric_l1["fd_rf"],
            "enacting_metric_l1.fd_rf",
        )
        for year in historical_years
    }
    projected_fd_rf = project_series_from_gdp(
        source=context.source,
        fu_code=context.fu_code,
        l2_method="UT(FD)",
        target_object="fd_rf",
        target_year=int(target_year),
        historical_years=historical_years,
        history_by_year=fd_rf_history,
        gdp_by_year=basis.gdp_by_year,
        gdp_target=gdp_target,
        selected_domains=(
            None
            if needs_global_fd_total
            else selected_values_for_level(
                filters=context.filters,
                level="r_f",
            )
        ),
        state=state,
    )

    if not needs_ut_fd:
        return (
            projected_fd_rf,
            pd.DataFrame(),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    plan = _fd_projection_plan_for_fu(fu_code=context.fu_code)

    # 2) Regress sector shares inside each receiving region.
    fd_rf_sp_template = basis.base_payload.enacting_metric_l2["fd_rf_sp"]
    share_sp_rf_history = {
        int(year): safe_share(
            _series(
                basis.payload_by_year[int(year)].enacting_metric_l2["fd_rf_sp"],
                "enacting_metric_l2.fd_rf_sp",
            ),
            _series(
                basis.payload_by_year[int(year)].enacting_metric_l1["fd_rf"],
                "enacting_metric_l1.fd_rf",
            ),
            level="r_f",
        )
        for year in historical_years
    }
    share_sp_rf = project_share_from_time_logit(
        source=context.source,
        fu_code=context.fu_code,
        l2_method="UT(FD)",
        target_object="fd_share_sp_given_rf",
        historical_years=historical_years,
        share_by_year=share_sp_rf_history,
        target_year=int(target_year),
        future_years=future_years,
        container_levels="r_f",
        category_level="s_p",
        selected_categories=selected_values_for_level(
            filters=context.filters,
            level="s_p",
        ),
        selected_containers=selected_values_for_levels(
            filters=context.filters,
            levels=["r_f"],
        ),
        state=state,
    )
    share_sp_rf = coerce_index_like(share_sp_rf, template=fd_rf_sp_template.index).fillna(0.0)
    rf_factors = projected_fd_rf.reindex(share_sp_rf.index.get_level_values("r_f")).to_numpy(
        dtype=float
    )
    projected_fd_rf_sp = (share_sp_rf * rf_factors).fillna(0.0)

    if not plan.needs_producer_share_by_rf_sp:
        return (
            projected_fd_rf,
            pd.DataFrame(),
            pd.Series(dtype=float),
            projected_fd_rf_sp,
        )

    # 3) Regress producer region shares inside each (receiving region, sector).
    fd_rp_sp_rf_template = basis.base_payload.enacting_metric_l2["fd_rp_sp_rf"]
    share_rp_history = {
        int(year): safe_share(
            _stack_series(
                _frame(
                    basis.payload_by_year[int(year)].enacting_metric_l2["fd_rp_sp_rf"],
                    "enacting_metric_l2.fd_rp_sp_rf",
                ),
                "enacting_metric_l2.fd_rp_sp_rf_stacked",
                names=["r_p", "s_p", "r_f"],
            ),
            _series(
                basis.payload_by_year[int(year)].enacting_metric_l2["fd_rf_sp"],
                "enacting_metric_l2.fd_rf_sp",
            ),
            level=["r_f", "s_p"],
        )
        .pipe(
            _reorder_levels,
            order=["r_f", "s_p", "r_p"],
            label="fd_share_rp_given_rf_sp",
        )
        .sort_index()
        for year in historical_years
    }
    share_rp = project_share_from_time_logit(
        source=context.source,
        fu_code=context.fu_code,
        l2_method="UT(FD)",
        target_object="fd_share_rp_given_rf_sp",
        historical_years=historical_years,
        share_by_year=share_rp_history,
        target_year=int(target_year),
        future_years=future_years,
        container_levels=["r_f", "s_p"],
        category_level="r_p",
        selected_categories=selected_values_for_level(
            filters=context.filters,
            level="r_p",
        ),
        selected_containers=selected_values_for_levels(
            filters=context.filters,
            levels=["r_f", "s_p"],
        ),
        state=state,
    )
    share_rp = coerce_index_like(
        share_rp,
        template=share_rp_history[int(historical_years[-1])].index,
    )
    share_rp = share_rp.fillna(0.0)
    rf_sp_index = pd.MultiIndex.from_arrays(
        [
            share_rp.index.get_level_values("r_f"),
            share_rp.index.get_level_values("s_p"),
        ],
        names=["r_f", "s_p"],
    )
    base_values = projected_fd_rf_sp.reindex(rf_sp_index).to_numpy(dtype=float)
    fd_rp_sp_rf_long = share_rp.to_numpy(dtype=float) * base_values
    fd_rp_sp_rf_series = pd.Series(
        fd_rp_sp_rf_long,
        index=pd.MultiIndex.from_arrays(
            [
                share_rp.index.get_level_values("r_p"),
                share_rp.index.get_level_values("s_p"),
                share_rp.index.get_level_values("r_f"),
            ],
            names=["r_p", "s_p", "r_f"],
        ),
    )
    projected_fd_rp_sp_rf = (
        fd_rp_sp_rf_series.unstack("r_f")
        .reindex(index=fd_rp_sp_rf_template.index, columns=fd_rp_sp_rf_template.columns)
        .fillna(0.0)
    )
    projected_fd_rp_sp = projected_fd_rp_sp_rf.sum(axis=1, min_count=1).fillna(0.0)
    return (
        projected_fd_rf,
        projected_fd_rp_sp_rf,
        projected_fd_rp_sp,
        projected_fd_rf_sp,
    )


def project_gva_payload(
    *,
    context,
    state,
    basis: RegressionBasis,
    historical_years: list[int],
    target_year: int,
    future_years: list[int],
    gdp_target: pd.Series,
    needs_global_gva_total: bool,
) -> tuple[pd.Series, pd.Series]:
    """Project GVA based MRIO enacting metrics used by UT(GVA)."""
    setattr(state, "runtime_proj_base", context.proj_base)
    setattr(
        state,
        "runtime_output_source",
        context.output_source,
    )
    setattr(state, "runtime_agg_version", context.agg_version)
    setattr(state, "runtime_agg_reg", context.agg_reg)
    setattr(state, "runtime_group_indices", context.group_indices)
    setattr(state, "runtime_l1_reg_aggreg", context.l1_reg_aggreg)
    # 1) Regress total GVA by producer region.
    gva_rp_history = {
        int(year): _series(
            basis.payload_by_year[int(year)].enacting_metric_l1["gva_rp"],
            "enacting_metric_l1.gva_rp",
        )
        for year in historical_years
    }
    projected_gva_rp = project_series_from_gdp(
        source=context.source,
        fu_code=context.fu_code,
        l2_method="UT(GVA)",
        target_object="gva_rp",
        target_year=int(target_year),
        historical_years=historical_years,
        history_by_year=gva_rp_history,
        gdp_by_year=basis.gdp_by_year,
        gdp_target=gdp_target,
        selected_domains=(
            None
            if needs_global_gva_total
            else selected_values_for_level(
                filters=context.filters,
                level="r_p",
            )
        ),
        state=state,
    )
    # 2) Regress sector shares inside each producer region.
    gva_rp_sp_template = basis.base_payload.enacting_metric_l2["gva_rp_sp"]
    share_sp_rp_history = {
        int(year): safe_share(
            _series(
                basis.payload_by_year[int(year)].enacting_metric_l2["gva_rp_sp"],
                "enacting_metric_l2.gva_rp_sp",
            ),
            _series(
                basis.payload_by_year[int(year)].enacting_metric_l1["gva_rp"],
                "enacting_metric_l1.gva_rp",
            ),
            level="r_p",
        )
        for year in historical_years
    }
    share_sp_rp = project_share_from_time_logit(
        source=context.source,
        fu_code=context.fu_code,
        l2_method="UT(GVA)",
        target_object="gva_share_sp_given_rp",
        historical_years=historical_years,
        share_by_year=share_sp_rp_history,
        target_year=int(target_year),
        future_years=future_years,
        container_levels="r_p",
        category_level="s_p",
        selected_categories=selected_values_for_level(
            filters=context.filters,
            level="s_p",
        ),
        selected_containers=selected_values_for_levels(
            filters=context.filters,
            levels=["r_p"],
        ),
        state=state,
    )
    share_sp_rp = coerce_index_like(share_sp_rp, template=gva_rp_sp_template.index).fillna(0.0)
    rp_factors = projected_gva_rp.reindex(share_sp_rp.index.get_level_values("r_p")).to_numpy(
        dtype=float
    )
    projected_gva_rp_sp = (share_sp_rp * rp_factors).fillna(0.0)
    return projected_gva_rp, projected_gva_rp_sp

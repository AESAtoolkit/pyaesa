"""x_to_rc payload projection for UT(TD) routes."""

from collections.abc import Hashable
from typing import Literal

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
    stack_series_payload as _stack_series,
)


def _summed_series(
    frame: pd.DataFrame,
    *,
    axis: Literal[0, 1, "index", "columns"],
    group_level: str | None = None,
) -> pd.Series:
    """Return one numeric Series after deterministic sum and optional grouping."""
    series = pd.Series(frame.sum(axis=axis, min_count=1), copy=False)
    if group_level is None:
        return series
    return pd.Series(series.groupby(level=group_level).sum(min_count=1), copy=False)


def _grouped_sp_transpose(frame: pd.DataFrame) -> pd.DataFrame:
    """Return frame grouped on `s_p` and transposed to `(r_c, s_p)` shape."""
    grouped = pd.DataFrame(frame.groupby(level="s_p").sum(min_count=1), copy=False)
    return pd.DataFrame(grouped.T, copy=False)


def _to_frame_label(value: object) -> Hashable:
    """Return one stable column label for one single column output frame."""
    return value if isinstance(value, Hashable) else str(value)


def project_x_to_rc_payload(
    *,
    context,
    state,
    basis: RegressionBasis,
    historical_years: list[int],
    target_year: int,
    future_years: list[int],
    gdp_target: pd.Series,
) -> pd.DataFrame:
    """Project x to rc utility MRIO enacting metric according to FU specific equations."""
    setattr(state, "runtime_proj_base", context.proj_base)
    setattr(
        state,
        "runtime_output_source",
        context.output_source,
    )
    setattr(state, "runtime_group_version", context.group_version)
    setattr(state, "runtime_group_reg", context.group_reg)
    setattr(state, "runtime_aggreg_indices", context.aggreg_indices)
    setattr(state, "runtime_l1_reg_aggreg", context.l1_reg_aggreg)
    x_template = basis.base_payload.utility["x_to_rc"]
    if context.fu_code == "L2.a.b":
        # Route L2.a.b: project x by producer region totals and sector shares.
        x_total_rp_history = {
            int(year): _summed_series(
                _frame(
                    basis.payload_by_year[int(year)].utility["x_to_rc"],
                    "utility.x_to_rc",
                ),
                axis=1,
                group_level="r_p",
            )
            for year in historical_years
        }
        projected_x_total_rp = project_series_from_gdp(
            source=context.source,
            fu_code=context.fu_code,
            l2_method="UT(TD)",
            target_object="x_total_out_rp",
            target_year=int(target_year),
            historical_years=historical_years,
            history_by_year=x_total_rp_history,
            gdp_by_year=basis.gdp_by_year,
            gdp_target=gdp_target,
            selected_domains=selected_values_for_level(
                filters=context.filters,
                level="r_p",
            ),
            state=state,
        )
        x_rp_sp_template = _summed_series(x_template, axis=1)
        share_sp_rp_history = {
            int(year): safe_share(
                _summed_series(
                    _frame(
                        basis.payload_by_year[int(year)].utility["x_to_rc"],
                        "utility.x_to_rc",
                    ),
                    axis=1,
                ),
                _summed_series(
                    _frame(
                        basis.payload_by_year[int(year)].utility["x_to_rc"],
                        "utility.x_to_rc",
                    ),
                    axis=1,
                    group_level="r_p",
                ),
                level="r_p",
            )
            for year in historical_years
        }
        share_sp_rp = project_share_from_time_logit(
            source=context.source,
            fu_code=context.fu_code,
            l2_method="UT(TD)",
            target_object="x_share_sp_given_rp",
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
        share_sp_rp = coerce_index_like(share_sp_rp, template=x_rp_sp_template.index).fillna(0.0)
        rp_factors = projected_x_total_rp.reindex(
            share_sp_rp.index.get_level_values("r_p")
        ).to_numpy(dtype=float)
        projected_x_rp_sp = pd.Series(
            share_sp_rp.to_numpy(dtype=float) * rp_factors,
            index=share_sp_rp.index,
        )
        out_col = _to_frame_label(x_template.columns[0]) if len(x_template.columns) else "__total__"
        return projected_x_rp_sp.to_frame(out_col).fillna(0.0)

    # Routes L2.b.b / L2.c.b: project x by consumer region totals.
    x_total_rc_history = {
        int(year): _summed_series(
            _frame(
                basis.payload_by_year[int(year)].utility["x_to_rc"],
                "utility.x_to_rc",
            ),
            axis=0,
        )
        for year in historical_years
    }
    projected_x_total_rc = project_series_from_gdp(
        source=context.source,
        fu_code=context.fu_code,
        l2_method="UT(TD)",
        target_object="x_total_rc",
        target_year=int(target_year),
        historical_years=historical_years,
        history_by_year=x_total_rc_history,
        gdp_by_year=basis.gdp_by_year,
        gdp_target=gdp_target,
        selected_domains=selected_values_for_level(
            filters=context.filters,
            level="r_c",
        ),
        state=state,
    )
    x_rc_sp_template = _stack_series(
        _grouped_sp_transpose(x_template),
        "x_template_grouped_sp_stacked",
        names=["r_c", "s_p"],
    )
    share_sp_rc_history = {
        int(year): safe_share(
            _stack_series(
                _grouped_sp_transpose(
                    _frame(
                        basis.payload_by_year[int(year)].utility["x_to_rc"],
                        "utility.x_to_rc",
                    )
                ),
                "utility.x_to_rc_grouped_sp_stacked",
                names=["r_c", "s_p"],
            ),
            _summed_series(
                _frame(
                    basis.payload_by_year[int(year)].utility["x_to_rc"],
                    "utility.x_to_rc",
                ),
                axis=0,
            ),
            level="r_c",
        )
        for year in historical_years
    }
    share_sp_rc = project_share_from_time_logit(
        source=context.source,
        fu_code=context.fu_code,
        l2_method="UT(TD)",
        target_object="x_share_sp_given_rc",
        historical_years=historical_years,
        share_by_year=share_sp_rc_history,
        target_year=int(target_year),
        future_years=future_years,
        container_levels="r_c",
        category_level="s_p",
        selected_categories=selected_values_for_level(
            filters=context.filters,
            level="s_p",
        ),
        selected_containers=selected_values_for_levels(
            filters=context.filters,
            levels=["r_c"],
        ),
        state=state,
    )
    share_sp_rc = coerce_index_like(share_sp_rc, template=x_rc_sp_template.index).fillna(0.0)
    rc_factors = projected_x_total_rc.reindex(share_sp_rc.index.get_level_values("r_c")).to_numpy(
        dtype=float
    )
    projected_x_rc_sp = pd.Series(
        share_sp_rc.to_numpy(dtype=float) * rc_factors,
        index=share_sp_rc.index,
    )
    if context.fu_code == "L2.c.b":
        # L2.c.b aggregates over producer region after x_rc_sp projection.
        rc_wide = projected_x_rc_sp.unstack("r_c").fillna(0.0)
        rp_anchor = (
            x_template.index.get_level_values("r_p")[0] if len(x_template.index) else "__rp__"
        )
        rc_wide.index = pd.MultiIndex.from_arrays(
            [[rp_anchor] * len(rc_wide.index), rc_wide.index],
            names=["r_p", "s_p"],
        )
        return rc_wide
    share_rp_template = (
        _stack_series(
            x_template,
            "x_template_stacked",
            names=["r_p", "s_p", "r_c"],
        )
        .pipe(
            _reorder_levels,
            order=["r_c", "s_p", "r_p"],
            label="x_template_stacked",
        )
        .sort_index()
    )
    # L2.b.b additionally projects producer region shares within (r_c, s_p).
    share_rp_history = {
        int(year): safe_share(
            _stack_series(
                _frame(
                    basis.payload_by_year[int(year)].utility["x_to_rc"],
                    "utility.x_to_rc",
                ),
                "utility.x_to_rc_stacked",
                names=["r_p", "s_p", "r_c"],
            )
            .pipe(
                _reorder_levels,
                order=["r_c", "s_p", "r_p"],
                label="utility.x_to_rc_stacked",
            )
            .sort_index(),
            _stack_series(
                _grouped_sp_transpose(
                    _frame(
                        basis.payload_by_year[int(year)].utility["x_to_rc"],
                        "utility.x_to_rc",
                    )
                ),
                "utility.x_to_rc_grouped_sp_stacked",
                names=["r_c", "s_p"],
            ),
            level=["r_c", "s_p"],
        )
        for year in historical_years
    }
    share_rp = project_share_from_time_logit(
        source=context.source,
        fu_code=context.fu_code,
        l2_method="UT(TD)",
        target_object="x_share_rp_given_rc_sp",
        historical_years=historical_years,
        share_by_year=share_rp_history,
        target_year=int(target_year),
        future_years=future_years,
        container_levels=["r_c", "s_p"],
        category_level="r_p",
        selected_categories=selected_values_for_level(
            filters=context.filters,
            level="r_p",
        ),
        selected_containers=selected_values_for_levels(
            filters=context.filters,
            levels=["r_c", "s_p"],
        ),
        state=state,
    )
    share_rp = coerce_index_like(share_rp, template=share_rp_template.index).fillna(0.0)
    rc_sp_index = pd.MultiIndex.from_arrays(
        [
            share_rp.index.get_level_values("r_c"),
            share_rp.index.get_level_values("s_p"),
        ],
        names=["r_c", "s_p"],
    )
    rc_sp_values = projected_x_rc_sp.reindex(rc_sp_index).to_numpy(dtype=float)
    projected_long = pd.Series(
        share_rp.to_numpy(dtype=float) * rc_sp_values,
        index=pd.MultiIndex.from_arrays(
            [
                share_rp.index.get_level_values("r_p"),
                share_rp.index.get_level_values("s_p"),
                share_rp.index.get_level_values("r_c"),
            ],
            names=["r_p", "s_p", "r_c"],
        ),
    )
    return (
        projected_long.unstack("r_c")
        .reindex(index=x_template.index, columns=x_template.columns)
        .fillna(0.0)
    )

"""Compute L2 allocation methods."""

import pandas as pd
from typing import cast

from .equations.ar_e import compute_ar_e_l2
from .equations.ut_fd import compute_ut_fd_l2
from .equations.ut_fda import compute_ut_fda_l2
from .equations.ut_gva import compute_ut_gva_l2
from .equations.ut_gvaa import compute_ut_gvaa_l2
from .equations.ut_td import compute_ut_td_l2
from .registry.registry import REGISTRY


def apply_l1_weights_to_preweighted(
    *,
    l2_method: str,
    fu_code: str,
    year: int,
    pre_weighted: pd.DataFrame,
    l1_weights: pd.Series,
    weight_axis: str | None = None,
    required_indices: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Apply L1 weights to a pre weighted L2 matrix using method contract."""
    # Weight axis is part of registry contract (r_f/r_p/r_u depending on L2 method/FU).
    pre_axis = (
        weight_axis
        if weight_axis is not None
        else REGISTRY.l2_weight_axis_for_method(l2_method, fu_code)
    )
    pre_series = pre_weighted.iloc[:, 0]

    weighted = pre_series.mul(l1_weights, level=pre_axis)
    required = (
        required_indices
        if required_indices is not None
        else REGISTRY.required_indices(
            l2_method,
            fu_code,
            l1_weighting=True,
        )
    )
    weighted = weighted.groupby(level=list(required), sort=False).sum(min_count=1)
    return weighted.to_frame(int(year))


def compute_l2_method(
    *,
    l2_method: str,
    fu_code: str,
    year: int,
    l1_weights: pd.Series | None,
    fd_rf: pd.Series,
    gva_rp: pd.Series,
    fd_rp_sp_rf: pd.DataFrame,
    fd_rp_sp: pd.Series,
    fd_rf_sp: pd.Series,
    gva_rp_sp: pd.Series,
    x_to_rc: pd.DataFrame,
    kappa: pd.DataFrame,
    omega_reg: pd.DataFrame,
    lcia: dict | None,
    reference_year: int | None,
    pre_weighting: bool = False,
) -> pd.DataFrame:
    """Compute one L2 method output for one studied year.

    Args:
        l2_method: L2 method name.
        fu_code: Functional unit code.
        year: Year of computation.
        l1_weights: L1 weights, if applicable.
        fd_rf: Final demand by consuming region.
        gva_rp: GVA by producing region.
        fd_rp_sp_rf: Final demand by producer and r_f.
        fd_rp_sp: Final demand by producer.
        fd_rf_sp: Final demand by (r_f, s_p).
        gva_rp_sp: GVA by producer.
        x_to_rc: Total demand absorption by r_c.
        kappa: Overlap adjusted distribution matrix.
        omega_reg: Upstream regional GVA shares.
        lcia: LCIA metrics dictionary.
        reference_year: Reference year for AR methods.
        pre_weighting: Whether to return pre weighting outputs.

    Returns:
        Wide DataFrame with one studied year column.
    """
    family = REGISTRY.method_family(l2_method, level="L2", fu_code=fu_code)
    # Family dispatch keeps L2 equation routing data driven through registry metadata.

    if family == "UT_FD":
        return compute_ut_fd_l2(
            fu_code=fu_code,
            year=year,
            l1_weights=l1_weights,
            fd_rf=fd_rf,
            fd_rp_sp_rf=fd_rp_sp_rf,
            fd_rp_sp=fd_rp_sp,
            fd_rf_sp=fd_rf_sp,
            pre_weighting=pre_weighting,
        )

    if family == "UT_FDA":
        return compute_ut_fda_l2(
            fu_code=fu_code,
            year=year,
            l1_weights=l1_weights,
            fd_rf=fd_rf,
            x_to_rc=x_to_rc,
            kappa=kappa,
            pre_weighting=pre_weighting,
        )

    if family == "UT_GVAA":
        return compute_ut_gvaa_l2(
            fu_code=fu_code,
            year=year,
            l1_weights=l1_weights,
            gva_rp=gva_rp,
            x_to_rc=x_to_rc,
            omega_reg=omega_reg,
            pre_weighting=pre_weighting,
        )

    if family == "UT_TD":
        return compute_ut_td_l2(
            fu_code=fu_code,
            year=year,
            fd_rf=fd_rf,
            x_to_rc=x_to_rc,
        )

    if family == "UT_GVA":
        return compute_ut_gva_l2(
            year=year,
            l1_weights=l1_weights,
            gva_rp=gva_rp,
            gva_rp_sp=gva_rp_sp,
            pre_weighting=pre_weighting,
        )

    return compute_ar_e_l2(
        l2_method=l2_method,
        fu_code=fu_code,
        l1_weights=l1_weights,
        lcia=cast(dict, lcia),
        reference_year=reference_year,
        pre_weighting=pre_weighting,
    )

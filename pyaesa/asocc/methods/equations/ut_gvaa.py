"""UT(GVAa) allocation method (L2)."""

from typing import cast

import numpy as np
import pandas as pd

from .ut_support import _get_x_vec, _stack_to_year


def _divide_frame_by_series(frame: pd.DataFrame, denom: pd.Series) -> pd.DataFrame:
    """Return one numeric frame divided by a region Series."""
    return pd.DataFrame(frame.div(denom, axis=1), copy=False)


def _collapse_weighted_result(
    *,
    weights: pd.DataFrame,
    l1_weights: pd.Series,
    year: int,
) -> pd.DataFrame:
    """Collapse one weighting frame to a one year result column."""
    aligned = pd.Series(l1_weights.reindex(weights.columns), copy=False)
    values = weights.to_numpy(dtype="float64", copy=False) @ aligned.to_numpy(
        dtype="float64",
        copy=False,
    )
    out = pd.Series(values, index=weights.index, dtype="float64")
    return out.to_frame(int(year))


def _weighted_omega_by_rc(
    *,
    omega_reg: pd.DataFrame,
    x_to_rc: pd.DataFrame,
) -> pd.DataFrame:
    """Return weighted omega values by (r_c, r_p, s_p) with r_u columns."""
    if x_to_rc.empty:
        return pd.DataFrame()
    omega_t = omega_reg.T

    # Align omega on producer index
    omega_aligned = omega_t.reindex(x_to_rc.index)
    x_vals = x_to_rc.to_numpy(dtype="float64", copy=False)
    omega_vals = omega_aligned.to_numpy(dtype="float64", copy=False)
    weighted = x_vals[:, :, None] * omega_vals[:, None, :]
    weighted_2d = weighted.transpose(1, 0, 2).reshape(
        x_vals.shape[1] * x_vals.shape[0],
        omega_vals.shape[1],
    )

    rc_vals = np.repeat(x_to_rc.columns.to_numpy(), x_vals.shape[0])
    rp_vals = np.tile(x_to_rc.index.get_level_values(0).to_numpy(), x_vals.shape[1])
    sp_vals = np.tile(x_to_rc.index.get_level_values(1).to_numpy(), x_vals.shape[1])
    out_index = pd.MultiIndex.from_arrays(
        [rc_vals, rp_vals, sp_vals],
        names=["r_c", "r_p", "s_p"],
    )
    return pd.DataFrame(weighted_2d, index=out_index, columns=omega_aligned.columns)


def compute_ut_gvaa_l2(
    *,
    fu_code: str,
    year: int,
    l1_weights: pd.Series | None,
    gva_rp: pd.Series,
    x_to_rc: pd.DataFrame,
    omega_reg: pd.DataFrame,
    pre_weighting: bool,
) -> pd.DataFrame:
    """Compute UT(GVAa) for L2.

    Args:
        fu_code: Functional unit code.
        year: Year of computation.
        l1_weights: L1 weights series, if applicable.
        gva_rp: GVA by producing region.
        x_to_rc: Total demand absorption by r_c.
        omega_reg: Upstream regional GVA shares.
        pre_weighting: Whether to return pre weighting outputs.

    Returns:
        DataFrame with year as column (or r_u columns for pre weighting).
    """
    x_vec = _get_x_vec(x_to_rc)
    gva_safe = gva_rp.replace(0, pd.NA)
    weighted_by_rc = _weighted_omega_by_rc(omega_reg=omega_reg, x_to_rc=x_to_rc)
    if pre_weighting:
        if fu_code == "L2.a.b":
            weights = omega_reg.mul(x_vec, axis=1).T
            weights = _divide_frame_by_series(weights, gva_safe)
            return _stack_to_year(weights, year, "r_u")
        if fu_code == "L2.b.b":
            weights = _divide_frame_by_series(weighted_by_rc, gva_safe)
            return _stack_to_year(weights, year, "r_u")
        aggregated = pd.DataFrame(
            weighted_by_rc.groupby(level=["r_c", "s_p"]).sum(min_count=1),
            copy=False,
        )
        weights = _divide_frame_by_series(aggregated, gva_safe)
        return _stack_to_year(weights, year, "r_u")

    active_l1_weights = cast(pd.Series, l1_weights)
    if fu_code == "L2.a.b":
        weights = omega_reg.mul(x_vec, axis=1).T
        weights = _divide_frame_by_series(weights, gva_safe)
        return _collapse_weighted_result(
            weights=weights,
            l1_weights=active_l1_weights,
            year=year,
        )
    if fu_code == "L2.b.b":
        weights = _divide_frame_by_series(weighted_by_rc, gva_safe)
        return _collapse_weighted_result(
            weights=weights,
            l1_weights=active_l1_weights,
            year=year,
        )
    aggregated = pd.DataFrame(
        weighted_by_rc.groupby(level=["r_c", "s_p"]).sum(min_count=1),
        copy=False,
    )
    weights = _divide_frame_by_series(aggregated, gva_safe)
    return _collapse_weighted_result(weights=weights, l1_weights=active_l1_weights, year=year)

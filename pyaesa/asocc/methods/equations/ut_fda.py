"""UT(FDa) allocation method (L2)."""

from typing import cast

import pandas as pd

from .ut_support import _safe_divide_frame, _stack_to_year


def _adjust_td_to_fd(
    x_to_rc: pd.DataFrame,
    kappa: pd.DataFrame,
) -> pd.DataFrame:
    """Map TD destination to FD absorbers aggregated over r_c."""
    contrib_by_rc = _adjust_td_to_fd_by_rc(x_to_rc, kappa)
    if contrib_by_rc.empty:
        raise ValueError("No r_c columns available in x_to_rc for UT(FDa).")
    return cast(
        pd.DataFrame,
        contrib_by_rc.groupby(level=["r_p", "s_p"]).sum(min_count=1),
    )


def _adjust_td_to_fd_by_rc(
    x_to_rc: pd.DataFrame,
    kappa: pd.DataFrame,
) -> pd.DataFrame:
    """Map TD destination to FD absorbers retaining r_c."""
    if len(x_to_rc.columns) == 0:
        return pd.DataFrame()
    x_long = x_to_rc.stack(future_stack=True)
    x_long.index = x_long.index.set_names(["r_p", "s_p", "r_c"])
    kappa_idx = kappa.copy()
    kappa_index = cast(pd.MultiIndex, kappa_idx.index)
    kappa_idx.index = kappa_index.reorder_levels(["r_p", "s_p", "r_c"])
    contrib = kappa_idx.mul(x_long, axis=0)
    contrib.index = cast(pd.MultiIndex, contrib.index).reorder_levels(["r_c", "r_p", "s_p"])
    return contrib.sort_index()


def _adjust_td_to_fd_by_rc_sp(
    x_to_rc: pd.DataFrame,
    kappa: pd.DataFrame,
) -> pd.DataFrame:
    """Map TD destination to FD absorbers retaining r_c and s_p."""
    contrib_by_rc = _adjust_td_to_fd_by_rc(x_to_rc, kappa)
    if contrib_by_rc.empty:
        return pd.DataFrame()
    return cast(
        pd.DataFrame,
        contrib_by_rc.groupby(level=["r_c", "s_p"]).sum(min_count=1),
    )


def compute_ut_fda_l2(
    *,
    fu_code: str,
    year: int,
    l1_weights: pd.Series | None,
    fd_rf: pd.Series,
    x_to_rc: pd.DataFrame,
    kappa: pd.DataFrame,
    pre_weighting: bool,
) -> pd.DataFrame:
    """Compute UT(FDa) for L2.

    Args:
        fu_code: Functional unit code.
        year: Year of computation.
        l1_weights: L1 weights series, if applicable.
        fd_rf: Final demand by r_f.
        x_to_rc: Total demand absorption by r_c.
        kappa: Overlap adjusted distribution matrix.
        pre_weighting: Whether to return pre weighting outputs.

    Returns:
        DataFrame with year as column (or r_f columns for pre weighting).
    """
    if pre_weighting:
        if fu_code == "L2.a.b":
            contrib_all = _adjust_td_to_fd(x_to_rc, kappa)
            return _stack_to_year(_safe_divide_frame(contrib_all, fd_rf), year, "r_f")
        if fu_code == "L2.b.b":
            contrib = _adjust_td_to_fd_by_rc(x_to_rc, kappa)
            return _stack_to_year(_safe_divide_frame(contrib, fd_rf), year, "r_f")
        contrib = _adjust_td_to_fd_by_rc_sp(x_to_rc, kappa)
        return _stack_to_year(_safe_divide_frame(contrib, fd_rf), year, "r_f")

    active_l1_weights = cast(pd.Series, l1_weights)
    if fu_code == "L2.a.b":
        contrib_all = _adjust_td_to_fd(x_to_rc, kappa)
        weights = _safe_divide_frame(contrib_all, fd_rf)
        out = weights @ active_l1_weights
        return out.to_frame(int(year))
    if fu_code == "L2.b.b":
        contrib = _adjust_td_to_fd_by_rc(x_to_rc, kappa)
        weights_df = _safe_divide_frame(contrib, fd_rf)
        out = weights_df @ active_l1_weights
        return out.to_frame(int(year))
    contrib = _adjust_td_to_fd_by_rc_sp(x_to_rc, kappa)
    weights_df = _safe_divide_frame(contrib, fd_rf)
    out = weights_df @ active_l1_weights
    return out.to_frame(int(year))

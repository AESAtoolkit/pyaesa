"""UT(FD) allocation method (L2)."""

import pandas as pd

from .ut_support import (
    _safe_divide_frame,
    _safe_divide_series,
    _stack_to_year,
)


def compute_ut_fd_l2(
    *,
    fu_code: str,
    year: int,
    l1_weights: pd.Series | None,
    fd_rf: pd.Series,
    fd_rp_sp_rf: pd.DataFrame,
    fd_rp_sp: pd.Series,
    fd_rf_sp: pd.Series,
    pre_weighting: bool,
) -> pd.DataFrame:
    """Compute UT(FD) for L2.

    Args:
        fu_code: Functional unit code.
        year: Year of computation.
        l1_weights: L1 weights series, if applicable.
        fd_rf: Final demand by r_f.
        fd_rp_sp_rf: FD by producer and r_f.
        fd_rp_sp: FD by producer.
        fd_rf_sp: FD by (r_f, s_p).
        pre_weighting: Whether to return pre weighting outputs.

    Returns:
        DataFrame with year as column (or r_f columns for pre weighting).
    """
    fd_global = fd_rf.sum()
    if pre_weighting:
        if fu_code in {"L2.a.a", "L2.b.a"}:
            weights = _safe_divide_frame(fd_rp_sp_rf, fd_rf)
            return _stack_to_year(weights, year, "r_f")
        weights = _safe_divide_series(fd_rf_sp, fd_rf)
        return weights.to_frame(int(year))

    if fu_code == "L2.a.a":
        if l1_weights is not None:
            weights = _safe_divide_frame(fd_rp_sp_rf, fd_rf)
            out = weights.mul(l1_weights, axis=1).sum(axis=1)
        else:
            out = _safe_divide_series(fd_rp_sp, fd_global)
        return out.to_frame(int(year))

    if fu_code == "L2.b.a":
        if l1_weights is not None:
            weights = _safe_divide_frame(fd_rp_sp_rf, fd_rf)
            out = weights.mul(l1_weights, axis=1)
            return _stack_to_year(out, year, "r_f")
        out = fd_rp_sp_rf / fd_global if fd_global != 0 else fd_rp_sp_rf * 0.0
        return _stack_to_year(out, year, "r_f")

    if l1_weights is not None:
        weights = _safe_divide_series(fd_rf_sp, fd_rf)
        out = weights.mul(l1_weights, level="r_f")
    else:
        out = _safe_divide_series(fd_rf_sp, fd_global)
    return out.to_frame(int(year))

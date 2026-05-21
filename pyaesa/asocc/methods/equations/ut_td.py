"""UT(TD) allocation method (L2)."""

from typing import cast

import pandas as pd

from .ut_support import _get_x_vec, _safe_divide_series, _stack_frame_to_series


def compute_ut_td_l2(
    *,
    fu_code: str,
    year: int,
    fd_rf: pd.Series,
    x_to_rc: pd.DataFrame,
) -> pd.DataFrame:
    """Compute UT(TD) for L2.

    Args:
        fu_code: Functional unit code.
        year: Year of computation.
        fd_rf: Final demand by r_f.
        x_to_rc: Total demand absorption by r_c.

    Returns:
        DataFrame with year as column.
    """
    x_vec = _get_x_vec(x_to_rc)
    fd_global = fd_rf.sum()
    if fu_code == "L2.a.b":
        out = _safe_divide_series(x_vec, fd_global)
    elif fu_code == "L2.b.b":
        stacked = _stack_frame_to_series(x_to_rc)
        out = _safe_divide_series(stacked, fd_global)
    else:
        grouped = cast(pd.DataFrame, x_to_rc.groupby(level="s_p").sum(min_count=1))
        stacked = _stack_frame_to_series(grouped.T)
        out = _safe_divide_series(stacked, fd_global)
    return out.to_frame(int(year))

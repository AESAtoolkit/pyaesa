"""UT(GVA) allocation method (L2)."""

import pandas as pd

from .ut_support import _safe_divide_series


def compute_ut_gva_l2(
    *,
    year: int,
    l1_weights: pd.Series | None,
    gva_rp: pd.Series,
    gva_rp_sp: pd.Series,
    pre_weighting: bool,
) -> pd.DataFrame:
    """Compute UT(GVA) for L2.

    Args:
        year: Year of computation.
        l1_weights: L1 weights series, if applicable.
        gva_rp: GVA by producing region.
        gva_rp_sp: GVA by producer.
        pre_weighting: Whether to return pre weighting outputs.

    Returns:
        DataFrame with year as column.
    """
    gva_global = gva_rp.sum()
    if pre_weighting:
        weights = _safe_divide_series(gva_rp_sp, gva_rp)
        return weights.to_frame(int(year))
    if l1_weights is not None:
        weights = _safe_divide_series(gva_rp_sp, gva_rp)
        out = weights.mul(l1_weights, level="r_p")
    else:
        out = _safe_divide_series(gva_rp_sp, gva_global)
    return out.to_frame(int(year))

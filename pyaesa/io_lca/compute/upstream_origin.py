"""Origin of impacts output ownership for IO-LCA upstream analysis."""

from typing import cast

import pandas as pd

from pyaesa.io_lca.orchestration.io.method_support import origin_long_columns


def finalize_origin_rows(
    *,
    origin_rows: pd.DataFrame,
    selector_axes: tuple[str, ...],
) -> pd.DataFrame:
    """Aggregate additive origin rows into deterministic long output schema.

    Args:
        origin_rows: Concatenated rows produced during upstream stage compute.
        selector_axes: FU selector columns included in output.

    Returns:
        Aggregated origin table with deterministic column order.
    """
    if origin_rows.empty:
        return pd.DataFrame(columns=origin_long_columns(selector_axes))
    group_cols = origin_long_columns(selector_axes)[:-1]
    grouped = origin_rows.groupby(group_cols, dropna=False, as_index=False)[["lca_value"]].sum(
        min_count=1
    )
    out = cast(pd.DataFrame, grouped).sort_values(group_cols).reset_index(drop=True)
    return out.loc[:, origin_long_columns(selector_axes)]

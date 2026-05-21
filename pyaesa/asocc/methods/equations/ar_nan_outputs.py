"""Empty AR result shapes for studied years before a requested reference year.

Acquired rights (AR) methods are undefined when the studied year is earlier
than the selected reference year. The deterministic runner preserves those
requested AR scopes as public rows with the normal method specific identity
axes and ``NaN`` values.
"""

import numpy as np
import pandas as pd

from ..registry.registry import REGISTRY
from .ar_result_indexing import (
    _cached_index_for,
    _CachedIndexEntry,
    _IndexCacheKey,
    _store_cached_index,
)


def _stacked_nan_index(*, row_index: pd.Index, columns: pd.Index) -> pd.MultiIndex:
    """Return the stacked row by column index without stacking values."""
    row_count = len(row_index)
    column_count = len(columns)
    if isinstance(row_index, pd.MultiIndex):
        row_levels = list(row_index.levels)
        row_codes = [
            np.repeat(np.asarray(code, dtype=np.intp), column_count) for code in row_index.codes
        ]
        row_names = list(row_index.names)
    else:
        row_base_codes, row_base_levels = pd.factorize(row_index, sort=False)
        row_levels = [pd.Index(row_base_levels, name=row_index.name, copy=False)]
        row_codes = [np.repeat(np.asarray(row_base_codes, dtype=np.intp), column_count)]
        row_names = [row_index.name]
    if isinstance(columns, pd.MultiIndex):
        column_levels = list(columns.levels)
        column_codes = [
            np.tile(np.asarray(code, dtype=np.intp), row_count) for code in columns.codes
        ]
        column_names = list(columns.names)
    else:
        column_base_codes, column_base_levels = pd.factorize(columns, sort=False)
        column_levels = [pd.Index(column_base_levels, name=columns.name, copy=False)]
        column_codes = [np.tile(np.asarray(column_base_codes, dtype=np.intp), row_count)]
        column_names = [columns.name]
    return pd.MultiIndex(
        levels=[*row_levels, *column_levels],
        codes=[*row_codes, *column_codes],
        names=[*row_names, *column_names],
        verify_integrity=False,
    )


def _stack_matrix_to_year(matrix: pd.DataFrame, year: int) -> pd.DataFrame:
    """Stack a DataFrame to Series and return a year column."""
    stacked_index = _stacked_nan_index(row_index=matrix.index, columns=matrix.columns)
    return pd.DataFrame(
        {int(year): np.full(len(stacked_index), np.nan, dtype=np.float64)},
        index=stacked_index,
    )


def _nan_like_ar_l1(
    lcia_reg: pd.DataFrame,
    year: int,
    *,
    region_label: str | None = None,
    index_cache: dict[_IndexCacheKey, _CachedIndexEntry] | None = None,
) -> pd.DataFrame:
    """Create NaN AR L1 output using LCIA regional impacts shape."""
    region_name = region_label
    if not region_name:
        col_names = [str(name) for name in lcia_reg.columns.names]
        if len(col_names) == 1 and col_names[0] != "None":
            region_name = col_names[0]
        else:
            region_name = "region"
    cache_key: _IndexCacheKey | None = None
    stacked_index = None
    if index_cache is not None:
        cache_key = ("nan_ar_l1", id(lcia_reg.index), id(lcia_reg.columns), region_name)
        stacked_index = _cached_index_for(
            index_cache=index_cache,
            cache_key=cache_key,
            source_index=lcia_reg.index,
        )
    if stacked_index is None:
        stacked_index = _stacked_nan_index(row_index=lcia_reg.index, columns=lcia_reg.columns)
        stacked_index = stacked_index.set_names(["impact", region_name])
        _store_cached_index(
            index_cache=index_cache,
            cache_key=cache_key,
            source_index=lcia_reg.index,
            attached_index=stacked_index,
        )
    return pd.DataFrame(
        {int(year): np.full(len(stacked_index), np.nan, dtype=np.float64)},
        index=stacked_index,
    )


def _nan_like_ar_l2(
    *,
    l2_method: str,
    fu_code: str,
    lcia: dict,
    year: int,
    pre_weighting: bool,
) -> pd.DataFrame:
    """Create NaN AR L2 output using LCIA input shapes."""
    kind = REGISTRY.l1_kind_for_l2_method(l2_method)
    lcia_key: str | None = None
    if kind == "CBA_FD":
        cba_fd_keys = {
            ("L2.a.a", True): "e_cba_fd_rp_sp_rf",
            ("L2.a.a", False): "e_cba_fd_rp_sp",
            ("L2.b.a", True): "e_cba_fd_rp_sp_rf",
            ("L2.b.a", False): "e_cba_fd_rp_sp_rf",
            ("L2.c.a", True): "e_cba_fd_rf_sp",
            ("L2.c.a", False): "e_cba_fd_rf_sp",
        }
        lcia_key = cba_fd_keys[(fu_code, bool(pre_weighting))]
    elif kind == "CBA_TD":
        cba_td_keys = {
            "L2.a.b": "e_cba_td_rp_sp",
            "L2.b.b": "e_cba_td_rp_sp_rc",
            "L2.c.b": "e_cba_td_rc_sp",
        }
        lcia_key = cba_td_keys[fu_code]
    else:
        lcia_key = "e_pba_rp_sp"
    if lcia_key not in lcia:
        raise ValueError(
            f"LCIA payload missing required key '{lcia_key}' for method "
            f"'{l2_method}' on FU '{fu_code}'."
        )
    return _stack_matrix_to_year(lcia[lcia_key], year)

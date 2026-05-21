"""AR(E) allocation methods (L1 and L2)."""

import numpy as np
import pandas as pd

from .share_math import safe_divide_frame


def _stack_columns(
    frame: pd.DataFrame,
    *,
    col_names: list[str],
    year: int,
) -> pd.DataFrame:
    """Stack all column levels into the index and label the result."""
    return _stack_array_to_year(
        frame.to_numpy(dtype="float64", copy=False),
        row_index=frame.index,
        columns=frame.columns,
        col_names=col_names,
        year=year,
    )


def _stack_array_to_year(
    values: np.ndarray,
    *,
    row_index: pd.Index,
    columns: pd.Index,
    col_names: list[str],
    year: int,
    index_cache: dict[tuple[object, ...], object] | None = None,
) -> pd.DataFrame:
    """Stack one numeric matrix into the public one year frame shape."""
    cache_key = ("stack_array_to_year", id(row_index), id(columns), tuple(col_names))
    cached = index_cache.get(cache_key) if index_cache is not None else None
    if isinstance(cached, pd.MultiIndex):
        return pd.DataFrame({int(year): values.reshape(-1)}, index=cached)
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
    else:
        column_base_codes, column_base_levels = pd.factorize(columns, sort=False)
        column_levels = [pd.Index(column_base_levels, name=columns.name, copy=False)]
        column_codes = [np.tile(np.asarray(column_base_codes, dtype=np.intp), row_count)]
    stacked_index = pd.MultiIndex(
        levels=[*row_levels, *column_levels],
        codes=[*row_codes, *column_codes],
        names=[*row_names, *col_names],
        verify_integrity=False,
    )
    if index_cache is not None:
        index_cache[cache_key] = stacked_index
    return pd.DataFrame({int(year): values.reshape(-1)}, index=stacked_index)


def compute_ar_e_l1(
    *,
    year: int,
    lcia_reg: pd.DataFrame | None,
    lcia_reg_by_year: dict[int, pd.DataFrame] | None,
    reference_year: int | None,
    region_label: str = "region",
) -> pd.DataFrame:
    """Compute AR(E) for L1.

    Args:
        year: Studied year (output column label).
        lcia_reg: LCIA impacts by region for current year.
        lcia_reg_by_year: LCIA time series by year.
        reference_year: Reference year.

    Returns:
        DataFrame of AR shares indexed by impact and region.
    """
    if reference_year is None:
        raise ValueError("reference_year is required for AR methods.")
    if lcia_reg is None and lcia_reg_by_year is None:
        raise ValueError("LCIA regional impacts required for AR.")
    impacts = lcia_reg_by_year.get(reference_year) if lcia_reg_by_year else lcia_reg
    if impacts is None:
        raise ValueError("LCIA impacts missing for reference year.")
    share = safe_divide_frame(impacts, impacts.sum(axis=1), axis=0)
    return _stack_columns(
        share,
        col_names=[region_label],
        year=year,
    )


def compute_ar_e_l2(
    *,
    l2_method: str,
    fu_code: str,
    l1_weights: pd.Series | None,
    lcia: dict,
    reference_year: int | None,
    pre_weighting: bool = False,
) -> pd.DataFrame:
    """Compute AR(E) methods for L2."""
    if reference_year is None:
        raise ValueError("reference_year is required for AR methods.")
    if lcia is None:
        raise ValueError("LCIA metrics required for AR methods.")

    if "CBA_FD" in l2_method:
        return _compute_cba_fd_l2(
            fu_code=fu_code,
            l1_weights=l1_weights,
            lcia=lcia,
            reference_year=reference_year,
            pre_weighting=pre_weighting,
        )

    if "CBA_TD" in l2_method:
        return _compute_cba_td_l2(
            fu_code=fu_code,
            lcia=lcia,
            reference_year=reference_year,
        )

    return _compute_pba_l2(
        l1_weights=l1_weights,
        lcia=lcia,
        reference_year=reference_year,
        pre_weighting=pre_weighting,
    )


def _compute_cba_fd_l2(
    *,
    fu_code: str,
    l1_weights: pd.Series | None,
    lcia: dict,
    reference_year: int,
    pre_weighting: bool,
) -> pd.DataFrame:
    """Compute CBA_FD AR(E) for L2 FUs."""
    if fu_code == "L2.a.a":
        if pre_weighting:
            weights = safe_divide_frame(lcia["e_cba_fd_rp_sp_rf"], lcia["e_cba_fd_reg"], axis=1)
            return _stack_columns(weights, col_names=["r_f"], year=reference_year)
        if l1_weights is not None:
            weights = safe_divide_frame(lcia["e_cba_fd_rp_sp_rf"], lcia["e_cba_fd_reg"], axis=1)
            out = weights.mul(l1_weights, axis=1).sum(axis=1)
            return out.to_frame(reference_year)
        out = safe_divide_frame(lcia["e_cba_fd_rp_sp"], lcia["e_cba_fd_reg"].sum(axis=1), axis=0)
        return _stack_columns(out, col_names=["r_p", "s_p"], year=reference_year)

    if fu_code == "L2.b.a":
        numer = lcia["e_cba_fd_rp_sp_rf"]
        if pre_weighting:
            weights = safe_divide_frame(numer, lcia["e_cba_fd_reg"], axis=1)
            return _stack_columns(weights, col_names=["r_f"], year=reference_year)
        if l1_weights is not None:
            weights = safe_divide_frame(numer, lcia["e_cba_fd_reg"], axis=1)
            out = weights.mul(l1_weights, axis=1)
            return _stack_columns(out, col_names=["r_f"], year=reference_year)
        out = safe_divide_frame(numer, lcia["e_cba_fd_reg"].sum(axis=1), axis=0)
        return _stack_columns(out, col_names=["r_f"], year=reference_year)

    numer = lcia["e_cba_fd_rf_sp"]
    if pre_weighting:
        weights = safe_divide_frame(numer, lcia["e_cba_fd_reg"], axis=1, level="r_f")
        return _stack_columns(weights, col_names=["r_f", "s_p"], year=reference_year)
    if l1_weights is not None:
        weights = safe_divide_frame(numer, lcia["e_cba_fd_reg"], axis=1, level="r_f")
        out = weights.mul(l1_weights, axis=1, level="r_f")
        return _stack_columns(out, col_names=["r_f", "s_p"], year=reference_year)
    out = safe_divide_frame(numer, lcia["e_cba_fd_reg"].sum(axis=1), axis=0)
    return _stack_columns(out, col_names=["r_f", "s_p"], year=reference_year)


def _compute_cba_td_l2(
    *,
    fu_code: str,
    lcia: dict,
    reference_year: int,
) -> pd.DataFrame:
    """Compute CBA_TD AR(E) for L2 FUs."""
    denom = lcia["e_cba_fd_reg"].sum(axis=1)
    if fu_code == "L2.a.b":
        out = safe_divide_frame(lcia["e_cba_td_rp_sp"], denom, axis=0)
        return _stack_columns(out, col_names=["r_p", "s_p"], year=reference_year)
    if fu_code == "L2.b.b":
        out = safe_divide_frame(lcia["e_cba_td_rp_sp_rc"], denom, axis=0)
        return _stack_columns(out, col_names=["r_c"], year=reference_year)
    out = safe_divide_frame(lcia["e_cba_td_rc_sp"], denom, axis=0)
    return _stack_columns(out, col_names=["r_c", "s_p"], year=reference_year)


def _compute_pba_l2(
    *,
    l1_weights: pd.Series | None,
    lcia: dict,
    reference_year: int,
    pre_weighting: bool,
) -> pd.DataFrame:
    """Compute PBA AR(E) for L2 FUs."""
    numer = lcia["e_pba_rp_sp"]
    if pre_weighting:
        weights = safe_divide_frame(numer, lcia["e_pba_reg"], axis=1, level="r_p")
        return _stack_columns(weights, col_names=["r_p", "s_p"], year=reference_year)
    if l1_weights is not None:
        weights = safe_divide_frame(numer, lcia["e_pba_reg"], axis=1, level="r_p")
        out = weights.mul(l1_weights, axis=1, level="r_p")
        return _stack_columns(out, col_names=["r_p", "s_p"], year=reference_year)
    out = safe_divide_frame(numer, lcia["e_pba_reg"].sum(axis=1), axis=0)
    return _stack_columns(out, col_names=["r_p", "s_p"], year=reference_year)

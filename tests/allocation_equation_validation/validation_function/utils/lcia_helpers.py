"""Helpers for L2/L2*b LCIA checks with F_Y add back."""

from typing import Any, Callable, NamedTuple, cast

import numpy as np
import pandas as pd

from pyaesa.asocc.methods.lcia_inputs import load_impact_parent_mapping
from pyaesa.process.mrios.utils.io.paths import _get_year_saved_dir
from .io_helpers import method_label_from_row
from .l2_two_step_helpers import L1RegionWeightRequest, load_l1_region_weights

_CF_PARENT_CACHE: dict[tuple[str, str], pd.Series] = {}


class LciaWeightingRequest(NamedTuple):
    """Input contract for L1-weighted LCIA global share computations."""

    validation_project_name_root: str
    source: str
    agg_reg: bool | None
    group_indices: bool
    l1_mode: str
    output_format: str
    l1_method: str
    year: int
    impact: str
    reference_year: int | None
    lcia_method: str
    boundary: str
    matrix_version: str | None


class FyShareRowRequest(NamedTuple):
    """Input contract for row level L2 LCIA ``F_Y`` add back values."""

    row: pd.Series
    fu_code: str
    l2_bucket: str
    source: str
    year: int
    reference_year: int | None
    lcia_method: str
    l2_country_axis_fn: Callable[[str], str | None]
    matrix_version: str | None = None


def _impact_parent_map(*, source: str, lcia_method: str) -> pd.Series:
    """Return cached mapping from LCIA child impacts to parent impact labels."""
    cache_key = (str(source), str(lcia_method))
    if cache_key in _CF_PARENT_CACHE:
        return _CF_PARENT_CACHE[cache_key]
    mapping = load_impact_parent_mapping(source=source, lcia_method=lcia_method).astype("string")
    _CF_PARENT_CACHE[cache_key] = mapping
    return mapping


def _aggregate_impacts_to_parent(
    frame: pd.DataFrame,
    *,
    source: str,
    lcia_method: str,
) -> pd.DataFrame:
    """Aggregate LCIA rows from child impacts to parent impact categories."""
    mapping = _impact_parent_map(source=source, lcia_method=lcia_method)
    out = frame.copy()
    new_index = pd.Index(
        [str(mapping.get(str(idx), str(idx))) for idx in out.index],
        name=out.index.name,
    )
    out.index = new_index
    grouped = out.groupby(level=0).sum(min_count=1)
    return _ensure_frame(grouped)


def _ensure_frame(value: object) -> pd.DataFrame:
    """Return DataFrame payload for pickle/groupby results."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame().T
    return pd.DataFrame(cast(Any, value))


def _load_lcia_level1_parent_frames(
    *,
    source: str,
    year: int,
    lcia_method: str,
    boundary: str,
    matrix_version: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load LCIA F_Y and denominator frames and aggregate both to parent impacts."""
    year_dir = _get_year_saved_dir(source, year, matrix_version=matrix_version)
    base = year_dir / "enacting_metrics" / "level_1" / lcia_method
    fy = _ensure_frame(pd.read_pickle(base / "F_Y.pickle"))
    denom_name = "e_pba_reg.pickle" if boundary == "PBA" else "e_cba_fd_reg.pickle"
    denom = _ensure_frame(pd.read_pickle(base / denom_name))
    return (
        _aggregate_impacts_to_parent(
            fy,
            source=source,
            lcia_method=lcia_method,
        ),
        _aggregate_impacts_to_parent(
            denom,
            source=source,
            lcia_method=lcia_method,
        ),
    )


def _as_numeric_series(value: pd.Series | pd.DataFrame) -> pd.Series:
    """Return a numeric Series from a Series or DataFrame row selection."""
    if isinstance(value, pd.DataFrame):
        if value.shape[0] == 1:
            series = value.iloc[0]
        else:
            series = value.sum(axis=0, min_count=1)
    else:
        series = value
    return pd.Series(pd.to_numeric(series, errors="coerce"), copy=False).astype(float)


def _sum_series_min_count(series: pd.Series) -> float:
    """Return numeric series sum with one-value minimum count semantics."""
    values = pd.Series(pd.to_numeric(series, errors="coerce"), copy=False)
    finite = values.dropna()
    if finite.empty:
        return np.nan
    return float(np.asarray(finite, dtype=float).sum())


def fy_shares_by_region(
    source: str,
    year: int,
    lcia_method: str,
    boundary: str,  # "CBA_FD" | "PBA"
    matrix_version: str | None = None,
) -> pd.DataFrame:
    """Return F_Y share by (impact, region) for one LCIA method/boundary."""
    fy, denom = _load_lcia_level1_parent_frames(
        source=source,
        year=year,
        lcia_method=lcia_method,
        boundary=boundary,
        matrix_version=matrix_version,
    )

    # F_Y is stored by (region, category): aggregate all categories per region.
    fy_grouped = _ensure_frame(fy).T.groupby(level="region").sum(min_count=1)
    fy_reg_df = _ensure_frame(fy_grouped).T
    shares = fy_reg_df.reindex(columns=denom.columns).div(denom.replace(0, np.nan))
    return _ensure_frame(shares.fillna(0.0))


def weighted_l1_fy_global_share(
    request: LciaWeightingRequest,
) -> float:
    """Return global F_Y share weighted by L1 regional weights."""
    fy_year = (
        int(request.reference_year) if request.reference_year is not None else int(request.year)
    )
    shares = fy_shares_by_region(
        request.source,
        fy_year,
        request.lcia_method,
        request.boundary,
        matrix_version=request.matrix_version,
    )
    if request.impact not in shares.index:
        return 0.0
    fy_region = _as_numeric_series(shares.loc[request.impact])
    fy_region.index = fy_region.index.map(str)

    weights = load_l1_region_weights(
        L1RegionWeightRequest(
            validation_project_name_root=request.validation_project_name_root,
            source=request.source,
            agg_version=request.matrix_version,
            agg_reg=request.agg_reg,
            group_indices=request.group_indices,
            l1_mode=request.l1_mode,
            output_format=request.output_format,
            l1_method=request.l1_method,
            year=request.year,
            impact=request.impact,
            reference_year=request.reference_year,
        )
    )
    if weights is None or weights.empty:
        return np.nan
    common = weights.index.intersection(fy_region.index)
    if len(common) == 0:
        return np.nan
    weighted = weights.loc[common].astype(float) * fy_region.loc[common].astype(float)
    return _sum_series_min_count(pd.Series(weighted, copy=False))


def l1_weight_coverage_for_lcia_global(
    request: LciaWeightingRequest,
) -> tuple[float | None, float, list[str]]:
    """Return effective target, lost weight mass, and invalid regions.

    Invalid regions are those where the LCIA denominator for the selected
    impact/year is non finite or zero, so weighted L1 mass on those regions
    cannot contribute to normalized L2-vs global totals.
    """
    fy_year = (
        int(request.reference_year) if request.reference_year is not None else int(request.year)
    )
    _, denom = _load_lcia_level1_parent_frames(
        source=request.source,
        year=fy_year,
        lcia_method=request.lcia_method,
        boundary=request.boundary,
        matrix_version=request.matrix_version,
    )
    if request.impact not in denom.index:
        return None, 0.0, []
    denom_region = _as_numeric_series(denom.loc[request.impact])
    denom_region.index = denom_region.index.map(str)

    weights = load_l1_region_weights(
        L1RegionWeightRequest(
            validation_project_name_root=request.validation_project_name_root,
            source=request.source,
            agg_version=request.matrix_version,
            agg_reg=request.agg_reg,
            group_indices=request.group_indices,
            l1_mode=request.l1_mode,
            output_format=request.output_format,
            l1_method=request.l1_method,
            year=request.year,
            impact=request.impact,
            reference_year=request.reference_year,
        )
    )
    if weights is None or weights.empty:
        return None, 0.0, []

    common = weights.index.intersection(denom_region.index)
    if len(common) == 0:
        return None, 0.0, []

    denom_slice = denom_region.loc[common].astype(float)
    invalid_mask = denom_slice.isna() | ~np.isfinite(denom_slice) | np.isclose(denom_slice, 0.0)
    if not bool(invalid_mask.any()):
        effective = _sum_series_min_count(pd.Series(weights.loc[common].astype(float), copy=False))
        return effective, 0.0, []

    weights_common = weights.loc[common].astype(float)
    lost_weight = _sum_series_min_count(pd.Series(weights_common.loc[invalid_mask], copy=False))
    invalid_regions = sorted(str(r) for r in denom_slice.index[invalid_mask].tolist())
    effective = _sum_series_min_count(pd.Series(weights_common.loc[~invalid_mask], copy=False))
    return effective, lost_weight, invalid_regions


def _impact_share_row(*, shares: pd.DataFrame, impact: str) -> pd.Series | None:
    """Return numeric region share row for one impact, or ``None`` if missing."""
    if impact not in shares.index:
        return None
    impact_slice = shares.loc[impact]
    if isinstance(impact_slice, pd.DataFrame):
        raise RuntimeError(
            f"Expected one F_Y row per parent impact after LCIA mapping, got many for '{impact}'."
        )
    return pd.Series(pd.to_numeric(impact_slice, errors="coerce"), copy=False).astype(float)


def _fy_share_in_l1_bucket(*, request: FyShareRowRequest, impact_row: pd.Series) -> float:
    """Return F_Y share for one L2-in-L1 grouped row."""
    axis = request.l2_country_axis_fn(request.fu_code)
    if axis and axis in request.row.index:
        axis_value = str(request.row.get(axis))
        if axis_value in impact_row.index:
            return float(impact_row.loc[axis_value])
    # If the expected country axis is unavailable, avoid adding a global
    # fallback share to a country level row.
    return 0.0


def _fy_share_global_bucket(
    *,
    request: FyShareRowRequest,
    fy_year: int,
    boundary: str,
    impact: str,
) -> float:
    """Return global F_Y/denominator share for one parent impact."""
    fy, denom = _load_lcia_level1_parent_frames(
        source=request.source,
        year=fy_year,
        lcia_method=request.lcia_method,
        boundary=boundary,
        matrix_version=request.matrix_version,
    )
    fy_grouped = _ensure_frame(fy).T.groupby(level="region").sum(min_count=1)
    fy_reg_df = _ensure_frame(fy_grouped).T
    if impact not in fy_reg_df.index or impact not in denom.index:
        return 0.0
    fy_global = _sum_series_min_count(_as_numeric_series(fy_reg_df.loc[impact]))
    denom_global = _sum_series_min_count(_as_numeric_series(denom.loc[impact]))
    if not np.isfinite(denom_global) or np.isclose(denom_global, 0.0):
        return 0.0
    return fy_global / denom_global


def fy_share_for_row(
    request: FyShareRowRequest,
) -> float:
    """Return additive F_Y share for one aggregated L2 LCIA row."""
    impact = str(request.row.get("impact"))
    method_label = method_label_from_row(request.row)
    boundary = "PBA" if "PBA" in method_label else "CBA_FD"
    fy_year = (
        int(request.reference_year) if request.reference_year is not None else int(request.year)
    )
    shares = fy_shares_by_region(
        request.source,
        fy_year,
        request.lcia_method,
        boundary,
        matrix_version=request.matrix_version,
    )
    impact_row = _impact_share_row(shares=shares, impact=impact)
    if impact_row is None:
        return 0.0

    if request.l2_bucket == "l2_in_l1":
        return _fy_share_in_l1_bucket(request=request, impact_row=impact_row)

    if request.l2_bucket == "l2_vs_global":
        return _fy_share_global_bucket(
            request=request,
            fy_year=fy_year,
            boundary=boundary,
            impact=impact,
        )

    return 0.0

"""Expected overlap helpers for L2*b AR(E^{CBA_TD}) checks."""

import numpy as np
import pandas as pd

from pyaesa.process.mrios.utils.io.paths import _get_year_saved_dir

from .lcia_helpers import (
    _aggregate_impacts_to_parent,
    _ensure_frame,
    _impact_parent_map,
    _load_lcia_level1_parent_frames,
)

_L2B_AR_TD_CACHE: dict[tuple[str, int, int, str, str | None], pd.Series] = {}
# AR(E^{CBA_TD}) expected overlap uses one step world numerator for all L2*b FUs.
_L2B_AR_TD_WORLD_NUMERATOR_PICKLE = "e_cba_td_rp_sp.pickle"


def _numeric_series(value: object) -> pd.Series:
    """Return numeric pandas Series from scalar/array like results."""
    return pd.Series(pd.to_numeric(value, errors="coerce"), copy=False).astype(float)


def method_is_ar_cba_td(method_label: str) -> bool:
    """Return whether method label is AR(E^{CBA_TD})."""
    return "AR(E^{CBA_TD})" in str(method_label).strip()


def _global_series_by_impact(frame: pd.DataFrame) -> pd.Series:
    """Aggregate one impact indexed frame to world totals by impact."""
    out = _numeric_series(frame.sum(axis=1, min_count=1))
    if not isinstance(out.index, pd.MultiIndex):
        return out
    level = "impact" if "impact" in out.index.names else 0
    return _numeric_series(out.groupby(level=level).sum(min_count=1))


def _global_fy_series_by_impact(frame: pd.DataFrame) -> pd.Series:
    """Aggregate F_Y payload to world totals by impact."""
    fy_grouped = _ensure_frame(frame).T.groupby(level="region").sum(min_count=1)
    fy_reg = _ensure_frame(fy_grouped).T
    return _numeric_series(fy_reg.sum(axis=1, min_count=1))


def _load_ar_td_parent_numerator(
    *,
    source: str,
    ref_year: int,
    lcia_method: str,
    matrix_version: str | None,
) -> pd.DataFrame:
    """Load L2 AR(CBA_TD) numerator and aggregate child impacts to parent labels."""
    year_dir = _get_year_saved_dir(source, ref_year, matrix_version=matrix_version)
    base_l2 = year_dir / "enacting_metrics" / "level_2" / lcia_method
    payload = _ensure_frame(pd.read_pickle(base_l2 / _L2B_AR_TD_WORLD_NUMERATOR_PICKLE))
    return _aggregate_for_parent_impact(
        payload,
        source=source,
        lcia_method=lcia_method,
    )


def _combine_expected_series(
    *,
    numer: pd.DataFrame,
    fy: pd.DataFrame,
    denom: pd.DataFrame,
) -> pd.Series:
    """Combine numerator/F_Y/denominator to expected global overlap by impact."""
    numer_w = _global_series_by_impact(numer)
    denom_w = _global_series_by_impact(denom)
    fy_w = _global_fy_series_by_impact(fy)
    expected = _numeric_series(
        numer_w.add(fy_w, fill_value=np.nan).div(denom_w.replace(0.0, np.nan))
    )
    expected.index = expected.index.map(str)
    return expected


def load_l2_star_b_ar_cba_td_expected_global_by_impact(
    *,
    source: str,
    year: int,
    reference_year: int | None,
    lcia_method: str,
    matrix_version: str | None,
) -> pd.Series:
    """Return expected global overlap for AR(E^{CBA_TD}) by impact."""
    ref_year = int(reference_year) if reference_year is not None else int(year)
    key = (source, int(year), ref_year, lcia_method, matrix_version)
    cached = _L2B_AR_TD_CACHE.get(key)
    if cached is not None:
        return cached

    fy, denom = _load_lcia_level1_parent_frames(
        source=source,
        year=ref_year,
        lcia_method=lcia_method,
        boundary="CBA_FD",
        matrix_version=matrix_version,
    )
    numer = _load_ar_td_parent_numerator(
        source=source,
        ref_year=ref_year,
        lcia_method=lcia_method,
        matrix_version=matrix_version,
    )
    expected = _combine_expected_series(numer=numer, fy=fy, denom=denom)
    _L2B_AR_TD_CACHE[key] = expected
    return expected


def _aggregate_for_parent_impact(
    frame: pd.DataFrame,
    *,
    source: str,
    lcia_method: str,
) -> pd.DataFrame:
    """Aggregate impact labels to parent categories, preserving extra index levels."""
    if not isinstance(frame.index, pd.MultiIndex) or "impact" not in frame.index.names:
        return _aggregate_impacts_to_parent(
            frame,
            source=source,
            lcia_method=lcia_method,
        )

    mapping = _impact_parent_map(source=source, lcia_method=lcia_method)
    idx = frame.index
    level_pos = idx.names.index("impact")
    arrays = [idx.get_level_values(i).to_list() for i in range(idx.nlevels)]
    arrays[level_pos] = [str(mapping.get(str(v), str(v))) for v in arrays[level_pos]]
    out = frame.copy()
    out.index = pd.MultiIndex.from_arrays(arrays, names=list(idx.names))
    grouped = out.groupby(level=list(range(out.index.nlevels))).sum(min_count=1)
    return _ensure_frame(grouped)

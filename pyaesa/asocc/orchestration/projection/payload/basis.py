"""Shared basis and index utilities for projection payload builders."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyaesa.download.pop_gdp.contracts import GDP_WB_INDICATOR

from ....data.paths import _get_mrio_year_dir
from ....data.load_pop_gdp import _get_series_for_year
from ...yearly.shared.year_inputs import _MrioPayload
from ...yearly.shared.year_inputs import _load_year_mrio_payloads_required, build_l2_compute_inputs


@dataclass(frozen=True)
class RegressionBasis:
    """Shared historical basis used by regression payload projection."""

    gdp_by_year: dict[int, pd.Series]
    payload_by_year: dict[int, _MrioPayload]
    base_payload: _MrioPayload


def _numeric_series(series: pd.Series) -> pd.Series:
    """Return one numeric pandas Series with the same index/name."""
    values = np.asarray(series.to_numpy(copy=False), dtype=np.float64)
    return pd.Series(values, index=series.index, name=series.name, copy=False)


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one numeric pandas DataFrame with the same shape metadata."""
    values = np.asarray(frame.to_numpy(copy=False), dtype=np.float64)
    return pd.DataFrame(
        values,
        index=frame.index,
        columns=frame.columns,
        copy=False,
    )


def safe_share(
    numer: pd.Series,
    denom: pd.Series,
    *,
    level: str | list[str],
) -> pd.Series:
    """Compute a share series and align denominators on container levels.

    The function supports both single index and multi index numerators.
    For multi index inputs, ``level`` defines which index levels identify the
    denominator bucket (for example ``"r_f"`` or ``["r_c", "s_p"]``).
    """
    level_names = [level] if isinstance(level, str) else [str(name) for name in level]
    numer_clean = _numeric_series(numer)
    denom_clean = _numeric_series(denom).replace(0.0, np.nan)
    if isinstance(numer_clean.index, pd.MultiIndex):
        missing = [name for name in level_names if name not in numer_clean.index.names]
        if missing:
            raise ValueError(f"Numerator index is missing required level names: {missing}.")
        key_arrays = [numer_clean.index.get_level_values(name) for name in level_names]
        if len(level_names) == 1:
            key_index = pd.Index(key_arrays[0], name=level_names[0])
        else:
            key_index = pd.MultiIndex.from_arrays(key_arrays, names=level_names)
    else:
        if len(level_names) != 1:
            raise ValueError("Single index numerator cannot use multi-level share keys.")
        if numer_clean.index.name != level_names[0]:
            raise ValueError(
                f"Numerator index name does not match required share level '{level_names[0]}'."
            )
        key_index = pd.Index(numer_clean.index, name=level_names[0])

    if isinstance(denom_clean.index, pd.MultiIndex):
        missing = [name for name in level_names if name not in denom_clean.index.names]
        if missing:
            raise ValueError(f"Denominator index is missing required level names: {missing}.")
        denom_bucket = pd.Series(
            denom_clean.groupby(level=level_names).sum(min_count=1),
            copy=False,
        )
    else:
        if len(level_names) != 1:
            raise ValueError("Multi-level share denominator requires a MultiIndex denominator.")
        if denom_clean.index.name != level_names[0]:
            raise ValueError(
                f"Denominator index name does not match required share level '{level_names[0]}'."
            )
        denom_bucket = pd.Series(denom_clean, copy=False)
    matched = denom_bucket.reindex(key_index).to_numpy(dtype=float)
    denom_aligned = pd.Series(matched, index=numer_clean.index, dtype=float)
    out = numer_clean.div(denom_aligned)
    return out.replace([np.inf, -np.inf], np.nan)


def require_series(value: pd.Series, *, label: str) -> pd.Series:
    """Return a numeric series for a pyaesa owned projection payload."""
    del label
    return _numeric_series(value)


def require_frame(value: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Return a numeric frame for a pyaesa owned projection payload."""
    del label
    return _numeric_frame(value)


def coerce_index_like(series: pd.Series, *, template: pd.Index) -> pd.Series:
    """Align a projected series index with a template index."""
    if isinstance(template, pd.MultiIndex):
        if not isinstance(series.index, pd.MultiIndex):
            raise ValueError("Projected index must be MultiIndex to match template.")
        if series.index.nlevels != template.nlevels:
            raise ValueError("Projected index level count does not match template level count.")
        if list(series.index.names) != list(template.names):
            raise ValueError(
                "Projected index level names do not match template names. "
                f"projected={series.index.names}, template={template.names}"
            )
        return _numeric_series(series).reindex(template)
    if isinstance(series.index, pd.MultiIndex):
        raise ValueError("Projected index is MultiIndex but template is single index.")
    if series.index.name != template.name:
        raise ValueError(
            "Projected index name does not match template name. "
            f"projected={series.index.name}, template={template.name}"
        )
    return _numeric_series(series).reindex(template)


def _history_payloads_for_years(
    *,
    context,
    years: list[int],
    needs_fd_total: bool,
    needs_fd_detail: bool,
    needs_gva: bool,
    needs_x_to_rc: bool,
) -> dict[int, _MrioPayload]:
    """Load all historical MRIO payloads needed by projection."""
    out: dict[int, _MrioPayload] = {}
    for year in years:
        out[int(year)] = _load_projection_metric_subset(
            context=context,
            year=int(year),
            needs_fd_total=needs_fd_total,
            needs_fd_detail=needs_fd_detail,
            needs_gva=needs_gva,
            needs_x_to_rc=needs_x_to_rc,
        )
    return out


def _load_projection_metric_subset(
    *,
    context,
    year: int,
    needs_fd_total: bool,
    needs_fd_detail: bool,
    needs_gva: bool,
    needs_x_to_rc: bool,
) -> _MrioPayload:
    """Load one historical payload and retain only regression metrics."""
    saved_dir = _get_mrio_year_dir(
        source=context.source,
        year=int(year),
        group_version=context.group_version,
    )
    payload = _load_year_mrio_payloads_required(
        saved_dir=saved_dir,
        context=context,
        needs_mrio=True,
    )
    enacting_metric_l1: dict[str, pd.Series] = {}
    enacting_metric_l2: dict[str, pd.Series | pd.DataFrame] = {}
    utility: dict[str, pd.DataFrame] = {}
    if needs_fd_total:
        enacting_metric_l1["fd_rf"] = payload.enacting_metric_l1["fd_rf"]
    if needs_fd_detail:
        enacting_metric_l2["fd_rp_sp_rf"] = payload.enacting_metric_l2["fd_rp_sp_rf"]
        enacting_metric_l2["fd_rp_sp"] = payload.enacting_metric_l2["fd_rp_sp"]
        enacting_metric_l2["fd_rf_sp"] = payload.enacting_metric_l2["fd_rf_sp"]
    if needs_gva:
        enacting_metric_l1["gva_rp"] = payload.enacting_metric_l1["gva_rp"]
        enacting_metric_l2["gva_rp_sp"] = payload.enacting_metric_l2["gva_rp_sp"]
    if needs_x_to_rc:
        utility["x_to_rc"] = payload.utility["x_to_rc"]
    return _MrioPayload(
        enacting_metric_l1=enacting_metric_l1,
        enacting_metric_l2=enacting_metric_l2,
        utility=utility,
        l2_inputs=build_l2_compute_inputs(
            enacting_metric_l1={
                "fd_rf": enacting_metric_l1.get("fd_rf", pd.Series(dtype=float)),
                "gva_rp": enacting_metric_l1.get("gva_rp", pd.Series(dtype=float)),
            },
            enacting_metric_l2={
                "fd_rp_sp_rf": enacting_metric_l2.get("fd_rp_sp_rf", pd.DataFrame()),
                "fd_rp_sp": enacting_metric_l2.get("fd_rp_sp", pd.Series(dtype=float)),
                "fd_rf_sp": enacting_metric_l2.get("fd_rf_sp", pd.Series(dtype=float)),
                "gva_rp_sp": enacting_metric_l2.get("gva_rp_sp", pd.Series(dtype=float)),
            },
            utility={
                "x_to_rc": utility.get("x_to_rc", pd.DataFrame()),
            },
        ),
    )


def _load_gdp_series_for_history_year(*, context, year: int) -> pd.Series:
    return _get_series_for_year(
        df=context.wb_df,
        variable=GDP_WB_INDICATOR,
        year=int(year),
        source_key=context.source,
        group_version=context.group_version_reg,
        ssp_scenario=None,
        region_col_override=None,
    )


def regression_basis(
    *,
    context,
    state,
    historical_years: list[int],
    fit_end: int,
    needs_fd_total: bool,
    needs_fd_detail: bool,
    needs_gva: bool,
    needs_x_to_rc: bool,
) -> RegressionBasis:
    """Build and cache shared regression basis for future year payloads."""
    key = (
        str(context.source),
        str(context.group_version),
        str(context.group_version_reg),
        tuple(int(year) for year in historical_years),
        int(fit_end),
        str(context.fu_code),
        bool(needs_fd_total),
        bool(needs_fd_detail),
        bool(needs_gva),
        bool(needs_x_to_rc),
    )
    cached = state.projection_regression_basis_cache.get(key)
    if isinstance(cached, RegressionBasis):
        return cached
    payload_by_year = _history_payloads_for_years(
        context=context,
        years=historical_years,
        needs_fd_total=needs_fd_total,
        needs_fd_detail=needs_fd_detail,
        needs_gva=needs_gva,
        needs_x_to_rc=needs_x_to_rc,
    )
    gdp_by_year = {
        int(hist_year): _load_gdp_series_for_history_year(context=context, year=int(hist_year))
        for hist_year in historical_years
    }
    basis = RegressionBasis(
        gdp_by_year=gdp_by_year,
        payload_by_year=payload_by_year,
        base_payload=payload_by_year[int(fit_end)],
    )
    state.projection_regression_basis_cache[key] = basis
    return basis

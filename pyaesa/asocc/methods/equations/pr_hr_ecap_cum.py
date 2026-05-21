"""PR-HR(Ecap,cum) allocation method (L1)."""

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

from pyaesa.asocc.data.region_group_mapping import load_region_group_mapping
from pyaesa.asocc.runtime.methods.fallback_policy import (
    resolve_latest_previous_nonzero_series,
)
from pyaesa.asocc.runtime.selection.normalize import (
    normalize_l1_reg_mode_required,
)

_Rp1FallbackCallback = Callable[[list[str], int, int], None]
_PerCapCache = dict[int, pd.DataFrame]


def _responsibility_period_years(*, rps: pd.DataFrame, impact: object) -> int | None:
    """Return one responsibility period length for one impact."""
    value = rps.loc[impact, "responsibility_period_years"]
    if bool(pd.isna(value)):
        return None
    numeric_scalar = float(value)
    return int(numeric_scalar)


def _collect_parent_cumulative_per_cap(
    *,
    impact_year: int,
    impact_df: pd.DataFrame,
    population_by_year: dict[int, pd.Series],
    lcia_reg_by_year: dict[int, pd.DataFrame],
    rps: pd.DataFrame,
    impact_parent_map: pd.Series,
    available_years: list[int],
    fallback_callback: _Rp1FallbackCallback | None = None,
) -> dict[str, pd.Series]:
    """Build cumulative per capita impacts by parent impact."""
    parent_cum: dict[str, pd.Series] = {}
    rp1_fallback_impacts: dict[tuple[int, int], set[str]] = {}
    min_year = min(available_years)
    available_set = set(available_years)
    per_cap_cache: _PerCapCache = {}
    for impact in rps.index:
        if impact not in impact_df.index:
            continue
        rp_years = _responsibility_period_years(rps=rps, impact=impact)
        if rp_years is None:
            continue
        if rp_years == 1:
            # For RP=1, the cumulative PR-HR denominator equals the requested
            # single year impact per capita. If EXIOBASE stores an all zero
            # row for that year, the row is treated as a data error placeholder
            # and the latest previous non zero year is reused instead.
            cumulative_per_cap = _resolve_rp1_per_cap_for_year(
                impact=str(impact),
                impact_year=int(impact_year),
                population_by_year=population_by_year,
                lcia_reg_by_year=lcia_reg_by_year,
                available_years=available_years,
                per_cap_cache=per_cap_cache,
                fallback_callback=lambda impacts, target_year, fallback_year: (
                    rp1_fallback_impacts.setdefault(
                        (int(target_year), int(fallback_year)),
                        set(),
                    ).update(str(value) for value in impacts)
                ),
            )
            if cumulative_per_cap is None:
                continue
        else:
            start_year = max(min_year, impact_year - rp_years + 1)
            # Responsibility window is clipped to available historical horizon.
            year_window = list(range(start_year, impact_year + 1))
            missing = [y for y in year_window if y not in available_set]
            if missing:
                raise ValueError(
                    f"Missing years in responsibility period for impact {impact}: {missing}"
                )
            per_cap_list: list[pd.Series] = []
            for year_item in year_window:
                if year_item in lcia_reg_by_year and year_item in population_by_year:
                    per_cap_list.append(
                        cast(
                            pd.Series,
                            _impact_per_cap_for_year(
                                impact=str(impact),
                                year_item=int(year_item),
                                population_by_year=population_by_year,
                                lcia_reg_by_year=lcia_reg_by_year,
                                per_cap_cache=per_cap_cache,
                            ),
                        )
                    )
            if not per_cap_list:
                continue
            cumulative_per_cap = per_cap_list[0].copy()
            for per_cap in per_cap_list[1:]:
                cumulative_per_cap = cumulative_per_cap.add(per_cap, fill_value=0.0)
        # Aggregate child impacts to parent impacts before share normalization.
        parent = impact_parent_map.get(impact)
        if parent is None or (isinstance(parent, float) and pd.isna(parent)):
            continue
        parents = (
            [str(parent)]
            if not isinstance(parent, pd.Series)
            else [str(p) for p in parent.dropna().astype(str).unique()]
        )
        for parent_key in parents:
            if parent_key in parent_cum:
                parent_cum[parent_key] = parent_cum[parent_key].add(
                    cumulative_per_cap, fill_value=0.0
                )
            else:
                parent_cum[parent_key] = cumulative_per_cap.copy()
    if fallback_callback is not None:
        for (target_year, fallback_year), impacts in sorted(rp1_fallback_impacts.items()):
            fallback_callback(
                sorted(str(value) for value in impacts),
                int(target_year),
                int(fallback_year),
            )
    return parent_cum


def _clone_parent_cum(
    parent_cum: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Return a deep enough copy for safe cache reuse."""
    return {str(parent): series.copy() for parent, series in parent_cum.items()}


def _resolve_parent_keys(
    *,
    impact_parent_map: pd.Series,
    impact: str,
) -> list[str]:
    """Return one or more parent labels for an impact."""
    parent = impact_parent_map.get(impact)
    if parent is None or (isinstance(parent, float) and pd.isna(parent)):
        return []
    if isinstance(parent, pd.Series):
        return [str(v) for v in parent.dropna().astype(str).unique()]
    return [str(parent)]


def _has_single_year_responsibility(
    rps: pd.DataFrame,
) -> bool:
    """Return whether the responsibility settings include any RP=1 impacts."""
    rp_values = pd.to_numeric(rps["responsibility_period_years"], errors="raise")
    return bool(np.any(pd.Series(rp_values, copy=False).to_numpy(dtype=np.float64) == 1.0))


def _impact_per_cap_for_year(
    *,
    impact: str,
    year_item: int,
    population_by_year: dict[int, pd.Series],
    lcia_reg_by_year: dict[int, pd.DataFrame],
    per_cap_cache: _PerCapCache,
) -> pd.Series | None:
    """Return one impact per capita vector for one year."""
    if year_item not in lcia_reg_by_year or year_item not in population_by_year:
        return None
    return _per_cap_frame_for_year(
        year_item=year_item,
        population_by_year=population_by_year,
        lcia_reg_by_year=lcia_reg_by_year,
        per_cap_cache=per_cap_cache,
    ).loc[impact]


def _per_cap_frame_for_year(
    *,
    year_item: int,
    population_by_year: dict[int, pd.Series],
    lcia_reg_by_year: dict[int, pd.DataFrame],
    per_cap_cache: _PerCapCache,
) -> pd.DataFrame:
    """Return one all impact per capita matrix for one LCIA year."""
    cached = per_cap_cache.get(int(year_item))
    if cached is not None:
        return cached
    impact_frame = lcia_reg_by_year[int(year_item)]
    if isinstance(impact_frame.index, pd.MultiIndex) and "impact" in impact_frame.index.names:
        impact_frame = cast(
            pd.DataFrame,
            impact_frame.groupby(level="impact", sort=False).sum(min_count=1),
        )
    elif impact_frame.index.has_duplicates:
        impact_frame = cast(
            pd.DataFrame,
            impact_frame.groupby(level=0, sort=False).sum(min_count=1),
        )
    numer_values = impact_frame.to_numpy(dtype=np.float64, copy=False)
    population_values = cast(
        pd.Series,
        population_by_year[int(year_item)].reindex(impact_frame.columns),
    ).to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    values = np.full(numer_values.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(numer_values) & ~np.isnan(population_values[np.newaxis, :])
    valid &= population_values[np.newaxis, :] != 0.0
    np.divide(numer_values, population_values[np.newaxis, :], out=values, where=valid)
    values[~np.isfinite(values)] = np.nan
    out = pd.DataFrame(values, index=impact_frame.index, columns=impact_frame.columns)
    per_cap_cache[int(year_item)] = out
    return out


def _is_all_zero_series(series: pd.Series) -> bool:
    """Return whether all non missing entries are exactly zero."""
    valid = series.dropna()
    return bool(not valid.empty and bool((valid == 0).all()))


def _resolve_rp1_per_cap_for_year(
    *,
    impact: str,
    impact_year: int,
    population_by_year: dict[int, pd.Series],
    lcia_reg_by_year: dict[int, pd.DataFrame],
    available_years: list[int],
    per_cap_cache: _PerCapCache,
    fallback_callback: _Rp1FallbackCallback | None,
) -> pd.Series | None:
    """Return one RP=1 per cap series with explicit zero placeholder handling.

    For RP=1, the cumulative PR-HR denominator is the single year impact per
    capita for the requested LCIA year. When that requested year is present but
    every non missing regional value is exactly zero, the zero row is treated as
    a missing data placeholder (EXIOBASE data error) and the latest previous
    non zero year is reused. If no previous non zero year exists, the all zero series
    is kept unchanged.
    """
    resolved_per_cap, fallback_resolution = resolve_latest_previous_nonzero_series(
        requested_year=int(impact_year),
        available_years=available_years,
        load_series=lambda year_item: _impact_per_cap_for_year(
            impact=impact,
            year_item=int(year_item),
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            per_cap_cache=per_cap_cache,
        ),
        is_zero_placeholder=_is_all_zero_series,
    )
    if fallback_resolution is not None and fallback_callback is not None:
        fallback_callback(
            [str(impact)],
            int(fallback_resolution.requested_year),
            int(fallback_resolution.resolved_year),
        )
    return resolved_per_cap


def _collect_parent_cumulative_per_cap_incremental(
    *,
    impact_year: int,
    impact_df: pd.DataFrame,
    population_by_year: dict[int, pd.Series],
    lcia_reg_by_year: dict[int, pd.DataFrame],
    rps: pd.DataFrame,
    impact_parent_map: pd.Series,
    available_years: list[int],
    previous_parent_cum: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Update parent cumulative per capita using previous impact year state."""
    parent_cum = _clone_parent_cum(previous_parent_cum)
    per_cap_cache: _PerCapCache = {}
    min_year = min(available_years)
    available_set = set(available_years)
    prev_year = int(impact_year) - 1
    for impact in rps.index:
        if impact not in impact_df.index:
            continue
        rp_years = _responsibility_period_years(rps=rps, impact=impact)
        if rp_years is None:
            continue
        start_year = max(min_year, impact_year - rp_years + 1)
        year_window = list(range(start_year, impact_year + 1))
        missing = [y for y in year_window if y not in available_set]
        if missing:
            raise ValueError(
                f"Missing years in responsibility period for impact {impact}: {missing}"
            )
        parent_keys = _resolve_parent_keys(
            impact_parent_map=impact_parent_map,
            impact=str(impact),
        )
        if not parent_keys:
            continue

        add_per_cap = _impact_per_cap_for_year(
            impact=str(impact),
            year_item=int(impact_year),
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            per_cap_cache=per_cap_cache,
        )
        if add_per_cap is not None:
            for parent_key in parent_keys:
                if parent_key in parent_cum:
                    parent_cum[parent_key] = parent_cum[parent_key].add(add_per_cap, fill_value=0.0)
                else:
                    parent_cum[parent_key] = add_per_cap.copy()

        prev_start = max(min_year, prev_year - rp_years + 1)
        if start_year > prev_start:
            drop_per_cap = _impact_per_cap_for_year(
                impact=str(impact),
                year_item=int(prev_start),
                population_by_year=population_by_year,
                lcia_reg_by_year=lcia_reg_by_year,
                per_cap_cache=per_cap_cache,
            )
            if drop_per_cap is not None:
                for parent_key in parent_keys:
                    if parent_key in parent_cum:
                        parent_cum[parent_key] = parent_cum[parent_key].sub(
                            drop_per_cap, fill_value=0.0
                        )
                    else:
                        parent_cum[parent_key] = drop_per_cap.mul(-1.0)
    return parent_cum


def _resolve_parent_cumulative_per_cap(
    *,
    impact_year: int,
    impact_df: pd.DataFrame,
    population_by_year: dict[int, pd.Series],
    lcia_reg_by_year: dict[int, pd.DataFrame],
    rps: pd.DataFrame,
    impact_parent_map: pd.Series,
    available_years: list[int],
    parent_cum_cache: dict[int, dict[str, pd.Series]] | None,
    fallback_callback: _Rp1FallbackCallback | None = None,
) -> dict[str, pd.Series]:
    """Return cached parent cumulative per capita or compute and cache it."""
    if parent_cum_cache is None:
        return _collect_parent_cumulative_per_cap(
            impact_year=impact_year,
            impact_df=impact_df,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps=rps,
            impact_parent_map=impact_parent_map,
            available_years=available_years,
            fallback_callback=fallback_callback,
        )
    cached = parent_cum_cache.get(int(impact_year))
    if cached is not None:
        return _clone_parent_cum(cached)
    if _has_single_year_responsibility(rps):
        # RP=1 is rebuilt directly per year because its cumulative denominator
        # is defined by the requested single year impact per capita, including
        # the explicit fallback to the latest previous non zero year when an
        # EXIOBASE all zero row is treated as a missing data placeholder.
        parent_cum = _collect_parent_cumulative_per_cap(
            impact_year=impact_year,
            impact_df=impact_df,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps=rps,
            impact_parent_map=impact_parent_map,
            available_years=available_years,
            fallback_callback=fallback_callback,
        )
        parent_cum_cache[int(impact_year)] = _clone_parent_cum(parent_cum)
        return _clone_parent_cum(parent_cum)
    prev_cached = parent_cum_cache.get(int(impact_year) - 1)
    if prev_cached is not None:
        parent_cum = _collect_parent_cumulative_per_cap_incremental(
            impact_year=impact_year,
            impact_df=impact_df,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps=rps,
            impact_parent_map=impact_parent_map,
            available_years=available_years,
            previous_parent_cum=prev_cached,
        )
    else:
        parent_cum = _collect_parent_cumulative_per_cap(
            impact_year=impact_year,
            impact_df=impact_df,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps=rps,
            impact_parent_map=impact_parent_map,
            available_years=available_years,
            fallback_callback=fallback_callback,
        )
    parent_cum_cache[int(impact_year)] = _clone_parent_cum(parent_cum)
    return _clone_parent_cum(parent_cum)


def build_parent_cumulative_per_cap(
    *,
    impact_year: int,
    population_by_year: dict[int, pd.Series],
    lcia_reg_by_year: dict[int, pd.DataFrame],
    rps_df: pd.DataFrame,
    impact_parent_map: pd.Series,
    available_years: list[int],
    parent_cum_cache: dict[int, dict[str, pd.Series]] | None = None,
    fallback_callback: _Rp1FallbackCallback | None = None,
) -> dict[str, pd.Series]:
    """Build (or reuse) parent cumulative per capita impacts for one year."""
    effective_year = int(impact_year)
    if effective_year not in lcia_reg_by_year:
        raise ValueError(f"LCIA data missing for year {effective_year}.")
    impact_df = lcia_reg_by_year[effective_year]
    rps = rps_df.copy()
    if "responsibility_period_years" not in rps.columns:
        raise ValueError("RPS file missing required responsibility_period_years column.")
    rps = rps.set_index("impact")
    return _resolve_parent_cumulative_per_cap(
        impact_year=effective_year,
        impact_df=impact_df,
        population_by_year=population_by_year,
        lcia_reg_by_year=lcia_reg_by_year,
        rps=rps,
        impact_parent_map=impact_parent_map,
        available_years=available_years,
        parent_cum_cache=parent_cum_cache,
        fallback_callback=fallback_callback,
    )


def _build_parent_share_frame(
    *,
    year: int,
    population: pd.Series,
    parent_cum: dict[str, pd.Series],
    region_label: str,
) -> pd.DataFrame:
    """Convert parent cumulative per capita values to PR-HR shares."""
    parent_keys = [str(parent_key) for parent_key in parent_cum]
    region_index = population.index
    cumulative = pd.DataFrame(parent_cum, index=region_index)
    population_values = population.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    cumulative_values = cumulative.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    inverse_values = np.full(cumulative_values.shape, np.nan, dtype=np.float64)
    valid = (
        ~np.isnan(population_values[:, np.newaxis])
        & ~np.isnan(cumulative_values)
        & (cumulative_values != 0.0)
    )
    np.divide(
        population_values[:, np.newaxis],
        cumulative_values,
        out=inverse_values,
        where=valid,
    )
    inverse_values[~np.isfinite(inverse_values)] = np.nan
    valid_inverse = ~np.isnan(inverse_values)
    totals = np.nansum(inverse_values, axis=0)
    share_values = np.full(inverse_values.shape, np.nan, dtype=np.float64)
    non_zero_total = totals != 0.0
    np.divide(
        inverse_values,
        totals[np.newaxis, :],
        out=share_values,
        where=valid_inverse & non_zero_total[np.newaxis, :],
    )
    zero_total = (totals == 0.0) & valid_inverse.any(axis=0)
    share_values[:, zero_total] = np.where(valid_inverse[:, zero_total], 0.0, np.nan)
    parent_count = len(parent_keys)
    region_count = len(region_index)
    index = pd.MultiIndex(
        levels=[pd.Index(parent_keys, name="impact"), pd.Index(region_index, name=region_label)],
        codes=[
            np.repeat(np.arange(parent_count, dtype=np.intp), region_count),
            np.tile(np.arange(region_count, dtype=np.intp), parent_count),
        ],
        names=["impact", region_label],
        verify_integrity=False,
    )
    return pd.DataFrame(
        {int(year): share_values.T.reshape(parent_count * region_count)},
        index=index,
    )


def _apply_post_grouping(
    *,
    frame: pd.DataFrame,
    source_key: str,
    group_version: str,
    region_label: str,
) -> pd.DataFrame:
    """Apply post aggregation grouping by MRIO mapping and sum duplicates."""
    mapping = load_region_group_mapping(
        source_key=source_key,
        group_version=group_version,
    )
    region_vals = frame.index.get_level_values(region_label)
    regions = region_vals.map(lambda code: mapping.get(code, code))
    out = frame.copy()
    out.index = pd.MultiIndex.from_arrays(
        [out.index.get_level_values("impact"), regions],
        names=["impact", region_label],
    )
    return pd.DataFrame(
        out.groupby(level=["impact", region_label]).sum(min_count=1),
        copy=False,
    )


def compute_pr_hr(
    *,
    year: int,
    impact_year: int | None = None,
    population: pd.Series,
    population_by_year: dict[int, pd.Series],
    lcia_reg_by_year: dict[int, pd.DataFrame],
    rps_df: pd.DataFrame,
    impact_parent_map: pd.Series,
    available_years: list[int],
    source_key: str,
    group_version: str | None,
    aggregation_mode: str,
    region_label: str = "region",
    parent_cum_cache: dict[int, dict[str, pd.Series]] | None = None,
    fallback_callback: _Rp1FallbackCallback | None = None,
) -> pd.DataFrame:
    """Compute PR-HR(Ecap,cum) shares for a year.

    Args:
        year: Year of computation.
        impact_year: Optional LCIA year used for cumulative impacts. When
            omitted, uses ``year``.
        population: Population series by region (studied year).
        population_by_year: Population series by year.
        lcia_reg_by_year: LCIA time series by year.
        rps_df: Responsibility period settings per impact.
        impact_parent_map: Mapping from impact child to parent.
        available_years: MRIO years available for responsibility windows.

    Returns:
        DataFrame indexed by (impact, region).
    """
    effective_year = int(impact_year) if impact_year is not None else int(year)
    parent_cum = build_parent_cumulative_per_cap(
        impact_year=effective_year,
        population_by_year=population_by_year,
        lcia_reg_by_year=lcia_reg_by_year,
        rps_df=rps_df,
        impact_parent_map=impact_parent_map,
        available_years=available_years,
        parent_cum_cache=parent_cum_cache,
        fallback_callback=fallback_callback,
    )
    if not parent_cum:
        return pd.DataFrame()
    out = _build_parent_share_frame(
        year=year,
        population=population,
        parent_cum=parent_cum,
        region_label=region_label,
    )
    mode = normalize_l1_reg_mode_required(aggregation_mode)
    if mode == "post" and group_version:
        out = _apply_post_grouping(
            frame=out,
            source_key=source_key,
            group_version=group_version,
            region_label=region_label,
        )
    return out

"""Scope selection for enacting metric LCIA-derived metrics."""

from dataclasses import dataclass

import pandas as pd

from ....io.metadata import RunContext, RunState


@dataclass(frozen=True)
class _DirectPerCapScope:
    lcia_method: str
    effective_year: int
    lcia_data: dict


@dataclass(frozen=True)
class _PrHrCumulativeScope:
    lcia_method: str
    lcia_kind: str
    effective_year: int
    lcia_reg_by_year: dict[int, pd.DataFrame]
    available_years: list[int]
    rps_df: pd.DataFrame
    impact_parent_map: pd.Series


def _resolve_enacting_metric_output_scenario(
    *,
    context: RunContext,
    year: int,
    ssp_scenario: str | None,
) -> str | None:
    """Resolve enacting metric output scenario key for one year."""
    return None if str(int(year)) in context.wb_df.columns else ssp_scenario


def _resolve_requested_effective_year(
    *,
    lcia_method: str,
    year: int,
    lcia_effective_year_by_method: dict[str, int] | None,
) -> int:
    """Resolve requested LCIA effective year for one method."""
    if lcia_effective_year_by_method is None:
        return int(year)
    return int(lcia_effective_year_by_method.get(lcia_method, year))


def _iter_direct_percap_scopes(
    *,
    year: int,
    lcia_by_method: dict[str, dict] | None,
    lcia_effective_year_by_method: dict[str, int] | None,
) -> list[_DirectPerCapScope]:
    """Return direct LCIA per-cap scopes for methods still active at the run year."""
    if not lcia_by_method:
        return []
    scopes: list[_DirectPerCapScope] = []
    for lcia_method, lcia_data in lcia_by_method.items():
        effective_year = _resolve_requested_effective_year(
            lcia_method=str(lcia_method),
            year=int(year),
            lcia_effective_year_by_method=lcia_effective_year_by_method,
        )
        if effective_year != int(year):
            continue
        scopes.append(
            _DirectPerCapScope(
                lcia_method=str(lcia_method),
                effective_year=effective_year,
                lcia_data=lcia_data,
            )
        )
    return scopes


def _resolve_pr_hr_base_inputs(
    *,
    context: RunContext,
    state: RunState,
    ssp_scenario: str | None,
    use_original_domain: bool,
    lcia_by_method: dict[str, dict] | None,
) -> tuple[dict, dict[int, pd.Series], list[str]]:
    """Resolve shared PR-HR cumulative base inputs for enacting metric LCIA outputs."""
    lcia_store = state.lcia_timeseries_original if use_original_domain else state.lcia_timeseries
    population_by_year = (
        state.pr_post_pop_series_by_ssp_scenario.get(ssp_scenario, {})
        if use_original_domain
        else state.pop_series_by_ssp_scenario.get(ssp_scenario, {})
    )
    if lcia_by_method:
        lcia_methods_in_scope = [str(lcia_method) for lcia_method in lcia_by_method.keys()]
    else:
        lcia_methods_in_scope = [str(lcia_method) for lcia_method in (context.lcia_methods or [])]
    return lcia_store, population_by_year, lcia_methods_in_scope


def _iter_pr_hr_cumulative_scopes(
    *,
    state: RunState,
    year: int,
    lcia_kinds: set[str],
    lcia_store: dict,
    lcia_methods_in_scope: list[str],
    lcia_effective_year_by_method: dict[str, int] | None,
) -> list[_PrHrCumulativeScope]:
    """Return PR-HR cumulative scopes for methods/kinds with usable LCIA history."""
    scopes: list[_PrHrCumulativeScope] = []
    for lcia_method in lcia_methods_in_scope:
        requested_effective_year = _resolve_requested_effective_year(
            lcia_method=str(lcia_method),
            year=int(year),
            lcia_effective_year_by_method=lcia_effective_year_by_method,
        )
        if requested_effective_year != int(year):
            continue
        method_timeseries = lcia_store.get(lcia_method, {})
        rps_df = state.rps_by_method.get(lcia_method)
        impact_parent_map = state.cf_by_method.get(lcia_method)
        if rps_df is None or impact_parent_map is None:
            continue
        for lcia_kind in lcia_kinds:
            lcia_reg_by_year = method_timeseries.get(lcia_kind)
            if not lcia_reg_by_year:
                continue
            available_years = sorted(int(y) for y in lcia_reg_by_year.keys())
            effective_years = [y for y in available_years if y <= int(year)]
            if not effective_years:
                continue
            scopes.append(
                _PrHrCumulativeScope(
                    lcia_method=str(lcia_method),
                    lcia_kind=str(lcia_kind),
                    effective_year=max(effective_years),
                    lcia_reg_by_year=lcia_reg_by_year,
                    available_years=available_years,
                    rps_df=rps_df,
                    impact_parent_map=impact_parent_map,
                )
            )
    return scopes

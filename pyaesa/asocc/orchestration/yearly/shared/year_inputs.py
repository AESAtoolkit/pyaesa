"""Shared per year/per scenario input loading for orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, cast, overload

import pandas as pd

from pyaesa.download.pop_gdp.contracts import (
    GDP_SSP_INDICATOR,
    POP_SSP_INDICATOR,
)
from pyaesa.download.pop_gdp.contracts import (
    GDP_WB_INDICATOR,
    POP_WB_INDICATOR,
)
from pyaesa.process.mrios.utils.aggregation.aggregation import read_agg_map
from pyaesa.process.mrios.utils.io.paths import _get_agg_map_path

from ...method_scope import _unique_l2_methods_in_scope
from ....data.load_mrio import (
    _load_enacting_metric_l1_metric,
    _load_enacting_metric_l2_metric,
    _load_utility_metric,
)
from ....data.load_pop_gdp import _get_pr_iso3_inputs, _get_series_for_year
from ....io.metadata import RunContext, RunState
from ....methods.registry.registry import REGISTRY
from ..l2.l2_types import _L2ComputeInputs


@dataclass(frozen=True)
class _MrioPayload:
    enacting_metric_l1: dict[str, pd.Series]
    enacting_metric_l2: dict[str, pd.DataFrame | pd.Series]
    utility: dict[str, pd.DataFrame]
    l2_inputs: _L2ComputeInputs


@dataclass(frozen=True)
class _ScenarioInputs:
    pop_series: pd.Series
    gdp_series: pd.Series
    pop_series_original: pd.Series | None
    pop_iso: pd.Series
    gdp_iso: pd.Series
    iso_to_mrio: pd.Series
    use_ssp: bool


class _ScenarioRunContext(NamedTuple):
    context: RunContext
    state: RunState
    year: int
    ssp_scenario: str | None


@dataclass(frozen=True)
class _PopGdpSource:
    pop_df: pd.DataFrame
    gdp_df: pd.DataFrame
    pr_df: pd.DataFrame
    pop_var: str
    gdp_var: str
    region_override: str | None
    agg_version_reg: str | None
    scenario_arg: str | None
    use_ssp: bool
    needs_pr_post_unaggregated: bool


def build_l2_compute_inputs(
    *,
    enacting_metric_l1: dict[str, pd.Series],
    enacting_metric_l2: dict[str, pd.DataFrame | pd.Series],
    utility: dict[str, pd.DataFrame],
) -> _L2ComputeInputs:
    """Build the typed L2 compute payload owned by MRIO payload construction."""
    return _L2ComputeInputs(
        fd_rf=enacting_metric_l1["fd_rf"],
        gva_rp=enacting_metric_l1["gva_rp"],
        fd_rp_sp_rf=cast(pd.DataFrame, enacting_metric_l2["fd_rp_sp_rf"]),
        fd_rp_sp=cast(pd.Series, enacting_metric_l2["fd_rp_sp"]),
        fd_rf_sp=cast(pd.Series, enacting_metric_l2["fd_rf_sp"]),
        gva_rp_sp=cast(pd.Series, enacting_metric_l2["gva_rp_sp"]),
        x_to_rc=utility["x_to_rc"],
        kappa=utility.get("kappa", pd.DataFrame()),
        omega_reg=utility.get("omega_reg", pd.DataFrame()),
    )


def _load_reg_agg_map(*, context: RunContext, state: RunState) -> dict[str, str]:
    """Load optional regional aggregate map."""
    if not context.agg_version_reg:
        return {}
    cache_key = (str(context.source), context.agg_version_reg)
    cached = state.reg_agg_map_cache.get(cache_key)
    if cached is not None:
        return cached
    map_path = _get_agg_map_path(
        context.source,
        kind="reg",
        agg_version=context.agg_version_reg,
    )
    map_df = read_agg_map(map_path)
    mapping = dict(zip(map_df["original_classification"], map_df["aggregated_mrio"]))
    state.reg_agg_map_cache[cache_key] = mapping
    return mapping


def _required_mrio_metric_keys(*, context: RunContext) -> tuple[set[str], set[str], set[str]]:
    """Return required MRIO metric keys for this branch."""
    l1_keys: set[str] = set()
    l2_keys: set[str] = set()
    util_keys: set[str] = set()
    selected_l2 = _unique_l2_methods_in_scope(
        selected_l2_one_step=context.selected_l2_one_step,
        combined=context.combined,
    )
    for l2_method in selected_l2:
        family = REGISTRY.method_family(l2_method, level="L2", fu_code=context.fu_code)
        # Enacting output requirements (registry defined).
        base_keys = set(REGISTRY.l2_base_enacting_metrics(l2_method, fu_code=context.fu_code))
        l1_keys.update(base_keys.intersection({"fd_rf", "gva_rp"}))
        l2_keys.update(base_keys.intersection({"fd_rp_sp_rf", "fd_rp_sp", "fd_rf_sp", "gva_rp_sp"}))
        if {"x_rp_sp", "x_rp_sp_rc", "x_rc_sp"}.intersection(base_keys):
            util_keys.add("x_to_rc")
        # Compute path minimums by family.
        if family == "UT_FD":
            l1_keys.add("fd_rf")
            l2_keys.update({"fd_rp_sp_rf", "fd_rp_sp", "fd_rf_sp"})
        elif family == "UT_FDA":
            l1_keys.add("fd_rf")
            util_keys.update({"x_to_rc", "kappa"})
        elif family == "UT_GVAA":
            l1_keys.add("gva_rp")
            util_keys.update({"x_to_rc", "omega_reg"})
            # L2.a.b identity closure compares UT(GVAa) against UT(GVA) using
            # the same L1 weights, which requires producer level GVA at level 2.
            if context.fu_code == "L2.a.b":
                l2_keys.add("gva_rp_sp")
        elif family == "UT_GVA":
            l1_keys.add("gva_rp")
            l2_keys.add("gva_rp_sp")
        elif family == "UT_TD":
            l1_keys.add("fd_rf")
            util_keys.add("x_to_rc")
    return l1_keys, l2_keys, util_keys


@overload
def _load_year_mrio_payloads_required(
    *,
    saved_dir: Path,
    context: RunContext,
    needs_mrio: Literal[True],
) -> _MrioPayload: ...


@overload
def _load_year_mrio_payloads_required(
    *,
    saved_dir: Path,
    context: RunContext,
    needs_mrio: Literal[False],
) -> None: ...


def _load_year_mrio_payloads_required(
    *,
    saved_dir: Path,
    context: RunContext,
    needs_mrio: bool,
) -> _MrioPayload | None:
    """Load only MRIO enacting metrics required for this branch/method selection."""
    if not needs_mrio:
        return None
    req_l1, req_l2, req_util = _required_mrio_metric_keys(context=context)
    enacting_metric_l1: dict[str, pd.Series] = {
        "fd_rf": pd.Series(dtype=float),
        "gva_rp": pd.Series(dtype=float),
    }
    enacting_metric_l2: dict[str, pd.DataFrame | pd.Series] = {
        "fd_rp_sp_rf": pd.DataFrame(),
        "fd_rp_sp": pd.Series(dtype=float),
        "fd_rf_sp": pd.Series(dtype=float),
        "gva_rp_sp": pd.Series(dtype=float),
    }
    utility: dict[str, pd.DataFrame] = {
        "x_to_rc": pd.DataFrame(),
        "kappa": pd.DataFrame(),
        "omega_reg": pd.DataFrame(),
    }
    for key in sorted(req_l1):
        enacting_metric_l1[key] = _load_enacting_metric_l1_metric(saved_dir, key)
    for key in sorted(req_l2):
        enacting_metric_l2[key] = _load_enacting_metric_l2_metric(saved_dir, key)
    for key in sorted(req_util):
        utility[key] = _load_utility_metric(saved_dir, key)
    return _MrioPayload(
        enacting_metric_l1=enacting_metric_l1,
        enacting_metric_l2=enacting_metric_l2,
        utility=utility,
        l2_inputs=build_l2_compute_inputs(
            enacting_metric_l1=enacting_metric_l1,
            enacting_metric_l2=enacting_metric_l2,
            utility=utility,
        ),
    )


def _resolve_pop_gdp_source(run_ctx: _ScenarioRunContext) -> _PopGdpSource:
    """Resolve pop/gdp input tables and indicators for one scenario/year."""
    context = run_ctx.context
    year_col = str(int(run_ctx.year))
    use_ssp = year_col not in context.wb_df.columns
    base_df = context.ssp_df if use_ssp else context.wb_df
    raw_df = context.ssp_df_raw if use_ssp else context.wb_df_raw
    selected_df = raw_df if context.l1_only_no_mrio else base_df
    region_override = "iso3_code" if context.l1_only_no_mrio else None
    pr_mode = context.l1_reg_aggreg
    needs_pr_post_unaggregated = (
        context.agg_version_reg is not None
        and pr_mode == "post"
        and context.use_original_l1_post_domain
    )
    return _PopGdpSource(
        pop_df=selected_df,
        gdp_df=selected_df,
        pr_df=raw_df,
        pop_var=POP_SSP_INDICATOR if use_ssp else POP_WB_INDICATOR,
        gdp_var=GDP_SSP_INDICATOR if use_ssp else GDP_WB_INDICATOR,
        region_override=region_override,
        agg_version_reg=(None if context.l1_only_no_mrio else context.agg_version_reg),
        scenario_arg=run_ctx.ssp_scenario if use_ssp else None,
        use_ssp=use_ssp,
        needs_pr_post_unaggregated=needs_pr_post_unaggregated,
    )


def _load_scenario_population_gdp(
    *,
    run_ctx: _ScenarioRunContext,
) -> _ScenarioInputs:
    """Load population/GDP series and PR ISO inputs for one scenario/year."""
    context = run_ctx.context
    state = run_ctx.state
    year = run_ctx.year
    source = _resolve_pop_gdp_source(run_ctx)

    pop_series = _get_series_for_year(
        df=source.pop_df,
        variable=source.pop_var,
        year=year,
        source_key=context.source,
        agg_version=source.agg_version_reg,
        ssp_scenario=source.scenario_arg,
        region_col_override=source.region_override,
    )
    gdp_series = _get_series_for_year(
        df=source.gdp_df,
        variable=source.gdp_var,
        year=year,
        source_key=context.source,
        agg_version=source.agg_version_reg,
        ssp_scenario=source.scenario_arg,
        region_col_override=source.region_override,
    )
    state.pop_series_by_ssp_scenario.setdefault(run_ctx.ssp_scenario, {})[year] = pop_series
    state.gdp_series_by_ssp_scenario.setdefault(run_ctx.ssp_scenario, {})[year] = gdp_series
    pop_series_original: pd.Series | None = None
    if source.needs_pr_post_unaggregated:
        pop_series_unaggregated = _get_series_for_year(
            df=source.pop_df,
            variable=source.pop_var,
            year=year,
            source_key=context.source,
            agg_version=None,
            ssp_scenario=source.scenario_arg,
            region_col_override=source.region_override,
        )
        state.pr_post_pop_series_by_ssp_scenario.setdefault(run_ctx.ssp_scenario, {})[year] = (
            pop_series_unaggregated
        )
        pop_series_original = pop_series_unaggregated

    pop_iso, gdp_iso, iso_to_mrio = _get_pr_iso3_inputs(
        df=source.pr_df,
        year=year,
        source_key=context.source,
        gdp_variable=source.gdp_var,
        pop_variable=source.pop_var,
        ssp_scenario=source.scenario_arg,
        region_col_override=source.region_override,
    )
    return _ScenarioInputs(
        pop_series=pop_series,
        gdp_series=gdp_series,
        pop_series_original=pop_series_original,
        pop_iso=pop_iso,
        gdp_iso=gdp_iso,
        iso_to_mrio=iso_to_mrio,
        use_ssp=source.use_ssp,
    )

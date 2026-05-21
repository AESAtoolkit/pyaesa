"""Base and LCIA enacting metric recording."""

from ...method_scope import _unique_l2_methods_in_scope
from ....data.load_mrio import _metric_to_series
from ....io.metadata import EnactingMetricKey, RunContext, RunState
from ....methods.run_ut import _compute_ut_l2_preweight, _ut_preweight_cache_key
from ....methods.lcia_key_selection import required_lcia_metric_keys_for_context
from ....methods.registry.registry import REGISTRY
from ..l2.l2_slicing import _slice_l2_inputs_for_compute
from ..shared.year_inputs import build_l2_compute_inputs
from .enacting_metric_common import (
    _record_enacting_metric_input,
    _slice_enacting_metric_payload_for_run,
    _store_enacting_metric_input,
)


def record_base_enacting_metrics(
    *,
    context: RunContext,
    state: RunState,
    year: int,
    ssp_scenario: str | None,
    enacting_metric_l1: dict,
    enacting_metric_l2: dict,
    utility: dict,
) -> None:
    """Record non LCIA MRIO enacting metrics required by selected L2 methods."""
    required = _required_base_enacting_metric_keys(context=context, year=year)
    if not required:
        return
    scenario_key = None if str(int(year)) in context.wb_df.columns else ssp_scenario
    x_to_rc = utility["x_to_rc"]
    payload_getters = {
        "fd_rf": lambda: enacting_metric_l1["fd_rf"],
        "gva_rp": lambda: enacting_metric_l1["gva_rp"],
        "fd_rp_sp_rf": lambda: enacting_metric_l2["fd_rp_sp_rf"],
        "fd_rp_sp": lambda: enacting_metric_l2["fd_rp_sp"],
        "fd_rf_sp": lambda: enacting_metric_l2["fd_rf_sp"],
        "gva_rp_sp": lambda: enacting_metric_l2["gva_rp_sp"],
        "x_rp_sp": lambda: x_to_rc.sum(axis=1),
        "x_rp_sp_rc": lambda: x_to_rc.stack(future_stack=True),
        "x_rc_sp": lambda: x_to_rc.groupby(level="s_p").sum(min_count=1).T.stack(future_stack=True),
    }
    for key in sorted(required):
        payload = payload_getters[key]()
        series = _metric_to_series(key, payload)
        level = (
            "level_2"
            if key
            in {
                "fd_rp_sp_rf",
                "fd_rp_sp",
                "fd_rf_sp",
                "gva_rp_sp",
                "x_rp_sp",
                "x_rp_sp_rc",
                "x_rc_sp",
            }
            else "level_1"
        )
        _record_enacting_metric_input(
            context=context,
            state=state,
            key=EnactingMetricKey(metric=key, ssp_scenario=scenario_key),
            year=year,
            series=series,
            level=level,
        )


def _required_base_enacting_metric_keys(*, context: RunContext, year: int) -> set[str]:
    """Return non LCIA enacting metric keys required by selected L2 methods."""
    l2_selected = set(
        _unique_l2_methods_in_scope(
            selected_l2_one_step=context.selected_l2_one_step,
            combined=context.combined,
        )
    )
    projection_context = context.projection_context
    if (
        projection_context is not None
        and projection_context.enabled
        and projection_context.is_future_year(int(year))
    ):
        # For future years, keep only methods that consume base projected
        # MRIO enacting metrics in this branch.
        # Historical reuse routes read their own persisted reuse outputs.
        l2_selected = {
            name
            for name in l2_selected
            if projection_context.route_for_l2_method(name) != "historical_reuse"
        }
    if not l2_selected:
        return set()
    # Registry input requirements keep UT family metric rules centralized.
    required: set[str] = set()
    for name in l2_selected:
        required.update(REGISTRY.l2_base_enacting_metrics(name, fu_code=context.fu_code))
    return required


def record_lcia_enacting_metrics(
    *,
    context: RunContext,
    state: RunState,
    year: int,
    lcia_by_method: dict[str, dict] | None,
    lcia_effective_year_by_method: dict[str, int] | None = None,
) -> None:
    """Record LCIA based enacting metrics required by selected methods."""
    if not lcia_by_method:
        return
    l1_keys, l2_keys = _required_lcia_enacting_metric_keys(context=context)
    required_keys = l1_keys | l2_keys
    if not required_keys:
        return
    for lcia_method, lcia_data in lcia_by_method.items():
        effective_year = (
            int(lcia_effective_year_by_method.get(lcia_method, year))
            if lcia_effective_year_by_method is not None
            else int(year)
        )
        # LCIA enacting metrics are persisted only for years where LCIA is
        # available for the method. Future years that reuse frozen LCIA
        # values do not add duplicate columns.
        if effective_year != int(year):
            continue
        for key in required_keys:
            if key not in lcia_data:
                continue
            sliced_payload = _slice_enacting_metric_payload_for_run(
                context=context,
                payload=lcia_data[key],
            )
            series = _metric_to_series(key, sliced_payload)
            level = "level_2" if key in l2_keys else "level_1"
            _store_enacting_metric_input(
                state=state,
                key=EnactingMetricKey(metric=key, lcia_method=lcia_method),
                year=effective_year,
                series=series,
                level=level,
            )


def _required_lcia_enacting_metric_keys(*, context: RunContext) -> tuple[set[str], set[str]]:
    """Return LCIA enacting metric keys needed by selected methods for this run."""
    return required_lcia_metric_keys_for_context(
        context=context,
        registry=REGISTRY,
    )


def record_adjusted_ut_preweights(
    *,
    context: RunContext,
    state: RunState,
    year: int,
    enacting_metric_l1: dict,
    enacting_metric_l2: dict,
    utility: dict,
) -> None:
    """Record derived preweighted UT MRIO enacting metrics once per year."""
    inputs = build_l2_compute_inputs(
        enacting_metric_l1=enacting_metric_l1,
        enacting_metric_l2=enacting_metric_l2,
        utility=utility,
    )
    inputs = _slice_l2_inputs_for_compute(context=context, inputs=inputs)
    l2_combined_methods = {l2_method for l2_method, _ in context.combined}
    for l2_method in ("UT(FDa)", "UT(GVAa)"):
        if l2_method not in l2_combined_methods:
            continue
        pre = _compute_ut_l2_preweight(
            context=context,
            l2_method=l2_method,
            year=year,
            enacting_metric_l1={"fd_rf": inputs.fd_rf, "gva_rp": inputs.gva_rp},
            enacting_metric_l2={
                "fd_rp_sp_rf": inputs.fd_rp_sp_rf,
                "fd_rp_sp": inputs.fd_rp_sp,
                "fd_rf_sp": inputs.fd_rf_sp,
                "gva_rp_sp": inputs.gva_rp_sp,
            },
            utility={
                "x_to_rc": inputs.x_to_rc,
                "kappa": inputs.kappa,
                "omega_reg": inputs.omega_reg,
            },
        )
        cache_key = _ut_preweight_cache_key(
            l2_method=l2_method,
            fu_code=context.fu_code,
            year=year,
        )
        for scenario_cache in state.preweight_cache_by_ssp_scenario.values():
            scenario_cache.setdefault(cache_key, pre)
        series = pre.iloc[:, 0]
        suffix = "_".join(str(n) for n in series.index.names if n)
        base = "fda" if l2_method == "UT(FDa)" else "gvaa"
        key = f"{base}_{suffix}"
        _record_enacting_metric_input(
            context=context,
            state=state,
            key=EnactingMetricKey(metric=key),
            year=year,
            series=series,
            level="level_2",
        )

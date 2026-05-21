from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from pyaesa.asocc.io.metadata import RunState
from pyaesa.asocc.methods.lcia_key_selection import required_lcia_metric_keys_for_context
from pyaesa.asocc.orchestration.setup.loading.loading import _load_source_tables
from pyaesa.asocc.orchestration.yearly.l1.l1_compute import (
    _compute_l1_for_year,
    _compute_l1_non_lcia_method,
)
from pyaesa.asocc.orchestration.yearly.l1.l1_lcia_compute import _compute_l1_lcia_method
from pyaesa.asocc.orchestration.yearly.l1.l1_lcia_ar import _compute_l1_ar_lcia_method
from pyaesa.asocc.orchestration.yearly.l1.l1_lcia_inputs import _iter_lcia_method_inputs
from pyaesa.asocc.orchestration.yearly.l1.l1_lcia_reference_policy import (
    emit_pr_hr_lcia_freeze_notice_if_needed,
    resolve_reference_years_for_ar,
)
from pyaesa.asocc.orchestration.yearly.l1.l1_lcia_standard import (
    _compute_l1_standard_lcia_method,
)
from pyaesa.asocc.orchestration.yearly.l1.l1_population_inputs import (
    _load_ar_reference_population,
    _load_population_for_year,
    _resolve_l1_population_inputs,
)
from pyaesa.asocc.orchestration.yearly.l1.l1_pr_hr_setup import (
    _prepare_pr_hr_parent_cumulative_runtime,
    _prepare_pr_hr_standard_inputs,
)
from pyaesa.asocc.orchestration.yearly.l1.l1_slicing import (
    _l2_axes_for_l1_method,
    _single_axis_or_none,
    _slice_l1_frame_for_compute,
)
from pyaesa.asocc.orchestration.yearly.l1.l1_store import (
    _build_l1_output_spec,
    _store_l1_frame,
)
from pyaesa.asocc.orchestration.yearly.l1.l1_types import (
    _L1RunContext,
    _LciaMethodInputs,
    _L1StorePayload,
)


class _RecorderLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str) -> None:
        self.messages.append(str(message))


def _context(*, source: str = "oecd_v2025", logger: Any | None = None, **overrides: Any) -> Any:
    wb_df, ssp_df, wb_df_raw, ssp_df_raw = _load_source_tables(source=source)
    payload: dict[str, Any] = {
        "source": source,
        "group_version": None,
        "group_version_reg": None,
        "wb_df": wb_df,
        "ssp_df": ssp_df,
        "wb_df_raw": wb_df_raw,
        "ssp_df_raw": ssp_df_raw,
        "historical_years": [2005, 2006],
        "resolved_years": [2005, 2006, 2030],
        "persisted_years": [2005, 2006, 2030],
        "reference_years": [2005],
        "selected_l1": [],
        "selected_l2_one_step": [],
        "combined": [],
        "lcia_methods": ["gwp100_lcia"],
        "needs_lcia": True,
        "use_original_l1_post_domain": False,
        "fu_code": "L2.a.a",
        "filters": {
            "r_p": None,
            "s_p": None,
            "r_c": None,
            "r_f": None,
            "r_u": None,
        },
        "l1_reg_aggreg": "post",
        "aggreg_indices": False,
        "l1_only_no_mrio": False,
        "logger": _RecorderLogger() if logger is None else logger,
    }
    payload.update(overrides)
    return cast(Any, SimpleNamespace(**payload))


def _run(
    *,
    context: Any,
    state: RunState | None = None,
    year: int = 2030,
    scenario: str | None = "SSP2",
    pop_series: pd.Series | None = None,
    pop_series_original: pd.Series | None = None,
    pr_pop: pd.Series | None = None,
    pr_gdp: pd.Series | None = None,
    pr_to_mrio: pd.Series | None = None,
) -> _L1RunContext:
    active_state = RunState() if state is None else state
    current_pop = pd.Series(dtype=float) if pop_series is None else pop_series
    return _L1RunContext(
        context=context,
        state=active_state,
        year=year,
        ssp_scenario=scenario,
        pop_series=current_pop,
        pop_series_original=pop_series_original,
        pr_pop=pr_pop,
        pr_gdp=pr_gdp,
        pr_to_mrio=pr_to_mrio,
        l1_results_year={},
    )


def test_l1_population_inputs_cover_grouped_original_and_iso3_contracts(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    grouped_context = _context(group_version_reg="demo_reg")

    population_hist = _load_population_for_year(
        context=grouped_context,
        year=2005,
        ssp_scenario="SSP2",
        group_version_reg="demo_reg",
    )
    population_future = _load_population_for_year(
        context=grouped_context,
        year=2030,
        ssp_scenario="SSP2",
        group_version_reg="demo_reg",
    )
    population_future_original = _load_population_for_year(
        context=grouped_context,
        year=2030,
        ssp_scenario="SSP2",
        group_version_reg=None,
    )
    population_hist_original = _load_population_for_year(
        context=grouped_context,
        year=2005,
        ssp_scenario="SSP2",
        group_version_reg=None,
    )

    assert population_hist.to_dict() == {"EU": 10.0, "NAM": 20.0}
    assert population_future.to_dict() == {"EU": 35.0, "NAM": 45.0}

    state = RunState(
        pop_series_by_ssp_scenario={"SSP2": {2030: population_future}},
        pr_post_pop_series_by_ssp_scenario={
            "SSP2": {
                2005: population_hist_original,
                2030: population_future_original,
            }
        },
    )
    run = _run(
        context=grouped_context,
        state=state,
        pop_series=population_future,
        pop_series_original=population_future_original,
    )

    grouped_series, grouped_by_year = _resolve_l1_population_inputs(
        run=run,
        use_original_domain=False,
    )
    original_series, original_by_year = _resolve_l1_population_inputs(
        run=run,
        use_original_domain=True,
    )
    reference_population = _load_ar_reference_population(
        run=run,
        ref_year=2005,
        use_original_domain=True,
    )
    cached_reference_population = _load_ar_reference_population(
        run=run,
        ref_year=2005,
        use_original_domain=True,
    )

    assert grouped_series.to_dict() == {"EU": 35.0, "NAM": 45.0}
    assert list(grouped_by_year) == [2030]
    assert original_series.to_dict() == {"FR": 35.0, "US": 45.0}
    assert sorted(original_by_year) == [2005, 2006, 2030]
    assert reference_population.to_dict() == {"FR": 10.0, "US": 20.0}
    assert cached_reference_population.equals(reference_population)
    assert 2005 in state.pr_post_pop_series_by_ssp_scenario[None]

    iso3_context = _context(source="iso3", l1_only_no_mrio=True)
    assert _load_population_for_year(
        context=iso3_context,
        year=2005,
        ssp_scenario="SSP2",
        group_version_reg=None,
    ).to_dict() == {"FRA": 10.0, "USA": 20.0}
    assert _load_population_for_year(
        context=iso3_context,
        year=2030,
        ssp_scenario="SSP2",
        group_version_reg=None,
    ).to_dict() == {"FRA": 35.0, "USA": 45.0}


def test_l1_slicing_and_store_cover_axes_filters_and_cached_specs() -> None:
    axis_context = _context(
        needs_lcia=False,
        selected_l1=["EG(Pop)"],
        combined=[
            ("UT(FD)", "EG(Pop)"),
            ("AR(E^{CBA_FD})", "EG(Pop)"),
            ("UT(FD)", "PR(GDPcap)"),
        ],
        fu_code="L2.a.a",
        filters={
            "r_p": ["EU"],
            "s_p": ["D"],
            "r_c": None,
            "r_f": ["EU"],
            "r_u": None,
        },
    )
    slicing_context = _context(
        needs_lcia=False,
        fu_code="L2.a.b",
        filters={
            "r_p": ["EU"],
            "s_p": ["D"],
            "r_c": None,
            "r_f": ["EU"],
            "r_u": None,
        },
    )
    state = RunState(
        l1_results_by_ssp_scenario={"SSP2": {}},
        output_spec_cache={},
    )
    run = _run(context=slicing_context, state=state)
    store_run = _run(context=axis_context, state=state)

    axis_by_l2 = _l2_axes_for_l1_method(run=_run(context=axis_context), l1_method="EG(Pop)")
    assert axis_by_l2 == {"AR(E^{CBA_FD})": "r_f", "UT(FD)": "r_f"}
    assert (
        _l2_axes_for_l1_method(
            run=_run(context=_context(fu_code="L1.a")),
            l1_method="EG(Pop)",
        )
        == {}
    )
    assert _single_axis_or_none(axis_by_l2) == "r_f"
    assert _single_axis_or_none({"left": "r_f", "right": "r_p"}) is None
    assert _single_axis_or_none({}) is None

    multi_index_frame = pd.DataFrame(
        {"2030": [1.0, 2.0, 3.0, 4.0]},
        index=pd.MultiIndex.from_tuples(
            [("EU", "D"), ("EU", "X"), ("NAM", "D"), ("NAM", "X")],
            names=["r_f", "s_p"],
        ),
    )
    simple_index_frame = pd.DataFrame(
        {"2030": [10.0, 20.0]},
        index=pd.Index(["EU", "NAM"], name="r_p"),
    )
    unmatched_simple_index_frame = pd.DataFrame(
        {"2030": [100.0, 200.0]},
        index=pd.Index(["A", "B"], name="impact"),
    )

    preserved = _slice_l1_frame_for_compute(run=run, frame=multi_index_frame)
    simple = _slice_l1_frame_for_compute(run=run, frame=simple_index_frame)
    unmatched_simple = _slice_l1_frame_for_compute(run=run, frame=unmatched_simple_index_frame)

    assert list(preserved.index.get_level_values("r_f")) == ["EU", "NAM"]
    assert list(preserved.index.get_level_values("s_p")) == ["D", "D"]
    assert simple.index.tolist() == ["EU"]
    assert unmatched_simple.equals(unmatched_simple_index_frame)
    assert _slice_l1_frame_for_compute(run=run, frame=pd.DataFrame()).empty

    output_frame = pd.DataFrame(
        {"reference_year": [2005, 2005], "2030": [5.0, 6.0]},
        index=pd.Index(["EU", "NAM"], name="r_f"),
    )
    value_frame = pd.DataFrame(
        {"2030": [50.0, 60.0]},
        index=pd.Index(["EU", "NAM"], name="r_f"),
    )
    spec_first = _build_l1_output_spec(
        l1_method="EG(Pop)",
        lcia_method=None,
        frame=output_frame,
        ssp_scenario="SSP2",
        grouped_mode=False,
        state=state,
    )
    spec_second = _build_l1_output_spec(
        l1_method="EG(Pop)",
        lcia_method=None,
        frame=output_frame,
        ssp_scenario="SSP2",
        grouped_mode=False,
        state=state,
    )

    assert spec_first is spec_second
    assert spec_first.identifier_columns == ("r_f", "reference_year")

    _store_l1_frame(
        run=store_run,
        payload=_L1StorePayload(
            resolved_name="EG(Pop)",
            lcia_method=None,
            frame=output_frame,
            year_key="EG(Pop)",
            value_frame=value_frame,
        ),
    )

    stored_frames = state.l1_results_by_ssp_scenario["SSP2"][spec_first]
    assert len(stored_frames) == 1
    assert stored_frames[0].index.tolist() == ["EU"]
    assert stored_frames[0]["reference_year"].tolist() == [2005]
    assert store_run.l1_results_year["EG(Pop)"].index.tolist() == ["EU"]
    assert store_run.l1_results_year["EG(Pop)"]["2030"].tolist() == [50.0]
    assert spec_first.route.ssp_scenario == "SSP2"

    support_context = _context(persisted_years=[2030])
    support_state = RunState()
    support_run = _run(context=support_context, state=support_state, year=2005)
    _store_l1_frame(
        run=support_run,
        payload=_L1StorePayload(
            resolved_name="EG(Pop)",
            lcia_method=None,
            frame=output_frame,
            year_key="support_value",
            value_frame=value_frame,
        ),
    )
    _store_l1_frame(
        run=support_run,
        payload=_L1StorePayload(
            resolved_name="EG(Pop)",
            lcia_method=None,
            frame=output_frame,
            year_key="support_frame",
            value_frame=None,
        ),
    )
    assert support_state.l1_results_by_ssp_scenario == {}
    assert support_run.l1_results_year["support_value"]["2030"].tolist() == [50.0, 60.0]
    assert support_run.l1_results_year["support_frame"]["reference_year"].tolist() == [2005, 2005]


def test_l1_lcia_input_and_pr_hr_runtime_cover_validation_and_prefill(
    allocation_dummy_repo,
) -> None:
    logger = _RecorderLogger()
    context = _context(
        logger=logger,
        selected_l1=["PR-HR(Ecap,cum^{CBA_FD})"],
        use_original_l1_post_domain=True,
    )
    state = RunState(
        pop_series_by_ssp_scenario={"SSP2": {2030: pd.Series({"EU": 10.0})}},
        pr_post_pop_series_by_ssp_scenario={
            None: {
                2005: pd.Series({"FR": 2.0}),
            }
        },
        l1_results_by_ssp_scenario={"SSP2": {}},
        output_spec_cache={},
        rps_by_method={
            "gwp100_lcia": pd.DataFrame(
                {"impact": ["climate_child"], "responsibility_period_years": [1]}
            )
        },
        cf_by_method={"gwp100_lcia": pd.Series({"climate_child": "climate_parent"})},
    )
    run = _run(context=context, state=state)

    pr_hr_inputs = _iter_lcia_method_inputs(
        run=run,
        l1_method="PR-HR(Ecap,cum^{CBA_FD})",
        lcia_by_method={"gwp100_lcia": {}},
        lcia_by_method_original=None,
    )
    assert len(pr_hr_inputs) == 1
    lcia_inputs = pr_hr_inputs[0]
    assert lcia_inputs.lcia_kind == "CBA_FD"
    assert lcia_inputs.impact_year == 2006
    assert sorted(cast(dict[int, pd.DataFrame], lcia_inputs.lcia_reg_by_year)) == [2005, 2006]

    scenario_runtime = _prepare_pr_hr_parent_cumulative_runtime(
        context=context,
        state=state,
        lcia_method="gwp100_lcia",
        lcia_kind="CBA_FD",
        ssp_scenario="SSP2",
        population_by_year={2030: pd.Series({"EU": 10.0})},
        impact_year=2030,
        use_original_domain=True,
        include_wb_history_for_scenario=True,
    )
    assert sorted(scenario_runtime.population_by_year) == [2005, 2006, 2030]
    assert sorted(state.pr_post_pop_series_by_ssp_scenario[None]) == [2005, 2006]
    assert scenario_runtime.fallback_callback is not None
    scenario_runtime.fallback_callback(["climate_child"], 2030, 2006)
    assert state.pr_hr_rp1_zero_fallback_pending == {
        ("gwp100_lcia", ("climate_child",), 2030, 2006, True, "SSP2"): {"CBA_FD"}
    }
    base_runtime = _prepare_pr_hr_parent_cumulative_runtime(
        context=context,
        state=state,
        lcia_method="gwp100_lcia",
        lcia_kind="CBA_FD",
        ssp_scenario=None,
        population_by_year={2006: pd.Series({"EU": 10.0})},
        impact_year=2006,
        use_original_domain=True,
        include_wb_history_for_scenario=True,
    )
    assert sorted(base_runtime.population_by_year) == [2005, 2006]
    unmerged_runtime = _prepare_pr_hr_parent_cumulative_runtime(
        context=context,
        state=state,
        lcia_method="gwp100_lcia",
        lcia_kind="CBA_FD",
        ssp_scenario="SSP2",
        population_by_year={2030: pd.Series({"EU": 10.0})},
        impact_year=2006,
        use_original_domain=True,
        include_wb_history_for_scenario=False,
    )
    assert sorted(unmerged_runtime.population_by_year) == [2030]

    standard_runtime = _prepare_pr_hr_standard_inputs(
        run=run,
        lcia_inputs=lcia_inputs,
        population_by_year={2030: pd.Series({"EU": 10.0})},
        use_original_domain=True,
    )
    assert sorted(standard_runtime.population_by_year) == [2005, 2006, 2030]
    assert standard_runtime.parent_cum_cache == {}

    state.pr_post_pop_series_by_ssp_scenario["SSP2"] = {2030: pd.Series({"FR": 10.0, "US": 20.0})}
    pr_hr_run = _run(
        context=context,
        state=state,
        pop_series=pd.Series({"EU": 10.0}),
        pop_series_original=pd.Series({"FR": 10.0, "US": 20.0}),
    )
    _compute_l1_standard_lcia_method(
        run=pr_hr_run,
        l1_method="PR-HR(Ecap,cum^{CBA_FD})",
        lcia_inputs=lcia_inputs,
        region_label_override=None,
    )
    assert "PR-HR(Ecap,cum^{CBA_FD})_gwp100_lcia" in pr_hr_run.l1_results_year
    assert sorted(standard_runtime.parent_cum_cache) == [2006]

    non_pr_context = _context(
        selected_l1=["PR(GDPcap)"],
        use_original_l1_post_domain=False,
    )
    non_pr_state = RunState(
        pop_series_by_ssp_scenario={"SSP2": {2030: pd.Series({"EU": 35.0, "NAM": 45.0})}},
        l1_results_by_ssp_scenario={"SSP2": {}},
        output_spec_cache={},
    )
    non_pr_run = _run(
        context=non_pr_context,
        state=non_pr_state,
        pop_series=pd.Series({"EU": 35.0, "NAM": 45.0}),
        pr_pop=pd.Series({"FRA": 35.0, "USA": 45.0}),
        pr_gdp=pd.Series({"FRA": 350.0, "USA": 900.0}),
        pr_to_mrio=pd.Series({"FRA": "FR", "USA": "US"}),
    )
    _compute_l1_standard_lcia_method(
        run=non_pr_run,
        l1_method="PR(GDPcap)",
        lcia_inputs=_LciaMethodInputs(
            lcia_method="gwp100_lcia",
            lcia_kind="CBA_FD",
            lcia_reg=pd.DataFrame(
                [[10.0, 20.0]],
                index=pd.Index(["climate_child"], name="impact"),
                columns=pd.Index(["FR", "US"], name="r_f"),
            ),
            lcia_reg_by_year=None,
            rps_df=None,
            impact_parent_map=None,
            resolved_name="PR(GDPcap)",
            impact_year=2005,
        ),
        region_label_override=None,
    )

    assert "PR(GDPcap)_gwp100_lcia" in non_pr_run.l1_results_year
    dispatch_state = RunState(
        pop_series_by_ssp_scenario={"SSP2": {2030: pd.Series({"EU": 10.0})}},
        pr_post_pop_series_by_ssp_scenario={
            None: {
                2005: pd.Series({"FR": 2.0}),
                2006: pd.Series({"US": 3.0}),
            },
            "SSP2": {2030: pd.Series({"FR": 10.0, "US": 20.0})},
        },
        l1_results_by_ssp_scenario={"SSP2": {}},
        output_spec_cache={},
        rps_by_method={
            "gwp100_lcia": pd.DataFrame(
                {"impact": ["climate_child"], "responsibility_period_years": [1]}
            )
        },
        cf_by_method={"gwp100_lcia": pd.Series({"climate_child": "climate_parent"})},
    )
    dispatch_run = _run(
        context=context,
        state=dispatch_state,
        pop_series=pd.Series({"EU": 10.0}),
        pop_series_original=pd.Series({"FR": 10.0, "US": 20.0}),
    )
    _compute_l1_lcia_method(
        run=dispatch_run,
        l1_method="PR-HR(Ecap,cum^{CBA_FD})",
        lcia_by_method={"gwp100_lcia": {}},
        lcia_by_method_original=None,
    )
    assert "PR-HR(Ecap,cum^{CBA_FD})_gwp100_lcia" in dispatch_run.l1_results_year

    allocation_dummy_repo.set_lcia_methods(
        source="oecd_v2025",
        matrix_version=None,
        methods=["gwp100_lcia"],
        available_years_by_method={"gwp100_lcia": []},
    )
    empty_lcia_inputs = _iter_lcia_method_inputs(
        run=_run(
            context=context,
            state=RunState(
                pop_series_by_ssp_scenario={"SSP2": {2030: pd.Series({"EU": 10.0})}},
                l1_results_by_ssp_scenario={"SSP2": {}},
                output_spec_cache={},
            ),
        ),
        l1_method="AR(E^{CBA_FD})",
        lcia_by_method={},
        lcia_by_method_original=None,
    )
    assert empty_lcia_inputs == []


def test_required_lcia_metric_keys_for_context_skips_non_lcia_combined_methods() -> None:
    class _RecordingRegistry:
        def __init__(self) -> None:
            self.l1_metric_calls: list[tuple[str, str, str | None, bool | None]] = []
            self.l2_metric_calls: list[tuple[str, str, str | None, bool | None]] = []

        @staticmethod
        def method_requires_lcia(method: str, fu_code: str | None) -> bool:
            del fu_code
            return method.startswith("needs_")

        def lcia_enacting_metric_l1_metrics(
            self,
            method: str,
            *,
            level: str,
            fu_code: str | None = None,
            l1_weighting: bool | None = None,
        ) -> set[str]:
            self.l1_metric_calls.append((method, level, fu_code, l1_weighting))
            return {f"l1::{method}::{level}::{l1_weighting}"}

        def lcia_enacting_metric_l2_metrics(
            self,
            method: str,
            *,
            level: str,
            fu_code: str | None = None,
            l1_weighting: bool | None = None,
        ) -> set[str]:
            self.l2_metric_calls.append((method, level, fu_code, l1_weighting))
            return {f"l2::{method}::{level}::{l1_weighting}"}

    context = SimpleNamespace(
        selected_l1=["needs_l1", "skip_l1"],
        selected_l2_one_step=["needs_l2_one_step", "skip_l2_one_step"],
        combined=[("needs_l2_combined", "EG(Pop)"), ("skip_l2_combined", "EG(Pop)")],
        fu_code="L2.a.a",
    )
    registry = _RecordingRegistry()

    l1_keys, l2_keys = required_lcia_metric_keys_for_context(
        context=context,
        registry=registry,
    )

    assert l1_keys == {
        "l1::needs_l1::L1::None",
        "l1::needs_l2_one_step::L2::False",
        "l1::needs_l2_combined::L2::True",
    }
    assert l2_keys == {
        "l2::needs_l2_one_step::L2::False",
        "l2::needs_l2_combined::L2::True",
    }
    assert registry.l1_metric_calls == [
        ("needs_l1", "L1", None, None),
        ("needs_l2_one_step", "L2", "L2.a.a", False),
        ("needs_l2_combined", "L2", "L2.a.a", True),
    ]
    assert registry.l2_metric_calls == [
        ("needs_l2_one_step", "L2", "L2.a.a", False),
        ("needs_l2_combined", "L2", "L2.a.a", True),
    ]


def test_l1_compute_covers_non_lcia_alias_routing_and_missing_lcia_skip(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    alias_context = _context(
        needs_lcia=False,
        selected_l1=["EG(Pop)"],
        combined=[("UT(FD)", "EG(Pop)"), ("AR(E^{CBA_FD})", "EG(Pop)")],
        fu_code="L2.a.a",
    )
    alias_population = _load_population_for_year(
        context=alias_context,
        year=2030,
        ssp_scenario="SSP2",
        group_version_reg=alias_context.group_version_reg,
    )
    alias_state = RunState(
        pop_series_by_ssp_scenario={"SSP2": {2030: alias_population}},
        l1_results_by_ssp_scenario={"SSP2": {}},
        output_spec_cache={},
    )
    alias_run = _run(context=alias_context, state=alias_state, pop_series=alias_population)

    _compute_l1_non_lcia_method(run=alias_run, l1_method="EG(Pop)")

    assert sorted(alias_run.l1_results_year) == [
        "EG(Pop)",
        "EG(Pop)__for__AR(E^{CBA_FD})",
        "EG(Pop)__for__UT(FD)",
    ]
    assert not alias_run.l1_results_year["EG(Pop)"].empty
    assert len(alias_state.l1_results_by_ssp_scenario["SSP2"]) == 1

    l1_only_context = _context(
        needs_lcia=False,
        selected_l1=["EG(Pop)"],
        combined=[],
        fu_code="L1.a",
    )
    l1_only_population = _load_population_for_year(
        context=l1_only_context,
        year=2030,
        ssp_scenario="SSP2",
        group_version_reg=l1_only_context.group_version_reg,
    )
    l1_only_state = RunState(
        pop_series_by_ssp_scenario={"SSP2": {2030: l1_only_population}},
        l1_results_by_ssp_scenario={"SSP2": {}},
        output_spec_cache={},
    )
    l1_only_run = _run(
        context=l1_only_context,
        state=l1_only_state,
        pop_series=l1_only_population,
    )

    _compute_l1_non_lcia_method(run=l1_only_run, l1_method="EG(Pop)")

    assert list(l1_only_run.l1_results_year) == ["EG(Pop)"]

    skip_context = _context(
        selected_l1=["AR(E^{CBA_FD})", "EG(Pop)"],
        selected_l2_one_step=[],
        combined=[],
        needs_lcia=True,
        fu_code="L2.a.a",
    )
    skip_population = _load_population_for_year(
        context=skip_context,
        year=2030,
        ssp_scenario="SSP2",
        group_version_reg=skip_context.group_version_reg,
    )
    skip_state = RunState(
        pop_series_by_ssp_scenario={"SSP2": {2030: skip_population}},
        l1_results_by_ssp_scenario={"SSP2": {}},
        output_spec_cache={},
    )
    skip_run = _run(context=skip_context, state=skip_state, pop_series=skip_population)

    results = _compute_l1_for_year(
        run=skip_run,
        lcia_by_method=None,
        lcia_by_method_original=None,
    )

    assert skip_state.skipped_years[2030] == "LCIA unavailable"
    assert list(results) == ["EG(Pop)"]

    preserved_skip_state = RunState(
        pop_series_by_ssp_scenario={"SSP2": {2030: skip_population}},
        l1_results_by_ssp_scenario={"SSP2": {}},
        output_spec_cache={},
        skipped_years={2030: "already skipped"},
    )
    preserved_skip_run = _run(
        context=skip_context,
        state=preserved_skip_state,
        pop_series=skip_population,
    )
    preserved_results = _compute_l1_for_year(
        run=preserved_skip_run,
        lcia_by_method=None,
        lcia_by_method_original=None,
    )
    assert preserved_skip_state.skipped_years[2030] == "already skipped"
    assert list(preserved_results) == ["EG(Pop)"]

    lcia_context = _context(
        selected_l1=["AR(E^{CBA_FD})"],
        selected_l2_one_step=[],
        combined=[],
        needs_lcia=True,
        fu_code="L2.a.a",
    )
    lcia_population = _load_population_for_year(
        context=lcia_context,
        year=2030,
        ssp_scenario="SSP2",
        group_version_reg=lcia_context.group_version_reg,
    )
    lcia_state = RunState(
        pop_series_by_ssp_scenario={"SSP2": {2030: lcia_population}},
        l1_results_by_ssp_scenario={"SSP2": {}},
        output_spec_cache={},
    )
    lcia_run = _run(context=lcia_context, state=lcia_state, pop_series=lcia_population)
    lcia_results = _compute_l1_for_year(
        run=lcia_run,
        lcia_by_method={
            "gwp100_lcia": {
                "e_cba_fd_reg": pd.DataFrame(
                    [[10.0, 20.0]],
                    index=pd.Index(["climate_child"], name="impact"),
                    columns=pd.Index(["FR", "US"], name="r_f"),
                )
            }
        },
        lcia_by_method_original=None,
    )
    assert "AR(E^{CBA_FD})_gwp100_lcia_ref_2005" in lcia_results


def test_l1_ar_lcia_covers_invariant_cache_and_missing_reference_notice(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    lcia_payload = {
        "gwp100_lcia": {
            "e_cba_fd_reg": pd.DataFrame(
                [[10.0, 20.0]],
                index=pd.Index(["climate_child"], name="impact"),
                columns=pd.Index(["FR", "US"], name="r_f"),
            )
        }
    }
    context = _context(
        selected_l1=["AR(E^{CBA_FD})"],
        reference_years=[2005],
        r_p=None,
    )
    population = _load_population_for_year(
        context=context,
        year=2030,
        ssp_scenario="SSP2",
        group_version_reg=context.group_version_reg,
    )
    state = RunState(
        pop_series_by_ssp_scenario={"SSP2": {2030: population}},
        l1_results_by_ssp_scenario={"SSP2": {}},
        output_spec_cache={},
    )
    run = _run(context=context, state=state, pop_series=population)
    lcia_inputs = _iter_lcia_method_inputs(
        run=run,
        l1_method="AR(E^{CBA_FD})",
        lcia_by_method=lcia_payload,
        lcia_by_method_original=None,
    )[0]
    invariant_cache_key = (
        2030,
        "AR(E^{CBA_FD})",
        "AR(E^{CBA_FD})",
        "gwp100_lcia",
        "CBA_FD",
        2005,
        None,
        False,
        "None",
        "post",
    )
    cached_frame = pd.DataFrame(
        {"2005": [1.0], "reference_year": [2005]},
        index=pd.Index(["FR"], name="r_f"),
    )
    cached_value = pd.DataFrame(
        {"2005": [1.0]},
        index=pd.Index(["FR"], name="r_f"),
    )
    state.l1_invariant_cache[invariant_cache_key] = (cached_frame, cached_value)

    _compute_l1_ar_lcia_method(
        run=run,
        l1_method="AR(E^{CBA_FD})",
        lcia_inputs=lcia_inputs,
        region_label_override=None,
    )

    assert "AR(E^{CBA_FD})_gwp100_lcia_ref_2005" in run.l1_results_year
    assert run.l1_results_year["AR(E^{CBA_FD})_gwp100_lcia_ref_2005"].equals(cached_value)

    no_ref_context = _context(
        selected_l1=["AR(E^{CBA_FD})"],
        reference_years=[2035],
        r_p=None,
    )
    no_ref_population = _load_population_for_year(
        context=no_ref_context,
        year=2030,
        ssp_scenario="SSP2",
        group_version_reg=no_ref_context.group_version_reg,
    )
    no_ref_state = RunState(
        pop_series_by_ssp_scenario={"SSP2": {2030: no_ref_population}},
        l1_results_by_ssp_scenario={"SSP2": {}},
        output_spec_cache={},
    )
    no_ref_run = _run(
        context=no_ref_context,
        state=no_ref_state,
        pop_series=no_ref_population,
    )
    no_ref_inputs = _iter_lcia_method_inputs(
        run=no_ref_run,
        l1_method="AR(E^{CBA_FD})",
        lcia_by_method=lcia_payload,
        lcia_by_method_original=None,
    )[0]

    _compute_l1_ar_lcia_method(
        run=no_ref_run,
        l1_method="AR(E^{CBA_FD})",
        lcia_inputs=no_ref_inputs,
        region_label_override=None,
    )

    assert no_ref_run.l1_results_year == {}
    assert "ar-no-refs:gwp100_lcia" in no_ref_state.notices_emitted


def test_l1_ar_lcia_covers_ar_ecap_original_domain_population_reference(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    lcia_payload = {
        "gwp100_lcia": {
            "e_cba_fd_reg": pd.DataFrame(
                [[10.0, 20.0]],
                index=pd.Index(["climate_child"], name="impact"),
                columns=pd.Index(["FR", "US"], name="r_f"),
            )
        }
    }
    context = _context(
        selected_l1=["AR(Ecap^{CBA_FD})"],
        reference_years=[2005],
        use_original_l1_post_domain=True,
        r_p=None,
    )
    grouped_population = _load_population_for_year(
        context=context,
        year=2030,
        ssp_scenario="SSP2",
        group_version_reg=context.group_version_reg,
    )
    original_population = grouped_population * 2.0
    state = RunState(
        pop_series_by_ssp_scenario={"SSP2": {2030: original_population}},
        l1_results_by_ssp_scenario={"SSP2": {}},
        output_spec_cache={},
    )
    run = _run(
        context=context,
        state=state,
        pop_series=grouped_population,
        pop_series_original=original_population,
    )
    lcia_inputs = _iter_lcia_method_inputs(
        run=run,
        l1_method="AR(Ecap^{CBA_FD})",
        lcia_by_method=lcia_payload,
        lcia_by_method_original=lcia_payload,
    )[0]

    _compute_l1_ar_lcia_method(
        run=run,
        l1_method="AR(Ecap^{CBA_FD})",
        lcia_inputs=lcia_inputs,
        region_label_override=None,
    )

    assert "AR(Ecap^{CBA_FD})_gwp100_lcia_ref_2005" in run.l1_results_year
    assert not run.l1_results_year["AR(Ecap^{CBA_FD})_gwp100_lcia_ref_2005"].empty


def test_lcia_reference_policy_emits_pr_hr_gap_notices(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    logger = _RecorderLogger()
    state = RunState(notices_emitted=set())
    run = _run(
        context=_context(
            logger=logger,
            historical_years=[2005, 2006],
            resolved_years=[2005, 2006, 2030],
        ),
        state=state,
        year=2006,
    )

    emit_pr_hr_lcia_freeze_notice_if_needed(
        run=run,
        lcia_method="gwp100_lcia",
        last_timeseries_year=2006,
    )
    assert logger.messages == []

    emit_pr_hr_lcia_freeze_notice_if_needed(
        run=run,
        lcia_method="gwp100_lcia",
        last_timeseries_year=2005,
    )

    assert any("gwp100_lcia" in message for message in logger.messages)

    future_logger = _RecorderLogger()
    future_state = RunState(notices_emitted=set())
    future_run = _run(
        context=_context(
            logger=future_logger,
            historical_years=[2005, 2006],
            resolved_years=[2030],
        ),
        state=future_state,
        year=2030,
    )
    emit_pr_hr_lcia_freeze_notice_if_needed(
        run=future_run,
        lcia_method="gwp100_lcia",
        last_timeseries_year=2005,
    )
    emit_pr_hr_lcia_freeze_notice_if_needed(
        run=future_run,
        lcia_method="gwp100_lcia",
        last_timeseries_year=2005,
    )

    assert len(future_logger.messages) == 1
    assert "2006" in future_logger.messages[0]

    prefixed_state = RunState(notices_emitted={"pr-hr-lcia-gap:gwp100_lcia:2006"})
    prefixed_logger = _RecorderLogger()
    prefixed_run = _run(
        context=_context(
            logger=prefixed_logger,
            historical_years=[2005, 2006],
            resolved_years=[2030],
        ),
        state=prefixed_state,
        year=2030,
    )
    emit_pr_hr_lcia_freeze_notice_if_needed(
        run=prefixed_run,
        lcia_method="gwp100_lcia",
        last_timeseries_year=2005,
    )
    assert prefixed_logger.messages == []


def test_ar_reference_policy_emits_default_and_requested_clipping_notices(
    allocation_dummy_repo,
) -> None:
    allocation_dummy_repo.set_lcia_methods(
        source="oecd_v2025",
        matrix_version=None,
        methods=["gwp100_lcia"],
        available_years_by_method={"gwp100_lcia": [2005]},
    )
    lcia_inputs = _LciaMethodInputs(
        lcia_method="gwp100_lcia",
        lcia_kind="CBA_FD",
        lcia_reg=pd.DataFrame(),
        lcia_reg_by_year=None,
        rps_df=None,
        impact_parent_map=None,
        resolved_name="AR(E^{CBA_FD})",
    )

    logger = _RecorderLogger()
    state = RunState(notices_emitted=set())
    run = _run(
        context=_context(
            logger=logger,
            historical_years=[2005, 2006],
            resolved_years=[2006],
            reference_years=None,
            selected_l1=["AR(E^{CBA_FD})"],
        ),
        state=state,
        year=2006,
    )
    refs = resolve_reference_years_for_ar(
        run=run,
        lcia_inputs=lcia_inputs,
        use_original_domain=False,
    )

    assert refs == [2005]
    assert any("2005" in message for message in logger.messages)

    requested_logger = _RecorderLogger()
    requested_state = RunState(notices_emitted=set())
    requested_run = _run(
        context=_context(
            logger=requested_logger,
            historical_years=[2005, 2006],
            resolved_years=[2006],
            reference_years=[2005, 2006],
            selected_l1=["AR(E^{CBA_FD})"],
        ),
        state=requested_state,
        year=2006,
    )
    requested_refs = resolve_reference_years_for_ar(
        run=requested_run,
        lcia_inputs=lcia_inputs,
        use_original_domain=False,
    )

    assert requested_refs == [2005]
    assert any("2005" in message for message in requested_logger.messages)

    future_logger = _RecorderLogger()
    future_state = RunState(notices_emitted=set())
    future_run = _run(
        context=_context(
            logger=future_logger,
            historical_years=[2005],
            resolved_years=[2030],
            reference_years=None,
            selected_l1=["AR(E^{CBA_FD})"],
        ),
        state=future_state,
        year=2030,
    )
    future_refs = resolve_reference_years_for_ar(
        run=future_run,
        lcia_inputs=lcia_inputs,
        use_original_domain=False,
    )
    assert future_refs == [2005]
    assert any("2005" in message for message in future_logger.messages)

    cached_logger = _RecorderLogger()
    cached_state = RunState(notices_emitted=set())
    cached_context = _context(
        logger=cached_logger,
        historical_years=[2005, 2006],
        resolved_years=[2006],
        reference_years=None,
        selected_l1=["AR(E^{CBA_FD})"],
    )
    cached_key = ("gwp100_lcia", "CBA_FD", False, None, (2005, 2006))
    cached_state.ar_valid_refs_cache[cached_key] = ([2005], [])
    cached_run = _run(context=cached_context, state=cached_state, year=2006)
    assert resolve_reference_years_for_ar(
        run=cached_run,
        lcia_inputs=lcia_inputs,
        use_original_domain=False,
    ) == [2005]
    assert any(
        "LCIA data is missing for year 2006" in message for message in cached_logger.messages
    )

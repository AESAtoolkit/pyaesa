from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa.asocc.io.metadata import EnactingMetricKey, RunState
from pyaesa.asocc.orchestration.yearly.enacting_metric import enacting_metric_base as base_mod
from pyaesa.asocc.orchestration.yearly.enacting_metric import enacting_metric_common as common_mod
from pyaesa.asocc.orchestration.yearly.enacting_metric import (
    enacting_metric_lcia_percap as percap_mod,
)
from pyaesa.asocc.orchestration.yearly.enacting_metric import (
    enacting_metric_lcia_passes as passes_mod,
)
from pyaesa.asocc.orchestration.yearly.enacting_metric import (
    enacting_metric_lcia_policy as policy_mod,
)
from pyaesa.asocc.orchestration.yearly.enacting_metric import enacting_metric_pr as pr_mod
from pyaesa.asocc.orchestration.yearly.enacting_metric import (
    enacting_metric_lcia_routing as routing_mod,
)
from pyaesa.asocc.orchestration.yearly.enacting_metric import (
    enacting_metric_lcia_selection as scope_mod,
)
from pyaesa.asocc.orchestration.yearly.enacting_metric import (
    enacting_metric_lcia_shaping as shaping_mod,
)
from pyaesa.process.mrios.utils.io.paths import _get_group_map_path


def _context(**overrides: Any) -> Any:
    payload = {
        "source": "oecd_v2025",
        "group_version_reg": None,
        "fu_code": "L2.a.a",
        "filters": {
            "r_p": ["FR"],
            "s_p": ["D"],
            "r_c": None,
            "r_f": ["FR"],
            "r_u": None,
        },
        "wb_df": pd.DataFrame(columns=["2005"]),
        "historical_years": [2005],
        "lcia_methods": ["gwp100_lcia"],
        "l1_only_no_mrio": False,
        "logger": SimpleNamespace(warning=lambda _message: None),
    }
    payload.update(overrides)
    return cast(
        Any,
        SimpleNamespace(**payload),
    )


def _impact_frame(*, axis: str) -> pd.DataFrame:
    return pd.DataFrame(
        [[10.0, 20.0]],
        index=pd.Index(["AAL"], name="impact"),
        columns=pd.Index(["FR", "DE"], name=axis),
    )


def _ut_payload() -> dict[str, object]:
    return {
        "fd_rf": pd.Series([4.0, 6.0], index=pd.Index(["FR", "US"], name="r_f")),
        "gva_rp": pd.Series([5.0, 5.0], index=pd.Index(["FR", "US"], name="r_p")),
        "fd_rp_sp_rf": pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=pd.MultiIndex.from_tuples(
                [("FR", "A"), ("US", "A")],
                names=["r_p", "s_p"],
            ),
            columns=pd.Index(["FR", "US"], name="r_f"),
        ),
        "fd_rp_sp": pd.Series(
            [3.0, 7.0],
            index=pd.MultiIndex.from_tuples(
                [("FR", "A"), ("US", "A")],
                names=["r_p", "s_p"],
            ),
        ),
        "fd_rf_sp": pd.Series(
            [2.0, 8.0],
            index=pd.MultiIndex.from_tuples(
                [("FR", "A"), ("US", "A")],
                names=["r_f", "s_p"],
            ),
        ),
        "gva_rp_sp": pd.Series(
            [6.0, 7.0],
            index=pd.MultiIndex.from_tuples(
                [("FR", "A"), ("US", "A")],
                names=["r_p", "s_p"],
            ),
        ),
        "x_to_rc": pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=pd.MultiIndex.from_tuples(
                [("FR", "A"), ("US", "A")],
                names=["r_p", "s_p"],
            ),
            columns=pd.Index(["FR", "US"], name="r_c"),
        ),
        "kappa": pd.DataFrame(
            [[0.5, 0.5], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9]],
            index=pd.MultiIndex.from_tuples(
                [
                    ("FR", "FR", "A"),
                    ("US", "FR", "A"),
                    ("FR", "US", "A"),
                    ("US", "US", "A"),
                ],
                names=["r_c", "r_p", "s_p"],
            ),
            columns=pd.Index(["FR", "US"], name="r_f"),
        ),
        "omega_reg": pd.DataFrame(
            [[0.6, 0.4], [0.4, 0.6]],
            index=pd.Index(["FR", "US"], name="r_u"),
            columns=pd.MultiIndex.from_tuples(
                [("FR", "A"), ("US", "A")],
                names=["r_p", "s_p"],
            ),
        ),
    }


def test_enacting_metric_lcia_selection_and_routing_cover_selection_branches() -> None:
    context = _context(wb_df=pd.DataFrame(columns=["2005", "2006"]), lcia_methods=["m1", "m2"])
    state = RunState(
        pop_series_by_ssp_scenario={
            None: {2005: pd.Series([1.0], index=pd.Index(["FR"], name="r_f"))}
        },
        pr_post_pop_series_by_ssp_scenario={
            "SSP2": {2005: pd.Series([2.0], index=pd.Index(["FR"], name="r_f"))}
        },
        lcia_timeseries={"m1": {"CBA_FD": {2005: _impact_frame(axis="r_f")}}},
        lcia_timeseries_original={"m2": {"PBA": {2004: _impact_frame(axis="r_p")}}},
        rps_by_method={
            "m2": pd.DataFrame({"impact": ["AAL"], "responsibility_period_years": [1]}),
            "m5": pd.DataFrame({"impact": ["AAL"], "responsibility_period_years": [1]}),
        },
        cf_by_method={
            "m2": pd.Series({"AAL": "AAL"}),
            "m5": pd.Series({"AAL": "AAL"}),
        },
    )

    assert (
        scope_mod._resolve_enacting_metric_output_scenario(  # noqa: SLF001
            context=context,
            year=2005,
            ssp_scenario="SSP2",
        )
        is None
    )
    assert (
        scope_mod._resolve_enacting_metric_output_scenario(  # noqa: SLF001
            context=_context(wb_df=pd.DataFrame(columns=["2005"])),
            year=2030,
            ssp_scenario="SSP2",
        )
        == "SSP2"
    )
    assert (
        scope_mod._resolve_requested_effective_year(  # noqa: SLF001
            lcia_method="m1",
            year=2030,
            lcia_effective_year_by_method=None,
        )
        == 2030
    )
    assert (
        scope_mod._resolve_requested_effective_year(  # noqa: SLF001
            lcia_method="m1",
            year=2030,
            lcia_effective_year_by_method={"m1": 2025},
        )
        == 2025
    )

    assert (
        scope_mod._iter_direct_percap_scopes(  # noqa: SLF001
            year=2005,
            lcia_by_method=None,
            lcia_effective_year_by_method=None,
        )
        == []
    )
    direct_scopes = scope_mod._iter_direct_percap_scopes(  # noqa: SLF001
        year=2005,
        lcia_by_method={
            "m1": {"e_cba_fd_reg": _impact_frame(axis="r_f")},
            "m2": {"e_cba_fd_reg": _impact_frame(axis="r_f")},
        },
        lcia_effective_year_by_method={"m1": 2005, "m2": 2004},
    )
    assert [(scope.lcia_method, scope.effective_year) for scope in direct_scopes] == [("m1", 2005)]

    lcia_store, population_by_year, methods_in_scope = scope_mod._resolve_pr_hr_base_inputs(  # noqa: SLF001
        context=context,
        state=state,
        ssp_scenario="SSP2",
        use_original_domain=True,
        lcia_by_method=None,
    )
    assert lcia_store == state.lcia_timeseries_original
    assert list(population_by_year) == [2005]
    assert methods_in_scope == ["m1", "m2"]

    filtered_store, _, filtered_methods = scope_mod._resolve_pr_hr_base_inputs(  # noqa: SLF001
        context=context,
        state=state,
        ssp_scenario=None,
        use_original_domain=False,
        lcia_by_method={"m1": {}},
    )
    assert filtered_store == state.lcia_timeseries
    assert filtered_methods == ["m1"]

    cumulative_scopes = scope_mod._iter_pr_hr_cumulative_scopes(  # noqa: SLF001
        state=state,
        year=2005,
        lcia_kinds={"CBA_FD", "PBA"},
        lcia_store={
            "m1": {"CBA_FD": {2005: _impact_frame(axis="r_f")}},
            "m2": {"PBA": {2004: _impact_frame(axis="r_p"), 2005: _impact_frame(axis="r_p")}},
            "m3": {"PBA": {}},
            "m4": {"CBA_FD": {2005: _impact_frame(axis="r_f")}},
            "m5": {"PBA": {2006: _impact_frame(axis="r_p")}},
        },
        lcia_methods_in_scope=["m1", "m2", "m3", "m4", "m5", "missing"],
        lcia_effective_year_by_method={
            "m1": 2005,
            "m2": 2005,
            "m3": 2005,
            "m4": 2004,
            "m5": 2005,
        },
    )
    assert [
        (scope.lcia_method, scope.lcia_kind, scope.effective_year) for scope in cumulative_scopes
    ] == [("m2", "PBA", 2005)]

    assert routing_mod._iter_lcia_percap_metric_pairs(required_kinds=set()) == []  # noqa: SLF001
    assert routing_mod._iter_lcia_percap_metric_pairs(  # noqa: SLF001
        required_kinds={"PBA"}
    ) == [
        routing_mod._PerCapMetricPair(
            source_metric="e_pba_reg",
            output_metric="e_pba_reg_cap",
            region_label="r_p",
        )
    ]
    pairs = routing_mod._iter_lcia_percap_metric_pairs(  # noqa: SLF001
        required_kinds={"CBA_FD", "CBA_TD", "PBA"}
    )
    assert [(pair.source_metric, pair.output_metric, pair.region_label) for pair in pairs] == [
        ("e_cba_fd_reg", "e_cba_fd_reg_cap", "r_f"),
        ("e_pba_reg", "e_pba_reg_cap", "r_p"),
    ]
    assert routing_mod._resolve_pr_hr_cumulative_metric_contract(  # noqa: SLF001
        lcia_kind="PBA"
    ) == routing_mod._PrHrCumulativeMetricContract(
        output_metric="e_pba_reg_cap_cum",
        region_label="r_p",
    )
    assert routing_mod._resolve_pr_hr_cumulative_metric_contract(  # noqa: SLF001
        lcia_kind="CBA_FD"
    ) == routing_mod._PrHrCumulativeMetricContract(
        output_metric="e_cba_fd_reg_cap_cum",
        region_label="r_f",
    )


def test_enacting_metric_lcia_shaping_and_common_cover_grouping_and_filtering(
    project_repo: Path,
) -> None:
    del project_repo
    group_map_path = _get_group_map_path("oecd_v2025", kind="reg", group_version="demo")
    group_map_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "original_classification": ["FR", "DE"],
            "grouped_mrio": ["EU", "EU"],
        }
    ).to_csv(group_map_path, index=False)

    lcia_frame = _impact_frame(axis="r_f")
    population = pd.Series([2.0, 4.0], index=pd.Index(["FR", "DE"]))
    aligned = shaping_mod._align_population_for_lcia_frame(  # noqa: SLF001
        population=population,
        lcia_frame=lcia_frame,
    )
    assert aligned.index.name == "r_f"
    unchanged_population = shaping_mod._align_population_for_lcia_frame(  # noqa: SLF001
        population=population,
        lcia_frame=pd.DataFrame(
            [[1.0, 2.0]],
            columns=pd.MultiIndex.from_tuples([("FR", "A"), ("DE", "B")], names=["r_f", "s_p"]),
        ),
    )
    assert unchanged_population.index.name is None

    per_cap = shaping_mod._shape_lcia_percap_series(  # noqa: SLF001
        lcia_frame=lcia_frame,
        population=pd.Series([2.0, 0.0], index=pd.Index(["FR", "DE"])),
        output_metric="e_cba_fd_reg_cap",
        region_label="r_f",
        use_original_domain=True,
        source_key="oecd_v2025",
        group_version="demo",
    )
    assert per_cap.index.names == ["impact", "r_f", "grouped_mrio_code"]
    assert float(per_cap.loc[("AAL", "FR", "EU")]) == pytest.approx(5.0)
    assert pd.isna(per_cap.loc[("AAL", "DE", "EU")])

    cumulative_series = shaping_mod._shape_pr_hr_cumulative_series(  # noqa: SLF001
        parent_cum={"AAL": pd.Series([1.0, 2.0], index=pd.Index(["FR", "DE"], name="r_f"))},
        contract=routing_mod._PrHrCumulativeMetricContract(
            output_metric="e_cba_fd_reg_cap_cum",
            region_label="r_f",
        ),
        use_original_domain=True,
        source_key="oecd_v2025",
        group_version="demo",
    )
    assert cumulative_series.index.names == ["impact", "r_f", "grouped_mrio_code"]
    assert float(cumulative_series.loc[("AAL", "DE", "EU")]) == pytest.approx(2.0)

    multiindex_series = pd.Series(
        [1.0, 2.0],
        index=pd.MultiIndex.from_tuples(
            [("AAL", "FR"), ("AAL", "DE")],
            names=["impact", "r_f"],
        ),
    )
    assert common_mod._append_grouped_mrio_code_level(  # noqa: SLF001
        series=multiindex_series,
        region_label="r_f",
        source_key="oecd_v2025",
        group_version=None,
    ).equals(multiindex_series)
    assert common_mod._append_grouped_mrio_code_level(  # noqa: SLF001
        series=pd.Series([1.0], index=pd.Index(["FR"], name="r_f")),
        region_label="r_f",
        source_key="oecd_v2025",
        group_version="demo",
    ).equals(pd.Series([1.0], index=pd.Index(["FR"], name="r_f")))
    assert common_mod._append_grouped_mrio_code_level(  # noqa: SLF001
        series=multiindex_series,
        region_label="r_p",
        source_key="oecd_v2025",
        group_version="demo",
    ).equals(multiindex_series)

    missing_series = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([("AAL", "US")], names=["impact", "r_f"]),
    )
    with pytest.raises(ValueError):
        common_mod._append_grouped_mrio_code_level(  # noqa: SLF001
            series=missing_series,
            region_label="r_f",
            source_key="oecd_v2025",
            group_version="demo",
        )

    slice_context = _context(
        filters={"r_p": ["EU"], "s_p": None, "r_c": None, "r_f": None, "r_u": None}
    )
    grouped_series = common_mod._append_grouped_mrio_code_level(  # noqa: SLF001
        series=multiindex_series,
        region_label="r_f",
        source_key="oecd_v2025",
        group_version="demo",
    )
    sliced = common_mod._slice_enacting_metric_series_for_run(  # noqa: SLF001
        context=slice_context,
        series=grouped_series,
    )
    assert sliced.index.get_level_values("grouped_mrio_code").tolist() == ["EU", "EU"]
    mrio_filtered = common_mod._slice_enacting_metric_series_for_run(  # noqa: SLF001
        context=_context(
            filters={"r_p": ["FR"], "s_p": None, "r_c": None, "r_f": None, "r_u": None}
        ),
        series=pd.Series(
            [1.0, 2.0],
            index=pd.MultiIndex.from_tuples(
                [("AAL", "FR"), ("AAL", "DE")],
                names=["impact", "mrio_code"],
            ),
        ),
    )
    assert mrio_filtered.index.get_level_values("mrio_code").tolist() == ["FR"]
    ungrouped_multiindex = common_mod._slice_enacting_metric_series_for_run(  # noqa: SLF001
        context=_context(filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None, "r_u": None}),
        series=pd.Series(
            [1.0, 2.0],
            index=pd.MultiIndex.from_tuples(
                [("FR", "A"), ("DE", "B")],
                names=["r_f", "s_p"],
            ),
        ),
    )
    assert list(ungrouped_multiindex.index) == [("FR", "A"), ("DE", "B")]

    single_index_context = _context(
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": ["FR"], "r_u": None}
    )
    filtered_single = common_mod._slice_enacting_metric_series_for_run(  # noqa: SLF001
        context=single_index_context,
        series=pd.Series([1.0, 2.0], index=pd.Index(["FR", "DE"], name="r_f")),
    )
    assert filtered_single.index.tolist() == ["FR"]
    unchanged_single = common_mod._slice_enacting_metric_series_for_run(  # noqa: SLF001
        context=_context(filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None, "r_u": None}),
        series=pd.Series([1.0, 2.0], index=pd.Index(["FR", "DE"], name="r_f")),
    )
    assert unchanged_single.index.tolist() == ["FR", "DE"]
    with pytest.raises(ValueError):
        common_mod._slice_enacting_metric_series_for_run(  # noqa: SLF001
            context=single_index_context,
            series=pd.Series([1.0], index=pd.Index(["FR"])),
        )

    keep_full_weight = common_mod._slice_enacting_metric_series_for_run(  # noqa: SLF001
        context=_context(
            fu_code="L2.a.b",
            filters={"r_p": None, "s_p": None, "r_c": None, "r_f": ["DE"], "r_u": ["DE"]},
        ),
        series=pd.Series(
            [1.0, 2.0],
            index=pd.MultiIndex.from_tuples([("FR", "A"), ("DE", "B")], names=["r_f", "r_u"]),
        ),
    )
    assert list(keep_full_weight.index) == [("FR", "A"), ("DE", "B")]

    state = RunState()
    key = EnactingMetricKey(metric="demo")
    common_mod._record_enacting_metric_input(  # noqa: SLF001
        context=_context(
            filters={"r_p": None, "s_p": None, "r_c": None, "r_f": ["FR"], "r_u": None}
        ),
        state=state,
        key=key,
        year=2005,
        series=multiindex_series,
        level="level_1",
    )
    assert state.enacting_metric_levels[key] == "level_1"
    assert state.enacting_metric_inputs[key][2005].index.get_level_values("r_f").tolist() == ["FR"]
    with pytest.raises(ValueError):
        common_mod._record_enacting_metric_input(  # noqa: SLF001
            context=_context(),
            state=state,
            key=key,
            year=2006,
            series=multiindex_series,
            level="level_2",
        )

    assert common_mod._l1_kinds_for_selected_method(  # noqa: SLF001
        context=_context(),
        l1_method="AR(E^{CBA_FD})",
    ) == {"CBA_FD"}


def test_enacting_metric_lcia_direct_pass_records_filtered_per_cap_outputs() -> None:
    state = RunState()
    context = _context(
        filters={"r_p": ["FR"], "s_p": None, "r_c": None, "r_f": ["FR"], "r_u": None}
    )
    pop_series = pd.Series([2.0, 4.0], index=pd.Index(["FR", "DE"], name="r_f"))
    pairs = routing_mod._iter_lcia_percap_metric_pairs(required_kinds={"CBA_FD", "PBA"})  # noqa: SLF001

    passes_mod._record_direct_lcia_percap_pass(  # noqa: SLF001
        context=context,
        state=state,
        year=2005,
        scenario_key=None,
        lcia_by_method={"gwp100_lcia": {"e_cba_fd_reg": _impact_frame(axis="r_f")}},
        pop_series=pop_series,
        pairs=pairs,
        use_original_domain=False,
        lcia_effective_year_by_method=None,
    )

    direct_key = EnactingMetricKey(
        metric="e_cba_fd_reg_cap",
        lcia_method="gwp100_lcia",
        ssp_scenario=None,
    )
    assert direct_key in state.enacting_metric_inputs
    assert state.enacting_metric_inputs[direct_key][2005].index.get_level_values(
        "r_f"
    ).tolist() == ["FR"]
    missing_key = EnactingMetricKey(
        metric="e_pba_reg_cap",
        lcia_method="gwp100_lcia",
        ssp_scenario=None,
    )
    assert missing_key not in state.enacting_metric_inputs


def test_enacting_metric_lcia_pr_hr_pass_records_cumulative_outputs_and_skips_invalid_runtime() -> (
    None
):
    population = pd.Series([2.0, 4.0], index=pd.Index(["FR", "DE"], name="r_f"))
    lcia_reg_by_year = {2005: _impact_frame(axis="r_f")}
    good_state = RunState(
        pop_series_by_ssp_scenario={None: {2005: population}},
        pr_post_pop_series_by_ssp_scenario={None: {2005: population}},
        lcia_timeseries={"gwp100_lcia": {"CBA_FD": lcia_reg_by_year}},
        rps_by_method={
            "gwp100_lcia": pd.DataFrame({"impact": ["AAL"], "responsibility_period_years": [1]})
        },
        cf_by_method={"gwp100_lcia": pd.Series({"AAL": "AAL"})},
    )

    passes_mod._record_pr_hr_cumulative_pass(  # noqa: SLF001
        context=_context(wb_df=pd.DataFrame(columns=["2005"])),
        state=good_state,
        year=2005,
        ssp_scenario=None,
        scenario_key=None,
        lcia_by_method=None,
        cumulative_kinds={"CBA_FD"},
        use_original_domain=False,
        lcia_effective_year_by_method=None,
    )

    cumulative_key = EnactingMetricKey(
        metric="e_cba_fd_reg_cap_cum",
        lcia_method="gwp100_lcia",
        ssp_scenario=None,
    )
    assert cumulative_key in good_state.enacting_metric_inputs
    assert good_state.enacting_metric_inputs[cumulative_key][2005].index.get_level_values(
        "r_f"
    ).tolist() == ["FR"]

    bad_state = RunState(
        pop_series_by_ssp_scenario={None: {2005: population}},
        pr_post_pop_series_by_ssp_scenario={None: {2005: population}},
        lcia_timeseries={"gwp100_lcia": {"CBA_FD": lcia_reg_by_year}},
        rps_by_method={"gwp100_lcia": pd.DataFrame({"impact": ["AAL"]})},
        cf_by_method={"gwp100_lcia": pd.Series({"AAL": "AAL"})},
    )

    passes_mod._record_pr_hr_cumulative_pass(  # noqa: SLF001
        context=_context(wb_df=pd.DataFrame(columns=["2005"])),
        state=bad_state,
        year=2005,
        ssp_scenario=None,
        scenario_key=None,
        lcia_by_method=None,
        cumulative_kinds={"CBA_FD"},
        use_original_domain=False,
        lcia_effective_year_by_method=None,
    )

    assert bad_state.enacting_metric_inputs == {}

    empty_parent_state = RunState(
        pop_series_by_ssp_scenario={None: {2005: population}},
        pr_post_pop_series_by_ssp_scenario={None: {2005: population}},
        lcia_timeseries={"gwp100_lcia": {"CBA_FD": lcia_reg_by_year}},
        rps_by_method={
            "gwp100_lcia": pd.DataFrame({"impact": ["AAL"], "responsibility_period_years": [1]})
        },
        cf_by_method={"gwp100_lcia": pd.Series({"AAL": None})},
    )

    passes_mod._record_pr_hr_cumulative_pass(  # noqa: SLF001
        context=_context(wb_df=pd.DataFrame(columns=["2005"])),
        state=empty_parent_state,
        year=2005,
        ssp_scenario=None,
        scenario_key=None,
        lcia_by_method=None,
        cumulative_kinds={"CBA_FD"},
        use_original_domain=False,
        lcia_effective_year_by_method=None,
    )

    assert empty_parent_state.enacting_metric_inputs == {}


def test_enacting_metric_policy_and_base_cover_registry_and_projection_routes() -> None:
    assert policy_mod._required_lcia_percap_kinds(  # noqa: SLF001
        context=_context(selected_l1=["AR(Ecap^{PBA})", "PR(GDPcap)"], combined=[])
    ) == {"PBA"}
    assert policy_mod._required_pr_hr_cumulative_kinds(  # noqa: SLF001
        context=_context(
            selected_l1=["AR(Ecap^{PBA})"],
            combined=[("UT(FD)", "PR-HR(Ecap,cum^{CBA_FD})")],
        )
    ) == {"CBA_FD"}

    future_projection = SimpleNamespace(
        enabled=True,
        is_future_year=lambda year: int(year) == 2030,
        route_for_l2_method=lambda name: "historical_reuse" if name == "UT(FD)" else "regression",
    )
    assert (
        base_mod._required_base_enacting_metric_keys(  # noqa: SLF001
            context=_context(
                fu_code="L2.a.a",
                selected_l2_one_step=["UT(FD)"],
                combined=[],
                projection_context=future_projection,
            ),
            year=2030,
        )
        == set()
    )
    assert base_mod._required_base_enacting_metric_keys(  # noqa: SLF001
        context=_context(
            fu_code="L2.a.a",
            selected_l2_one_step=["UT(FD)"],
            combined=[],
            projection_context=future_projection,
        ),
        year=2005,
    ) == {"fd_rf", "fd_rp_sp", "fd_rp_sp_rf"}

    empty_state = RunState()
    base_mod.record_base_enacting_metrics(
        context=_context(
            fu_code="L2.a.a",
            wb_df=pd.DataFrame(columns=["2005"]),
            selected_l2_one_step=["AR(E^{CBA_FD})"],
            combined=[],
            projection_context=None,
        ),
        state=empty_state,
        year=2005,
        ssp_scenario="SSP2",
        enacting_metric_l1={},
        enacting_metric_l2={},
        utility={},
    )
    assert empty_state.enacting_metric_inputs == {}

    payload = _ut_payload()
    state = RunState()
    base_mod.record_base_enacting_metrics(
        context=_context(
            fu_code="L2.a.a",
            wb_df=pd.DataFrame(columns=["2005"]),
            selected_l2_one_step=["UT(FD)"],
            combined=[],
            projection_context=None,
        ),
        state=state,
        year=2005,
        ssp_scenario="SSP2",
        enacting_metric_l1={
            "fd_rf": payload["fd_rf"],
            "gva_rp": payload["gva_rp"],
        },
        enacting_metric_l2={
            "fd_rp_sp_rf": payload["fd_rp_sp_rf"],
            "fd_rp_sp": payload["fd_rp_sp"],
            "fd_rf_sp": payload["fd_rf_sp"],
            "gva_rp_sp": payload["gva_rp_sp"],
        },
        utility={
            "x_to_rc": payload["x_to_rc"],
        },
    )
    assert (
        state.enacting_metric_levels[EnactingMetricKey(metric="fd_rf", ssp_scenario=None)]
        == "level_1"
    )
    assert (
        state.enacting_metric_levels[EnactingMetricKey(metric="fd_rp_sp", ssp_scenario=None)]
        == "level_2"
    )
    assert state.enacting_metric_inputs[EnactingMetricKey(metric="fd_rp_sp_rf", ssp_scenario=None)][
        2005
    ].index.names == ["r_p", "s_p", "r_f"]


def test_enacting_metric_lcia_recorders_cover_required_key_and_preweight_paths() -> None:
    no_lcia_state = RunState()
    base_mod.record_lcia_enacting_metrics(
        context=_context(
            fu_code="L2.a.a",
            selected_l1=["AR(Ecap^{PBA})"],
            selected_l2_one_step=[],
            combined=[],
        ),
        state=no_lcia_state,
        year=2030,
        lcia_by_method=None,
    )
    assert no_lcia_state.enacting_metric_inputs == {}

    base_mod.record_lcia_enacting_metrics(
        context=_context(
            fu_code="L2.a.a",
            selected_l1=["EG(Pop)"],
            selected_l2_one_step=["UT(FD)"],
            combined=[],
        ),
        state=no_lcia_state,
        year=2030,
        lcia_by_method={"skip_method": {"e_pba_reg": _impact_frame(axis="r_p")}},
    )
    assert no_lcia_state.enacting_metric_inputs == {}

    state = RunState()
    base_mod.record_lcia_enacting_metrics(
        context=_context(
            fu_code="L2.a.a",
            selected_l1=["AR(Ecap^{PBA})"],
            selected_l2_one_step=["AR(E^{CBA_FD})"],
            combined=[("AR(E^{CBA_FD})", "AR(E^{CBA_FD})")],
        ),
        state=state,
        year=2030,
        lcia_by_method={
            "gwp100_lcia": {
                "e_pba_reg": _impact_frame(axis="r_p"),
                "e_cba_fd_reg": _impact_frame(axis="r_f"),
                "e_cba_fd_rp_sp": pd.DataFrame(
                    [[11.0, 15.0]],
                    index=pd.Index(["AAL"], name="impact"),
                    columns=pd.MultiIndex.from_tuples(
                        [("FR", "A"), ("US", "A")],
                        names=["r_p", "s_p"],
                    ),
                ),
                "e_cba_fd_rp_sp_rf": pd.DataFrame(
                    [[1.0, 2.0], [3.0, 4.0]],
                    index=pd.MultiIndex.from_tuples(
                        [("AAL", "FR", "A"), ("AAL", "US", "A")],
                        names=["impact", "r_p", "s_p"],
                    ),
                    columns=pd.Index(["FR", "US"], name="r_f"),
                ),
            },
            "skip_method": {
                "e_pba_reg": _impact_frame(axis="r_p"),
            },
            "partial_method": {
                "e_pba_reg": _impact_frame(axis="r_p"),
            },
        },
        lcia_effective_year_by_method={
            "gwp100_lcia": 2030,
            "skip_method": 2029,
            "partial_method": 2030,
        },
    )
    assert (
        state.enacting_metric_levels[
            EnactingMetricKey(metric="e_pba_reg", lcia_method="gwp100_lcia")
        ]
        == "level_1"
    )
    assert (
        state.enacting_metric_levels[
            EnactingMetricKey(metric="e_cba_fd_rp_sp", lcia_method="gwp100_lcia")
        ]
        == "level_2"
    )
    assert EnactingMetricKey(metric="e_pba_reg", lcia_method="skip_method") not in (
        state.enacting_metric_inputs
    )

    full_weight_axis_state = RunState()
    base_mod.record_lcia_enacting_metrics(
        context=_context(
            fu_code="L2.c.b",
            selected_l1=["AR(Ecap^{CBA_FD})"],
            selected_l2_one_step=[],
            combined=[],
        ),
        state=full_weight_axis_state,
        year=2030,
        lcia_by_method={
            "gwp100_lcia": {
                "e_cba_fd_reg": _impact_frame(axis="r_f"),
            }
        },
    )
    full_weight_axis_key = EnactingMetricKey(
        metric="e_cba_fd_reg",
        lcia_method="gwp100_lcia",
    )
    assert full_weight_axis_state.enacting_metric_inputs[full_weight_axis_key][
        2030
    ].index.get_level_values("r_f").tolist() == ["FR", "DE"]

    all_scope_state = RunState()
    base_mod.record_lcia_enacting_metrics(
        context=_context(
            filters={
                "r_p": None,
                "s_p": None,
                "r_c": None,
                "r_f": None,
                "r_u": None,
            },
            selected_l1=[],
            selected_l2_one_step=["AR(E^{CBA_FD})"],
            combined=[],
        ),
        state=all_scope_state,
        year=2030,
        lcia_by_method={
            "gwp100_lcia": {
                "e_cba_fd_rp_sp": pd.DataFrame(
                    [[11.0, 15.0]],
                    index=pd.Index(["AAL"], name="impact"),
                    columns=pd.MultiIndex.from_tuples(
                        [("FR", "A"), ("US", "B")],
                        names=["r_p", "s_p"],
                    ),
                ),
            }
        },
    )
    all_scope_key = EnactingMetricKey(
        metric="e_cba_fd_rp_sp",
        lcia_method="gwp100_lcia",
    )
    assert all_scope_state.enacting_metric_inputs[all_scope_key][2030].index.tolist() == [
        ("AAL", "FR", "A"),
        ("AAL", "US", "B"),
    ]

    payload = _ut_payload()
    preweight_state = RunState()
    preweight_state.preweight_cache_by_ssp_scenario = {None: {}, "SSP2": {}}
    base_mod.record_adjusted_ut_preweights(
        context=_context(
            fu_code="L2.a.b",
            filters={
                "r_p": ["FR"],
                "s_p": ["A"],
                "r_c": ["FR"],
                "r_f": ["FR"],
                "r_u": None,
            },
            combined=[("UT(FDa)", "EG(Pop)"), ("UT(GVAa)", "EG(Pop)")],
        ),
        state=preweight_state,
        year=2005,
        enacting_metric_l1={
            "fd_rf": payload["fd_rf"],
            "gva_rp": payload["gva_rp"],
        },
        enacting_metric_l2={
            "fd_rp_sp_rf": payload["fd_rp_sp_rf"],
            "fd_rp_sp": payload["fd_rp_sp"],
            "fd_rf_sp": payload["fd_rf_sp"],
            "gva_rp_sp": payload["gva_rp_sp"],
        },
        utility={
            "x_to_rc": payload["x_to_rc"],
            "kappa": payload["kappa"],
            "omega_reg": payload["omega_reg"],
        },
    )
    assert EnactingMetricKey(metric="fda_r_p_s_p_r_f") in preweight_state.enacting_metric_inputs
    assert EnactingMetricKey(metric="gvaa_r_p_s_p_r_u") in preweight_state.enacting_metric_inputs
    assert set(preweight_state.preweight_cache_by_ssp_scenario[None]) == {
        ("UT(FDa)", "L2.a.b", 2005),
        ("UT(GVAa)", "L2.a.b", 2005),
    }
    assert set(preweight_state.preweight_cache_by_ssp_scenario["SSP2"]) == {
        ("UT(FDa)", "L2.a.b", 2005),
        ("UT(GVAa)", "L2.a.b", 2005),
    }

    skipped_preweight_state = RunState()
    base_mod.record_adjusted_ut_preweights(
        context=_context(
            fu_code="L2.a.b",
            filters={
                "r_p": ["FR"],
                "s_p": ["A"],
                "r_c": ["FR"],
                "r_f": ["FR"],
                "r_u": None,
            },
            combined=[("UT(FDa)", "EG(Pop)")],
        ),
        state=skipped_preweight_state,
        year=2005,
        enacting_metric_l1={
            "fd_rf": payload["fd_rf"],
            "gva_rp": payload["gva_rp"],
        },
        enacting_metric_l2={
            "fd_rp_sp_rf": payload["fd_rp_sp_rf"],
            "fd_rp_sp": payload["fd_rp_sp"],
            "fd_rf_sp": payload["fd_rf_sp"],
            "gva_rp_sp": payload["gva_rp_sp"],
        },
        utility={
            "x_to_rc": payload["x_to_rc"],
            "kappa": payload["kappa"],
            "omega_reg": payload["omega_reg"],
        },
    )
    assert (
        EnactingMetricKey(metric="fda_r_p_s_p_r_f")
        in skipped_preweight_state.enacting_metric_inputs
    )
    assert EnactingMetricKey(metric="gvaa_r_p_s_p_r_u") not in (
        skipped_preweight_state.enacting_metric_inputs
    )


def test_pr_and_lcia_percap_recorders_cover_direct_and_grouped_routes() -> None:
    grouped_context = _context(
        wb_df=pd.DataFrame(columns=["2005"]),
        group_version_reg="demo",
        l1_only_no_mrio=False,
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None, "r_u": None},
    )
    grouped_state = RunState()
    pr_mod.record_pr_enacting_metrics(
        context=grouped_context,
        state=grouped_state,
        year=2030,
        ssp_scenario="SSP2",
        reg_group_map={"FR": "EU", "US": "NAM"},
        pop_iso=pd.Series([2.0, 0.0], index=pd.Index(["FRA", "USA"], name="iso3")),
        gdp_iso=pd.Series([20.0, 40.0], index=pd.Index(["FRA", "USA"], name="iso3")),
        iso_to_mrio=pd.Series(["FR", "US"], index=pd.Index(["FRA", "USA"], name="iso3")),
    )
    grouped_key = EnactingMetricKey(metric="gdp_capita", ssp_scenario="SSP2")
    assert grouped_state.enacting_metric_inputs[grouped_key][2030].index.names == [
        "iso3_code",
        "mrio_code",
        "grouped_mrio_code",
    ]
    assert pd.isna(grouped_state.enacting_metric_inputs[grouped_key][2030].iloc[1])

    l1_only_state = RunState()
    pr_mod.record_pr_enacting_metrics(
        context=_context(wb_df=pd.DataFrame(columns=["2005"]), l1_only_no_mrio=True),
        state=l1_only_state,
        year=2005,
        ssp_scenario="SSP2",
        reg_group_map={},
        pop_iso=pd.Series([2.0], index=pd.Index(["FRA"], name="iso3")),
        gdp_iso=pd.Series([20.0], index=pd.Index(["FRA"], name="iso3")),
        iso_to_mrio=pd.Series(["FR"], index=pd.Index(["FRA"], name="iso3")),
    )
    assert l1_only_state.enacting_metric_inputs[
        EnactingMetricKey(metric="population", ssp_scenario=None)
    ][2005].index.tolist() == ["FRA"]

    assert pr_mod._build_pr_index(  # noqa: SLF001
        pop_index=pd.Index(["FRA"], name="iso3"),
        iso_to_mrio=pd.Series(["FR"], index=pd.Index(["FRA"], name="iso3")),
        reg_group_map={},
        include_grouped_col=False,
    ).names == ["iso3_code", "mrio_code"]
    with pytest.raises(ValueError):
        pr_mod._build_pr_index(  # noqa: SLF001
            pop_index=pd.Index(["FRA", "USA"], name="iso3"),
            iso_to_mrio=pd.Series(["FR"], index=pd.Index(["FRA"], name="iso3")),
            reg_group_map={},
            include_grouped_col=False,
        )
    with pytest.raises(ValueError):
        pr_mod._build_pr_index(  # noqa: SLF001
            pop_index=pd.Index(["FRA"], name="iso3"),
            iso_to_mrio=pd.Series(["FR"], index=pd.Index(["FRA"], name="iso3")),
            reg_group_map={"US": "NAM"},
            include_grouped_col=True,
        )

    direct_state = RunState()
    percap_mod.record_lcia_percap_enacting_metrics(
        context=_context(
            wb_df=pd.DataFrame(columns=["2005"]),
            selected_l1=["AR(Ecap^{PBA})"],
            combined=[],
        ),
        state=direct_state,
        year=2030,
        ssp_scenario="SSP2",
        lcia_by_method={"gwp100_lcia": {"e_pba_reg": _impact_frame(axis="r_p")}},
        pop_series=pd.Series([2.0, 4.0], index=pd.Index(["FR", "DE"], name="r_p")),
        use_original_domain=False,
    )
    assert (
        EnactingMetricKey(
            metric="e_pba_reg_cap",
            lcia_method="gwp100_lcia",
            ssp_scenario="SSP2",
        )
        in direct_state.enacting_metric_inputs
    )

    cumulative_population = pd.Series([2.0, 4.0], index=pd.Index(["FR", "DE"], name="r_f"))
    cumulative_state = RunState(
        pop_series_by_ssp_scenario={
            None: {2005: cumulative_population},
            "SSP2": {2030: cumulative_population},
        },
        pr_post_pop_series_by_ssp_scenario={
            None: {2005: cumulative_population},
            "SSP2": {2030: cumulative_population},
        },
        lcia_timeseries={"gwp100_lcia": {"CBA_FD": {2030: _impact_frame(axis="r_f")}}},
        rps_by_method={
            "gwp100_lcia": pd.DataFrame({"impact": ["AAL"], "responsibility_period_years": [1]})
        },
        cf_by_method={"gwp100_lcia": pd.Series({"AAL": "AAL"})},
    )
    percap_mod.record_lcia_percap_enacting_metrics(
        context=_context(
            wb_df=pd.DataFrame(columns=["2005"]),
            selected_l1=["PR-HR(Ecap,cum^{CBA_FD})"],
            combined=[],
        ),
        state=cumulative_state,
        year=2030,
        ssp_scenario="SSP2",
        lcia_by_method=None,
        pop_series=cumulative_population,
        use_original_domain=False,
    )
    assert (
        EnactingMetricKey(
            metric="e_cba_fd_reg_cap_cum",
            lcia_method="gwp100_lcia",
            ssp_scenario="SSP2",
        )
        in cumulative_state.enacting_metric_inputs
    )

from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.download.pop_gdp.contracts import (
    GDP_SSP_INDICATOR,
    GDP_WB_INDICATOR,
    POP_SSP_INDICATOR,
    POP_WB_INDICATOR,
)
from pyaesa.asocc.io.metadata import RunContext, RunState
from pyaesa.asocc.orchestration.projection.config.types import ProjectionContext
from pyaesa.asocc.orchestration.setup.loading.loading import _load_source_tables
from pyaesa.asocc.orchestration.yearly import run_year as run_year_mod
from pyaesa.asocc.orchestration.yearly.shared import scenario_processing as scenario_mod
from pyaesa.asocc.orchestration.yearly.shared import year_inputs as year_inputs_mod
from pyaesa.asocc.orchestration.yearly.shared.year_inputs import (
    _MrioPayload,
    _ScenarioRunContext,
)
from pyaesa.asocc.data.paths import _get_mrio_year_dir
from pyaesa.asocc.runtime.output.contracts import OutputRoute, OutputSpec


class _RecorderLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(str(message))

    def warning(self, message: str) -> None:
        self.warning_messages.append(str(message))


class _ProgressRecorder:
    def __init__(self) -> None:
        self.started: list[int] = []
        self.completed: list[int] = []

    def begin_year(self, year: int) -> None:
        self.started.append(int(year))

    def complete_year(self, year: int) -> None:
        self.completed.append(int(year))


def _wb_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": POP_WB_INDICATOR, "iso3_code": "FRA", "oecd_code": "FR", "2005": 1.0},
            {"variable": POP_WB_INDICATOR, "iso3_code": "DEU", "oecd_code": "DE", "2005": 2.0},
            {"variable": GDP_WB_INDICATOR, "iso3_code": "FRA", "oecd_code": "FR", "2005": 10.0},
            {"variable": GDP_WB_INDICATOR, "iso3_code": "DEU", "oecd_code": "DE", "2005": 20.0},
        ]
    )


def _ssp_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": POP_SSP_INDICATOR,
                "scenario": "SSP2",
                "iso3_code": "FRA",
                "oecd_code": "FR",
                "2030": 3.0,
            },
            {
                "variable": POP_SSP_INDICATOR,
                "scenario": "SSP2",
                "iso3_code": "DEU",
                "oecd_code": "DE",
                "2030": 4.0,
            },
            {
                "variable": GDP_SSP_INDICATOR,
                "scenario": "SSP2",
                "iso3_code": "FRA",
                "oecd_code": "FR",
                "2030": 30.0,
            },
            {
                "variable": GDP_SSP_INDICATOR,
                "scenario": "SSP2",
                "iso3_code": "DEU",
                "oecd_code": "DE",
                "2030": 40.0,
            },
        ]
    )


def _projection_context(route_by_name: dict[str, str]) -> ProjectionContext:
    return ProjectionContext(
        enabled=True,
        mode="regression",
        max_historical_year=2005,
        future_years=(2030,),
        reg_window=(2000, 2005),
        l2_reuse_years=(2005,),
        ut_methods_in_scope=tuple(route_by_name),
        l2_method_route_by_name=route_by_name,
    )


def _context(
    *,
    tmp_path: Path,
    selected_l1: list[str],
    selected_l2_one_step: list[str],
    combined: list[tuple[str, str]],
    projection_context: ProjectionContext | None = None,
    intermediate_outputs: bool = False,
) -> RunContext:
    wb_df = _wb_frame()
    ssp_df = _ssp_frame()
    return RunContext(
        project_name="scenario_processing_demo",
        source="oecd_v2025",
        fu_code="L2.a.a",
        group_version=None,
        group_version_reg=None,
        group_reg=False,
        group_sec=False,
        lcia_method=None,
        years_input=[2005, 2030],
        reference_years_input=None,
        ssp_scenario="SSP2",
        is_exio=False,
        l1_lcia_kind="CBA_FD",
        lcia_methods=[],
        selected_l1=selected_l1,
        combined=combined,
        selected_l2_one_step=selected_l2_one_step,
        required_indices=set(),
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None, "r_u": None},
        studied_indices_tag="demo",
        proj_base=tmp_path,
        logger=_RecorderLogger(),
        requested_years=[2005, 2030],
        resolved_years=[2005, 2030],
        persisted_years=[2005, 2030],
        historical_years=[2005],
        reference_years=None,
        ssp_scenario_options=[None, "SSP2"],
        run_signature={"selected_methods": {"L1": selected_l1, "L2": selected_l2_one_step}},
        needs_lcia=False,
        repo_root=tmp_path,
        wb_df=wb_df,
        ssp_df=ssp_df,
        wb_df_raw=wb_df.copy(),
        ssp_df_raw=ssp_df.copy(),
        selected_methods={"L1": selected_l1, "L2": selected_l2_one_step},
        l1_kinds_needed=set(),
        l1_only_no_mrio=False,
        l1_reg_aggreg="post",
        use_original_l1_post_domain=False,
        variant_tag=None,
        aggreg_indices=False,
        output_format="csv",
        intermediate_outputs=intermediate_outputs,
        output_source_label="oecd_v2025",
        projection_context=projection_context,
        ssp_scenario_options_by_year={2030: ["SSP2"]},
    )


def _package_run_context(
    *,
    tmp_path: Path,
    source: str,
    group_version: str | None,
    group_version_reg: str | None,
    selected_l1: list[str],
    selected_l2_one_step: list[str],
    combined: list[tuple[str, str]],
    historical_years: list[int],
    reference_years: list[int] | None,
    needs_lcia: bool,
    use_original_l1_post_domain: bool,
    intermediate_outputs: bool,
    ssp_scenario_options_by_year: dict[int, list[str | None]],
) -> RunContext:
    wb_df, ssp_df, wb_df_raw, ssp_df_raw = _load_source_tables(source=source)
    return RunContext(
        project_name="scenario_processing_demo",
        source=source,
        fu_code="L1.b",
        group_version=group_version,
        group_version_reg=group_version_reg,
        group_reg=group_version_reg is not None,
        group_sec=False,
        lcia_method=None,
        years_input=[*historical_years, 2030],
        reference_years_input=reference_years,
        ssp_scenario="SSP2",
        is_exio=source.startswith("exiobase_"),
        l1_lcia_kind="CBA_FD",
        lcia_methods=["gwp100_lcia"],
        selected_l1=selected_l1,
        combined=combined,
        selected_l2_one_step=selected_l2_one_step,
        required_indices=set(),
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None, "r_u": None},
        studied_indices_tag="demo",
        proj_base=tmp_path,
        logger=_RecorderLogger(),
        requested_years=[2030],
        resolved_years=[2030],
        persisted_years=[2030],
        historical_years=historical_years,
        reference_years=reference_years,
        ssp_scenario_options=[None, "SSP2"],
        run_signature={"selected_methods": {"L1": selected_l1, "L2": selected_l2_one_step}},
        needs_lcia=needs_lcia,
        repo_root=tmp_path,
        wb_df=wb_df,
        ssp_df=ssp_df,
        wb_df_raw=wb_df_raw,
        ssp_df_raw=ssp_df_raw,
        selected_methods={"L1": selected_l1, "L2": selected_l2_one_step},
        l1_kinds_needed=set(),
        l1_only_no_mrio=False,
        l1_reg_aggreg="post",
        use_original_l1_post_domain=use_original_l1_post_domain,
        variant_tag=None,
        aggreg_indices=False,
        output_format="csv",
        intermediate_outputs=intermediate_outputs,
        output_source_label=source,
        projection_context=None,
        ssp_scenario_options_by_year=ssp_scenario_options_by_year,
    )


def _state() -> RunState:
    state = RunState()
    state.l1_results_by_ssp_scenario = {None: {}, "SSP2": {}}
    state.l2_results_by_ssp_scenario = {None: {}, "SSP2": {}}
    state.pop_series_by_ssp_scenario = {None: {}, "SSP2": {}}
    state.gdp_series_by_ssp_scenario = {None: {}, "SSP2": {}}
    return state


def _output_spec(*, scenario_dependent: bool) -> OutputSpec:
    return OutputSpec(
        l1_l2_method="demo",
        l2_method=None,
        l1_method="AR(E^{CBA_FD})",
        file_stem="demo",
        route=OutputRoute(
            level="level_1",
            bucket=None,
            source="oecd_v2025",
            grouped_mode=False,
            variant_tag=None,
            ssp_scenario=None,
            lcia_method=None,
        ),
        scenario_dependent=scenario_dependent,
        identifier_columns=("r_p",),
    )


def _ar_lcia_payload() -> dict[str, dict[str, pd.DataFrame]]:
    return {
        "gwp100_lcia": {
            "e_cba_fd_reg": pd.DataFrame(
                [[10.0, 20.0]],
                index=pd.Index(["climate_child"], name="impact"),
                columns=pd.Index(["FR", "US"], name="r_f"),
            ),
            "e_cba_fd_rp_sp_rf": pd.DataFrame(
                [[1.0, 2.0], [3.0, 4.0]],
                index=pd.MultiIndex.from_tuples(
                    [("climate_child", "FR", "A"), ("climate_child", "US", "A")],
                    names=["impact", "r_p", "s_p"],
                ),
                columns=pd.Index(["FR", "US"], name="r_f"),
            ),
        }
    }


def test_scenario_processing_contracts_cover_projection_selection_and_caches(
    tmp_path: Path,
) -> None:
    context = _context(
        tmp_path=tmp_path,
        selected_l1=["AR(E^{CBA_FD})", "PR(GDPcap)"],
        selected_l2_one_step=["UT(FD)", "UT(GVAa)"],
        combined=[("UT(FD)", "PR(GDPcap)")],
        projection_context=_projection_context(
            {"UT(FD)": "regression", "UT(GVAa)": "historical_reuse"}
        ),
    )
    run_ctx = _ScenarioRunContext(context=context, state=_state(), year=2030, ssp_scenario="SSP2")

    assert (
        scenario_mod._needs_regression_projection_payload(  # noqa: SLF001
            context=context,
            year=2030,
        )
        is True
    )
    assert (
        scenario_mod._needs_regression_projection_payload(  # noqa: SLF001
            context=context,
            year=2005,
        )
        is False
    )
    assert (
        scenario_mod._needs_regression_projection_payload(  # noqa: SLF001
            context=_context(
                tmp_path=tmp_path,
                selected_l1=[],
                selected_l2_one_step=["UT(GVAa)"],
                combined=[],
                projection_context=None,
            ),
            year=2030,
        )
        is False
    )

    assert scenario_mod._select_scenario_methods(  # noqa: SLF001
        run_ctx=run_ctx,
        process_invariant_methods=True,
    ) == (
        ["AR(E^{CBA_FD})", "PR(GDPcap)"],
        ["UT(FD)", "UT(GVAa)"],
        [("UT(FD)", "PR(GDPcap)")],
    )
    assert scenario_mod._select_scenario_methods(  # noqa: SLF001
        run_ctx=run_ctx,
        process_invariant_methods=False,
    ) == (
        ["PR(GDPcap)"],
        ["UT(FD)"],
        [("UT(FD)", "PR(GDPcap)")],
    )

    assert scenario_mod._l1_key_matches_method(  # noqa: SLF001
        key="AR(E^{CBA_FD})",
        l1_method="AR(E^{CBA_FD})",
    )
    assert scenario_mod._l1_key_matches_method(  # noqa: SLF001
        key="AR(E^{CBA_FD})__for__UT(FD)",
        l1_method="AR(E^{CBA_FD})",
    )
    assert not scenario_mod._l1_key_matches_method(  # noqa: SLF001
        key="PR(GDPcap)",
        l1_method="AR(E^{CBA_FD})",
    )

    state = _state()
    l1_results_year = {
        "AR(E^{CBA_FD})": pd.DataFrame({"value": [1.0]}),
        "AR(E^{CBA_FD})__for__UT(FD)": pd.DataFrame({"value": [2.0]}),
        "PR(GDPcap)": pd.DataFrame({"value": [3.0]}),
    }
    scenario_mod._cache_invariant_l1_results_for_year(  # noqa: SLF001
        run_ctx=run_ctx._replace(state=state),
        l1_results_year=l1_results_year,
    )
    assert sorted(state.l1_year_invariant_cache[2030]) == [
        "AR(E^{CBA_FD})",
        "AR(E^{CBA_FD})__for__UT(FD)",
    ]

    merged = {"AR(E^{CBA_FD})": pd.DataFrame({"value": [9.0]})}
    scenario_mod._merge_cached_invariant_l1_results(  # noqa: SLF001
        run_ctx=run_ctx._replace(state=state),
        l1_results_year=merged,
    )
    assert merged["AR(E^{CBA_FD})"].iloc[0, 0] == 9.0
    assert "AR(E^{CBA_FD})__for__UT(FD)" in merged

    empty_inputs = scenario_mod._build_empty_l2_inputs()  # noqa: SLF001
    assert empty_inputs.fd_rf.empty
    assert empty_inputs.fd_rp_sp_rf.empty
    assert empty_inputs.x_to_rc.empty


def test_process_scenario_for_year_covers_no_payload_return_and_cleanup(tmp_path: Path) -> None:
    context = _context(
        tmp_path=tmp_path,
        selected_l1=[],
        selected_l2_one_step=[],
        combined=[],
        projection_context=None,
        intermediate_outputs=False,
    )
    state = _state()
    invariant_spec = _output_spec(scenario_dependent=False)
    scenario_spec = _output_spec(scenario_dependent=True)
    state.l1_results_by_ssp_scenario[None] = {
        invariant_spec: [pd.DataFrame({"value": [1.0]})],
        scenario_spec: [pd.DataFrame({"value": [2.0]})],
    }
    state.l1_year_invariant_cache[2005] = {"cached": pd.DataFrame({"value": [3.0]})}
    run_ctx = _ScenarioRunContext(context=context, state=state, year=2005, ssp_scenario=None)

    scenario_mod._process_scenario_for_year(  # noqa: SLF001
        run_ctx=run_ctx,
        lcia_by_method=None,
        lcia_by_method_original=None,
        lcia_effective_year_by_method=None,
        lcia_effective_year_by_method_original=None,
        reg_group_map={},
        mrio_payload=None,
        l2_inputs_sliced=None,
        process_invariant_methods=False,
    )

    assert 2005 in state.pop_series_by_ssp_scenario[None]
    assert 2005 in state.gdp_series_by_ssp_scenario[None]
    assert list(state.l1_results_by_ssp_scenario[None]) == [scenario_spec]


def test_process_scenario_for_year_covers_future_payload_path(tmp_path: Path) -> None:
    context = _context(
        tmp_path=tmp_path,
        selected_l1=[],
        selected_l2_one_step=[],
        combined=[],
        projection_context=None,
        intermediate_outputs=True,
    )
    state = _state()
    run_ctx = _ScenarioRunContext(context=context, state=state, year=2030, ssp_scenario="SSP2")

    scenario_mod._process_scenario_for_year(  # noqa: SLF001
        run_ctx=run_ctx,
        lcia_by_method=None,
        lcia_by_method_original=None,
        lcia_effective_year_by_method=None,
        lcia_effective_year_by_method_original=None,
        reg_group_map={},
        mrio_payload=_MrioPayload(
            enacting_metric_l1={},
            enacting_metric_l2={},
            utility={},
            l2_inputs=scenario_mod._build_empty_l2_inputs(),  # noqa: SLF001
        ),
        l2_inputs_sliced=scenario_mod._build_empty_l2_inputs(),  # noqa: SLF001
        process_invariant_methods=False,
    )

    assert 2030 in state.pop_series_by_ssp_scenario["SSP2"]
    assert 2030 in state.gdp_series_by_ssp_scenario["SSP2"]
    assert state.enacting_metric_inputs
    assert any(key.ssp_scenario == "SSP2" for key in state.enacting_metric_inputs)


def test_process_scenario_for_year_loads_projection_payload_when_future_mrio_is_missing(
    allocation_dummy_repo,
    tmp_path: Path,
) -> None:
    context = _package_run_context(
        tmp_path=tmp_path,
        source="oecd_v2025",
        group_version=None,
        group_version_reg=None,
        selected_l1=[],
        selected_l2_one_step=["UT(FD)"],
        combined=[],
        historical_years=[2005],
        reference_years=None,
        needs_lcia=False,
        use_original_l1_post_domain=False,
        intermediate_outputs=False,
        ssp_scenario_options_by_year={2030: ["SSP2"]},
    )
    context.fu_code = "L2.a.a"
    context.projection_context = ProjectionContext(
        enabled=True,
        mode="regression",
        max_historical_year=2005,
        future_years=(2030,),
        reg_window=(2005, 2005),
        l2_reuse_years=(2005,),
        ut_methods_in_scope=("UT(FD)",),
        l2_method_route_by_name={"UT(FD)": "regression"},
    )
    state = _state()
    year_dir = _get_mrio_year_dir(
        source=context.source,
        year=2005,
        group_version=context.group_version,
    )
    payload = year_inputs_mod._load_year_mrio_payloads_required(  # noqa: SLF001
        saved_dir=year_dir,
        context=context,
        needs_mrio=True,
    )
    assert payload is not None
    state.projection_payload_cache[(2030, "SSP2")] = payload

    scenario_mod._process_scenario_for_year(  # noqa: SLF001
        run_ctx=_ScenarioRunContext(context=context, state=state, year=2030, ssp_scenario="SSP2"),
        lcia_by_method=None,
        lcia_by_method_original=None,
        lcia_effective_year_by_method=None,
        lcia_effective_year_by_method_original=None,
        reg_group_map={},
        mrio_payload=None,
        l2_inputs_sliced=None,
        process_invariant_methods=True,
    )

    assert (2030, "SSP2") in state.projection_payload_cache
    assert state.l2_results_by_ssp_scenario["SSP2"]


def test_process_scenario_for_year_skips_base_enacting_metric_for_wb_backed_payload(
    allocation_dummy_repo,
    tmp_path: Path,
) -> None:
    del allocation_dummy_repo
    context = _package_run_context(
        tmp_path=tmp_path,
        source="oecd_v2025",
        group_version=None,
        group_version_reg=None,
        selected_l1=[],
        selected_l2_one_step=["UT(FD)"],
        combined=[],
        historical_years=[2005],
        reference_years=None,
        needs_lcia=False,
        use_original_l1_post_domain=False,
        intermediate_outputs=True,
        ssp_scenario_options_by_year={2005: [None]},
    )
    context.fu_code = "L2.a.a"
    state = _state()
    year_dir = _get_mrio_year_dir(
        source=context.source,
        year=2005,
        group_version=context.group_version,
    )
    payload = year_inputs_mod._load_year_mrio_payloads_required(  # noqa: SLF001
        saved_dir=year_dir,
        context=context,
        needs_mrio=True,
    )
    assert payload is not None
    sliced_inputs = scenario_mod._slice_l2_inputs_for_compute(  # noqa: SLF001
        context=context,
        inputs=payload.l2_inputs,
    )

    scenario_mod._process_scenario_for_year(  # noqa: SLF001
        run_ctx=_ScenarioRunContext(context=context, state=state, year=2005, ssp_scenario=None),
        lcia_by_method=None,
        lcia_by_method_original=None,
        lcia_effective_year_by_method=None,
        lcia_effective_year_by_method_original=None,
        reg_group_map={},
        mrio_payload=payload,
        l2_inputs_sliced=sliced_inputs,
        process_invariant_methods=False,
    )

    recorded_metrics = {key.metric for key in state.enacting_metric_inputs}
    assert {"population", "gdp_capita"}.issubset(recorded_metrics)
    assert "fd_rf" not in recorded_metrics


def test_process_scenario_for_year_runs_partial_l2_path_for_ar_without_mrio_payload(
    allocation_dummy_repo,
    tmp_path: Path,
) -> None:
    del allocation_dummy_repo
    context = _package_run_context(
        tmp_path=tmp_path,
        source="oecd_v2025",
        group_version=None,
        group_version_reg=None,
        selected_l1=[],
        selected_l2_one_step=["AR(E^{CBA_FD})"],
        combined=[],
        historical_years=[2005],
        reference_years=[2005],
        needs_lcia=True,
        use_original_l1_post_domain=False,
        intermediate_outputs=False,
        ssp_scenario_options_by_year={2005: [None]},
    )
    context.fu_code = "L2.a.a"
    state = _state()

    assert (
        run_year_mod._process_year(  # noqa: SLF001
            context=context,
            state=state,
            year=2005,
        )
        is True
    )
    assert state.l2_results_by_ssp_scenario[None]


def test_process_scenario_for_year_runs_partial_l2_path_for_ar_without_direct_mrio_payload(
    allocation_dummy_repo,
    tmp_path: Path,
) -> None:
    del allocation_dummy_repo
    context = _package_run_context(
        tmp_path=tmp_path,
        source="oecd_v2025",
        group_version=None,
        group_version_reg=None,
        selected_l1=[],
        selected_l2_one_step=["AR(E^{CBA_FD})"],
        combined=[],
        historical_years=[2005],
        reference_years=[2005],
        needs_lcia=True,
        use_original_l1_post_domain=False,
        intermediate_outputs=False,
        ssp_scenario_options_by_year={2005: [None]},
    )
    context.fu_code = "L2.a.a"
    state = _state()

    scenario_mod._process_scenario_for_year(  # noqa: SLF001
        run_ctx=_ScenarioRunContext(context=context, state=state, year=2005, ssp_scenario=None),
        lcia_by_method=_ar_lcia_payload(),
        lcia_by_method_original=None,
        lcia_effective_year_by_method=None,
        lcia_effective_year_by_method_original=None,
        reg_group_map={},
        mrio_payload=None,
        l2_inputs_sliced=None,
        process_invariant_methods=True,
    )

    assert state.l2_results_by_ssp_scenario[None]


def test_year_input_cover_group_maps_metric_loading_and_source_resolution(
    allocation_dummy_repo,
    tmp_path: Path,
) -> None:
    del allocation_dummy_repo
    grouped_context = _context(
        tmp_path=tmp_path,
        selected_l1=[],
        selected_l2_one_step=["UT(FD)", "UT(FDa)", "UT(GVAa)", "UT(GVA)", "UT(TD)"],
        combined=[],
        projection_context=None,
    )
    grouped_context.group_version_reg = "demo_reg"
    grouped_context.use_original_l1_post_domain = True
    for frame_name in ("wb_df", "ssp_df", "wb_df_raw", "ssp_df_raw"):
        grouped_frame = getattr(grouped_context, frame_name).replace({"DE": "US", "DEU": "USA"})
        setattr(grouped_context, frame_name, grouped_frame)
    state = _state()

    mapping = year_inputs_mod._load_reg_group_map(  # noqa: SLF001
        context=grouped_context,
        state=state,
    )
    cached_mapping = year_inputs_mod._load_reg_group_map(  # noqa: SLF001
        context=grouped_context,
        state=state,
    )
    assert mapping == {"FR": "EU", "US": "NAM"}
    assert cached_mapping == mapping
    assert (
        year_inputs_mod._load_reg_group_map(  # noqa: SLF001
            context=_context(
                tmp_path=tmp_path,
                selected_l1=[],
                selected_l2_one_step=[],
                combined=[],
                projection_context=None,
            ),
            state=_state(),
        )
        == {}
    )

    fd_context = _context(
        tmp_path=tmp_path,
        selected_l1=[],
        selected_l2_one_step=["UT(FD)"],
        combined=[],
        projection_context=None,
    )
    l1_keys, l2_keys, util_keys = year_inputs_mod._required_mrio_metric_keys(  # noqa: SLF001
        context=fd_context,
    )
    assert l1_keys == {"fd_rf"}
    assert l2_keys == {"fd_rp_sp_rf", "fd_rp_sp", "fd_rf_sp"}
    assert util_keys == set()

    utility_context = _context(
        tmp_path=tmp_path,
        selected_l1=[],
        selected_l2_one_step=["UT(FDa)", "UT(GVAa)", "UT(TD)"],
        combined=[],
        projection_context=None,
    )
    utility_context.fu_code = "L2.a.b"
    l1_keys, l2_keys, util_keys = year_inputs_mod._required_mrio_metric_keys(  # noqa: SLF001
        context=utility_context,
    )
    assert l1_keys == {"fd_rf", "gva_rp"}
    assert l2_keys == {"gva_rp_sp"}
    assert util_keys == {"x_to_rc", "kappa", "omega_reg"}

    utility_context_alt = _context(
        tmp_path=tmp_path,
        selected_l1=[],
        selected_l2_one_step=["UT(GVAa)"],
        combined=[],
        projection_context=None,
    )
    utility_context_alt.fu_code = "L2.b.b"
    l1_keys, l2_keys, util_keys = year_inputs_mod._required_mrio_metric_keys(  # noqa: SLF001
        context=utility_context_alt,
    )
    assert l1_keys == {"gva_rp"}
    assert l2_keys == set()
    assert util_keys == {"x_to_rc", "omega_reg"}

    gva_context = _context(
        tmp_path=tmp_path,
        selected_l1=[],
        selected_l2_one_step=["UT(GVA)"],
        combined=[],
        projection_context=None,
    )
    gva_context.fu_code = "L2.a.c"
    l1_keys, l2_keys, util_keys = year_inputs_mod._required_mrio_metric_keys(  # noqa: SLF001
        context=gva_context,
    )
    assert l1_keys == {"gva_rp"}
    assert l2_keys == {"gva_rp_sp"}
    assert util_keys == set()

    ar_context = _context(
        tmp_path=tmp_path,
        selected_l1=[],
        selected_l2_one_step=["AR(E^{CBA_FD})"],
        combined=[],
        projection_context=None,
    )
    l1_keys, l2_keys, util_keys = year_inputs_mod._required_mrio_metric_keys(  # noqa: SLF001
        context=ar_context,
    )
    assert l1_keys == set()
    assert l2_keys == set()
    assert util_keys == set()

    year_dir = _get_mrio_year_dir(
        source=fd_context.source,
        year=2005,
        group_version=fd_context.group_version,
    )
    assert (
        year_inputs_mod._load_year_mrio_payloads_required(  # noqa: SLF001
            saved_dir=year_dir,
            context=fd_context,
            needs_mrio=False,
        )
        is None
    )
    payload = year_inputs_mod._load_year_mrio_payloads_required(  # noqa: SLF001
        saved_dir=year_dir,
        context=fd_context,
        needs_mrio=True,
    )
    assert payload is not None
    assert not payload.enacting_metric_l1["fd_rf"].empty
    assert not payload.enacting_metric_l2["fd_rp_sp_rf"].empty

    utility_payload = year_inputs_mod._load_year_mrio_payloads_required(  # noqa: SLF001
        saved_dir=year_dir,
        context=utility_context,
        needs_mrio=True,
    )
    assert utility_payload is not None
    assert not utility_payload.enacting_metric_l1["fd_rf"].empty
    assert not utility_payload.enacting_metric_l1["gva_rp"].empty
    assert not utility_payload.enacting_metric_l2["gva_rp_sp"].empty
    assert not utility_payload.utility["x_to_rc"].empty
    assert not utility_payload.utility["kappa"].empty
    assert not utility_payload.utility["omega_reg"].empty

    gva_payload = year_inputs_mod._load_year_mrio_payloads_required(  # noqa: SLF001
        saved_dir=year_dir,
        context=gva_context,
        needs_mrio=True,
    )
    assert gva_payload is not None
    assert not gva_payload.enacting_metric_l1["gva_rp"].empty
    assert not gva_payload.enacting_metric_l2["gva_rp_sp"].empty

    historical_ctx = _ScenarioRunContext(
        context=grouped_context,
        state=state,
        year=2005,
        ssp_scenario="SSP2",
    )
    historical_source = year_inputs_mod._resolve_pop_gdp_source(historical_ctx)  # noqa: SLF001
    assert historical_source.use_ssp is False
    assert historical_source.scenario_arg is None
    assert historical_source.group_version_reg == "demo_reg"

    future_ctx = _ScenarioRunContext(
        context=grouped_context,
        state=state,
        year=2030,
        ssp_scenario="SSP2",
    )
    future_source = year_inputs_mod._resolve_pop_gdp_source(future_ctx)  # noqa: SLF001
    assert future_source.use_ssp is True
    assert future_source.scenario_arg == "SSP2"
    assert future_source.needs_pr_post_ungrouped is True

    iso3_context = _context(
        tmp_path=tmp_path,
        selected_l1=[],
        selected_l2_one_step=[],
        combined=[],
        projection_context=None,
    )
    iso3_context.source = "iso3"
    iso3_context.l1_only_no_mrio = True
    iso3_source = year_inputs_mod._resolve_pop_gdp_source(  # noqa: SLF001
        _ScenarioRunContext(
            context=iso3_context,
            state=_state(),
            year=2030,
            ssp_scenario="SSP2",
        )
    )
    assert iso3_source.region_override == "iso3_code"
    assert iso3_source.group_version_reg is None

    scenario_inputs = year_inputs_mod._load_scenario_population_gdp(  # noqa: SLF001
        run_ctx=future_ctx,
    )
    assert scenario_inputs.use_ssp is True
    assert scenario_inputs.pop_series_original is not None
    assert 2030 in state.pr_post_pop_series_by_ssp_scenario["SSP2"]


def test_run_year_cover_latest_existing_year_lookup_and_missing_year_skip(
    allocation_dummy_repo,
    tmp_path: Path,
) -> None:
    del allocation_dummy_repo
    context = _context(
        tmp_path=tmp_path,
        selected_l1=[],
        selected_l2_one_step=[],
        combined=[],
        projection_context=None,
    )
    existing_dir = _get_mrio_year_dir(
        source=context.source,
        year=2005,
        group_version=context.group_version,
    )
    existing_dir.mkdir(parents=True, exist_ok=True)

    latest_year, latest_dir = run_year_mod._latest_existing_year_dir(  # noqa: SLF001
        source=context.source,
        historical_years=[2004, 2005],
        year=2030,
        group_version=context.group_version,
    )
    assert latest_year == 2005
    assert latest_dir == existing_dir
    assert run_year_mod._latest_existing_year_dir(  # noqa: SLF001
        source=context.source,
        historical_years=[1990],
        year=1990,
        group_version=context.group_version,
    ) == (None, None)

    state = _state()
    assert (
        run_year_mod._process_year(  # noqa: SLF001
            context=context,
            state=state,
            year=2030,
        )
        is False
    )
    assert 2030 in state.skipped_years


def test_process_year_covers_l1_only_success_progress_and_gc_path(
    allocation_dummy_repo,
    tmp_path: Path,
) -> None:
    del allocation_dummy_repo
    context = _context(
        tmp_path=tmp_path,
        selected_l1=["EG(Pop)"],
        selected_l2_one_step=[],
        combined=[],
        projection_context=None,
    )
    context.l1_only_no_mrio = True
    context.needs_lcia = False
    context.ssp_scenario_options_by_year = {2005: [None]}
    state = _state()
    state.processed_years = list(range(1, 24))
    progress = _ProgressRecorder()

    assert (
        run_year_mod._process_year(  # noqa: SLF001
            context=context,
            state=state,
            year=2005,
            progress=cast(Any, progress),
        )
        is True
    )
    assert progress.started == [2005]
    assert progress.completed == [2005]
    assert state.processed_years[-1] == 2005
    assert len(state.processed_years) == 24


def test_process_year_covers_intermediate_recording_and_success_without_progress(
    allocation_dummy_repo,
    tmp_path: Path,
) -> None:
    del allocation_dummy_repo
    context = _context(
        tmp_path=tmp_path,
        selected_l1=[],
        selected_l2_one_step=["UT(FD)"],
        combined=[],
        projection_context=None,
        intermediate_outputs=True,
    )
    context.needs_lcia = False
    context.ssp_scenario_options_by_year = {2005: [None]}
    state = _state()

    assert (
        run_year_mod._process_year(  # noqa: SLF001
            context=context,
            state=state,
            year=2005,
        )
        is True
    )
    assert state.processed_years == [2005]
    assert state.l2_results_by_ssp_scenario[None]
    assert state.enacting_metric_inputs


def test_process_year_emits_notice_when_required_mrio_metrics_are_missing(
    allocation_dummy_repo,
    tmp_path: Path,
) -> None:
    del allocation_dummy_repo
    context = _context(
        tmp_path=tmp_path,
        selected_l1=[],
        selected_l2_one_step=["UT(FD)"],
        combined=[],
        projection_context=None,
        intermediate_outputs=False,
    )
    context.needs_lcia = False
    context.ssp_scenario_options_by_year = {2005: [None]}
    year_dir = _get_mrio_year_dir(
        source=context.source,
        year=2005,
        group_version=context.group_version,
    )
    (year_dir / "enacting_metrics" / "level_1" / "fd_rf.pickle").unlink()
    state = _state()

    assert (
        run_year_mod._process_year(  # noqa: SLF001
            context=context,
            state=state,
            year=2005,
        )
        is True
    )
    assert state.processed_years == [2005]
    assert "l2-mrio-enacting-metric-metrics-missing" in state.notices_emitted
    assert context.logger.warning_messages


def test_process_year_covers_grouped_and_original_lcia_future_year_fallback(
    allocation_dummy_repo,
    tmp_path: Path,
) -> None:
    del allocation_dummy_repo
    context = _package_run_context(
        tmp_path=tmp_path,
        source="exiobase_396_ixi",
        group_version="oecd_d",
        group_version_reg="demo_reg",
        selected_l1=["AR(Ecap^{PBA})"],
        selected_l2_one_step=[],
        combined=[],
        historical_years=[2005, 2006],
        reference_years=[2005],
        needs_lcia=True,
        use_original_l1_post_domain=True,
        intermediate_outputs=False,
        ssp_scenario_options_by_year={2030: ["SSP2"]},
    )
    state = _state()

    assert (
        run_year_mod._process_year(  # noqa: SLF001
            context=context,
            state=state,
            year=2030,
        )
        is True
    )
    assert state.processed_years == [2030]
    assert state.l1_results_by_ssp_scenario["SSP2"]
    assert 2030 in state.pr_post_pop_series_by_ssp_scenario["SSP2"]


def test_process_year_covers_original_domain_current_year_lcia_loading(
    allocation_dummy_repo,
    tmp_path: Path,
) -> None:
    del allocation_dummy_repo
    context = _package_run_context(
        tmp_path=tmp_path,
        source="exiobase_396_ixi",
        group_version="oecd_d",
        group_version_reg="demo_reg",
        selected_l1=["AR(Ecap^{PBA})"],
        selected_l2_one_step=[],
        combined=[],
        historical_years=[2005, 2006],
        reference_years=[2005],
        needs_lcia=True,
        use_original_l1_post_domain=True,
        intermediate_outputs=False,
        ssp_scenario_options_by_year={2005: [None]},
    )
    state = _state()

    assert (
        run_year_mod._process_year(  # noqa: SLF001
            context=context,
            state=state,
            year=2005,
        )
        is True
    )
    assert state.processed_years == []
    assert state.l1_results_by_ssp_scenario[None] == {}
    assert ("gwp100_lcia", "PBA", True, None, (2005,)) in state.ar_valid_refs_cache
    assert 2005 in state.pr_post_pop_series_by_ssp_scenario[None]

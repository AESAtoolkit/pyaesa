from types import SimpleNamespace
from typing import Any

import pandas as pd

from pyaesa.asocc.orchestration.yearly.l2 import l2_outputs as outputs_mod
from pyaesa.asocc.orchestration.yearly.l2 import l2_reuse_frames as reuse_frames_mod
from pyaesa.asocc.orchestration.yearly.l2 import l2_types as types_mod
from pyaesa.asocc.runtime.output.contracts import OutputSpec


def _payload() -> dict[str, Any]:
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
        "lcia": {
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
            "e_cba_fd_rp_sp": pd.DataFrame(
                [[11.0, 15.0]],
                index=pd.Index(["climate_child"], name="impact"),
                columns=pd.MultiIndex.from_tuples(
                    [("FR", "A"), ("US", "A")],
                    names=["r_p", "s_p"],
                ),
            ),
            "e_cba_td_rp_sp": pd.DataFrame(
                [[3.0, 4.0]],
                index=pd.Index(["climate_child"], name="impact"),
                columns=pd.MultiIndex.from_tuples(
                    [("FR", "A"), ("US", "A")],
                    names=["r_p", "s_p"],
                ),
            ),
            "e_pba_rp_sp": pd.DataFrame(
                [[5.0, 7.0]],
                index=pd.Index(["climate_child"], name="impact"),
                columns=pd.MultiIndex.from_tuples(
                    [("FR", "A"), ("US", "A")],
                    names=["r_p", "s_p"],
                ),
            ),
            "e_pba_reg": pd.DataFrame(
                [[6.0, 8.0]],
                index=pd.Index(["climate_child"], name="impact"),
                columns=pd.Index(["FR", "US"], name="r_p"),
            ),
        },
    }


def _inputs() -> types_mod._L2ComputeInputs:
    payload = _payload()
    return types_mod._L2ComputeInputs(
        fd_rf=payload["fd_rf"],
        gva_rp=payload["gva_rp"],
        fd_rp_sp_rf=payload["fd_rp_sp_rf"],
        fd_rp_sp=payload["fd_rp_sp"],
        fd_rf_sp=payload["fd_rf_sp"],
        gva_rp_sp=payload["gva_rp_sp"],
        x_to_rc=payload["x_to_rc"],
        kappa=payload["kappa"],
        omega_reg=payload["omega_reg"],
    )


def _projection_context(*, route: str, mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        mode=mode,
        reg_window=(2005, 2006),
        is_future_year=lambda year: int(year) >= 2030,
        route_for_l2_method=lambda _l2_method: route,
    )


def _run(
    *,
    fu_code: str,
    year: int,
    ssp_scenario: str | None,
    intermediate_outputs: bool,
    projection_context: SimpleNamespace | None = None,
) -> types_mod._L2RunContext:
    context: Any = SimpleNamespace(
        source="oecd_v2025",
        group_version=None,
        aggreg_indices=False,
        needs_lcia=True,
        lcia_methods=["gwp100_lcia"],
        logger=SimpleNamespace(warning=lambda _message: None),
        proj_base=SimpleNamespace(name="demo"),
        fu_code=fu_code,
        selected_l1=[],
        selected_l2_one_step=[],
        combined=[],
        output_summed=False,
        intermediate_outputs=intermediate_outputs,
        wb_df=pd.DataFrame(columns=["2005", "2006"]),
        projection_context=projection_context,
        historical_years=[2005, 2006],
        persisted_years=[year],
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None, "r_u": None},
    )
    state: Any = SimpleNamespace(
        l2_results_by_ssp_scenario={ssp_scenario: {}},
        pre_weighting_written_by_ssp_scenario={ssp_scenario: set()},
        output_spec_cache={},
        ut_reuse_preweight_cache={},
        ar_l2_cache_by_ssp_scenario={None: {}},
        preweight_cache_by_ssp_scenario={None: {}},
        empty_ref_years={},
        output_index_level_cache={},
        notices_emitted=set(),
        lcia_metadata_cache={},
        lcia_available_years_cache={},
        lcia_method_payload_cache={},
        cf_by_method={},
        lcia_units={},
        skipped_years={},
        runtime_progress=None,
        runtime_source_prefix=None,
    )
    return types_mod._L2RunContext(
        context=context,
        state=state,
        year=year,
        ssp_scenario=ssp_scenario,
        lcia_by_method=None,
        l1_results_year={},
        inputs=_inputs(),
    )


def _indexed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {2030: [1.0], "reference_year": [2005]},
        index=pd.Index(["FR"], name="r_p"),
    )


def test_l2_output_spec_builders_and_subfolder() -> None:
    frame = _indexed_frame()
    state = SimpleNamespace(output_spec_cache={})

    spec_l2_in_l1 = outputs_mod._build_l2_output_spec(
        spec=outputs_mod._L2KeySpec(
            route="l2_in_l1",
            l2_method="UT(FD)",
            l1_method=None,
            lcia_method_name="gwp100_lcia",
            ssp_scenario=None,
            scenario_dependent=False,
            grouped_mode=False,
        ),
        frame=frame,
        state=state,
    )
    assert isinstance(spec_l2_in_l1, OutputSpec)
    assert spec_l2_in_l1.file_stem == "l2_UT(FD)__gwp100_lcia"
    assert "reference_year" in spec_l2_in_l1.identifier_columns

    spec_l2_vs = outputs_mod._build_l2_output_spec(
        spec=outputs_mod._L2KeySpec(
            route="l2_vs_global",
            l2_method="UT(FD)",
            l1_method=None,
            lcia_method_name=None,
            ssp_scenario="SSP2",
            scenario_dependent=True,
            grouped_mode=True,
            file_suffix="_x",
        ),
        frame=frame,
        state=state,
    )
    assert spec_l2_vs.file_stem == "UT(FD)"
    assert spec_l2_vs.persisted_stem == "UT(FD)"
    assert spec_l2_vs.file_name == "UT(FD)__ssp2__x.csv"

    spec_pair = outputs_mod._build_l2_output_spec(
        spec=outputs_mod._L2KeySpec(
            route="l2_vs_global",
            l2_method="UT(FD)",
            l1_method="EG(Pop)",
            lcia_method_name=None,
            ssp_scenario=None,
            scenario_dependent=False,
            grouped_mode=False,
        ),
        frame=frame,
        state=state,
    )
    assert spec_pair.file_stem == "EG(Pop)_UT(FD)"

    cached = outputs_mod._build_l2_output_spec(
        spec=outputs_mod._L2KeySpec(
            route="l2_vs_global",
            l2_method="UT(FD)",
            l1_method="EG(Pop)",
            lcia_method_name=None,
            ssp_scenario=None,
            scenario_dependent=False,
            grouped_mode=False,
        ),
        frame=frame,
        state=state,
    )
    assert cached is spec_pair


def test_l2_write_cover_real_scenario_routing_and_intermediate_flags() -> None:
    run = _run(
        fu_code="L2.a.b",
        year=2030,
        ssp_scenario="SSP2",
        intermediate_outputs=True,
        projection_context=_projection_context(route="historical_reuse", mode="historical_reuse"),
    )
    slice_spec = types_mod._L2SliceSpec(
        l2_method="UT(FDa)",
        l1_name="EG(Pop)",
        l1_name_resolved="EG(Pop)",
        lcia_key="gwp100_lcia",
        lcia_data=_payload()["lcia"],
        ref_year=2005,
        treat_as_one_step=False,
    )
    result = _indexed_frame()

    result_with_reuse = reuse_frames_mod._attach_l2_reuse_year_level(  # noqa: SLF001
        frame=result,
        l2_reuse_year=2020,
    )

    outputs_mod._write_l2_vs_global(
        run=run,
        slice_spec=slice_spec,
        result=result_with_reuse,
    )
    outputs_mod._write_l2_utility_propagation_contrib(
        run=run,
        slice_spec=slice_spec,
        result=result_with_reuse,
    )

    stored_specs = list(run.state.l2_results_by_ssp_scenario["SSP2"])
    assert any(spec.route.bucket == "l2_vs_global" for spec in stored_specs)
    assert any(spec.route.bucket == "utility_propagation_contrib" for spec in stored_specs)
    assert any(spec.route.projection_subfolder == "historical_reuse" for spec in stored_specs)
    assert all("l2_reuse_year" in spec.identifier_columns for spec in stored_specs)
    assert all("__l2_reuse_year_2020" not in spec.file_name for spec in stored_specs)
    assert any(spec.file_name.endswith("__per_rf.csv") for spec in stored_specs)

    support_run = _run(
        fu_code="L2.a.b",
        year=2005,
        ssp_scenario="SSP2",
        intermediate_outputs=True,
        projection_context=_projection_context(route="historical_reuse", mode="historical_reuse"),
    )
    support_run.context.persisted_years = [2030]
    outputs_mod._write_l2_vs_global(
        run=support_run,
        slice_spec=slice_spec,
        result=result_with_reuse,
    )
    outputs_mod._write_l2_utility_propagation_contrib(
        run=support_run,
        slice_spec=slice_spec,
        result=result_with_reuse,
    )
    assert support_run.state.l2_results_by_ssp_scenario["SSP2"] == {}

    run_no_intermediate = _run(
        fu_code="L2.a.b",
        year=2030,
        ssp_scenario="SSP2",
        intermediate_outputs=False,
        projection_context=_projection_context(route="historical_reuse", mode="historical_reuse"),
    )
    outputs_mod._write_l2_utility_propagation_contrib(
        run=run_no_intermediate,
        slice_spec=slice_spec,
        result=result,
    )
    assert run_no_intermediate.state.l2_results_by_ssp_scenario["SSP2"] == {}


def test_l2_write_preweight_covers_ut_and_ar_paths(allocation_dummy_repo) -> None:
    payload = _payload()

    run_ut = _run(
        fu_code="L2.a.b",
        year=2005,
        ssp_scenario=None,
        intermediate_outputs=False,
    )
    ut_slice = types_mod._L2SliceSpec(
        l2_method="UT(FDa)",
        l1_name="EG(Pop)",
        l1_name_resolved="EG(Pop)",
        lcia_key="gwp100_lcia",
        lcia_data=payload["lcia"],
        ref_year=None,
        treat_as_one_step=False,
    )
    outputs_mod._write_l2_preweight(run=run_ut, slice_spec=ut_slice)
    assert run_ut.state.pre_weighting_written_by_ssp_scenario[None]
    assert ("preweight", "UT(FDa)", None, 2005) in run_ut.state.ut_reuse_preweight_cache
    written_before = len(run_ut.state.l2_results_by_ssp_scenario[None])
    outputs_mod._write_l2_preweight(run=run_ut, slice_spec=ut_slice)
    assert len(run_ut.state.l2_results_by_ssp_scenario[None]) == written_before

    run_ut_support = _run(
        fu_code="L2.a.b",
        year=2005,
        ssp_scenario=None,
        intermediate_outputs=False,
    )
    run_ut_support.context.persisted_years = [2030]
    outputs_mod._write_l2_preweight(run=run_ut_support, slice_spec=ut_slice)
    assert run_ut_support.state.pre_weighting_written_by_ssp_scenario[None]
    assert ("preweight", "UT(FDa)", None, 2005) in run_ut_support.state.ut_reuse_preweight_cache
    assert run_ut_support.state.l2_results_by_ssp_scenario[None] == {}

    run_non_persisted_reuse = _run(
        fu_code="L2.a.a",
        year=2005,
        ssp_scenario=None,
        intermediate_outputs=True,
        projection_context=SimpleNamespace(l2_reuse_years_for=lambda: [2005]),
    )
    run_non_persisted_reuse.context.persisted_years = []
    outputs_mod._write_l2_historical_reuse_preweight(
        run=run_non_persisted_reuse,
        slice_spec=ut_slice,
    )
    assert run_non_persisted_reuse.state.l2_results_by_ssp_scenario[None] == {}

    run_ut_fd = _run(
        fu_code="L2.a.a",
        year=2005,
        ssp_scenario=None,
        intermediate_outputs=False,
    )
    ut_fd_slice = types_mod._L2SliceSpec(
        l2_method="UT(FD)",
        l1_name="EG(Pop)",
        l1_name_resolved="EG(Pop)",
        lcia_key=None,
        lcia_data=None,
        ref_year=None,
        treat_as_one_step=False,
    )
    outputs_mod._write_l2_preweight(run=run_ut_fd, slice_spec=ut_fd_slice)
    assert run_ut_fd.state.pre_weighting_written_by_ssp_scenario[None]

    run_ar = _run(
        fu_code="L2.a.b",
        year=2030,
        ssp_scenario="SSP2",
        intermediate_outputs=True,
    )
    run_ar.context.source = "exiobase_396_ixi"
    run_ar.context.selected_l2_one_step = ["AR(E^{CBA_TD})"]
    run_ar.context.combined = [("AR(E^{CBA_TD})", "EG(Pop)")]
    ar_slice = types_mod._L2SliceSpec(
        l2_method="AR(E^{CBA_TD})",
        l1_name="EG(Pop)",
        l1_name_resolved="EG(Pop)",
        lcia_key="gwp100_lcia",
        lcia_data=payload["lcia"],
        ref_year=2005,
        treat_as_one_step=False,
    )
    outputs_mod._write_l2_preweight(run=run_ar, slice_spec=ar_slice)
    assert any(
        spec.route.bucket == "l2_in_l1" for spec in run_ar.state.l2_results_by_ssp_scenario["SSP2"]
    )


def test_attach_l2_reuse_year_level_covers_all_index_shapes() -> None:
    frame = pd.DataFrame({"value": [1.0]}, index=pd.Index(["FR"], name="r_p"))

    assert reuse_frames_mod._attach_l2_reuse_year_level(frame=frame, l2_reuse_year=None) is frame  # noqa: SLF001

    simple = reuse_frames_mod._attach_l2_reuse_year_level(frame=frame, l2_reuse_year=2005)  # noqa: SLF001
    assert list(simple.index.names) == ["r_p", "l2_reuse_year"]
    assert simple.index.get_level_values("l2_reuse_year").tolist() == [2005]

    named_reuse = pd.DataFrame(
        {"value": [2.0]},
        index=pd.Index(["FR"], name="l2_reuse_year"),
    )
    named_out = reuse_frames_mod._attach_l2_reuse_year_level(  # noqa: SLF001
        frame=named_reuse,
        l2_reuse_year=2005,
    )
    assert list(named_out.index) == [2005]

    existing_reuse = pd.DataFrame(
        {"value": [3.0]},
        index=pd.MultiIndex.from_tuples(
            [("FR", 1990)],
            names=["r_p", "l2_reuse_year"],
        ),
    )
    replaced = reuse_frames_mod._attach_l2_reuse_year_level(  # noqa: SLF001
        frame=existing_reuse,
        l2_reuse_year=2030,
    )
    assert replaced.index.get_level_values("l2_reuse_year").tolist() == [2030]
    assert replaced.index.get_level_values("r_p").tolist() == ["FR"]

    multi = pd.DataFrame(
        {"value": [4.0]},
        index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"]),
    )
    multi_out = reuse_frames_mod._attach_l2_reuse_year_level(  # noqa: SLF001
        frame=multi,
        l2_reuse_year=2030,
    )
    assert list(multi_out.index.names) == ["r_p", "s_p", "l2_reuse_year"]
    assert multi_out.index.get_level_values("l2_reuse_year").tolist() == [2030]

    combined = reuse_frames_mod._combine_l2_reuse_year_frames(  # noqa: SLF001
        frames_by_l2_reuse_year=[
            (2020, frame),
            (2021, pd.DataFrame({"value": [2.0]}, index=pd.Index(["FR"], name="r_p"))),
        ],
        reference_year=2019,
    )
    assert list(combined.index.names) == ["r_p", "reference_year", "l2_reuse_year"]
    assert combined.index.get_level_values("l2_reuse_year").tolist() == [2020, 2021]
    assert combined["value"].tolist() == [1.0, 2.0]

    combined_multi = reuse_frames_mod._combine_l2_reuse_year_frames(  # noqa: SLF001
        frames_by_l2_reuse_year=[
            (2020, multi),
            (
                2021,
                pd.DataFrame(
                    {"value": [5.0]},
                    index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"]),
                ),
            ),
        ],
    )
    assert list(combined_multi.index.names) == ["r_p", "s_p", "l2_reuse_year"]
    assert combined_multi["value"].tolist() == [4.0, 5.0]

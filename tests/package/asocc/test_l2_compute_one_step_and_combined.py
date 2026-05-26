from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from pyaesa.asocc.methods.compute_l2 import compute_l2_method
from pyaesa.asocc.orchestration.yearly.l2 import (
    l2_compute as compute_mod,
)
from pyaesa.asocc.orchestration.yearly.l2 import (
    l2_compute_combined as combined_mod,
)
from pyaesa.asocc.orchestration.yearly.l2 import (
    l2_compute_one_step as one_step_mod,
)
from pyaesa.asocc.orchestration.yearly.l2 import l2_types as types_mod


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


def _projection_context(*, route: str, mode: str, l2_reuse_years: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        mode=mode,
        reg_window=(2005, 2006),
        is_future_year=lambda year: int(year) >= 2030,
        route_for_l2_method=lambda _l2_method: route,
        l2_reuse_years_for=lambda: list(l2_reuse_years),
    )


def _state(*, scenario: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        l2_results_by_ssp_scenario={scenario: {}},
        pre_weighting_written_by_ssp_scenario={scenario: set()},
        output_spec_cache={},
        ut_reuse_preweight_cache={},
        ut_reuse_one_step_cache={},
        preweight_cache_by_ssp_scenario={scenario: {}},
        ar_l2_cache_by_ssp_scenario={None: {}},
        lcia_sliced_payload_cache={},
        skipped_years={},
        empty_ref_years={},
        output_index_level_cache={},
        ut_gvaa_identity_closure_rows=[],
        notices_emitted=set(),
        lcia_metadata_cache={},
        lcia_available_years_cache={},
        lcia_method_payload_cache={},
        cf_by_method={},
        lcia_units={},
        runtime_progress=None,
        runtime_source_prefix=None,
    )


def _run(
    *,
    fu_code: str,
    year: int,
    scenario: str | None,
    one_step_methods: list[str],
    combined_methods: list[tuple[str, str]],
    projection_context: SimpleNamespace | None = None,
    lcia_by_method: dict[str, dict] | None = None,
    l1_results_year: dict[str, pd.DataFrame] | None = None,
    reference_years: list[int] | None = None,
) -> types_mod._L2RunContext:
    context: Any = SimpleNamespace(
        source="oecd_v2025",
        agg_version=None,
        group_indices=False,
        needs_lcia=True,
        lcia_methods=["gwp100_lcia"],
        logger=SimpleNamespace(warning=lambda _message: None),
        proj_base=SimpleNamespace(name="demo"),
        fu_code=fu_code,
        selected_l1=[],
        selected_l2_one_step=list(one_step_methods),
        combined=list(combined_methods),
        output_summed=False,
        projection_context=projection_context,
        wb_df=pd.DataFrame(columns=["2005", "2006"]),
        historical_years=[2005, 2006],
        persisted_years=[year],
        reference_years=reference_years,
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None, "r_u": None},
        intermediate_outputs=True,
    )
    return types_mod._L2RunContext(
        context=context,
        state=cast(Any, _state(scenario=scenario)),
        year=year,
        ssp_scenario=scenario,
        lcia_by_method=lcia_by_method,
        l1_results_year={} if l1_results_year is None else l1_results_year,
        inputs=_inputs(),
    )


def test_resolve_combined_slice_and_combined_runtime_paths(allocation_dummy_repo) -> None:
    payload = _payload()
    weights = pd.DataFrame({2030: [0.25, 0.75]}, index=pd.Index(["FR", "US"], name="r_f"))
    run = _run(
        fu_code="L2.a.a",
        year=2030,
        scenario=None,
        one_step_methods=[],
        combined_methods=[("UT(FD)", "EG(Pop)")],
        lcia_by_method={"gwp100_lcia": payload["lcia"]},
        l1_results_year={
            "EG(Pop)__for__UT(FD)": weights,
        },
        reference_years=[2005],
    )
    run.context.source = "exiobase_396_ixi"

    resolved = combined_mod._resolve_combined_slice(
        run=run,
        request=types_mod._CombinedSliceRequest(
            l2_method="UT(FD)",
            l1_name="EG(Pop)",
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
        ),
    )
    assert resolved is not None
    assert resolved[0].treat_as_one_step is False
    pd.testing.assert_frame_equal(resolved[1], weights)

    assert (
        combined_mod._resolve_combined_slice(
            run=run,
            request=types_mod._CombinedSliceRequest(
                l2_method="UT(FD)",
                l1_name="AR(E^{CBA_FD})",
                lcia_key=None,
                lcia_data=payload["lcia"],
                ref_year=2005,
            ),
        )
        is None
    )
    assert (
        combined_mod._resolve_combined_slice(
            run=run,
            request=types_mod._CombinedSliceRequest(
                l2_method="AR(E^{CBA_FD})",
                l1_name="EG(Pop)",
                lcia_key="gwp100_lcia",
                lcia_data=None,
                ref_year=2005,
            ),
        )
        is None
    )
    assert (
        combined_mod._resolve_combined_slice(
            run=run,
            request=types_mod._CombinedSliceRequest(
                l2_method="UT(FD)",
                l1_name="AR(E^{CBA_FD})",
                lcia_key=None,
                lcia_data=payload["lcia"],
                ref_year=2005,
            ),
        )
        is None
    )
    run = run._replace(l1_results_year={"AR(E^{CBA_FD})_gwp100_lcia_ref_2005": weights})
    resolved_ar_l1 = combined_mod._resolve_combined_slice(
        run=run,
        request=types_mod._CombinedSliceRequest(
            l2_method="UT(FD)",
            l1_name="AR(E^{CBA_FD})",
            lcia_key="gwp100_lcia",
            lcia_data=payload["lcia"],
            ref_year=2005,
        ),
    )
    assert resolved_ar_l1 is not None
    pd.testing.assert_frame_equal(resolved_ar_l1[1], weights)

    resolved_treat_as_one_step = combined_mod._resolve_combined_slice(
        run=run,
        request=types_mod._CombinedSliceRequest(
            l2_method="AR(E^{CBA_FD})",
            l1_name="AR(E^{CBA_FD})",
            lcia_key="gwp100_lcia",
            lcia_data=payload["lcia"],
            ref_year=2005,
        ),
    )
    assert resolved_treat_as_one_step is not None
    assert resolved_treat_as_one_step[0].treat_as_one_step is True
    assert resolved_treat_as_one_step[1] is None

    run_success = run._replace(l1_results_year={"EG(Pop)__for__UT(FD)": weights})
    combined_mod._compute_combined_methods(run_success)
    specs = list(run_success.state.l2_results_by_ssp_scenario[None])
    assert any(spec.route.bucket == "l2_vs_global" for spec in specs)
    assert any(spec.route.bucket == "l2_in_l1" for spec in specs)

    hist_preweight = compute_l2_method(
        l2_method="UT(FD)",
        fu_code="L2.a.a",
        year=2005,
        l1_weights=None,
        fd_rf=run.inputs.fd_rf,
        gva_rp=run.inputs.gva_rp,
        fd_rp_sp_rf=run.inputs.fd_rp_sp_rf,
        fd_rp_sp=run.inputs.fd_rp_sp,
        fd_rf_sp=run.inputs.fd_rf_sp,
        gva_rp_sp=run.inputs.gva_rp_sp,
        x_to_rc=run.inputs.x_to_rc,
        kappa=run.inputs.kappa,
        omega_reg=run.inputs.omega_reg,
        lcia=None,
        reference_year=None,
        pre_weighting=True,
    )
    run_reuse = _run(
        fu_code="L2.a.a",
        year=2030,
        scenario=None,
        one_step_methods=[],
        combined_methods=[("UT(FD)", "EG(Pop)")],
        projection_context=_projection_context(
            route="historical_reuse",
            mode="historical_reuse",
            l2_reuse_years=[2005],
        ),
        l1_results_year={"EG(Pop)__for__UT(FD)": weights},
    )
    run_reuse.state.ut_reuse_preweight_cache[("preweight", "UT(FD)", None, 2005)] = hist_preweight
    combined_mod._compute_combined_methods(run_reuse)
    reuse_specs = list(run_reuse.state.l2_results_by_ssp_scenario[None])
    assert any(spec.route.bucket == "l2_vs_global" for spec in reuse_specs)
    assert any(spec.route.bucket == "l2_in_l1" for spec in reuse_specs)

    run_missing_weights = _run(
        fu_code="L2.a.a",
        year=2030,
        scenario=None,
        one_step_methods=[],
        combined_methods=[("UT(FD)", "EG(Pop)")],
        lcia_by_method={"gwp100_lcia": payload["lcia"]},
        l1_results_year={},
        reference_years=[2005],
    )
    combined_mod._compute_combined_methods(run_missing_weights)
    assert run_missing_weights.state.l2_results_by_ssp_scenario[None] == {}

    run_treat_as_one_step = _run(
        fu_code="L2.a.c",
        year=2030,
        scenario=None,
        one_step_methods=[],
        combined_methods=[("AR(E^{PBA})", "AR(E^{PBA})")],
        lcia_by_method={"gwp100_lcia": payload["lcia"]},
        reference_years=[2005],
    )
    run_treat_as_one_step.context.source = "exiobase_396_ixi"
    combined_mod._compute_combined_methods(run_treat_as_one_step)
    treat_specs = list(run_treat_as_one_step.state.l2_results_by_ssp_scenario[None])
    assert len(treat_specs) == 1
    assert treat_specs[0].route.bucket == "l2_vs_global"


def test_one_step_runtime_paths_cover_default_and_historical_reuse() -> None:
    payload = _payload()
    run_ar = _run(
        fu_code="L2.a.a",
        year=2005,
        scenario=None,
        one_step_methods=["AR(E^{CBA_FD})"],
        combined_methods=[],
        lcia_by_method={"gwp100_lcia": payload["lcia"]},
        reference_years=[2005],
    )
    run_ar.context.source = "exiobase_396_ixi"
    one_step_mod._compute_one_step_methods(run_ar)
    ar_spec = next(iter(run_ar.state.l2_results_by_ssp_scenario[None]))
    assert ar_spec.l2_method == "AR(E^{CBA_FD})"
    assert "reference_year" in ar_spec.identifier_columns

    run_hist = _run(
        fu_code="L2.a.a",
        year=2005,
        scenario=None,
        one_step_methods=["UT(FD)"],
        combined_methods=[],
    )
    one_step_mod._compute_one_step_methods(run_hist)
    assert run_hist.state.l2_results_by_ssp_scenario[None]
    assert ("one_step", "UT(FD)", None, 2005) in run_hist.state.ut_reuse_one_step_cache

    hist_result = pd.DataFrame({2005: [1.0]}, index=pd.Index(["FR"], name="r_p"))
    run_reuse = _run(
        fu_code="L2.a.b",
        year=2030,
        scenario=None,
        one_step_methods=["UT(TD)"],
        combined_methods=[],
        projection_context=_projection_context(
            route="historical_reuse",
            mode="historical_reuse",
            l2_reuse_years=[2005],
        ),
    )
    run_reuse.state.ut_reuse_one_step_cache[("one_step", "UT(TD)", None, 2005)] = hist_result
    one_step_mod._compute_one_step_methods(run_reuse)
    reuse_spec = next(iter(run_reuse.state.l2_results_by_ssp_scenario[None]))
    reuse_result = run_reuse.state.l2_results_by_ssp_scenario[None][reuse_spec][0]
    assert reuse_spec.route.projection_subfolder == "historical_reuse"
    assert "l2_reuse_year" in reuse_spec.identifier_columns
    assert list(reuse_result.columns) == [2030]


def test_compute_l2_for_year_runs_real_combined_and_one_step_flows() -> None:
    weights = pd.DataFrame({2030: [0.25, 0.75]}, index=pd.Index(["FR", "US"], name="r_f"))
    run = _run(
        fu_code="L2.a.a",
        year=2030,
        scenario=None,
        one_step_methods=["UT(FD)"],
        combined_methods=[("UT(FD)", "EG(Pop)")],
        l1_results_year={"EG(Pop)__for__UT(FD)": weights},
    )
    compute_mod._compute_l2_for_year(run=run)
    specs = list(run.state.l2_results_by_ssp_scenario[None])
    assert any(spec.route.bucket == "l2_vs_global" for spec in specs)
    assert any(spec.route.bucket == "l2_in_l1" for spec in specs)

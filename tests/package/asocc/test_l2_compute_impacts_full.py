from types import SimpleNamespace
from typing import Any

import pandas as pd

from pyaesa.asocc.methods.compute_l2 import compute_l2_method
from pyaesa.asocc.orchestration.yearly.l2 import (
    l2_compute_impacts as impacts_mod,
)
from pyaesa.asocc.orchestration.yearly.l2 import (
    l2_impact_support as impact_support_mod,
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


def _run(
    *,
    fu_code: str,
    year: int,
    scenario: str | None,
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
        intermediate_outputs=True,
        projection_context=projection_context,
        wb_df=pd.DataFrame(columns=["2005", "2006"]),
        historical_years=[2005, 2006],
        persisted_years=[year],
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None, "r_u": None},
    )
    state: Any = SimpleNamespace(
        l2_results_by_ssp_scenario={scenario: {}},
        output_spec_cache={},
        ut_reuse_preweight_cache={},
        preweight_cache_by_ssp_scenario={scenario: {}},
        ar_l2_cache_by_ssp_scenario={None: {}},
        empty_ref_years={},
        output_index_level_cache={},
        ut_gvaa_identity_closure_rows=[],
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
        ssp_scenario=scenario,
        lcia_by_method={"gwp100_lcia": _payload()["lcia"]},
        l1_results_year={},
        inputs=_inputs(),
    )


def _slice_spec(
    l2_method: str,
    *,
    l1_name: str | None,
    l1_name_resolved: str | None,
    lcia_key: str | None,
    lcia_data: dict | None,
    ref_year: int | None,
) -> types_mod._L2SliceSpec:
    return types_mod._L2SliceSpec(
        l2_method=l2_method,
        l1_name=l1_name,
        l1_name_resolved=l1_name_resolved,
        lcia_key=lcia_key,
        lcia_data=lcia_data,
        ref_year=ref_year,
        treat_as_one_step=False,
    )


def _weights_frame(year: int, weights: pd.Series) -> pd.DataFrame:
    return weights.to_frame(name=int(year))


def test_l2_compute_impacts_small() -> None:
    assert (
        impact_support_mod._should_write_historical_reuse_utility_contrib(
            slice_spec=_slice_spec(
                "UT(FDa)",
                l1_name=None,
                l1_name_resolved=None,
                lcia_key=None,
                lcia_data=None,
                ref_year=None,
            )
        )
        is True
    )
    assert (
        impact_support_mod._should_write_historical_reuse_utility_contrib(
            slice_spec=_slice_spec(
                "UT(FDa)",
                l1_name="AR(E^{CBA_FD})",
                l1_name_resolved="AR(E^{CBA_FD})",
                lcia_key="gwp100_lcia",
                lcia_data=_payload()["lcia"],
                ref_year=2005,
            )
        )
        is False
    )

    run = _run(fu_code="L2.a.b", year=2030, scenario="SSP2")
    ut_enacting_metric_l1, ut_enacting_metric_l2, ut_utility = impact_support_mod._ut_input_maps(
        run=run
    )
    assert set(ut_enacting_metric_l1) == {"fd_rf", "gva_rp"}
    assert set(ut_enacting_metric_l2) == {"fd_rp_sp_rf", "fd_rp_sp", "fd_rf_sp", "gva_rp_sp"}
    assert set(ut_utility) == {"x_to_rc", "kappa", "omega_reg"}


def test_l2_compute_impacts_non_historical_batch_ut_with_reference_levels() -> None:
    run = _run(fu_code="L2.a.b", year=2030, scenario="SSP2")
    weights = pd.DataFrame(
        {2030: [0.25, 0.75, 0.40, 0.60]},
        index=pd.MultiIndex.from_tuples(
            [
                ("climate", "FR"),
                ("climate", "US"),
                ("water", "FR"),
                ("water", "US"),
            ],
            names=["impact", "r_f"],
        ),
    )
    impacts_mod.compute_combined_impact_results(
        run=run,
        slice_spec=_slice_spec(
            "UT(FDa)",
            l1_name="AR(E^{CBA_FD})",
            l1_name_resolved="AR(E^{CBA_FD})",
            lcia_key="gwp100_lcia",
            lcia_data=_payload()["lcia"],
            ref_year=2005,
        ),
        l1_weights=weights,
    )

    stored = run.state.l2_results_by_ssp_scenario["SSP2"]
    assert len(stored) == 2
    specs = list(stored)
    assert any(spec.route.bucket == "l2_vs_global" for spec in specs)
    assert any(spec.route.bucket == "utility_propagation_contrib" for spec in specs)
    for frames in stored.values():
        for frame in frames:
            assert "reference_year" in frame.index.names


def test_l2_compute_impacts_historical_reuse_batch_ut_writes_per_l2_reuse_year() -> None:
    run = _run(
        fu_code="L2.a.b",
        year=2030,
        scenario=None,
        projection_context=_projection_context(
            route="historical_reuse",
            mode="historical_reuse",
            l2_reuse_years=[2005, 2006],
        ),
    )
    weights = pd.DataFrame(
        {2030: [0.25, 0.75, 0.40, 0.60]},
        index=pd.MultiIndex.from_tuples(
            [
                ("climate", "FR"),
                ("climate", "US"),
                ("water", "FR"),
                ("water", "US"),
            ],
            names=["impact", "r_f"],
        ),
    )
    for l2_reuse_year in [2005, 2006]:
        run.state.ut_reuse_preweight_cache[("preweight", "UT(FDa)", None, l2_reuse_year)] = (
            compute_l2_method(
                l2_method="UT(FDa)",
                fu_code="L2.a.b",
                year=l2_reuse_year,
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
        )

    impacts_mod.compute_combined_impact_results(
        run=run,
        slice_spec=_slice_spec(
            "UT(FDa)",
            l1_name="PR(GDPcap)",
            l1_name_resolved="PR(GDPcap)",
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
        ),
        l1_weights=weights,
    )

    specs = list(run.state.l2_results_by_ssp_scenario[None])
    assert len(specs) == 2
    assert sum(spec.route.bucket == "l2_vs_global" for spec in specs) == 1
    assert sum(spec.route.bucket == "utility_propagation_contrib" for spec in specs) == 1
    assert all(spec.route.projection_subfolder == "historical_reuse" for spec in specs)
    assert all("l2_reuse_year" in spec.identifier_columns for spec in specs)

    run_no_hist_contrib = _run(
        fu_code="L2.a.b",
        year=2030,
        scenario=None,
        projection_context=_projection_context(
            route="historical_reuse",
            mode="historical_reuse",
            l2_reuse_years=[2005],
        ),
    )
    run_no_hist_contrib.context.intermediate_outputs = False
    run_no_hist_contrib.state.ut_reuse_preweight_cache[("preweight", "UT(FDa)", None, 2005)] = (
        compute_l2_method(
            l2_method="UT(FDa)",
            fu_code="L2.a.b",
            year=2005,
            l1_weights=None,
            fd_rf=run_no_hist_contrib.inputs.fd_rf,
            gva_rp=run_no_hist_contrib.inputs.gva_rp,
            fd_rp_sp_rf=run_no_hist_contrib.inputs.fd_rp_sp_rf,
            fd_rp_sp=run_no_hist_contrib.inputs.fd_rp_sp,
            fd_rf_sp=run_no_hist_contrib.inputs.fd_rf_sp,
            gva_rp_sp=run_no_hist_contrib.inputs.gva_rp_sp,
            x_to_rc=run_no_hist_contrib.inputs.x_to_rc,
            kappa=run_no_hist_contrib.inputs.kappa,
            omega_reg=run_no_hist_contrib.inputs.omega_reg,
            lcia=None,
            reference_year=None,
            pre_weighting=True,
        )
    )
    impacts_mod.compute_combined_impact_results(
        run=run_no_hist_contrib,
        slice_spec=_slice_spec(
            "UT(FDa)",
            l1_name="PR(GDPcap)",
            l1_name_resolved="PR(GDPcap)",
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
        ),
        l1_weights=weights,
    )
    no_hist_contrib_specs = list(run_no_hist_contrib.state.l2_results_by_ssp_scenario[None])
    assert len(no_hist_contrib_specs) == 1
    assert no_hist_contrib_specs[0].route.bucket == "l2_vs_global"


def test_l2_compute_impacts_historical_reuse_gvaa_closure_writes_contrib_and_closure() -> None:
    run = _run(
        fu_code="L2.a.b",
        year=2030,
        scenario=None,
        projection_context=_projection_context(
            route="historical_reuse",
            mode="historical_reuse",
            l2_reuse_years=[2005],
        ),
    )
    run.state.ut_reuse_preweight_cache[("preweight", "UT(GVAa)", None, 2005)] = compute_l2_method(
        l2_method="UT(GVAa)",
        fu_code="L2.a.b",
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

    impacts_mod.compute_combined_impact_results(
        run=run,
        slice_spec=_slice_spec(
            "UT(GVAa)",
            l1_name="PR(GDPcap)",
            l1_name_resolved="PR(GDPcap)",
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
        ),
        l1_weights=_weights_frame(
            run.year,
            pd.Series([0.40, 0.60], index=run.inputs.omega_reg.index),
        ),
    )

    stored = run.state.l2_results_by_ssp_scenario[None]
    stored_by_bucket = {spec.route.bucket: frames for spec, frames in stored.items()}
    assert set(stored_by_bucket) == {"l2_vs_global", "utility_propagation_contrib"}
    assert len(stored_by_bucket["l2_vs_global"]) == 1
    assert len(stored_by_bucket["utility_propagation_contrib"]) == 1
    written_spec = next(spec for spec in stored if spec.route.bucket == "l2_vs_global")
    assert "l2_reuse_year" in written_spec.identifier_columns
    assert set(stored_by_bucket["l2_vs_global"][0].index.get_level_values("l2_reuse_year")) == {
        2005
    }
    assert run.state.ut_gvaa_identity_closure_rows


def test_l2_compute_impacts_historical_reuse_unbatched_ut_without_contrib() -> None:
    run = _run(
        fu_code="L2.a.b",
        year=2030,
        scenario=None,
        projection_context=_projection_context(
            route="historical_reuse",
            mode="historical_reuse",
            l2_reuse_years=[2005],
        ),
    )
    run.context.intermediate_outputs = False
    run.state.ut_reuse_preweight_cache[("preweight", "UT(FDa)", None, 2005)] = compute_l2_method(
        l2_method="UT(FDa)",
        fu_code="L2.a.b",
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

    impacts_mod.compute_combined_impact_results(
        run=run,
        slice_spec=_slice_spec(
            "UT(FDa)",
            l1_name="PR(GDPcap)",
            l1_name_resolved="PR(GDPcap)",
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
        ),
        l1_weights=_weights_frame(run.year, pd.Series([0.25, 0.75], index=run.inputs.fd_rf.index)),
    )

    stored_specs = list(run.state.l2_results_by_ssp_scenario[None])
    assert len(stored_specs) == 1
    assert stored_specs[0].route.bucket == "l2_vs_global"


def test_l2_compute_impacts_handles_none_weight_slices() -> None:
    run = _run(fu_code="L2.a.a", year=2030, scenario=None)

    impacts_mod.compute_combined_impact_results(
        run=run,
        slice_spec=_slice_spec(
            "UT(FD)",
            l1_name=None,
            l1_name_resolved=None,
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
        ),
        l1_weights=None,
    )

    stored_specs = list(run.state.l2_results_by_ssp_scenario[None])
    assert len(stored_specs) == 1
    assert stored_specs[0].route.bucket == "l2_vs_global"


def test_l2_compute_impacts_skips_public_write_for_support_years() -> None:
    run = _run(fu_code="L2.a.a", year=2025, scenario=None)
    run.context.persisted_years = [2030]

    impacts_mod.compute_combined_impact_results(
        run=run,
        slice_spec=_slice_spec(
            "UT(FD)",
            l1_name=None,
            l1_name_resolved=None,
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
        ),
        l1_weights=None,
    )

    assert run.state.l2_results_by_ssp_scenario[None] == {}


def test_l2_compute_impacts_ar_and_gvaa_paths(allocation_dummy_repo) -> None:
    payload = _payload()
    run_ar = _run(fu_code="L2.a.a", year=2030, scenario=None)
    run_ar.context.source = "exiobase_396_ixi"
    run_ar.context.combined = [("AR(E^{CBA_FD})", "EG(Pop)")]
    ar_weights = pd.Series([0.25, 0.75], index=pd.Index(["FR", "US"], name="r_f"))
    impacts_mod.compute_combined_impact_results(
        run=run_ar,
        slice_spec=_slice_spec(
            "AR(E^{CBA_FD})",
            l1_name="EG(Pop)",
            l1_name_resolved="EG(Pop)",
            lcia_key="gwp100_lcia",
            lcia_data=payload["lcia"],
            ref_year=2005,
        ),
        l1_weights=_weights_frame(run_ar.year, ar_weights),
    )
    ar_spec = next(iter(run_ar.state.l2_results_by_ssp_scenario[None]))
    ar_frame = run_ar.state.l2_results_by_ssp_scenario[None][ar_spec][0]
    assert ar_spec.route.bucket == "l2_vs_global"
    assert "reference_year" in ar_frame.index.names

    run_gvaa = _run(fu_code="L2.a.b", year=2030, scenario="SSP2")
    gvaa_weights = pd.Series([0.40, 0.60], index=pd.Index(["FR", "US"], name="r_u"))
    impacts_mod.compute_combined_impact_results(
        run=run_gvaa,
        slice_spec=_slice_spec(
            "UT(GVAa)",
            l1_name="PR(GDPcap)",
            l1_name_resolved="PR(GDPcap)",
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
        ),
        l1_weights=_weights_frame(run_gvaa.year, gvaa_weights),
    )
    gvaa_specs = list(run_gvaa.state.l2_results_by_ssp_scenario["SSP2"])
    assert any(spec.route.bucket == "l2_vs_global" for spec in gvaa_specs)
    assert any(spec.route.bucket == "utility_propagation_contrib" for spec in gvaa_specs)

    run_gvaa_named = _run(fu_code="L2.a.b", year=2030, scenario="SSP2")
    gvaa_named_weights = pd.DataFrame(
        {2030: [0.40, 0.60, 0.55, 0.45]},
        index=pd.MultiIndex.from_tuples(
            [
                ("climate", "FR"),
                ("climate", "US"),
                ("water", "FR"),
                ("water", "US"),
            ],
            names=["impact", "r_u"],
        ),
    )
    impacts_mod.compute_combined_impact_results(
        run=run_gvaa_named,
        slice_spec=_slice_spec(
            "UT(GVAa)",
            l1_name="PR(GDPcap)",
            l1_name_resolved="PR(GDPcap)",
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
        ),
        l1_weights=gvaa_named_weights,
    )
    gvaa_named_store = run_gvaa_named.state.l2_results_by_ssp_scenario["SSP2"]
    gvaa_named_by_bucket = {spec.route.bucket: frames for spec, frames in gvaa_named_store.items()}
    assert set(gvaa_named_by_bucket) == {"l2_vs_global", "utility_propagation_contrib"}
    assert len(gvaa_named_by_bucket["l2_vs_global"]) == 2
    assert len(gvaa_named_by_bucket["utility_propagation_contrib"]) == 2
    assert {
        next(iter(frame.index.get_level_values("impact").unique()))
        for frame in gvaa_named_by_bucket["l2_vs_global"]
    } == {"climate", "water"}

    run_batch_ar = _run(fu_code="L2.a.c", year=2030, scenario=None)
    run_batch_ar.context.source = "exiobase_396_ixi"
    run_batch_ar.context.combined = [("AR(E^{PBA})", "PR(GDPcap)")]
    lcia_two_impacts = {
        **payload["lcia"],
        "e_pba_rp_sp": pd.DataFrame(
            [[5.0, 7.0], [2.0, 3.0]],
            index=pd.Index(["climate_parent", "water_parent"], name="impact"),
            columns=pd.MultiIndex.from_tuples(
                [("FR", "A"), ("US", "A")],
                names=["r_p", "s_p"],
            ),
        ),
        "e_pba_reg": pd.DataFrame(
            [[6.0, 8.0], [4.0, 5.0]],
            index=pd.Index(["climate_parent", "water_parent"], name="impact"),
            columns=pd.Index(["FR", "US"], name="r_p"),
        ),
    }
    batch_weights = pd.DataFrame(
        {2030: [0.25, 0.75, 0.40, 0.60]},
        index=pd.MultiIndex.from_tuples(
            [
                ("climate_parent", "FR"),
                ("climate_parent", "US"),
                ("water_parent", "FR"),
                ("water_parent", "US"),
            ],
            names=["impact", "r_p"],
        ),
    )
    impacts_mod.compute_combined_impact_results(
        run=run_batch_ar,
        slice_spec=_slice_spec(
            "AR(E^{PBA})",
            l1_name="PR(GDPcap)",
            l1_name_resolved="PR(GDPcap)",
            lcia_key="gwp100_lcia",
            lcia_data=lcia_two_impacts,
            ref_year=2005,
        ),
        l1_weights=batch_weights,
    )
    batch_ar_spec = next(iter(run_batch_ar.state.l2_results_by_ssp_scenario[None]))
    batch_ar_frame = run_batch_ar.state.l2_results_by_ssp_scenario[None][batch_ar_spec][0]
    assert batch_ar_spec.route.bucket == "l2_vs_global"
    assert "reference_year" in batch_ar_frame.index.names

    run_no_live_contrib = _run(fu_code="L2.a.b", year=2030, scenario="SSP2")
    run_no_live_contrib.context.intermediate_outputs = False
    impacts_mod.compute_combined_impact_results(
        run=run_no_live_contrib,
        slice_spec=_slice_spec(
            "UT(FDa)",
            l1_name="PR(GDPcap)",
            l1_name_resolved="PR(GDPcap)",
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
        ),
        l1_weights=_weights_frame(run_no_live_contrib.year, gvaa_weights.rename("r_f")),
    )
    no_live_specs = list(run_no_live_contrib.state.l2_results_by_ssp_scenario["SSP2"])
    assert len(no_live_specs) == 1
    assert no_live_specs[0].route.bucket == "l2_vs_global"

    run_batch_no_live_contrib = _run(fu_code="L2.a.b", year=2030, scenario="SSP2")
    run_batch_no_live_contrib.context.intermediate_outputs = False
    named_weights = pd.DataFrame(
        {2030: [0.25, 0.75, 0.40, 0.60]},
        index=pd.MultiIndex.from_tuples(
            [
                ("climate", "FR"),
                ("climate", "US"),
                ("water", "FR"),
                ("water", "US"),
            ],
            names=["impact", "r_f"],
        ),
    )
    impacts_mod.compute_combined_impact_results(
        run=run_batch_no_live_contrib,
        slice_spec=_slice_spec(
            "UT(FDa)",
            l1_name="PR(GDPcap)",
            l1_name_resolved="PR(GDPcap)",
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
        ),
        l1_weights=named_weights,
    )
    batch_no_live_specs = list(run_batch_no_live_contrib.state.l2_results_by_ssp_scenario["SSP2"])
    assert len(batch_no_live_specs) == 1
    assert batch_no_live_specs[0].route.bucket == "l2_vs_global"

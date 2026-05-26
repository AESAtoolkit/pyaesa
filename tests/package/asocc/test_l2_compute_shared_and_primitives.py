from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from pyaesa.asocc.orchestration.yearly.l2 import (
    l2_contracts as contracts_mod,
)
from pyaesa.asocc.orchestration.yearly.l2 import (
    l2_compute_primitives as primitives_mod,
)
from pyaesa.asocc.orchestration.yearly.l2 import (
    l2_compute_shared as shared_mod,
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


def _state(*, scenario: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        skipped_years={},
        lcia_sliced_payload_cache={},
        preweight_cache_by_ssp_scenario={scenario: {}},
        ar_l2_cache_by_ssp_scenario={None: {}},
        empty_ref_years={},
        output_index_level_cache={},
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
    year: int = 2030,
    scenario: str | None = None,
    reference_years: list[int] | None = None,
    historical_years: list[int] | None = None,
    filters: dict[str, list[str] | None] | None = None,
    lcia_by_method: dict[str, dict] | None = None,
) -> types_mod._L2RunContext:
    context: Any = SimpleNamespace(
        source="oecd_v2025",
        agg_version=None,
        needs_lcia=True,
        lcia_methods=["gwp100_lcia"],
        logger=SimpleNamespace(warning=lambda _message: None),
        fu_code=fu_code,
        selected_l1=[],
        selected_l2_one_step=[],
        combined=[],
        filters=(
            {
                "r_p": None,
                "s_p": None,
                "r_c": None,
                "r_f": None,
                "r_u": None,
            }
            if filters is None
            else filters
        ),
        reference_years=reference_years,
        historical_years=[2005, 2006] if historical_years is None else historical_years,
        persisted_years=[year],
    )
    return types_mod._L2RunContext(
        context=context,
        state=cast(Any, _state(scenario=scenario)),
        year=year,
        ssp_scenario=scenario,
        lcia_by_method=lcia_by_method,
        l1_results_year={},
        inputs=_inputs(),
    )


def test_l2_compute_shared_cover_real_registry_and_slice_paths() -> None:
    payload = _payload()
    run = _run(
        fu_code="L2.a.a",
        filters={
            "r_p": ["US", "FR"],
            "s_p": ["A"],
            "r_c": None,
            "r_f": None,
            "r_u": None,
        },
        reference_years=[2005, 2006],
        lcia_by_method={"gwp100_lcia": payload["lcia"]},
    )

    key = shared_mod._filters_cache_key(run=run)
    assert ("r_p", ("FR", "US")) in key
    assert ("s_p", ("A",)) in key

    assert shared_mod._lcia_items_for_method(run=run, l2_method="UT(FD)") == {None: None}

    run_no_lcia = _run(fu_code="L2.a.a", lcia_by_method=None)
    assert (
        shared_mod._lcia_items_for_method(
            run=run_no_lcia,
            l2_method="AR(E^{CBA_FD})",
        )
        == {}
    )
    assert run_no_lcia.state.skipped_years[2030] == "LCIA unavailable"

    first = shared_mod._lcia_items_for_method(
        run=run,
        l2_method="AR(E^{CBA_FD})",
    )
    second = shared_mod._lcia_items_for_method(
        run=run,
        l2_method="AR(E^{CBA_FD})",
    )
    assert list(first) == ["gwp100_lcia"]
    assert first == second
    assert len(run.state.lcia_sliced_payload_cache) == 1

    assert shared_mod._reference_years_for(
        run=run,
        l2_method="UT(FD)",
        l1_name=None,
    ) == [None]
    assert shared_mod._reference_years_for(
        run=run,
        l2_method="AR(E^{CBA_FD})",
        l1_name=None,
    ) == [2005, 2006]
    assert shared_mod._reference_years_for(
        run=run,
        l2_method="UT(FD)",
        l1_name="AR(E^{CBA_FD})",
    ) == [2005, 2006]
    assert shared_mod._reference_years_for(
        run=_run(
            fu_code="L2.a.a",
            year=2005,
            reference_years=[2005, 2006],
        ),
        l2_method="UT(FD)",
        l1_name="AR(E^{CBA_FD})",
    ) == [2005]
    assert shared_mod._reference_years_for(
        run=_run(
            fu_code="L2.a.a",
            year=2005,
            reference_years=None,
            historical_years=[2005, 2006],
        ),
        l2_method="AR(E^{CBA_FD})",
        l1_name=None,
    ) == [2005]

    cutoff_filtered_run = _run(
        fu_code="L2.a.a",
        historical_years=[2005, 2030],
        reference_years=None,
    )
    assert shared_mod._reference_years_for(
        run=cutoff_filtered_run,
        l2_method="AR(E^{CBA_FD})",
        l1_name=None,
    ) == [2005]
    assert shared_mod._reference_years_for(
        run=_run(
            fu_code="L2.a.a",
            reference_years=None,
            historical_years=[],
        ),
        l2_method="AR(E^{CBA_FD})",
        l1_name=None,
    ) == [None]
    # default reference-year filtering follows the source registry cutoff
    assert shared_mod._reference_years_for(
        run=_run(
            fu_code="L2.a.a",
            reference_years=None,
            historical_years=[2005, 2020, 2021],
        ),
        l2_method="AR(E^{CBA_FD})",
        l1_name=None,
    ) == [2005, 2020, 2021]

    assert (
        shared_mod._l1_weights_key_for_pair(
            base_key="EG(Pop)",
            l1_name="EG(Pop)",
            l2_method="UT(FD)",
        )
        == "EG(Pop)__for__UT(FD)"
    )
    assert (
        shared_mod._l1_weights_key_for_pair(
            base_key="AR(E^{CBA_FD})",
            l1_name="AR(E^{CBA_FD})",
            l2_method="UT(FD)",
        )
        == "AR(E^{CBA_FD})"
    )


def test_l2_compute_primitives_cover_ut_non_ut_and_ar_runtime_paths(
    allocation_dummy_repo,
) -> None:
    payload = _payload()
    weights_rf = pd.Series([0.25, 0.75], index=pd.Index(["FR", "US"], name="r_f"))
    weights_rp = pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_p"))

    run_ut = _run(fu_code="L2.a.b", year=2030, scenario="SSP2")
    ut_spec = types_mod._L2WeightSpec(
        slice_spec=types_mod._L2SliceSpec(
            l2_method="UT(FDa)",
            l1_name="EG(Pop)",
            l1_name_resolved="EG(Pop)",
            lcia_key=None,
            lcia_data=payload["lcia"],
            ref_year=None,
            treat_as_one_step=False,
        ),
        impact=None,
        weights=weights_rf,
    )
    ut_default = primitives_mod.compute_non_ar_or_ut_result(run=run_ut, weight_spec=ut_spec)
    assert list(ut_default.columns) == [2030]
    assert ut_default.index.names == ["r_p", "s_p"]

    run_non_ut = _run(fu_code="L2.a.c", year=2030, scenario=None)
    weighted_spec = types_mod._L2WeightSpec(
        slice_spec=types_mod._L2SliceSpec(
            l2_method="UT(GVA)",
            l1_name="EG(Pop)",
            l1_name_resolved="EG(Pop)",
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
            treat_as_one_step=False,
        ),
        impact=None,
        weights=weights_rp,
    )
    weighted = primitives_mod.compute_non_ar_or_ut_result(
        run=run_non_ut,
        weight_spec=weighted_spec,
    )
    assert list(weighted.columns) == [2030]

    one_step_spec = types_mod._L2WeightSpec(
        slice_spec=types_mod._L2SliceSpec(
            l2_method="UT(GVA)",
            l1_name="EG(Pop)",
            l1_name_resolved="EG(Pop)",
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
            treat_as_one_step=True,
        ),
        impact=None,
        weights=weights_rp,
    )
    one_step = primitives_mod.compute_non_ar_or_ut_result(
        run=run_non_ut,
        weight_spec=one_step_spec,
    )
    assert list(one_step.columns) == [2030]
    assert not weighted.equals(one_step)

    run_ar = _run(
        fu_code="L2.a.a",
        year=2030,
        scenario=None,
        reference_years=[2005],
        lcia_by_method={"gwp100_lcia": payload["lcia"]},
    )
    run_ar.context.source = "exiobase_396_ixi"
    run_ar.context.selected_l2_one_step = ["AR(E^{CBA_FD})"]
    run_ar.context.combined = [("AR(E^{CBA_FD})", "EG(Pop)")]
    ar_direct = primitives_mod.compute_ar_result(
        run=run_ar,
        weight_spec=types_mod._L2WeightSpec(
            slice_spec=types_mod._L2SliceSpec(
                l2_method="AR(E^{CBA_FD})",
                l1_name="AR(E^{CBA_FD})",
                l1_name_resolved="AR(E^{CBA_FD})",
                lcia_key="gwp100_lcia",
                lcia_data=payload["lcia"],
                ref_year=2005,
                treat_as_one_step=False,
            ),
            impact="climate_child",
            weights=None,
        ),
    )
    assert ar_direct is not None
    assert "impact" in ar_direct.index.names
    assert "reference_year" in ar_direct.index.names

    ar_direct_no_impact = primitives_mod.compute_ar_result(
        run=run_ar,
        weight_spec=types_mod._L2WeightSpec(
            slice_spec=types_mod._L2SliceSpec(
                l2_method="AR(E^{CBA_FD})",
                l1_name="AR(E^{CBA_FD})",
                l1_name_resolved="AR(E^{CBA_FD})",
                lcia_key="gwp100_lcia",
                lcia_data=payload["lcia"],
                ref_year=2005,
                treat_as_one_step=False,
            ),
            impact=None,
            weights=None,
        ),
    )
    assert ar_direct_no_impact is not None
    assert "impact" in ar_direct_no_impact.index.names
    assert pd.Index(ar_direct_no_impact.index.get_level_values("impact")).unique().tolist() == [
        "climate_parent"
    ]
    assert "reference_year" in ar_direct_no_impact.index.names

    ar_weighted = primitives_mod.compute_ar_result(
        run=run_ar,
        weight_spec=types_mod._L2WeightSpec(
            slice_spec=types_mod._L2SliceSpec(
                l2_method="AR(E^{CBA_FD})",
                l1_name="EG(Pop)",
                l1_name_resolved="EG(Pop)",
                lcia_key="gwp100_lcia",
                lcia_data=payload["lcia"],
                ref_year=2005,
                treat_as_one_step=False,
            ),
            impact=None,
            weights=weights_rf,
        ),
    )
    assert ar_weighted is not None
    assert "reference_year" in ar_weighted.index.names


def test_l2_contracts_cover_required_runtime_guards() -> None:
    inputs = _inputs()
    frame = pd.DataFrame({"2030": [1.0]})
    weights = pd.Series([1.0], index=pd.Index(["FR"], name="r_f"))

    assert contracts_mod.require_compute_inputs(inputs=inputs, where="x") is inputs

    assert contracts_mod.require_frame(frame=frame, where="x", subject="payload") is frame

    assert contracts_mod.require_ref_year(ref_year=2005, where="x") == 2005

    assert contracts_mod.require_required_indices(
        required_indices=("r_p", "s_p"),
        where="x",
    ) == ("r_p", "s_p")

    assert contracts_mod.require_weight_axis(weight_axis="r_f", where="x") == "r_f"

    assert contracts_mod.require_weights(weights=weights, where="x").equals(weights)


def test_l2_compute_primitives_cover_historical_and_ar_runtime_paths() -> None:
    payload = _payload()
    weights_rf = pd.Series([0.25, 0.75], index=pd.Index(["FR", "US"], name="r_f"))
    run_ut = _run(
        fu_code="L2.a.b",
        year=2005,
        scenario=None,
        historical_years=[2005],
    )
    ut_result = primitives_mod.compute_non_ar_or_ut_result(
        run=run_ut,
        weight_spec=types_mod._L2WeightSpec(
            slice_spec=types_mod._L2SliceSpec(
                l2_method="UT(FDa)",
                l1_name="EG(Pop)",
                l1_name_resolved="EG(Pop)",
                lcia_key=None,
                lcia_data=payload["lcia"],
                ref_year=None,
                treat_as_one_step=False,
            ),
            impact=None,
            weights=weights_rf,
        ),
    )
    assert ut_result.index.names == ["r_p", "s_p"]
    assert list(ut_result.columns) == [2005]
    run_weighted = _run(
        fu_code="L2.a.c",
        year=2005,
        scenario=None,
        historical_years=[2005],
    )
    weighted_result = primitives_mod.compute_non_ar_or_ut_result(
        run=run_weighted,
        weight_spec=types_mod._L2WeightSpec(
            slice_spec=types_mod._L2SliceSpec(
                l2_method="UT(GVA)",
                l1_name="EG(Pop)",
                l1_name_resolved="EG(Pop)",
                lcia_key=None,
                lcia_data=None,
                ref_year=None,
                treat_as_one_step=False,
            ),
            impact=None,
            weights=pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_p")),
        ),
    )
    assert list(weighted_result.columns) == [2005]

    ar_pba_run = _run(
        fu_code="L2.a.c",
        year=2030,
        scenario=None,
        reference_years=[2005],
        lcia_by_method={"gwp100_lcia": payload["lcia"]},
    )
    ar_pba_run.context.source = "exiobase_396_ixi"
    ar_pba_run.context.selected_l2_one_step = ["AR(E^{PBA})"]
    ar_pba_run.context.combined = [("AR(E^{PBA})", "PR(GDPcap)")]
    ar_pba_result = primitives_mod.compute_ar_result(
        run=ar_pba_run,
        weight_spec=types_mod._L2WeightSpec(
            slice_spec=types_mod._L2SliceSpec(
                l2_method="AR(E^{PBA})",
                l1_name="PR(GDPcap)",
                l1_name_resolved="PR(GDPcap)",
                lcia_key="gwp100_lcia",
                lcia_data=payload["lcia"],
                ref_year=2005,
                treat_as_one_step=False,
            ),
            impact="climate_child",
            weights=pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_p")),
        ),
    )
    assert "impact" in ar_pba_result.index.names
    assert "reference_year" in ar_pba_result.index.names

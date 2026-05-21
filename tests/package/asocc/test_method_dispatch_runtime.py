from types import SimpleNamespace
from typing import TypedDict

import pandas as pd
import pytest

from pyaesa.asocc.methods import ar_cache
from pyaesa.asocc.methods import compute_l1
from pyaesa.asocc.methods import compute_l2
from pyaesa.asocc.methods import run_ar
from pyaesa.asocc.methods import run_ut
from pyaesa.asocc.data import reference_payloads as reference_payloads_mod


class _Logger:
    def warning(self, _message: str) -> None:
        return None


def _lcia_reg_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [[10.0, 20.0]],
        index=pd.Index(["climate_child"], name="impact"),
        columns=pd.Index(["FR", "US"], name="r_f"),
    )


def _lcia_reg_by_year() -> dict[int, pd.DataFrame]:
    return {
        2019: pd.DataFrame(
            [[4.0, 8.0]],
            index=pd.Index(["climate_child"], name="impact"),
            columns=pd.Index(["FR", "US"], name="r_f"),
        ),
        2020: pd.DataFrame(
            [[6.0, 12.0]],
            index=pd.Index(["climate_child"], name="impact"),
            columns=pd.Index(["FR", "US"], name="r_f"),
        ),
    }


def _base_l1_inputs() -> dict:
    population = pd.Series([10.0, 20.0], index=pd.Index(["FR", "US"], name="r_f"))
    population_ref = pd.Series([9.0, 18.0], index=pd.Index(["FR", "US"], name="r_f"))
    population_by_year = {
        2019: pd.Series([2.0, 4.0], index=pd.Index(["FR", "US"], name="r_f")),
        2020: pd.Series([3.0, 6.0], index=pd.Index(["FR", "US"], name="r_f")),
    }
    pr_pop = pd.Series([10.0, 30.0], index=pd.Index(["FRA", "USA"], name="iso3"))
    pr_gdp = pd.Series([100.0, 300.0], index=pd.Index(["FRA", "USA"], name="iso3"))
    pr_to_mrio = pd.Series(["FR", "US"], index=pd.Index(["FRA", "USA"], name="iso3"))
    rps_df = pd.DataFrame(
        {
            "impact": ["climate_child"],
            "responsibility_period_years": [2],
        }
    )
    impact_parent_map = pd.Series(
        ["climate_parent"],
        index=pd.Index(["climate_child"], name="impact"),
    )
    return {
        "year": 2020,
        "population": population,
        "population_by_year": population_by_year,
        "population_ref": population_ref,
        "pr_pop": pr_pop,
        "pr_gdp": pr_gdp,
        "pr_to_mrio": pr_to_mrio,
        "source_key": "oecd_v2025",
        "group_version_reg": None,
        "l1_reg_aggreg": "pre",
        "region_label_override": None,
        "lcia_reg": _lcia_reg_frame(),
        "lcia_reg_by_year": _lcia_reg_by_year(),
        "rps_df": rps_df,
        "impact_parent_map": impact_parent_map,
        "available_years": [2019, 2020],
        "reference_year": 2019,
        "impact_year": 2020,
        "pr_hr_parent_cum_cache": {},
    }


class _UtPayload(TypedDict):
    fd_rf: pd.Series
    gva_rp: pd.Series
    fd_rp_sp_rf: pd.DataFrame
    fd_rp_sp: pd.Series
    fd_rf_sp: pd.Series
    gva_rp_sp: pd.Series
    x_to_rc: pd.DataFrame
    kappa: pd.DataFrame
    omega_reg: pd.DataFrame
    lcia: dict[str, pd.DataFrame]


def _ut_payload() -> _UtPayload:
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


def _ar_context(*, fu_code: str, selected_l1: list[str] | None = None) -> SimpleNamespace:
    selected_l2 = []
    if fu_code == "L2.a.a":
        selected_l2 = ["AR(E^{CBA_FD})"]
    elif fu_code == "L2.a.b":
        selected_l2 = ["AR(E^{CBA_TD})"]
    elif fu_code == "L2.a.c":
        selected_l2 = ["AR(E^{PBA})"]
    return SimpleNamespace(
        source="exiobase_396_ixi",
        group_version=None,
        group_version_reg=None,
        fu_code=fu_code,
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None, "r_u": None},
        needs_lcia=True,
        lcia_methods=["gwp100_lcia"],
        selected_l1=[] if selected_l1 is None else list(selected_l1),
        selected_l2_one_step=selected_l2,
        combined=[],
        logger=_Logger(),
        l1_reg_aggreg="pre",
        historical_years=[2005, 2006],
    )


def _ar_state() -> SimpleNamespace:
    pop_series = {
        2005: pd.Series([10.0, 20.0], index=pd.Index(["FR", "US"], name="r_f")),
        2006: pd.Series([11.0, 21.0], index=pd.Index(["FR", "US"], name="r_f")),
    }
    return SimpleNamespace(
        notices_emitted=set(),
        lcia_metadata_cache={},
        lcia_available_years_cache={},
        lcia_method_payload_cache={},
        skipped_years={},
        cf_by_method={},
        lcia_units={},
        runtime_progress=None,
        runtime_source_prefix=None,
        ar_l2_cache_by_ssp_scenario={None: {}},
        ar_l1_cache_by_ssp_scenario={None: {}},
        empty_ref_years={},
        output_index_level_cache={},
        preweight_cache_by_ssp_scenario={None: {}},
        pop_series_by_ssp_scenario={None: pop_series},
    )


def test_resolve_l1_region_label_paths() -> None:
    assert (
        compute_l1.resolve_l1_region_label(
            l1_method="AR(E^{PBA})",
            fu_code="L1.a",
        )
        == "r_p"
    )
    assert (
        compute_l1.resolve_l1_region_label(
            l1_method="AR(E^{CBA_FD})",
            fu_code="L1.a",
        )
        == "r_f"
    )
    assert compute_l1.resolve_l1_region_label(l1_method="EG(Pop)", fu_code="L1.a") == "r_f"
    assert compute_l1.resolve_l1_region_label(l1_method="EG(Pop)", fu_code="L1.b") == "r_p"
    assert compute_l1.resolve_l1_region_label(l1_method="EG(Pop)", fu_code="L2.a.a") == "r_f"
    assert compute_l1.resolve_l1_region_label(l1_method="EG(Pop)", fu_code="L2.a.c") == "r_p"


def test_aggregate_l1_regions_post_covers_single_and_multiindex_validation(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError, match="r_f"):
        compute_l1._aggregate_l1_regions_post(  # noqa: SLF001
            frame=pd.DataFrame(
                {2020: [1.0]},
                index=pd.MultiIndex.from_tuples(
                    [("climate_child", "FR")],
                    names=["impact", "r_x"],
                ),
            ),
            source_key="oecd_v2025",
            group_version_reg="demo_reg",
            region_label="r_f",
        )

    with pytest.raises(ValueError, match="r_x"):
        compute_l1._aggregate_l1_regions_post(  # noqa: SLF001
            frame=pd.DataFrame(
                {2020: [1.0]},
                index=pd.Index(["FR"], name="r_x"),
            ),
            source_key="oecd_v2025",
            group_version_reg="demo_reg",
            region_label="r_f",
        )

    aggregated = compute_l1._aggregate_l1_regions_post(  # noqa: SLF001
        frame=pd.DataFrame(
            {2020: [1.0, 2.0]},
            index=pd.Index(["FR", "US"], name="r_f"),
        ),
        source_key="oecd_v2025",
        group_version_reg="demo_reg",
        region_label="r_f",
    )
    assert list(aggregated.index) == ["EU", "NAM"]
    assert list(aggregated[2020]) == [1.0, 2.0]


def test_compute_l1_method_with_real_families(allocation_dummy_repo) -> None:
    base = _base_l1_inputs()

    eg = compute_l1.compute_l1_method(
        l1_method="EG(Pop)",
        fu_code="L1.a",
        **base,
    )
    assert eg.index.name == "r_f"

    pr = compute_l1.compute_l1_method(
        l1_method="PR(GDPcap)",
        fu_code="L1.a",
        **{**base, "group_version_reg": "demo_reg"},
    )
    assert pr.index.name == "r_f"
    assert pr[2020].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        compute_l1.compute_l1_method(
            l1_method="PR(GDPcap)",
            fu_code="L1.a",
            **{**base, "pr_pop": None},
        )

    pr_hr = compute_l1.compute_l1_method(
        l1_method="PR-HR(Ecap,cum^{CBA_FD})",
        fu_code="L1.a",
        **{**base, "population": base["population_by_year"][2020]},
    )
    assert pr_hr.index.names == ["impact", "r_f"]
    with pytest.raises(ValueError):
        compute_l1.compute_l1_method(
            l1_method="PR-HR(Ecap,cum^{CBA_FD})",
            fu_code="L1.a",
            **{**base, "lcia_reg_by_year": None},
        )
    with pytest.raises(ValueError):
        compute_l1.compute_l1_method(
            l1_method="PR-HR(Ecap,cum^{CBA_FD})",
            fu_code="L1.a",
            **{**base, "population_by_year": None},
        )

    ar_ecap = compute_l1.compute_l1_method(
        l1_method="AR(Ecap^{CBA_FD})",
        fu_code="L1.a",
        **{
            **base,
            "group_version_reg": "demo_reg",
            "l1_reg_aggreg": "post",
        },
    )
    assert ar_ecap.index.names == ["impact", "r_f"]
    assert sorted(ar_ecap.index.get_level_values("r_f").unique()) == ["EU", "NAM"]
    with pytest.raises(ValueError, match="l1_reg_aggreg"):
        compute_l1.compute_l1_method(
            l1_method="AR(Ecap^{CBA_FD})",
            fu_code="L1.a",
            **{**base, "l1_reg_aggreg": "bad"},
        )

    ar = compute_l1.compute_l1_method(
        l1_method="AR(E^{CBA_FD})",
        fu_code="L1.a",
        **base,
    )
    assert ar.index.names == ["impact", "r_f"]


def test_compute_l2_method_and_preweighted_application() -> None:
    payload = _ut_payload()
    l1_weights_rf = pd.Series([0.25, 0.75], index=pd.Index(["FR", "US"], name="r_f"))
    l1_weights_rp = pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_p"))
    l1_weights_ru = pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_u"))

    ut_fd = compute_l2.compute_l2_method(
        l2_method="UT(FD)",
        fu_code="L2.a.a",
        year=2020,
        l1_weights=None,
        fd_rf=payload["fd_rf"],
        gva_rp=payload["gva_rp"],
        fd_rp_sp_rf=payload["fd_rp_sp_rf"],
        fd_rp_sp=payload["fd_rp_sp"],
        fd_rf_sp=payload["fd_rf_sp"],
        gva_rp_sp=payload["gva_rp_sp"],
        x_to_rc=payload["x_to_rc"],
        kappa=payload["kappa"],
        omega_reg=payload["omega_reg"],
        lcia=payload["lcia"],
        reference_year=2019,
        pre_weighting=False,
    )
    assert list(ut_fd.columns) == [2020]

    pre = compute_l2.compute_l2_method(
        l2_method="UT(FDa)",
        fu_code="L2.a.b",
        year=2020,
        l1_weights=None,
        fd_rf=payload["fd_rf"],
        gva_rp=payload["gva_rp"],
        fd_rp_sp_rf=payload["fd_rp_sp_rf"],
        fd_rp_sp=payload["fd_rp_sp"],
        fd_rf_sp=payload["fd_rf_sp"],
        gva_rp_sp=payload["gva_rp_sp"],
        x_to_rc=payload["x_to_rc"],
        kappa=payload["kappa"],
        omega_reg=payload["omega_reg"],
        lcia=payload["lcia"],
        reference_year=2019,
        pre_weighting=True,
    )
    weighted = compute_l2.apply_l1_weights_to_preweighted(
        l2_method="UT(FDa)",
        fu_code="L2.a.b",
        year=2020,
        pre_weighted=pre,
        l1_weights=l1_weights_rf,
    )
    assert list(weighted.columns) == [2020]

    assert list(
        compute_l2.compute_l2_method(
            l2_method="UT(GVAa)",
            fu_code="L2.a.b",
            year=2020,
            l1_weights=l1_weights_ru,
            fd_rf=payload["fd_rf"],
            gva_rp=payload["gva_rp"],
            fd_rp_sp_rf=payload["fd_rp_sp_rf"],
            fd_rp_sp=payload["fd_rp_sp"],
            fd_rf_sp=payload["fd_rf_sp"],
            gva_rp_sp=payload["gva_rp_sp"],
            x_to_rc=payload["x_to_rc"],
            kappa=payload["kappa"],
            omega_reg=payload["omega_reg"],
            lcia=payload["lcia"],
            reference_year=2019,
            pre_weighting=False,
        ).columns
    ) == [2020]
    assert list(
        compute_l2.compute_l2_method(
            l2_method="UT(TD)",
            fu_code="L2.a.b",
            year=2020,
            l1_weights=None,
            fd_rf=payload["fd_rf"],
            gva_rp=payload["gva_rp"],
            fd_rp_sp_rf=payload["fd_rp_sp_rf"],
            fd_rp_sp=payload["fd_rp_sp"],
            fd_rf_sp=payload["fd_rf_sp"],
            gva_rp_sp=payload["gva_rp_sp"],
            x_to_rc=payload["x_to_rc"],
            kappa=payload["kappa"],
            omega_reg=payload["omega_reg"],
            lcia=payload["lcia"],
            reference_year=2019,
            pre_weighting=False,
        ).columns
    ) == [2020]
    assert list(
        compute_l2.compute_l2_method(
            l2_method="UT(GVA)",
            fu_code="L2.a.c",
            year=2020,
            l1_weights=l1_weights_rp,
            fd_rf=payload["fd_rf"],
            gva_rp=payload["gva_rp"],
            fd_rp_sp_rf=payload["fd_rp_sp_rf"],
            fd_rp_sp=payload["fd_rp_sp"],
            fd_rf_sp=payload["fd_rf_sp"],
            gva_rp_sp=payload["gva_rp_sp"],
            x_to_rc=payload["x_to_rc"],
            kappa=payload["kappa"],
            omega_reg=payload["omega_reg"],
            lcia=payload["lcia"],
            reference_year=2019,
            pre_weighting=False,
        ).columns
    ) == [2020]
    assert list(
        compute_l2.compute_l2_method(
            l2_method="AR(E^{CBA_TD})",
            fu_code="L2.a.b",
            year=2020,
            l1_weights=None,
            fd_rf=payload["fd_rf"],
            gva_rp=payload["gva_rp"],
            fd_rp_sp_rf=payload["fd_rp_sp_rf"],
            fd_rp_sp=payload["fd_rp_sp"],
            fd_rf_sp=payload["fd_rf_sp"],
            gva_rp_sp=payload["gva_rp_sp"],
            x_to_rc=payload["x_to_rc"],
            kappa=payload["kappa"],
            omega_reg=payload["omega_reg"],
            lcia=payload["lcia"],
            reference_year=2019,
            pre_weighting=False,
        ).columns
    ) == [2019]
    with pytest.raises(ValueError):
        compute_l2.compute_l2_method(
            l2_method="AR(E^{CBA_TD})",
            fu_code="L2.a.b",
            year=2020,
            l1_weights=None,
            fd_rf=payload["fd_rf"],
            gva_rp=payload["gva_rp"],
            fd_rp_sp_rf=payload["fd_rp_sp_rf"],
            fd_rp_sp=payload["fd_rp_sp"],
            fd_rf_sp=payload["fd_rf_sp"],
            gva_rp_sp=payload["gva_rp_sp"],
            x_to_rc=payload["x_to_rc"],
            kappa=payload["kappa"],
            omega_reg=payload["omega_reg"],
            lcia=None,
            reference_year=2019,
            pre_weighting=False,
        )


def test_ar_cache_runtime(
    allocation_dummy_repo,
    allocation_dummy_repo_factory,
) -> None:
    context = _ar_context(fu_code="L2.a.b")
    state = _ar_state()

    lcia_ref = reference_payloads_mod.load_ar_l2_reference_lcia_payload(
        context=context,
        state=state,
        ref_year=2005,
        lcia_key="gwp100_lcia",
    )
    assert "e_cba_td_rp_sp" in lcia_ref

    unavailable_repo = allocation_dummy_repo_factory(name="ar_cache_unavailable_lcia")
    unavailable_context = _ar_context(fu_code="L2.a.b")
    unavailable_state = _ar_state()
    unavailable_repo.set_lcia_methods(
        source="exiobase_396_ixi",
        matrix_version=None,
        methods=["gwp100_lcia"],
        available_years_by_method={"gwp100_lcia": []},
    )
    with pytest.raises(ValueError, match="gwp100_lcia"):
        reference_payloads_mod.load_ar_l2_reference_lcia_payload(
            context=unavailable_context,
            state=unavailable_state,
            ref_year=2005,
            lcia_key="gwp100_lcia",
        )

    projected = ar_cache._project_cached_baseline_for_year(
        cache={},
        cache_key=("k",),
        year=2006,
        compute_baseline=lambda: pd.DataFrame({2005: [1.0]}, index=pd.Index(["FR"], name="r_f")),
    )
    assert list(projected.columns) == [2006]
    refreshed = ar_cache._project_cached_baseline_for_year(
        cache={("k",): pd.DataFrame({2005: [1.0]}, index=pd.Index(["FR"], name="r_f"))},
        cache_key=("k",),
        year=2007,
        compute_baseline=lambda: pd.DataFrame({2005: [9.0]}, index=pd.Index(["FR"], name="r_f")),
    )
    assert float(refreshed.iloc[0, 0]) == 1.0

    baseline = ar_cache._ensure_ar_l2_cached(
        context=context,
        state=state,
        ssp_scenario=None,
        cache_key=("AR(E^{CBA_TD})", "L2.a.b", "gwp100_lcia"),
        l2_method="AR(E^{CBA_TD})",
        ref_year=2005,
        lcia_key="gwp100_lcia",
        l1_weights=None,
        pre_weighting=True,
        force_recompute=True,
    )
    assert list(baseline.columns) == [2005]
    cached = ar_cache._ensure_ar_l2_cached(
        context=context,
        state=state,
        ssp_scenario=None,
        cache_key=("AR(E^{CBA_TD})", "L2.a.b", "gwp100_lcia"),
        l2_method="AR(E^{CBA_TD})",
        ref_year=2005,
        lcia_key="gwp100_lcia",
        l1_weights=None,
        pre_weighting=True,
        force_recompute=False,
    )
    assert cached.equals(baseline)


def test_run_ar_runtime_paths(allocation_dummy_repo) -> None:
    context_l2 = _ar_context(fu_code="L2.b.a")
    context_l2.selected_l2_one_step = []
    context_l2.combined = [("AR(E^{CBA_FD})", "EG(Pop)")]
    state_l2 = _ar_state()
    lcia_ref = reference_payloads_mod.load_ar_l2_reference_lcia_payload(
        context=context_l2,
        state=state_l2,
        ref_year=2005,
        lcia_key="gwp100_lcia",
    )
    assert (
        run_ar._compute_ar_l2_result(
            context=context_l2,
            state=state_l2,
            cache_key=("AR(E^{CBA_FD})", "L2.b.a", "gwp100_lcia"),
            l2_method="AR(E^{CBA_FD})",
            year=2004,
            ref_year=2005,
            lcia_data=None,
            l1_weights=None,
        )
        is None
    )

    early = run_ar._compute_ar_l2_result(
        context=context_l2,
        state=state_l2,
        cache_key=("AR(E^{CBA_FD})", "L2.b.a", "gwp100_lcia"),
        l2_method="AR(E^{CBA_FD})",
        year=2004,
        ref_year=2005,
        lcia_data=lcia_ref,
        l1_weights=None,
    )
    assert early is not None
    assert 2005 in state_l2.empty_ref_years

    weighted = run_ar._compute_ar_l2_result(
        context=context_l2,
        state=state_l2,
        cache_key=("AR(E^{CBA_FD})", "L2.b.a", "gwp100_lcia"),
        l2_method="AR(E^{CBA_FD})",
        year=2006,
        ref_year=2005,
        lcia_data=lcia_ref,
        l1_weights=pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_f")),
    )
    assert weighted is not None
    assert list(weighted.columns) == [2006]
    assert "reference_year" in weighted.index.names

    assert (
        run_ar._compute_ar_l2_preweight(
            context=context_l2,
            state=state_l2,
            cache_key=("AR(E^{CBA_FD})", "L2.b.a", "gwp100_lcia", "pre"),
            l2_method="AR(E^{CBA_FD})",
            year=2004,
            ref_year=2005,
            lcia_data=None,
        )
        is None
    )
    preweight_early = run_ar._compute_ar_l2_preweight(
        context=context_l2,
        state=state_l2,
        cache_key=("AR(E^{CBA_FD})", "L2.b.a", "gwp100_lcia", "pre"),
        l2_method="AR(E^{CBA_FD})",
        year=2004,
        ref_year=2005,
        lcia_data=lcia_ref,
    )
    assert preweight_early is not None
    assert list(preweight_early.columns) == [2004]
    preweight_projected = run_ar._compute_ar_l2_preweight(
        context=context_l2,
        state=state_l2,
        cache_key=("AR(E^{CBA_FD})", "L2.b.a", "gwp100_lcia", "pre"),
        l2_method="AR(E^{CBA_FD})",
        year=2006,
        ref_year=2005,
        lcia_data=lcia_ref,
    )
    assert preweight_projected is not None
    assert list(preweight_projected.columns) == [2006]

    context_l1 = _ar_context(fu_code="L1.a", selected_l1=["AR(E^{CBA_FD})"])
    state_l1 = _ar_state()
    lcia_reg = _lcia_reg_frame()
    pop_series = pd.Series([10.0, 20.0], index=pd.Index(["FR", "US"], name="r_f"))
    early_l1 = run_ar._compute_ar_l1_result(
        context=context_l1,
        state=state_l1,
        ssp_scenario=None,
        cache_key=("AR(E^{CBA_FD})", "L1.a"),
        l1_method="AR(E^{CBA_FD})",
        year=2004,
        ref_year=2005,
        lcia_method="gwp100_lcia",
        lcia_kind="CBA_FD",
        lcia_reg=lcia_reg,
        lcia_reg_by_year={2005: lcia_reg},
        rps_df=None,
        impact_parent_map=None,
        pop_series=pop_series,
        pop_ref=pop_series,
        pr_pop=None,
        pr_gdp=None,
        pr_to_mrio=None,
        region_label_override=None,
        use_original_domain=False,
    )
    assert list(early_l1.columns) == [2004]
    assert 2005 in state_l1.empty_ref_years

    ar_ecap = run_ar._compute_ar_l1_result(
        context=context_l1,
        state=state_l1,
        ssp_scenario=None,
        cache_key=("AR(Ecap^{CBA_FD})", "L1.a"),
        l1_method="AR(Ecap^{CBA_FD})",
        year=2006,
        ref_year=2005,
        lcia_method="gwp100_lcia",
        lcia_kind="CBA_FD",
        lcia_reg=lcia_reg,
        lcia_reg_by_year={2005: lcia_reg},
        rps_df=None,
        impact_parent_map=None,
        pop_series=pop_series,
        pop_ref=pop_series,
        pr_pop=None,
        pr_gdp=None,
        pr_to_mrio=None,
        region_label_override=None,
        use_original_domain=False,
    )
    assert list(ar_ecap.columns) == [2006]

    ar_standard = run_ar._compute_ar_l1_result(
        context=context_l1,
        state=state_l1,
        ssp_scenario=None,
        cache_key=("AR(E^{CBA_FD})", "L1.a"),
        l1_method="AR(E^{CBA_FD})",
        year=2006,
        ref_year=2005,
        lcia_method="gwp100_lcia",
        lcia_kind="CBA_FD",
        lcia_reg=lcia_reg,
        lcia_reg_by_year={2005: lcia_reg},
        rps_df=None,
        impact_parent_map=None,
        pop_series=pop_series,
        pop_ref=pop_series,
        pr_pop=None,
        pr_gdp=None,
        pr_to_mrio=None,
        region_label_override=None,
        use_original_domain=False,
    )
    assert list(ar_standard.columns) == [2006]


def test_run_ut_runtime() -> None:
    payload = _ut_payload()
    context = SimpleNamespace(fu_code="L2.a.b")
    state = SimpleNamespace(preweight_cache_by_ssp_scenario={None: {}})

    preweight = run_ut._get_ut_l2_preweight(
        context=context,
        state=state,
        ssp_scenario=None,
        l2_method="UT(FDa)",
        year=2020,
        lcia_data=None,
        lcia_key=None,
        ref_year=2019,
        enacting_metric_l1={"fd_rf": payload["fd_rf"], "gva_rp": payload["gva_rp"]},
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
    assert list(preweight.columns) == [2020]
    cached = run_ut._get_ut_l2_preweight(
        context=context,
        state=state,
        ssp_scenario=None,
        l2_method="UT(FDa)",
        year=2020,
        lcia_data=None,
        lcia_key=None,
        ref_year=2019,
        enacting_metric_l1={"fd_rf": payload["fd_rf"], "gva_rp": payload["gva_rp"]},
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
    assert cached.equals(preweight)
    collapsed_key = run_ut._ut_preweight_cache_key(
        l2_method="UT(FDa)",
        fu_code="L2.a.b",
        year=2020,
    )
    assert list(state.preweight_cache_by_ssp_scenario[None]) == [collapsed_key]
    lcia_labelled = run_ut._get_ut_l2_preweight(
        context=context,
        state=state,
        ssp_scenario=None,
        l2_method="UT(FDa)",
        year=2020,
        lcia_data={"unused": pd.DataFrame()},
        lcia_key="pb_lcia",
        ref_year=2005,
        enacting_metric_l1={"fd_rf": payload["fd_rf"], "gva_rp": payload["gva_rp"]},
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
    assert lcia_labelled is preweight

    weighted = run_ut._weight_ut_contribution_from_preweight(
        context=context,
        l2_method="UT(FDa)",
        year=2020,
        weights=pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_f")),
        pre_weighted=preweight,
    )
    assert list(weighted.columns) == [2020]

    from_preweight = run_ut._compute_ut_weighted_contribution_from_preweight(
        context=context,
        state=state,
        ssp_scenario=None,
        l2_method="UT(FDa)",
        year=2020,
        lcia_data=None,
        lcia_key=None,
        ref_year=2019,
        weights=pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_f")),
        enacting_metric_l1={"fd_rf": payload["fd_rf"], "gva_rp": payload["gva_rp"]},
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
    assert list(from_preweight.columns) == [2020]

    reduced = run_ut._compute_ut_weighted_from_preweight(
        context=context,
        state=state,
        ssp_scenario=None,
        l2_method="UT(FDa)",
        year=2020,
        lcia_data=None,
        lcia_key=None,
        ref_year=2019,
        weights=pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_f")),
        enacting_metric_l1={"fd_rf": payload["fd_rf"], "gva_rp": payload["gva_rp"]},
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
        precomputed_contrib=None,
    )
    assert list(reduced.columns) == [2020]

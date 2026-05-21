from typing import Any, TypedDict, cast

import pandas as pd
import pytest

from pyaesa.asocc.methods.equations import ar_e as ar_e_mod
from pyaesa.asocc.methods.equations import ar_ecap as ar_ecap_mod
from pyaesa.asocc.methods.equations import ar_nan_outputs as ar_nan_outputs_mod
from pyaesa.asocc.methods.equations import ar_result_indexing as ar_result_indexing_mod
from pyaesa.asocc.methods.equations import eg_pop
from pyaesa.asocc.methods.equations import pr_gdpcap
from pyaesa.asocc.methods.equations import pr_hr_ecap_cum as pr_hr
from pyaesa.asocc.methods.equations import share_math
from pyaesa.asocc.methods.equations import ut_fd
from pyaesa.asocc.methods.equations import ut_fda
from pyaesa.asocc.methods.equations import ut_gvaa
from pyaesa.asocc.methods.equations import ut_support
from pyaesa.asocc.methods.equations import ut_td


def _lcia_payload() -> dict[str, pd.DataFrame]:
    idx_impact = pd.Index(["climate_child"], name="impact")
    cols_rp_sp = pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"])
    cols_rf_sp = pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_f", "s_p"])
    cols_rc_sp = pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_c", "s_p"])
    idx_impact_rp_sp = pd.MultiIndex.from_tuples(
        [("climate_child", "FR", "A"), ("climate_child", "US", "A")],
        names=["impact", "r_p", "s_p"],
    )
    return {
        "e_cba_fd_reg": pd.DataFrame(
            [[10.0, 20.0]],
            index=idx_impact,
            columns=pd.Index(["FR", "US"], name="r_f"),
        ),
        "e_cba_fd_rp_sp_rf": pd.DataFrame(
            [[5.0, 6.0], [7.0, 8.0]],
            index=idx_impact_rp_sp,
            columns=pd.Index(["FR", "US"], name="r_f"),
        ),
        "e_cba_fd_rp_sp": pd.DataFrame(
            [[11.0, 15.0]],
            index=idx_impact,
            columns=cols_rp_sp,
        ),
        "e_cba_fd_rf_sp": pd.DataFrame(
            [[4.0, 6.0]],
            index=idx_impact,
            columns=cols_rf_sp,
        ),
        "e_cba_td_rp_sp": pd.DataFrame(
            [[3.0, 4.0]],
            index=idx_impact,
            columns=cols_rp_sp,
        ),
        "e_cba_td_rp_sp_rc": pd.DataFrame(
            [[3.0, 4.0], [5.0, 6.0]],
            index=idx_impact_rp_sp,
            columns=pd.Index(["FR", "US"], name="r_c"),
        ),
        "e_cba_td_rc_sp": pd.DataFrame(
            [[3.0, 4.0]],
            index=idx_impact,
            columns=cols_rc_sp,
        ),
        "e_pba_reg": pd.DataFrame(
            [[10.0, 20.0]],
            index=idx_impact,
            columns=pd.Index(["FR", "US"], name="r_p"),
        ),
        "e_pba_rp_sp": pd.DataFrame(
            [[5.0, 7.0]],
            index=idx_impact,
            columns=cols_rp_sp,
        ),
    }


def _pop_gdp_iso() -> tuple[pd.Series, pd.Series, pd.Series]:
    pop_iso = pd.Series([10.0, 30.0], index=pd.Index(["FRA", "USA"], name="iso3"))
    gdp_iso = pd.Series([100.0, 300.0], index=pd.Index(["FRA", "USA"], name="iso3"))
    iso_to_mrio = pd.Series(["FR", "US"], index=pd.Index(["FRA", "USA"], name="iso3"))
    return pop_iso, gdp_iso, iso_to_mrio


def _pr_hr_inputs() -> tuple[
    dict[int, pd.Series], dict[int, pd.DataFrame], pd.DataFrame, pd.Series, list[int]
]:
    population_by_year = {
        2019: pd.Series([2.0, 4.0], index=pd.Index(["FR", "US"], name="r_f")),
        2020: pd.Series([3.0, 6.0], index=pd.Index(["FR", "US"], name="r_f")),
    }
    lcia_reg_by_year = {
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
    available_years = [2019, 2020]
    return (
        population_by_year,
        lcia_reg_by_year,
        rps_df,
        impact_parent_map,
        available_years,
    )


class _UtInputs(TypedDict):
    fd_rf: pd.Series
    fd_rp_sp_rf: pd.DataFrame
    fd_rp_sp: pd.Series
    fd_rf_sp: pd.Series
    x_to_rc: pd.DataFrame
    kappa: pd.DataFrame
    gva_rp: pd.Series
    omega_reg: pd.DataFrame


def _ut_inputs() -> _UtInputs:
    return {
        "fd_rf": pd.Series([4.0, 6.0], index=pd.Index(["FR", "US"], name="r_f")),
        "fd_rp_sp_rf": pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"]),
            columns=pd.Index(["FR", "US"], name="r_f"),
        ),
        "fd_rp_sp": pd.Series(
            [3.0, 7.0],
            index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"]),
        ),
        "fd_rf_sp": pd.Series(
            [2.0, 8.0],
            index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_f", "s_p"]),
        ),
        "x_to_rc": pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"]),
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
        "gva_rp": pd.Series([5.0, 5.0], index=pd.Index(["FR", "US"], name="r_p")),
        "omega_reg": pd.DataFrame(
            [[0.6, 0.4], [0.4, 0.6]],
            index=pd.Index(["FR", "US"], name="r_u"),
            columns=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"]),
        ),
    }


def test_ar_result_indexing_returns_original_frame_without_trailing_levels() -> None:
    frame = pd.DataFrame({2030: [1.0]}, index=pd.Index(["FR"], name="r_f"))
    assert ar_result_indexing_mod._attach_trailing_constant_levels(  # noqa: SLF001
        result=frame,
        levels=(),
    ).equals(frame)


def test_ar_equations_and() -> None:
    lcia = _lcia_payload()
    pop = pd.Series([10.0, 20.0], index=pd.Index(["FR", "US"], name="r_f"))
    pop_ref = pd.Series([9.0, 19.0], index=pd.Index(["FR", "US"], name="r_f"))

    with pytest.raises(ValueError):
        ar_e_mod.compute_ar_e_l1(
            year=2030,
            lcia_reg=lcia["e_cba_fd_reg"],
            lcia_reg_by_year=None,
            reference_year=None,
        )
    out_l1 = ar_e_mod.compute_ar_e_l1(
        year=2030,
        lcia_reg=lcia["e_cba_fd_reg"],
        lcia_reg_by_year=None,
        reference_year=2000,
        region_label="r_f",
    )
    assert list(out_l1.columns) == [2030]

    with pytest.raises(ValueError):
        ar_ecap_mod.compute_ar_ecap_l1(
            year=2030,
            population=pop,
            population_ref=None,
            lcia_reg=lcia["e_cba_fd_reg"],
            lcia_reg_by_year=None,
            reference_year=2000,
        )
    with pytest.raises(ValueError):
        ar_ecap_mod.compute_ar_ecap_l1(
            year=2030,
            population=pop,
            population_ref=pop_ref,
            lcia_reg=lcia["e_cba_fd_reg"],
            lcia_reg_by_year=None,
            reference_year=None,
        )
    out_ecap = ar_ecap_mod.compute_ar_ecap_l1(
        year=2030,
        population=pop,
        population_ref=pop_ref,
        lcia_reg=lcia["e_cba_fd_reg"],
        lcia_reg_by_year=None,
        reference_year=2000,
        region_label="r_f",
    )
    assert list(out_ecap.columns) == [2030]
    stack_cache: dict[tuple[object, ...], object] = {}
    cached_ecap_first = ar_ecap_mod.compute_ar_ecap_l1(
        year=2030,
        population=pop,
        population_ref=pop_ref,
        lcia_reg=lcia["e_cba_fd_reg"],
        lcia_reg_by_year=None,
        reference_year=2000,
        region_label="r_f",
        index_cache=stack_cache,
    )
    cached_ecap_second = ar_ecap_mod.compute_ar_ecap_l1(
        year=2031,
        population=pop,
        population_ref=pop_ref,
        lcia_reg=lcia["e_cba_fd_reg"],
        lcia_reg_by_year=None,
        reference_year=2000,
        region_label="r_f",
        index_cache=stack_cache,
    )
    assert cached_ecap_first.index is cached_ecap_second.index

    for l2_method, fu_code in [
        ("AR(E^{CBA_FD})", "L2.a.a"),
        ("AR(E^{CBA_FD})", "L2.b.a"),
        ("AR(E^{CBA_FD})", "L2.c.a"),
        ("AR(E^{CBA_TD})", "L2.a.b"),
        ("AR(E^{CBA_TD})", "L2.b.b"),
        ("AR(E^{CBA_TD})", "L2.c.b"),
        ("AR(E^{PBA})", "L2.a.a"),
    ]:
        out = ar_e_mod.compute_ar_e_l2(
            l2_method=l2_method,
            fu_code=fu_code,
            l1_weights=None,
            lcia=lcia,
            reference_year=2000,
            pre_weighting=fu_code == "L2.a.a",
        )
        assert list(out.columns) == [2000]

    assert ar_result_indexing_mod._apply_impact_level(
        pd.DataFrame({"2030": [1.0]}, index=pd.Index(["FR"], name="r_p")),
        "climate_parent",
    ).index.names == ["impact", "r_p"]
    assert (
        "reference_year"
        in ar_result_indexing_mod._add_reference_level(
            pd.DataFrame({"2030": [1.0]}, index=pd.Index(["FR"], name="r_p")),
            2000,
        ).index.names
    )
    assert list(
        ar_nan_outputs_mod._stack_matrix_to_year(
            pd.DataFrame({"FR": [1.0]}, index=pd.Index(["impact"], name="impact")),
            2030,
        ).columns
    ) == [2030]
    assert list(
        ar_nan_outputs_mod._nan_like_ar_l1(
            lcia["e_cba_fd_reg"],
            2030,
            region_label="r_f",
        ).columns
    ) == [2030]
    assert list(
        ar_nan_outputs_mod._nan_like_ar_l2(
            l2_method="AR(E^{CBA_TD})",
            fu_code="L2.c.b",
            lcia=lcia,
            year=2030,
            pre_weighting=False,
        ).columns
    ) == [2030]


def test_pr_equations(allocation_dummy_repo) -> None:
    pop_iso, gdp_iso, iso_to_mrio = _pop_gdp_iso()
    population_by_year, lcia_reg_by_year, rps_df, impact_parent_map, available_years = (
        _pr_hr_inputs()
    )

    grouped_pre = pr_gdpcap._compute_pre_aggregated_share(
        pop_iso=pop_iso,
        gdp_iso=gdp_iso,
        iso_to_mrio=iso_to_mrio,
        source_key="oecd_v2025",
        group_version="demo_reg",
    )
    assert list(grouped_pre.index) == ["EU", "NAM"]
    out_pre = pr_gdpcap.compute_pr_gdpcap(
        pop_iso=pop_iso,
        gdp_iso=gdp_iso,
        iso_to_mrio=iso_to_mrio,
        year=2020,
        source_key="oecd_v2025",
        group_version="demo_reg",
        aggregation_mode="pre",
        region_label="r_f",
    )
    assert out_pre.index.name == "r_f"
    out_post = pr_gdpcap.compute_pr_gdpcap(
        pop_iso=pop_iso,
        gdp_iso=gdp_iso,
        iso_to_mrio=iso_to_mrio,
        year=2020,
        source_key="oecd_v2025",
        group_version="demo_reg",
        aggregation_mode="post",
        region_label="r_f",
    )
    assert out_post.sum().iloc[0] == pytest.approx(1.0)

    parent_cum = pr_hr.build_parent_cumulative_per_cap(
        impact_year=2020,
        population_by_year=population_by_year,
        lcia_reg_by_year=lcia_reg_by_year,
        rps_df=rps_df,
        impact_parent_map=impact_parent_map,
        available_years=available_years,
        parent_cum_cache={},
    )
    assert "climate_parent" in parent_cum
    cache = {2019: {"climate_parent": parent_cum["climate_parent"].copy()}}
    grouped = pr_hr.compute_pr_hr(
        year=2020,
        impact_year=2020,
        population=population_by_year[2020],
        population_by_year=population_by_year,
        lcia_reg_by_year=lcia_reg_by_year,
        rps_df=rps_df,
        impact_parent_map=impact_parent_map,
        available_years=available_years,
        source_key="oecd_v2025",
        group_version="demo_reg",
        aggregation_mode="post",
        region_label="r_f",
        parent_cum_cache=cache,
    )
    assert {2019, 2020}.issubset(set(cache))
    assert grouped.index.get_level_values("r_f").unique().tolist() == ["EU", "NAM"]
    cached_grouped = pr_hr.compute_pr_hr(
        year=2020,
        impact_year=2020,
        population=population_by_year[2020],
        population_by_year=population_by_year,
        lcia_reg_by_year=lcia_reg_by_year,
        rps_df=rps_df,
        impact_parent_map=impact_parent_map,
        available_years=available_years,
        source_key="oecd_v2025",
        group_version="demo_reg",
        aggregation_mode="post",
        region_label="r_f",
        parent_cum_cache=cache,
    )
    pd.testing.assert_frame_equal(cached_grouped, grouped)


def test_ut_equations() -> None:
    payload = _ut_inputs()
    w_rf = pd.Series([0.25, 0.75], index=pd.Index(["FR", "US"], name="r_f"))
    w_ru = pd.Series([0.25, 0.75], index=pd.Index(["FR", "US"], name="r_u"))

    assert list(
        ut_fd.compute_ut_fd_l2(
            fu_code="L2.a.a",
            year=2020,
            l1_weights=None,
            fd_rf=payload["fd_rf"],
            fd_rp_sp_rf=payload["fd_rp_sp_rf"],
            fd_rp_sp=payload["fd_rp_sp"],
            fd_rf_sp=payload["fd_rf_sp"],
            pre_weighting=False,
        ).columns
    ) == [2020]
    assert list(
        ut_support._stack_to_year(
            pd.DataFrame(
                [[1.0, 2.0]],
                index=pd.Index(["FR"], name="r_p"),
                columns=pd.Index(["FR", "US"], name="r_f"),
            ),
            2020,
            "r_f",
        ).columns
    ) == [2020]
    assert list(
        ut_td.compute_ut_td_l2(
            fu_code="L2.b.b",
            year=2020,
            fd_rf=payload["fd_rf"],
            x_to_rc=payload["x_to_rc"],
        ).columns
    ) == [2020]
    assert list(
        ut_fda.compute_ut_fda_l2(
            fu_code="L2.a.b",
            year=2020,
            l1_weights=w_rf,
            fd_rf=payload["fd_rf"],
            x_to_rc=payload["x_to_rc"],
            kappa=payload["kappa"],
            pre_weighting=False,
        ).columns
    ) == [2020]
    assert list(
        ut_gvaa.compute_ut_gvaa_l2(
            fu_code="L2.a.b",
            year=2020,
            l1_weights=w_ru,
            gva_rp=payload["gva_rp"],
            x_to_rc=payload["x_to_rc"],
            omega_reg=payload["omega_reg"],
            pre_weighting=False,
        ).columns
    ) == [2020]

    empty_adjust = ut_fda._adjust_td_to_fd_by_rc(
        pd.DataFrame(index=payload["x_to_rc"].index),
        payload["kappa"],
    )
    assert empty_adjust.empty


def test_share_and_ar_utilities() -> None:
    numer = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index(["a", "b"], name="k"))
    denom = pd.Series([1.0, 0.0], index=numer.index)
    divided = share_math.safe_divide_frame(numer, denom, axis=0)
    assert float(divided.loc["a", "x"]) == 1.0
    assert pd.isna(divided.loc["b", "x"])

    series_divided = share_math.safe_divide_series(
        pd.Series([2.0, 2.0]),
        pd.Series([2.0, 0.0]),
    )
    assert float(series_divided.iloc[0]) == 1.0
    assert pd.isna(series_divided.iloc[1])
    misaligned_divided = share_math.safe_divide_series(
        pd.Series([4.0], index=pd.Index(["a"], name="k")),
        pd.Series([2.0], index=pd.Index(["b"], name="k")),
    )
    assert misaligned_divided.isna().all()
    assert misaligned_divided.index.tolist() == ["a", "b"]

    normalized = share_math.normalize_share(pd.Series([1.0, 3.0], index=["a", "b"]))
    assert normalized.sum() == pytest.approx(1.0)
    assert share_math.normalize_share(pd.Series([0.0, 0.0], index=["a", "b"])).tolist() == [
        0.0,
        0.0,
    ]
    assert (
        share_math.normalize_share(pd.Series([float("nan"), float("nan")], dtype="float64"))
        .isna()
        .all()
    )

    base = pd.DataFrame({2020: [1.0]}, index=pd.Index(["FR"], name="r_f"))
    impacted = ar_result_indexing_mod._apply_impact_level(base, "climate")
    assert impacted.index.names == ["impact", "r_f"]

    existing_impact = pd.DataFrame(
        {2020: [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [("climate", "FR"), ("water", "US")],
            names=["impact", "r_f"],
        ),
    )
    assert ar_result_indexing_mod._apply_impact_level(existing_impact, "climate").shape[0] == 1
    impact_only = pd.DataFrame(
        {2020: [1.0, 2.0]},
        index=pd.Index(["climate", "water"], name="impact"),
    )
    assert list(ar_result_indexing_mod._apply_impact_level(impact_only, "climate").index) == [
        "climate"
    ]

    with_ref = ar_result_indexing_mod._add_reference_level(base, 2019)
    assert with_ref.index.names == ["r_f", "reference_year"]
    replaced_ref = ar_result_indexing_mod._add_reference_level(with_ref, 2020)
    replaced_ref_values = cast(
        list[int], replaced_ref.index.get_level_values("reference_year").tolist()
    )
    assert replaced_ref_values == [2020]
    replaced_single = ar_result_indexing_mod._add_reference_level(
        pd.DataFrame({2020: [1.0]}, index=pd.Index([2018], name="reference_year")),
        2021,
    )
    assert cast(int, replaced_single.index.tolist()[0]) == 2021

    assert (
        ar_result_indexing_mod._attach_impact_reference_levels(
            result=base,
            impact=None,
            reference_year=None,
        )
        is base
    )
    assert ar_result_indexing_mod._attach_impact_reference_levels(
        result=base,
        impact=None,
        reference_year=2019,
    ).index.names == ["r_f", "reference_year"]
    assert ar_result_indexing_mod._attach_impact_reference_levels(
        result=base,
        impact="climate",
        reference_year=None,
    ).index.names == ["impact", "r_f"]
    multi_base = pd.DataFrame(
        {2020: [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "B")], names=["r_f", "s_p"]),
    )
    trailed_multi = ar_result_indexing_mod._attach_trailing_constant_levels(
        result=multi_base,
        levels=(("reference_year", 2019),),
    )
    assert trailed_multi.index.names == ["r_f", "s_p", "reference_year"]
    attached_multi = ar_result_indexing_mod._attach_impact_reference_levels(
        result=multi_base,
        impact="climate",
        reference_year=2019,
    )
    assert attached_multi.index.names == ["impact", "r_f", "s_p", "reference_year"]
    assert ar_result_indexing_mod._attach_impact_reference_levels(
        result=with_ref,
        impact="climate",
        reference_year=2021,
    ).index.names == ["impact", "r_f", "reference_year"]
    already_both = pd.DataFrame(
        {2020: [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [("climate", "FR", 2019), ("water", "US", 2019)],
            names=["impact", "r_f", "reference_year"],
        ),
    )
    attached_both = ar_result_indexing_mod._attach_impact_reference_levels(
        result=already_both,
        impact="climate",
        reference_year=2021,
    )
    assert attached_both.index.names == ["impact", "r_f", "reference_year"]
    attached_both_ref_values = cast(
        list[int], attached_both.index.get_level_values("reference_year").tolist()
    )
    assert attached_both_ref_values == [2021]
    attached_ref_only = ar_result_indexing_mod._attach_impact_reference_levels(
        result=with_ref,
        impact=None,
        reference_year=2022,
    )
    attached_ref_only_values = cast(
        list[int], attached_ref_only.index.get_level_values("reference_year").tolist()
    )
    assert attached_ref_only_values == [2022]
    attached_impact_only = ar_result_indexing_mod._attach_impact_reference_levels(
        result=existing_impact,
        impact="climate",
        reference_year=None,
    )
    assert attached_impact_only.index.get_level_values("impact").tolist() == ["climate"]
    index_cache: dict[tuple[object, ...], ar_result_indexing_mod._CachedIndexEntry] = {}
    cached_first = ar_result_indexing_mod._attach_impact_reference_levels(
        result=base,
        impact="climate",
        reference_year=2019,
        trailing_levels=(("l2_reuse_year", 2015),),
        index_cache=index_cache,
    )
    cached_second = ar_result_indexing_mod._attach_impact_reference_levels(
        result=base,
        impact="climate",
        reference_year=2019,
        trailing_levels=(("l2_reuse_year", 2015),),
        index_cache=index_cache,
    )
    assert cached_first.index is cached_second.index
    trailing_cache: dict[tuple[object, ...], ar_result_indexing_mod._CachedIndexEntry] = {}
    trailing_first = ar_result_indexing_mod._attach_trailing_constant_levels(
        result=base,
        levels=(("reference_year", 2019),),
        index_cache=trailing_cache,
    )
    trailing_second = ar_result_indexing_mod._attach_trailing_constant_levels(
        result=base,
        levels=(("reference_year", 2019),),
        index_cache=trailing_cache,
    )
    assert trailing_first.index is trailing_second.index
    nan_cache: dict[tuple[object, ...], ar_nan_outputs_mod._CachedIndexEntry] = {}
    lcia_reg = pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.Index(["climate"], name="impact"),
        columns=pd.Index(["FR", "US"], name="r_f"),
    )
    nan_first = ar_nan_outputs_mod._nan_like_ar_l1(
        lcia_reg,
        2019,
        region_label="r_f",
        index_cache=nan_cache,
    )
    nan_second = ar_nan_outputs_mod._nan_like_ar_l1(
        lcia_reg,
        2020,
        region_label="r_f",
        index_cache=nan_cache,
    )
    assert nan_first.index is nan_second.index
    assert (
        ar_result_indexing_mod._cached_index_for(
            index_cache=None,
            cache_key=("x",),
            source_index=base.index,
        )
        is None
    )
    assert (
        ar_result_indexing_mod._cached_index_for(
            index_cache={("x",): (pd.Index(["FR"]), pd.Index(["FR"]))},
            cache_key=("x",),
            source_index=base.index,
        )
        is None
    )

    stacked = ar_nan_outputs_mod._stack_matrix_to_year(
        pd.DataFrame(
            [[1.0, 2.0]],
            index=pd.Index(["impact"], name="impact"),
            columns=pd.Index(["FR", "US"], name="r_f"),
        ),
        2020,
    )
    assert list(stacked.columns) == [2020]
    assert stacked.index.names == ["impact", "r_f"]

    named_nan = ar_nan_outputs_mod._nan_like_ar_l1(
        pd.DataFrame(
            [[1.0, 2.0]],
            index=pd.Index(["impact"], name="impact"),
            columns=pd.Index(["FR", "US"], name="r_f"),
        ),
        2020,
    )
    assert named_nan.index.names == ["impact", "r_f"]
    unnamed_nan = ar_nan_outputs_mod._nan_like_ar_l1(
        pd.DataFrame(
            [[1.0, 2.0]],
            index=pd.Index(["impact"], name="impact"),
            columns=pd.Index(["FR", "US"]),
        ),
        2020,
    )
    assert unnamed_nan.index.names == ["impact", "region"]

    fd_nan = ar_nan_outputs_mod._nan_like_ar_l2(
        l2_method="AR(E^{CBA_FD})",
        fu_code="L2.a.a",
        lcia={"e_cba_fd_rp_sp": _lcia_payload()["e_cba_fd_rp_sp"]},
        year=2020,
        pre_weighting=False,
    )
    assert fd_nan.index.names == ["impact", "r_p", "s_p"]
    td_nan = ar_nan_outputs_mod._nan_like_ar_l2(
        l2_method="AR(E^{CBA_TD})",
        fu_code="L2.b.b",
        lcia={"e_cba_td_rp_sp_rc": _lcia_payload()["e_cba_td_rp_sp_rc"]},
        year=2020,
        pre_weighting=False,
    )
    assert td_nan.index.names == ["impact", "r_p", "s_p", "r_c"]
    td_rp_sp_nan = ar_nan_outputs_mod._nan_like_ar_l2(
        l2_method="AR(E^{CBA_TD})",
        fu_code="L2.a.b",
        lcia={"e_cba_td_rp_sp": _lcia_payload()["e_cba_td_rp_sp"]},
        year=2020,
        pre_weighting=False,
    )
    assert td_rp_sp_nan.index.names == ["impact", "r_p", "s_p"]
    fd_rf_sp_nan = ar_nan_outputs_mod._nan_like_ar_l2(
        l2_method="AR(E^{CBA_FD})",
        fu_code="L2.c.a",
        lcia={"e_cba_fd_rf_sp": _lcia_payload()["e_cba_fd_rf_sp"]},
        year=2020,
        pre_weighting=False,
    )
    assert fd_rf_sp_nan.index.names == ["impact", "r_f", "s_p"]
    pba_nan = ar_nan_outputs_mod._nan_like_ar_l2(
        l2_method="AR(E^{PBA})",
        fu_code="L2.a.c",
        lcia={"e_pba_rp_sp": _lcia_payload()["e_pba_rp_sp"]},
        year=2020,
        pre_weighting=False,
    )
    assert pba_nan.index.names == ["impact", "r_p", "s_p"]
    with pytest.raises(ValueError):
        ar_nan_outputs_mod._nan_like_ar_l2(
            l2_method="AR(E^{PBA})",
            fu_code="L2.a.c",
            lcia={},
            year=2020,
            pre_weighting=False,
        )


def test_ar_equation_failure_and_weighting_branches() -> None:
    lcia = _lcia_payload()
    pop = pd.Series([10.0, 20.0], index=pd.Index(["FR", "US"], name="r_f"))
    pop_ref = pd.Series([9.0, 19.0], index=pd.Index(["FR", "US"], name="r_f"))

    with pytest.raises(ValueError):
        ar_e_mod.compute_ar_e_l1(
            year=2030,
            lcia_reg=None,
            lcia_reg_by_year=None,
            reference_year=2000,
        )
    with pytest.raises(ValueError):
        ar_e_mod.compute_ar_e_l1(
            year=2030,
            lcia_reg=None,
            lcia_reg_by_year={2001: lcia["e_cba_fd_reg"]},
            reference_year=2000,
        )
    with pytest.raises(ValueError):
        ar_ecap_mod.compute_ar_ecap_l1(
            year=2030,
            population=pop,
            population_ref=pop_ref,
            lcia_reg=None,
            lcia_reg_by_year=None,
            reference_year=2000,
        )
    with pytest.raises(ValueError):
        ar_ecap_mod.compute_ar_ecap_l1(
            year=2030,
            population=pop,
            population_ref=pop_ref,
            lcia_reg=None,
            lcia_reg_by_year={2001: lcia["e_cba_fd_reg"]},
            reference_year=2000,
        )
    with pytest.raises(ValueError):
        ar_ecap_mod.compute_ar_ecap_l1(
            year=2030,
            population=pop,
            population_ref=pd.Series([9.0, 10.0], index=pd.Index(["FR", "FR"], name="r_f")),
            lcia_reg=lcia["e_cba_fd_reg"],
            lcia_reg_by_year=None,
            reference_year=2000,
        )
    with pytest.raises(ValueError):
        ar_ecap_mod.compute_ar_ecap_l1(
            year=2030,
            population=pd.Series([9.0, 10.0], index=pd.Index(["FR", "FR"], name="r_f")),
            population_ref=pop_ref,
            lcia_reg=lcia["e_cba_fd_reg"],
            lcia_reg_by_year=None,
            reference_year=2000,
        )

    with pytest.raises(ValueError):
        ar_e_mod.compute_ar_e_l2(
            l2_method="AR(E^{CBA_FD})",
            fu_code="L2.a.a",
            l1_weights=None,
            lcia=lcia,
            reference_year=None,
        )
    with pytest.raises(ValueError):
        ar_e_mod.compute_ar_e_l2(
            l2_method="AR(E^{CBA_FD})",
            fu_code="L2.a.a",
            l1_weights=None,
            lcia=cast(Any, None),
            reference_year=2000,
        )
    out_a_weighted = ar_e_mod.compute_ar_e_l2(
        l2_method="AR(E^{CBA_FD})",
        fu_code="L2.a.a",
        l1_weights=pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_f")),
        lcia=lcia,
        reference_year=2000,
        pre_weighting=False,
    )
    assert list(out_a_weighted.columns) == [2000]
    out_a_unweighted = ar_e_mod.compute_ar_e_l2(
        l2_method="AR(E^{CBA_FD})",
        fu_code="L2.a.a",
        l1_weights=None,
        lcia=lcia,
        reference_year=2000,
        pre_weighting=False,
    )
    assert list(out_a_unweighted.columns) == [2000]
    out_b_pre = ar_e_mod.compute_ar_e_l2(
        l2_method="AR(E^{CBA_FD})",
        fu_code="L2.b.a",
        l1_weights=None,
        lcia=lcia,
        reference_year=2000,
        pre_weighting=True,
    )
    assert list(out_b_pre.columns) == [2000]
    out_b_weighted = ar_e_mod.compute_ar_e_l2(
        l2_method="AR(E^{CBA_FD})",
        fu_code="L2.b.a",
        l1_weights=pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_f")),
        lcia=lcia,
        reference_year=2000,
        pre_weighting=False,
    )
    assert list(out_b_weighted.columns) == [2000]
    out_c_pre = ar_e_mod.compute_ar_e_l2(
        l2_method="AR(E^{CBA_FD})",
        fu_code="L2.c.a",
        l1_weights=None,
        lcia=lcia,
        reference_year=2000,
        pre_weighting=True,
    )
    assert list(out_c_pre.columns) == [2000]
    out_c_weighted = ar_e_mod.compute_ar_e_l2(
        l2_method="AR(E^{CBA_FD})",
        fu_code="L2.c.a",
        l1_weights=pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_f")),
        lcia=lcia,
        reference_year=2000,
        pre_weighting=False,
    )
    assert list(out_c_weighted.columns) == [2000]
    out_pba_weighted = ar_e_mod.compute_ar_e_l2(
        l2_method="AR(E^{PBA})",
        fu_code="L2.a.c",
        l1_weights=pd.Series([0.4, 0.6], index=pd.Index(["FR", "US"], name="r_p")),
        lcia=lcia,
        reference_year=2000,
        pre_weighting=False,
    )
    assert list(out_pba_weighted.columns) == [2000]
    out_pba_unweighted = ar_e_mod.compute_ar_e_l2(
        l2_method="AR(E^{PBA})",
        fu_code="L2.a.c",
        l1_weights=None,
        lcia=lcia,
        reference_year=2000,
        pre_weighting=False,
    )
    assert list(out_pba_unweighted.columns) == [2000]


def test_pr_hr_additional_runtime_paths() -> None:
    population_by_year, lcia_reg_by_year, rps_df, impact_parent_map, available_years = (
        _pr_hr_inputs()
    )
    impact_df = lcia_reg_by_year[2020]

    with pytest.raises(ValueError):
        pr_hr.build_parent_cumulative_per_cap(
            impact_year=2021,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps_df=rps_df,
            impact_parent_map=impact_parent_map,
            available_years=available_years,
            parent_cum_cache={},
        )

    with pytest.raises(ValueError):
        pr_hr._collect_parent_cumulative_per_cap(
            impact_year=2020,
            impact_df=impact_df,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps=rps_df.set_index("impact"),
            impact_parent_map=impact_parent_map,
            available_years=[2018, 2020],
        )
    assert (
        pr_hr._collect_parent_cumulative_per_cap(
            impact_year=2020,
            impact_df=impact_df,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps=pd.DataFrame(
                {"responsibility_period_years": [pd.NA]},
                index=pd.Index(["climate_child"], name="impact"),
            ),
            impact_parent_map=impact_parent_map,
            available_years=available_years,
        )
        == {}
    )
    assert (
        pr_hr._collect_parent_cumulative_per_cap(
            impact_year=2020,
            impact_df=impact_df,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps=rps_df.set_index("impact"),
            impact_parent_map=pd.Series(
                [None],
                index=pd.Index(["climate_child"], name="impact"),
            ),
            available_years=available_years,
        )
        == {}
    )

    multi_parent_map = pd.Series(
        ["climate_parent", "climate_parent_2"],
        index=pd.Index(["climate_child", "climate_child"], name="impact"),
    )
    assert pr_hr._resolve_parent_keys(
        impact_parent_map=multi_parent_map,
        impact="climate_child",
    ) == ["climate_parent", "climate_parent_2"]
    assert (
        pr_hr._resolve_parent_keys(
            impact_parent_map=pd.Series(dtype=object),
            impact="missing",
        )
        == []
    )

    lcia_with_duplicates = {
        2020: pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=pd.MultiIndex.from_tuples(
                [("climate_child", "a"), ("climate_child", "b")],
                names=["impact", "detail"],
            ),
            columns=pd.Index(["FR", "US"], name="r_f"),
        )
    }
    per_cap = pr_hr._impact_per_cap_for_year(
        impact="climate_child",
        year_item=2020,
        population_by_year={2020: population_by_year[2020]},
        lcia_reg_by_year=lcia_with_duplicates,
        per_cap_cache={},
    )
    assert per_cap is not None
    assert per_cap.index.tolist() == ["FR", "US"]
    simple_duplicate_per_cap = pr_hr._impact_per_cap_for_year(
        impact="climate_child",
        year_item=2020,
        population_by_year={2020: population_by_year[2020]},
        lcia_reg_by_year={
            2020: pd.DataFrame(
                [[1.0, 2.0], [3.0, 4.0]],
                index=pd.Index(["climate_child", "climate_child"], name="impact"),
                columns=pd.Index(["FR", "US"], name="r_f"),
            )
        },
        per_cap_cache={},
    )
    assert simple_duplicate_per_cap is not None
    assert simple_duplicate_per_cap.tolist() == pytest.approx([4.0 / 3.0, 1.0])
    assert (
        pr_hr._impact_per_cap_for_year(
            impact="climate_child",
            year_item=2021,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            per_cap_cache={},
        )
        is None
    )

    incremental = pr_hr._collect_parent_cumulative_per_cap_incremental(
        impact_year=2020,
        impact_df=impact_df,
        population_by_year=population_by_year,
        lcia_reg_by_year=lcia_reg_by_year,
        rps=rps_df.assign(responsibility_period_years=1).set_index("impact"),
        impact_parent_map=impact_parent_map,
        available_years=available_years,
        previous_parent_cum={
            "climate_parent": pd.Series([5.0, 5.0], index=pd.Index(["FR", "US"], name="r_f"))
        },
    )
    assert "climate_parent" in incremental
    negative_incremental = pr_hr._collect_parent_cumulative_per_cap_incremental(
        impact_year=2020,
        impact_df=impact_df,
        population_by_year={2019: population_by_year[2019]},
        lcia_reg_by_year={2019: lcia_reg_by_year[2019], 2020: lcia_reg_by_year[2020]},
        rps=rps_df.assign(responsibility_period_years=1).set_index("impact"),
        impact_parent_map=impact_parent_map,
        available_years=available_years,
        previous_parent_cum={},
    )
    assert (negative_incremental["climate_parent"] < 0).all()
    with pytest.raises(ValueError):
        pr_hr._collect_parent_cumulative_per_cap_incremental(
            impact_year=2020,
            impact_df=impact_df,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps=rps_df.assign(responsibility_period_years=2).set_index("impact"),
            impact_parent_map=impact_parent_map,
            available_years=[2018, 2020],
            previous_parent_cum={},
        )
    assert pr_hr._resolve_parent_cumulative_per_cap(
        impact_year=2020,
        impact_df=impact_df,
        population_by_year=population_by_year,
        lcia_reg_by_year=lcia_reg_by_year,
        rps=rps_df.set_index("impact"),
        impact_parent_map=impact_parent_map,
        available_years=available_years,
        parent_cum_cache=None,
    )
    with pytest.raises(ValueError):
        pr_hr.build_parent_cumulative_per_cap(
            impact_year=2020,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps_df=pd.DataFrame({"impact": ["climate_child"]}),
            impact_parent_map=impact_parent_map,
            available_years=available_years,
            parent_cum_cache={},
        )
    empty_parent = pr_hr.compute_pr_hr(
        year=2020,
        impact_year=2020,
        population=population_by_year[2020],
        population_by_year=population_by_year,
        lcia_reg_by_year={
            2020: pd.DataFrame(
                [[4.0, 8.0]],
                index=pd.Index(["other_impact"], name="impact"),
                columns=pd.Index(["FR", "US"], name="r_f"),
            )
        },
        rps_df=rps_df,
        impact_parent_map=impact_parent_map,
        available_years=[2020],
        source_key="oecd_v2025",
        group_version=None,
        aggregation_mode="pre",
        region_label="r_f",
        parent_cum_cache={},
    )
    assert empty_parent.empty


def test_pr_hr_rp1_fallback_and_incremental_edge_paths() -> None:
    population_by_year, lcia_reg_by_year, rps_df, impact_parent_map, available_years = (
        _pr_hr_inputs()
    )
    impact_df = lcia_reg_by_year[2020]
    rp1_rps = pd.DataFrame(
        {"responsibility_period_years": [1]},
        index=pd.Index(["climate_child"], name="impact"),
    )
    zero_rp1_lcia = pd.DataFrame(
        [[0.0, 0.0]],
        index=pd.Index(["climate_child"], name="impact"),
        columns=pd.Index(["FR", "US"], name="r_f"),
    )

    rp1_fallback_calls: list[tuple[list[str], int, int]] = []
    resolved_rp1 = pr_hr._resolve_rp1_per_cap_for_year(
        impact="climate_child",
        impact_year=2020,
        population_by_year=population_by_year,
        lcia_reg_by_year={
            2019: lcia_reg_by_year[2019],
            2020: zero_rp1_lcia,
        },
        available_years=available_years,
        per_cap_cache={},
        fallback_callback=lambda impacts, target_year, fallback_year: rp1_fallback_calls.append(
            (impacts, target_year, fallback_year)
        ),
    )
    assert resolved_rp1 is not None
    assert resolved_rp1.tolist() == pytest.approx([2.0, 2.0])
    assert rp1_fallback_calls == [(["climate_child"], 2020, 2019)]

    collect_fallback_calls: list[tuple[list[str], int, int]] = []
    fallback_parent = pr_hr._collect_parent_cumulative_per_cap(
        impact_year=2020,
        impact_df=zero_rp1_lcia,
        population_by_year=population_by_year,
        lcia_reg_by_year={
            2019: lcia_reg_by_year[2019],
            2020: zero_rp1_lcia,
        },
        rps=rp1_rps,
        impact_parent_map=impact_parent_map,
        available_years=available_years,
        fallback_callback=lambda impacts, target_year, fallback_year: collect_fallback_calls.append(
            (impacts, target_year, fallback_year)
        ),
    )
    assert fallback_parent["climate_parent"].tolist() == pytest.approx([2.0, 2.0])
    assert collect_fallback_calls == [(["climate_child"], 2020, 2019)]

    assert (
        pr_hr._collect_parent_cumulative_per_cap(
            impact_year=2020,
            impact_df=impact_df,
            population_by_year={},
            lcia_reg_by_year={},
            rps=rp1_rps,
            impact_parent_map=impact_parent_map,
            available_years=[2020],
        )
        == {}
    )

    partial_parent = pr_hr._collect_parent_cumulative_per_cap(
        impact_year=2020,
        impact_df=impact_df,
        population_by_year={2020: population_by_year[2020]},
        lcia_reg_by_year={2020: lcia_reg_by_year[2020]},
        rps=rps_df.set_index("impact"),
        impact_parent_map=impact_parent_map,
        available_years=available_years,
    )
    assert partial_parent["climate_parent"].tolist() == pytest.approx([2.0, 2.0])

    assert (
        pr_hr._collect_parent_cumulative_per_cap(
            impact_year=2020,
            impact_df=impact_df,
            population_by_year={},
            lcia_reg_by_year={},
            rps=rps_df.set_index("impact"),
            impact_parent_map=impact_parent_map,
            available_years=available_years,
        )
        == {}
    )

    incremental_from_empty = pr_hr._collect_parent_cumulative_per_cap_incremental(
        impact_year=2020,
        impact_df=impact_df,
        population_by_year={2020: population_by_year[2020]},
        lcia_reg_by_year={2020: lcia_reg_by_year[2020]},
        rps=rp1_rps,
        impact_parent_map=impact_parent_map,
        available_years=available_years,
        previous_parent_cum={},
    )
    assert incremental_from_empty["climate_parent"].tolist() == pytest.approx([2.0, 2.0])

    rp1_cache: dict[int, dict[str, pd.Series]] = {}
    cached_parent = pr_hr._resolve_parent_cumulative_per_cap(
        impact_year=2020,
        impact_df=impact_df,
        population_by_year=population_by_year,
        lcia_reg_by_year=lcia_reg_by_year,
        rps=rp1_rps,
        impact_parent_map=impact_parent_map,
        available_years=available_years,
        parent_cum_cache=rp1_cache,
    )
    assert cached_parent["climate_parent"].tolist() == pytest.approx([2.0, 2.0])
    cached_parent["climate_parent"].loc["FR"] = -1.0
    assert rp1_cache[2020]["climate_parent"].loc["FR"] == pytest.approx(2.0)

    post_without_group = pr_hr.compute_pr_hr(
        year=2020,
        impact_year=2020,
        population=population_by_year[2020],
        population_by_year=population_by_year,
        lcia_reg_by_year=lcia_reg_by_year,
        rps_df=rps_df,
        impact_parent_map=impact_parent_map,
        available_years=available_years,
        source_key="oecd_v2025",
        group_version=None,
        aggregation_mode="post",
        region_label="r_f",
        parent_cum_cache={},
    )
    assert post_without_group.index.get_level_values("r_f").tolist() == ["FR", "US"]


def test_ut_additional_paths() -> None:
    payload = _ut_inputs()
    w_rf = pd.Series([0.25, 0.75], index=pd.Index(["FR", "US"], name="r_f"))
    w_ru = pd.Series([0.25, 0.75], index=pd.Index(["FR", "US"], name="r_u"))

    assert list(
        ut_fd.compute_ut_fd_l2(
            fu_code="L2.b.a",
            year=2020,
            l1_weights=w_rf,
            fd_rf=payload["fd_rf"],
            fd_rp_sp_rf=payload["fd_rp_sp_rf"],
            fd_rp_sp=payload["fd_rp_sp"],
            fd_rf_sp=payload["fd_rf_sp"],
            pre_weighting=False,
        ).columns
    ) == [2020]
    assert list(
        ut_fd.compute_ut_fd_l2(
            fu_code="L2.c.a",
            year=2020,
            l1_weights=None,
            fd_rf=payload["fd_rf"],
            fd_rp_sp_rf=payload["fd_rp_sp_rf"],
            fd_rp_sp=payload["fd_rp_sp"],
            fd_rf_sp=payload["fd_rf_sp"],
            pre_weighting=True,
        ).columns
    ) == [2020]
    with pytest.raises(ValueError):
        ut_fda._adjust_td_to_fd(
            pd.DataFrame(index=payload["x_to_rc"].index),
            payload["kappa"],
        )
    assert ut_fda._adjust_td_to_fd_by_rc_sp(
        pd.DataFrame(index=payload["x_to_rc"].index),
        payload["kappa"],
    ).empty
    assert list(
        ut_fda.compute_ut_fda_l2(
            fu_code="L2.b.b",
            year=2020,
            l1_weights=None,
            fd_rf=payload["fd_rf"],
            x_to_rc=payload["x_to_rc"],
            kappa=payload["kappa"],
            pre_weighting=True,
        ).columns
    ) == [2020]
    assert list(
        ut_fda.compute_ut_fda_l2(
            fu_code="L2.c.b",
            year=2020,
            l1_weights=None,
            fd_rf=payload["fd_rf"],
            x_to_rc=payload["x_to_rc"],
            kappa=payload["kappa"],
            pre_weighting=True,
        ).columns
    ) == [2020]
    assert list(
        ut_fda.compute_ut_fda_l2(
            fu_code="L2.c.b",
            year=2020,
            l1_weights=w_rf,
            fd_rf=payload["fd_rf"],
            x_to_rc=payload["x_to_rc"],
            kappa=payload["kappa"],
            pre_weighting=False,
        ).columns
    ) == [2020]
    assert ut_gvaa._weighted_omega_by_rc(
        omega_reg=payload["omega_reg"],
        x_to_rc=pd.DataFrame(),
    ).empty
    assert list(
        ut_gvaa.compute_ut_gvaa_l2(
            fu_code="L2.b.b",
            year=2020,
            l1_weights=None,
            gva_rp=payload["gva_rp"],
            x_to_rc=payload["x_to_rc"],
            omega_reg=payload["omega_reg"],
            pre_weighting=True,
        ).columns
    ) == [2020]
    assert list(
        ut_gvaa.compute_ut_gvaa_l2(
            fu_code="L2.c.b",
            year=2020,
            l1_weights=None,
            gva_rp=payload["gva_rp"],
            x_to_rc=payload["x_to_rc"],
            omega_reg=payload["omega_reg"],
            pre_weighting=True,
        ).columns
    ) == [2020]
    assert list(
        ut_gvaa.compute_ut_gvaa_l2(
            fu_code="L2.c.b",
            year=2020,
            l1_weights=w_ru,
            gva_rp=payload["gva_rp"],
            x_to_rc=payload["x_to_rc"],
            omega_reg=payload["omega_reg"],
            pre_weighting=False,
        ).columns
    ) == [2020]


def test_population_and_pr_gdpcap_edges() -> None:
    zero_population = pd.Series([0.0, 0.0], index=pd.Index(["FR", "US"], name="region"))
    zero_out = eg_pop.compute_eg_pop(
        population=zero_population,
        year=2030,
    )
    assert zero_out[2030].tolist() == [0.0, 0.0]
    renamed_out = eg_pop.compute_eg_pop(
        population=pd.Series([1.0, 3.0], index=pd.Index(["FR", "US"], name="r_f")),
        year=2030,
        region_label="region",
    )
    assert renamed_out.index.name == "region"

    assert pr_gdpcap._map_region_code("FR", {"US": "NAM"}) == "FR"


def test_pr_gdpcap_and_pr_hr_additional_paths(allocation_dummy_repo) -> None:
    pop_iso, gdp_iso, iso_to_mrio = _pop_gdp_iso()
    ungrouped = pr_gdpcap._compute_pre_aggregated_share(
        pop_iso=pop_iso,
        gdp_iso=gdp_iso,
        iso_to_mrio=iso_to_mrio,
        source_key="oecd_v2025",
        group_version=None,
    )
    assert ungrouped.index.tolist() == ["FR", "US"]

    allocation_dummy_repo.write_group_map(
        source="oecd_v2025",
        kind="reg",
        group_version="demo_collapse",
        mapping={"FR": "EU", "US": "EU"},
    )
    collapsed = pr_gdpcap._compute_pre_aggregated_share(
        pop_iso=pop_iso,
        gdp_iso=gdp_iso,
        iso_to_mrio=iso_to_mrio,
        source_key="oecd_v2025",
        group_version="demo_collapse",
    )
    assert collapsed.index.tolist() == ["EU"]

    population_by_year, lcia_reg_by_year, rps_df, impact_parent_map, available_years = (
        _pr_hr_inputs()
    )
    two_impact_df = pd.DataFrame(
        [[6.0, 12.0], [3.0, 9.0]],
        index=pd.Index(["climate_child", "water_child"], name="impact"),
        columns=pd.Index(["FR", "US"], name="r_f"),
    )
    two_parent_map = pd.Series(
        ["climate_parent", "climate_parent"],
        index=pd.Index(["climate_child", "water_child"], name="impact"),
    )
    two_rps = pd.DataFrame(
        {"responsibility_period_years": [1, 1]},
        index=pd.Index(["climate_child", "water_child"], name="impact"),
    )
    combined_parent = pr_hr._collect_parent_cumulative_per_cap(
        impact_year=2020,
        impact_df=two_impact_df,
        population_by_year=population_by_year,
        lcia_reg_by_year={2020: two_impact_df},
        rps=two_rps,
        impact_parent_map=two_parent_map,
        available_years=[2020],
    )
    assert combined_parent["climate_parent"].tolist() == pytest.approx([3.0, 3.5])

    impact_df = lcia_reg_by_year[2020]
    assert (
        pr_hr._collect_parent_cumulative_per_cap_incremental(
            impact_year=2020,
            impact_df=impact_df.loc[["climate_child"]],
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps=pd.DataFrame(
                {"responsibility_period_years": [1]},
                index=pd.Index(["other_impact"], name="impact"),
            ),
            impact_parent_map=impact_parent_map,
            available_years=available_years,
            previous_parent_cum={},
        )
        == {}
    )
    assert (
        pr_hr._collect_parent_cumulative_per_cap_incremental(
            impact_year=2020,
            impact_df=impact_df,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps=pd.DataFrame(
                {"responsibility_period_years": [pd.NA]},
                index=pd.Index(["climate_child"], name="impact"),
            ),
            impact_parent_map=impact_parent_map,
            available_years=available_years,
            previous_parent_cum={},
        )
        == {}
    )
    assert (
        pr_hr._collect_parent_cumulative_per_cap_incremental(
            impact_year=2020,
            impact_df=impact_df,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps=rps_df.set_index("impact"),
            impact_parent_map=pd.Series([None], index=pd.Index(["climate_child"], name="impact")),
            available_years=available_years,
            previous_parent_cum={},
        )
        == {}
    )

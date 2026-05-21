import pandas as pd
import pytest
from pathlib import Path
from types import SimpleNamespace

from pyaesa.asocc.orchestration.projection.payload import basis as basis_mod
from pyaesa.asocc.orchestration.projection.payload import builders_x_to_rc as x_to_rc_mod
from pyaesa.asocc.orchestration.projection.payload import cache as cache_mod
from pyaesa.asocc.orchestration.projection.payload import common as common_mod
from pyaesa.asocc.orchestration.projection.payload import payload_builders_l2 as l2_mod
from pyaesa.asocc.orchestration.yearly.shared.year_inputs import (
    _MrioPayload,
    build_l2_compute_inputs,
)


def _projection_state(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        notices_emitted=set(),
        projection_history_cache={},
        projection_regression_basis_cache={},
        projection_payload_cache={},
        regression_fit_cache={},
        regression_stats_rows=[],
        regression_fit_inputs_rows=[],
        regression_uncertainty_rows=[],
        mrio_units={},
        mrio_default_monetary_unit="USD_2021",
        runtime_output_source="oecd_v2025",
        runtime_proj_base=tmp_path,
    )


def _payload_for_scale(scale: float) -> _MrioPayload:
    regions = ["FR", "US"]
    sectors = ["D", "X"]
    rp_sp = pd.MultiIndex.from_product([regions, sectors], names=["r_p", "s_p"])
    rf_sp = pd.MultiIndex.from_product([regions, sectors], names=["r_f", "s_p"])
    fd_rp_sp_rf = pd.DataFrame(
        [
            [1.0 * scale, 2.0 * scale],
            [2.0 * scale, 1.0 * scale],
            [3.0 * scale, 4.0 * scale],
            [4.0 * scale, 3.0 * scale],
        ],
        index=rp_sp,
        columns=pd.Index(regions, name="r_f"),
    )
    x_to_rc = pd.DataFrame(
        [
            [3.0 * scale, 1.0 * scale],
            [1.0 * scale, 3.0 * scale],
            [4.0 * scale, 2.0 * scale],
            [2.0 * scale, 4.0 * scale],
        ],
        index=rp_sp,
        columns=pd.Index(regions, name="r_c"),
    )
    enacting_metric_l1 = {
        "fd_rf": pd.Series(
            [10.0 * scale, 20.0 * scale],
            index=pd.Index(regions, name="r_f"),
        ),
        "gva_rp": pd.Series(
            [30.0 * scale, 40.0 * scale],
            index=pd.Index(regions, name="r_p"),
        ),
    }
    enacting_metric_l2 = {
        "fd_rp_sp_rf": fd_rp_sp_rf,
        "fd_rp_sp": fd_rp_sp_rf.sum(axis=1),
        "fd_rf_sp": pd.Series(
            [4.0 * scale, 6.0 * scale, 8.0 * scale, 12.0 * scale],
            index=rf_sp,
        ),
        "gva_rp_sp": pd.Series(
            [9.0 * scale, 21.0 * scale, 16.0 * scale, 24.0 * scale],
            index=rp_sp,
        ),
    }
    utility = {"x_to_rc": x_to_rc}
    return _MrioPayload(
        enacting_metric_l1=enacting_metric_l1,
        enacting_metric_l2=enacting_metric_l2,
        utility=utility,
        l2_inputs=build_l2_compute_inputs(
            enacting_metric_l1=enacting_metric_l1,
            enacting_metric_l2=enacting_metric_l2,
            utility=utility,
        ),
    )


def _basis() -> basis_mod.RegressionBasis:
    years = [2018, 2019, 2020, 2021]
    payload_by_year = {year: _payload_for_scale(1.0 + float(year - 2018)) for year in years}
    return basis_mod.RegressionBasis(
        gdp_by_year={
            year: pd.Series(
                [100.0 + float(year - 2018), 200.0 + float(year - 2018)],
                index=pd.Index(["FR", "US"], name="r_p"),
            )
            for year in years
        },
        payload_by_year=payload_by_year,
        base_payload=payload_by_year[2021],
    )


def _context(
    tmp_path: Path,
    *,
    fu_code: str,
    selected_l2_one_step: list[str] | None = None,
    combined: list[tuple[str, str]] | None = None,
    projection_context: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        source="oecd_v2025",
        fu_code=fu_code,
        proj_base=tmp_path,
        output_source="oecd_v2025",
        group_version=None,
        group_version_reg=None,
        group_reg=False,
        aggreg_indices=False,
        l1_reg_aggreg="pre",
        filters={},
        selected_l2_one_step=[] if selected_l2_one_step is None else selected_l2_one_step,
        combined=[] if combined is None else combined,
        projection_context=projection_context,
    )


def test_projection_basis_cover_shape_validation_and_alignment() -> None:
    series = pd.Series([1, 2], index=pd.Index(["A", "B"], name="s_p"))
    frame = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=pd.Index([0, 1], name="year"))

    assert list(basis_mod.require_series(series, label="series").index) == ["A", "B"]
    frame_numeric = basis_mod.require_frame(frame, label="frame")
    assert frame_numeric.index.equals(frame.index)
    assert frame_numeric.columns.equals(frame.columns)
    assert frame_numeric.to_numpy().tolist() == [[1.0, 3.0], [2.0, 4.0]]

    multi_index = pd.MultiIndex.from_tuples(
        [("FR", "A"), ("US", "B")],
        names=["r_f", "s_p"],
    )
    projected = basis_mod.coerce_index_like(
        pd.Series([1.0, 2.0], index=multi_index),
        template=multi_index,
    )
    assert projected.index.equals(multi_index)
    assert basis_mod.coerce_index_like(
        pd.Series([1.0, 2.0], index=pd.Index(["FR", "US"], name="r_f")),
        template=pd.Index(["FR", "US"], name="r_f"),
    ).index.equals(pd.Index(["FR", "US"], name="r_f"))

    with pytest.raises(ValueError):
        basis_mod.coerce_index_like(  # noqa: SLF001
            pd.Series([1.0], index=pd.Index(["FR"], name="r_f")),
            template=multi_index,
        )
    with pytest.raises(ValueError):
        basis_mod.coerce_index_like(  # noqa: SLF001
            pd.Series(
                [1.0, 2.0],
                index=pd.MultiIndex.from_tuples([("FR",), ("US",)], names=["r_f"]),
            ),
            template=multi_index,
        )
    with pytest.raises(ValueError):
        basis_mod.coerce_index_like(  # noqa: SLF001
            pd.Series(
                [1.0, 2.0],
                index=pd.MultiIndex.from_tuples(
                    [("FR", "A"), ("US", "B")],
                    names=["r_p", "s_p"],
                ),
            ),
            template=multi_index,
        )
    with pytest.raises(ValueError):
        basis_mod.coerce_index_like(  # noqa: SLF001
            pd.Series([1.0, 2.0], index=multi_index),
            template=pd.Index(["FR", "US"], name="r_f"),
        )
    with pytest.raises(ValueError):
        basis_mod.coerce_index_like(  # noqa: SLF001
            pd.Series([1.0, 2.0], index=pd.Index(["FR", "US"], name="r_p")),
            template=pd.Index(["FR", "US"], name="r_f"),
        )


def test_projection_payload_cover_share_alignment_and_reordering() -> None:
    assert basis_mod.safe_share(
        numer=pd.Series([1.0, 2.0], index=pd.Index(["FR", "US"], name="r_f")),
        denom=pd.Series([2.0, 4.0], index=pd.Index(["FR", "US"], name="r_f")),
        level="r_f",
    ).tolist() == [0.5, 0.5]

    multi_numer = pd.Series(
        [1.0, 2.0],
        index=pd.MultiIndex.from_tuples(
            [("FR", "A"), ("US", "B")],
            names=["r_f", "s_p"],
        ),
    )
    multi_denom = pd.Series(
        [2.0, 4.0],
        index=pd.MultiIndex.from_tuples(
            [("FR", "A"), ("US", "B")],
            names=["r_f", "s_p"],
        ),
    )
    assert basis_mod.safe_share(
        numer=multi_numer,
        denom=multi_denom,
        level=["r_f", "s_p"],
    ).tolist() == [0.5, 0.5]

    with pytest.raises(ValueError):
        basis_mod.safe_share(  # noqa: SLF001
            numer=pd.Series([1.0], index=pd.Index(["FR"], name="r_f")),
            denom=pd.Series(
                [2.0],
                index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_f", "s_p"]),
            ),
            level=["r_f", "s_p"],
        )
    with pytest.raises(ValueError):
        basis_mod.safe_share(  # noqa: SLF001
            numer=pd.Series([1.0], index=pd.Index(["FR"], name="r_p")),
            denom=pd.Series([2.0], index=pd.Index(["FR"], name="r_f")),
            level="r_f",
        )
    with pytest.raises(ValueError):
        basis_mod.safe_share(  # noqa: SLF001
            numer=pd.Series(
                [1.0, 2.0],
                index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "B")], names=["r_p", "s_p"]),
            ),
            denom=multi_denom,
            level=["r_f", "s_p"],
        )
    with pytest.raises(ValueError):
        basis_mod.safe_share(  # noqa: SLF001
            numer=multi_numer,
            denom=pd.Series(
                [2.0, 4.0],
                index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "B")], names=["r_p", "s_p"]),
            ),
            level=["r_f", "s_p"],
        )
    with pytest.raises(ValueError):
        basis_mod.safe_share(  # noqa: SLF001
            numer=multi_numer,
            denom=pd.Series([2.0, 4.0], index=pd.Index(["FR", "US"], name="r_f")),
            level=["r_f", "s_p"],
        )
    with pytest.raises(ValueError):
        basis_mod.safe_share(  # noqa: SLF001
            numer=pd.Series([1.0], index=pd.Index(["FR"], name="r_f")),
            denom=pd.Series([2.0], index=pd.Index(["FR"], name="r_p")),
            level="r_f",
        )

    stacked = common_mod.stack_series_payload(
        pd.DataFrame([[1.0, 2.0]], index=pd.Index(["FR"], name="r_f"), columns=["A", "B"]),
        label="stacked",
        names=["r_f", "s_p"],
    )
    assert stacked.index.names == ["r_f", "s_p"]

    with pytest.raises(ValueError):
        common_mod.stack_series_payload(  # noqa: SLF001
            pd.DataFrame([[1.0, 2.0]], index=pd.Index(["FR"], name="r_f"), columns=["A", "B"]),
            label="stacked",
            names=["r_f"],
        )
    assert common_mod.reorder_series_levels_payload(
        pd.Series(
            [1.0, 2.0],
            index=pd.MultiIndex.from_tuples(
                [("FR", "A"), ("US", "B")],
                names=["r_f", "s_p"],
            ),
        ),
        order=["s_p", "r_f"],
        label="payload",
    ).index.names == ["s_p", "r_f"]

    with pytest.raises(ValueError):
        common_mod.reorder_series_levels_payload(  # noqa: SLF001
            pd.Series([1.0], index=pd.Index(["FR"], name="r_f")),
            order=["r_f"],
            label="payload",
        )
    with pytest.raises(ValueError):
        common_mod.reorder_series_levels_payload(  # noqa: SLF001
            pd.Series(
                [1.0, 2.0],
                index=pd.MultiIndex.from_tuples(
                    [("FR", "A"), ("US", "B")],
                    names=["r_f", "s_p"],
                ),
            ),
            order=["r_c"],
            label="payload",
        )


def test_regression_basis_loads_real_dummy_history_and_reuses_cache(
    allocation_dummy_repo,
) -> None:
    context = _context(
        allocation_dummy_repo.repo_root / "projection_basis",
        fu_code="L2.a.a",
        selected_l2_one_step=["UT(FD)"],
    )
    context.wb_df = pd.read_csv(
        allocation_dummy_repo.repo_root / "data_processed" / "pop_gdp" / "wb_processed.csv"
    )
    state = _projection_state(allocation_dummy_repo.repo_root / "projection_basis")

    basis = basis_mod.regression_basis(
        context=context,
        state=state,
        historical_years=[2005, 2006],
        fit_end=2006,
        needs_fd_total=True,
        needs_fd_detail=True,
        needs_gva=False,
        needs_x_to_rc=False,
    )
    assert list(basis.gdp_by_year[2005].index) == ["FR", "US"]
    assert not basis.base_payload.enacting_metric_l1["fd_rf"].empty
    assert "x_to_rc" not in basis.base_payload.utility
    compact_basis = basis_mod.regression_basis(
        context=context,
        state=state,
        historical_years=[2005],
        fit_end=2005,
        needs_fd_total=False,
        needs_fd_detail=False,
        needs_gva=True,
        needs_x_to_rc=True,
    )
    assert "fd_rf" not in compact_basis.base_payload.enacting_metric_l1
    assert "gva_rp" in compact_basis.base_payload.enacting_metric_l1
    assert "x_to_rc" in compact_basis.base_payload.utility

    reused = basis_mod.regression_basis(
        context=context,
        state=state,
        historical_years=[2005, 2006],
        fit_end=2006,
        needs_fd_total=True,
        needs_fd_detail=True,
        needs_gva=False,
        needs_x_to_rc=False,
    )
    assert reused is basis


def test_project_l2_payload_builders_cover_fu_routes(tmp_path: Path) -> None:
    basis = _basis()
    years = [2018, 2019, 2020, 2021]
    gdp_target = pd.Series([110.0, 210.0], index=pd.Index(["FR", "US"], name="r_p"))

    fd_context = _context(tmp_path / "fd", fu_code="L2.a.a")
    fd_state = _projection_state(tmp_path / "fd")
    fd_rf, fd_rp_sp_rf, fd_rp_sp, fd_rf_sp = l2_mod.project_fd_payload(
        context=fd_context,
        state=fd_state,
        basis=basis,
        historical_years=years,
        target_year=2030,
        future_years=[2030],
        gdp_target=gdp_target,
        needs_ut_fd=True,
        needs_global_fd_total=True,
    )
    assert not fd_rf.empty
    assert fd_rp_sp_rf.shape == basis.base_payload.enacting_metric_l2["fd_rp_sp_rf"].shape
    assert fd_rp_sp.index.equals(basis.base_payload.enacting_metric_l2["fd_rp_sp"].index)
    assert fd_rf_sp.index.equals(basis.base_payload.enacting_metric_l2["fd_rf_sp"].index)

    only_total = l2_mod.project_fd_payload(
        context=_context(tmp_path / "fd_total", fu_code="L2.a.a"),
        state=_projection_state(tmp_path / "fd_total"),
        basis=basis,
        historical_years=years,
        target_year=2030,
        future_years=[2030],
        gdp_target=gdp_target,
        needs_ut_fd=False,
        needs_global_fd_total=False,
    )
    assert only_total[1].empty
    assert only_total[2].empty
    assert only_total[3].empty

    no_producer_share = l2_mod.project_fd_payload(
        context=_context(tmp_path / "fd_c", fu_code="L2.c.a"),
        state=_projection_state(tmp_path / "fd_c"),
        basis=basis,
        historical_years=years,
        target_year=2030,
        future_years=[2030],
        gdp_target=gdp_target,
        needs_ut_fd=True,
        needs_global_fd_total=False,
    )
    assert no_producer_share[1].empty
    assert not no_producer_share[3].empty

    gva_rp, gva_rp_sp = l2_mod.project_gva_payload(
        context=_context(tmp_path / "gva", fu_code="L2.a.c"),
        state=_projection_state(tmp_path / "gva"),
        basis=basis,
        historical_years=years,
        target_year=2030,
        future_years=[2030],
        gdp_target=gdp_target,
        needs_global_gva_total=True,
    )
    assert not gva_rp.empty
    assert gva_rp_sp.index.equals(basis.base_payload.enacting_metric_l2["gva_rp_sp"].index)


def test_project_x_to_rc_payload_covers_fu_routes(tmp_path: Path) -> None:
    basis = _basis()
    years = [2018, 2019, 2020, 2021]
    gdp_target = pd.Series([110.0, 210.0], index=pd.Index(["FR", "US"], name="r_p"))

    for fu_code in ["L2.a.b", "L2.b.b", "L2.c.b"]:
        projected = x_to_rc_mod.project_x_to_rc_payload(
            context=_context(tmp_path / fu_code.replace(".", "_"), fu_code=fu_code),
            state=_projection_state(tmp_path / fu_code.replace(".", "_")),
            basis=basis,
            historical_years=years,
            target_year=2030,
            future_years=[2030],
            gdp_target=gdp_target,
        )
        assert not projected.empty

    assert x_to_rc_mod._to_frame_label(["not", "hashable"]) == "['not', 'hashable']"


def test_get_projected_payload_cache_and_validation_paths(tmp_path: Path) -> None:
    payload = _payload_for_scale(1.0)
    state = _projection_state(tmp_path)
    state.projection_payload_cache[(2030, "SSP2")] = payload
    assert (
        cache_mod.get_projected_payload(
            context=_context(tmp_path, fu_code="L2.a.a"),
            state=state,
            year=2030,
            ssp_scenario="SSP2",
            gdp_series=pd.Series(dtype=float),
        )
        is payload
    )

    empty_state = _projection_state(tmp_path / "empty")
    projection_context = SimpleNamespace(
        enabled=True,
        is_future_year=lambda year: True,
        mode="regression",
        reg_window=None,
        max_historical_year=2021,
        route_for_l2_method=lambda name: "regression",
        future_years=[2030],
    )
    context = _context(
        tmp_path / "empty",
        fu_code="L2.a.a",
        projection_context=projection_context,
    )
    empty_state.projection_regression_basis_cache[
        ("oecd_v2025", "None", "None", (2021,), 2021, "L2.a.a", False, False, False, False)
    ] = basis_mod.RegressionBasis(
        gdp_by_year={2021: pd.Series([1.0], index=pd.Index(["FR"], name="r_p"))},
        payload_by_year={2021: payload},
        base_payload=payload,
    )
    built = cache_mod.get_projected_payload(
        context=context,
        state=empty_state,
        year=2030,
        ssp_scenario=None,
        gdp_series=pd.Series([2.0], index=pd.Index(["FR"], name="r_p")),
    )
    assert built.enacting_metric_l1["fd_rf"].empty
    assert empty_state.projection_payload_cache[(2030, None)] is built


def test_get_projected_payload_projects_routed_ut_families(tmp_path: Path) -> None:
    basis = _basis()
    gdp_target = pd.Series([110.0, 210.0], index=pd.Index(["FR", "US"], name="r_p"))

    cases = [
        ("L2.a.a", "UT(FD)", "fd_rp_sp_rf"),
        ("L2.a.b", "UT(TD)", "x_to_rc"),
        ("L2.a.c", "UT(GVA)", "gva_rp_sp"),
    ]
    for fu_code, l2_method, projected_key in cases:
        state = _projection_state(tmp_path / projected_key)
        projection_context = SimpleNamespace(
            enabled=True,
            is_future_year=lambda year: True,
            mode="regression",
            reg_window=(2018, 2021),
            max_historical_year=2021,
            route_for_l2_method=lambda name: "regression",
            future_years=[2030],
        )
        context = _context(
            tmp_path / projected_key,
            fu_code=fu_code,
            selected_l2_one_step=[l2_method],
            projection_context=projection_context,
        )
        state.projection_regression_basis_cache[
            (
                "oecd_v2025",
                "None",
                "None",
                (2018, 2019, 2020, 2021),
                2021,
                fu_code,
                l2_method in {"UT(FD)", "UT(TD)"},
                l2_method == "UT(FD)",
                l2_method == "UT(GVA)",
                l2_method == "UT(TD)",
            )
        ] = basis

        built = cache_mod.get_projected_payload(
            context=context,
            state=state,
            year=2030,
            ssp_scenario="SSP2",
            gdp_series=gdp_target,
        )
        if projected_key == "x_to_rc":
            assert not built.utility[projected_key].empty
        else:
            assert not built.enacting_metric_l2[projected_key].empty

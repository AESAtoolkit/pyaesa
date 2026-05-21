from typing import Any, cast

import pytest

from pyaesa.asocc.orchestration.projection.config import config
from pyaesa.asocc.orchestration.projection.config import types as types_mod


def test_normalize_projection_mode() -> None:
    assert config._normalize_projection_mode(None) is None
    assert config._normalize_projection_mode(" Regression ") == "regression"
    with pytest.raises(ValueError):
        config._normalize_projection_mode("bad")


def test_normalize_year_selector_and_require_available() -> None:
    assert config._normalize_year_selector(value=None, name="x") == []
    assert config._normalize_year_selector(value=2020, name="x") == [2020]
    assert config._normalize_year_selector(value=range(2020, 2022), name="x") == [2020, 2021]
    assert config._normalize_year_selector(value=[2021, 2020, 2020], name="x") == [2020, 2021]

    with pytest.raises(ValueError):
        config._normalize_year_selector(value=cast(Any, (2020, 2022)), name="x")
    with pytest.raises(ValueError):
        config._normalize_year_selector(value=cast(Any, (2020, 2021, 2022)), name="x")
    with pytest.raises(ValueError):
        config._normalize_year_selector(value=cast(Any, (2022, 2020)), name="x")
    with pytest.raises(ValueError):
        config._normalize_year_selector(value=cast(Any, {"y": 2020}), name="x")

    config._require_years_available(years=[], historical_years=[2020], label="x")
    config._require_years_available(years=[2020], historical_years=[2020, 2021], label="x")
    with pytest.raises(ValueError):
        config._require_years_available(years=[2030], historical_years=[2020, 2021], label="x")
    with pytest.raises(ValueError):
        config._require_years_available(years=[2030], historical_years=[], label="x")


def test_scope() -> None:
    methods = config.list_ut_l2_methods_in_scope(
        fu_code="L2.a.a",
        selected_l2_one_step=["UT(FD)", "UT(FD)"],
        combined=[("UT(FD)", "AR(E^{CBA_FD})"), ("AR(E^{CBA_FD})", "AR(E^{CBA_FD})")],
    )
    assert methods == ["UT(FD)"]

    route = config.build_l2_method_route_by_name(
        ut_methods=["UT(FD)", "UT(FDa)"], mode="regression"
    )
    assert route["UT(FD)"] == "regression"
    assert route["UT(FDa)"] == "historical_reuse"

    hist, fut = config.split_historical_future_years(
        years=[2019, 2020, 2021], max_historical_year=2020
    )
    assert hist == [2019, 2020]
    assert fut == [2021]


def test_resolve_projection_context_paths() -> None:
    context = config.resolve_projection_context(
        source="oecd_v2025",
        fu_code="L2.a.a",
        resolved_years=[2020, 2030],
        historical_years=[2020],
        selected_l2_one_step=[],
        combined=[],
        projection_mode=None,
        reg_window=[2020],
        l2_reuse_years=None,
    )
    assert context.enabled is False
    assert context.future_years == (2030,)

    ctx_reg = config.resolve_projection_context(
        source="oecd_v2025",
        fu_code="L2.a.a",
        resolved_years=[2020, 2030],
        historical_years=[2019, 2020],
        selected_l2_one_step=["UT(FD)"],
        combined=[],
        projection_mode=None,
        reg_window=[2019, 2020],
        l2_reuse_years=None,
    )
    assert ctx_reg.enabled is True
    assert ctx_reg.mode == "regression"
    assert ctx_reg.l2_reuse_years == ()
    assert ctx_reg.l2_method_route_by_name == {"UT(FD)": "regression"}

    with pytest.raises(ValueError):
        config.resolve_projection_context(
            source="oecd_v2025",
            fu_code="L2.a.a",
            resolved_years=[2020, 2030],
            historical_years=[2020],
            selected_l2_one_step=["UT(FD)"],
            combined=[],
            projection_mode="regression",
            reg_window=[2021, 2020],
            l2_reuse_years=None,
        )

    ctx_reuse = config.resolve_projection_context(
        source="oecd_v2025",
        fu_code="L2.a.b",
        resolved_years=[2020, 2030],
        historical_years=[2018, 2019, 2020],
        selected_l2_one_step=["UT(FDa)"],
        combined=[],
        projection_mode="historical_reuse",
        reg_window=[2019, 2020],
        l2_reuse_years=[2018, 2019],
    )
    assert ctx_reuse.mode == "historical_reuse"
    assert ctx_reuse.l2_reuse_years == (2018, 2019)

    with pytest.raises(ValueError):
        config.resolve_projection_context(
            source="oecd_v2025",
            fu_code="L2.a.a",
            resolved_years=[2020, 2030],
            historical_years=[2019, 2020],
            selected_l2_one_step=["UT(FD)"],
            combined=[],
            projection_mode="regression",
            reg_window=[2019, 2020],
            l2_reuse_years=[2019],
        )

    ctx_default_reuse = config.resolve_projection_context(
        source="oecd_v2025",
        fu_code="L2.a.b",
        resolved_years=[2020, 2030],
        historical_years=[2018, 2019, 2020],
        selected_l2_one_step=["UT(FDa)"],
        combined=[],
        projection_mode="historical_reuse",
        reg_window=[2019, 2020],
        l2_reuse_years=None,
    )
    assert ctx_default_reuse.l2_reuse_years == (2019, 2020)

    ctx_iso3 = config.resolve_projection_context(
        source="iso3",
        fu_code="L2.a.a",
        resolved_years=[2020, 2030],
        historical_years=[2018, 2020],
        selected_l2_one_step=["UT(FD)"],
        combined=[],
        projection_mode=None,
        reg_window=None,
        l2_reuse_years=None,
    )
    assert ctx_iso3.reg_window == (2020, 2020)

    with pytest.raises(ValueError):
        config.resolve_projection_context(
            source="oecd_v2025",
            fu_code="L2.a.a",
            resolved_years=[2020, 2030],
            historical_years=[2019, 2020],
            selected_l2_one_step=["UT(FD)"],
            combined=[],
            projection_mode="regression",
            reg_window=[],
            l2_reuse_years=None,
        )


def test_required_projection_years() -> None:
    disabled = types_mod.ProjectionContext(
        enabled=False,
        mode=None,
        max_historical_year=2020,
        future_years=(),
        reg_window=None,
        l2_reuse_years=(),
        ut_methods_in_scope=(),
        l2_method_route_by_name={},
    )

    regression = types_mod.ProjectionContext(
        enabled=True,
        mode="regression",
        max_historical_year=2020,
        future_years=(2030,),
        reg_window=(2019, 2020),
        l2_reuse_years=(),
        ut_methods_in_scope=("UT(FD)",),
        l2_method_route_by_name={"UT(FD)": "regression"},
    )

    reuse_with_adjusted = types_mod.ProjectionContext(
        enabled=True,
        mode="historical_reuse",
        max_historical_year=2020,
        future_years=(2030,),
        reg_window=(2019, 2020),
        l2_reuse_years=(2018, 2019),
        ut_methods_in_scope=("UT(FDa)",),
        l2_method_route_by_name={"UT(FDa)": "historical_reuse"},
    )

    adjusted_in_regression_mode = types_mod.ProjectionContext(
        enabled=True,
        mode="regression",
        max_historical_year=2020,
        future_years=(2030,),
        reg_window=(2019, 2020),
        l2_reuse_years=(2018, 2019),
        ut_methods_in_scope=("UT(FDa)",),
        l2_method_route_by_name={"UT(FDa)": "historical_reuse"},
    )

    assert config.required_projection_years(projection_context=disabled) == []
    assert config.required_projection_years(projection_context=regression) == [2019, 2020]
    assert config.required_projection_years(projection_context=reuse_with_adjusted) == [2018, 2019]
    assert config.required_projection_years(projection_context=adjusted_in_regression_mode) == [
        2018,
        2019,
        2020,
    ]


def test_projection_types() -> None:
    context = types_mod.ProjectionContext(
        enabled=True,
        mode="regression",
        max_historical_year=2020,
        future_years=(2030,),
        reg_window=(2019, 2020),
        l2_reuse_years=(2019,),
        ut_methods_in_scope=("UT(FD)",),
        l2_method_route_by_name={"UT(FD)": "regression"},
    )
    assert context.is_future_year(2030) is True
    assert context.is_future_year(2020) is False
    assert context.route_for_l2_method("UT(FD)") == "regression"
    assert context.route_for_l2_method("unknown") is None
    assert context.l2_reuse_years_for() == (2019,)

    stats = types_mod.RegressionStatsRow(
        projection_branch="b",
        source="oecd",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        model_type="ols",
        target_object="y",
        domain_key="k",
        fit_start_year=2019,
        fit_end_year=2020,
        n_obs=2,
        intercept=1.0,
        slope=2.0,
        r_squared=0.9,
        p_value_slope=0.01,
    )
    assert stats.as_dict()["slope"] == 2.0

    fit = types_mod.RegressionFitInputRow(
        projection_branch="b",
        source="oecd",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        model_type="ols",
        target_object="y",
        domain_key="k",
        fit_start_year=2019,
        fit_end_year=2020,
        fit_year=2019,
        x_value=1.0,
        y_value=2.0,
        y_kind="value",
        ratio_value=2.0,
        numerator_value=4.0,
        denominator_value=2.0,
    )
    assert fit.as_dict()["denominator_value"] == 2.0

    uncertainty = types_mod.RegressionUncertaintyRow(
        projection_branch="b",
        source="oecd",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        model_type="ols",
        target_object="y",
        domain_key="k",
        fit_start_year=2019,
        fit_end_year=2020,
        n_obs=3,
        sigma2_hat=0.5,
        df_resid=1,
        x_mean=0.0,
        ssx=2.0,
        x_min=-1.0,
        x_max=1.0,
        years_used="2019-2021",
        notes="ols_mean_var_simple",
    )
    assert uncertainty.as_dict()["years_used"] == "2019-2021"

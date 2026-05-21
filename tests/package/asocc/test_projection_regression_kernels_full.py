from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from pyaesa.asocc.orchestration.projection.regression import (
    level_ols_gdp_scaled_projection as level_mod,
)
from pyaesa.asocc.orchestration.projection.regression import (
    regression_core_utils as core_mod,
)
from pyaesa.asocc.orchestration.projection.regression import (
    share_fit_containers as share_fit_containers_mod,
)
from pyaesa.asocc.orchestration.projection.regression import (
    share_logit_time_projection as share_mod,
)
from pyaesa.asocc.runtime.paths.deterministic import (
    projection_clipping_log_path,
    share_fit_window_log_path,
)


def _state(*, runtime_proj_base: Path | None = None) -> SimpleNamespace:
    state = SimpleNamespace(
        notices_emitted=set(),
        regression_fit_cache={},
        regression_stats_rows=[],
        regression_fit_inputs_rows=[],
        regression_uncertainty_rows=[],
        mrio_units={},
        mrio_default_monetary_unit="USD_2021",
        runtime_output_source="oecd_v2025",
    )
    if runtime_proj_base is not None:
        state.runtime_proj_base = runtime_proj_base
    return state


def test_regression_core_cover_strict_and_upsert_paths() -> None:
    with pytest.raises(ValueError):
        core_mod.fit_simple_ols(x=np.array([1.0]), y=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        core_mod.fit_simple_ols(x=np.array([1.0]), y=np.array([2.0]))

    with pytest.raises(ValueError):
        core_mod.compute_ols_uncertainty_scalars(
            x=np.array([1.0, 2.0]),
            y=np.array([1.0]),
            intercept=0.0,
            slope=1.0,
        )
    with pytest.raises(ValueError):
        core_mod.compute_ols_uncertainty_scalars(
            x=np.array([1.0, 2.0]),
            y=np.array([1.0, 2.0]),
            intercept=0.0,
            slope=1.0,
        )
    with pytest.raises(ValueError):
        core_mod.compute_ols_uncertainty_scalars(
            x=np.array([2.0, 2.0, 2.0]),
            y=np.array([1.0, 2.0, 3.0]),
            intercept=0.0,
            slope=1.0,
        )
    sigma2_hat, x_mean, ssx, df_resid, x_min, x_max = core_mod.compute_ols_uncertainty_scalars(
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([1.0, 3.0, 5.0]),
        intercept=1.0,
        slope=2.0,
    )
    assert sigma2_hat == 0.0
    assert x_mean == 1.0
    assert ssx == 2.0
    assert df_resid == 1
    assert x_min == 0.0
    assert x_max == 2.0

    x, y = core_mod.coerce_numeric_pairs(
        x_values=[1, 2, 3],
        y_values=[2, float("nan"), 6],
    )
    assert x.tolist() == [1.0, 3.0]
    assert y.tolist() == [2.0, 6.0]
    with pytest.raises(ValueError):
        core_mod.coerce_numeric_pairs(
            x_values=[1, "x", 3],
            y_values=[2, 4, 6],
        )
    with pytest.raises(ValueError):
        core_mod.coerce_numeric_scalar("bad")
    assert core_mod.serialize_years(years=[2002, 2001, 2000]) == "2000-2002"
    assert core_mod.serialize_years(years=[2000, 2002, 2003, 2005]) == "2000, 2002-2003, 2005"
    assert core_mod.serialize_years(years=[]) == ""

    state = _state()
    core_mod.emit_regression_start_notice(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        model_type="ols_level",
        target_object="fd_rf",
        state=state,
    )
    core_mod.emit_regression_start_notice(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        model_type="ols_level",
        target_object="fd_rf",
        state=state,
    )
    assert len(state.notices_emitted) == 1

    row_a = {
        "source": "oecd",
        "fu_code": "L2.a.a",
        "l2_method": "UT(FD)",
        "model_type": "m",
        "target_object": "x",
        "domain_key": "FR",
        "fit_start_year": 2000,
        "fit_end_year": 2010,
        "n_obs": 2,
    }
    row_b = dict(row_a)
    row_b["domain_key"] = "DE"
    row_b["n_obs"] = 3
    core_mod.append_regression_row(state=state, row=row_a)
    core_mod.append_regression_row(state=state, row=row_b)
    assert [row["n_obs"] for row in state.regression_stats_rows] == [2, 3]

    fit_a = {
        **row_a,
        "fit_year": 2005,
        "x_value": 1.0,
    }
    fit_b = dict(fit_a)
    fit_b["fit_year"] = 2006
    fit_b["x_value"] = 2.0
    core_mod.append_regression_fit_input_row(state=state, row=fit_a)
    core_mod.append_regression_fit_input_row(state=state, row=fit_b)
    assert [row["x_value"] for row in state.regression_fit_inputs_rows] == [1.0, 2.0]

    unc_a = dict(row_a)
    unc_a["sigma2_hat"] = 1.0
    unc_b = dict(unc_a)
    unc_b["domain_key"] = "DE"
    unc_b["sigma2_hat"] = 2.0
    core_mod.append_regression_uncertainty_row(state=state, row=unc_a)
    core_mod.append_regression_uncertainty_row(state=state, row=unc_b)
    assert [row["sigma2_hat"] for row in state.regression_uncertainty_rows] == [1.0, 2.0]

    built_row = core_mod.build_regression_row(
        source="oecd",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        model_type="ols_level",
        target_object="fd_rf",
        domain_key="FR",
        fit_start_year=2000,
        fit_end_year=2010,
        n_obs=3,
        intercept=1.0,
        slope=2.0,
        r_squared=0.9,
        p_value_slope=0.1,
    )
    assert built_row["domain_key"] == "FR"
    built_fit = core_mod.build_regression_fit_input_row(
        source="oecd",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        model_type="ols_level",
        target_object="fd_rf",
        domain_key="FR",
        fit_start_year=2000,
        fit_end_year=2010,
        fit_year=2008,
        x_value=1.0,
        y_value=2.0,
        y_kind="level",
    )
    assert built_fit["fit_year"] == 2008
    built_unc = core_mod.build_regression_uncertainty_row(
        source="oecd",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        model_type="ols_level",
        target_object="fd_rf",
        domain_key="FR",
        fit_start_year=2000,
        fit_end_year=2010,
        n_obs=3,
        sigma2_hat=0.5,
        df_resid=1,
        x_mean=1.0,
        ssx=2.0,
        x_min=0.0,
        x_max=2.0,
        years_used="[2000,2001,2002]",
        notes="ols_mean_var_simple",
    )
    assert built_unc["notes"] == "ols_mean_var_simple"

    clipped = core_mod.clip_share_values(pd.Series([0.0, 1.0, 0.5]))
    assert float(clipped.iloc[0]) > 0.0
    assert float(clipped.iloc[1]) < 1.0


def test_share_fit_container_functions_cover_validation_paths() -> None:
    assert share_fit_containers_mod.as_level_list("r_f") == ["r_f"]
    assert share_fit_containers_mod.selected_signature(["B", "A"]) == ("A", "B")
    plain = pd.Series([1.0, 2.0], index=pd.Index(["A", "B"], name="s_p"))
    cat_map = share_fit_containers_mod.container_category_map(
        template=plain,
        container_levels=["r_f"],
        category_level="s_p",
    )
    assert cat_map == {tuple(): ["A", "B"]}
    assert share_fit_containers_mod.container_label(tuple()) == "global"
    assert share_fit_containers_mod.as_selected_set(cast(list[str], ["FR", 1])) == {"FR", "1"}

    valid_map = {
        ("FR",): {
            "emit": ["A"],
            "baseline": "A",
            "coefs": {},
            "structural_zero_categories": [],
            "last_vector": pd.Series({"A": 1.0}),
            "all_fitted": True,
        }
    }
    assert share_fit_containers_mod.share_fit_map_or_none(valid_map) is not None
    assert share_fit_containers_mod.share_fit_map_or_none("bad") is None
    invalid = {
        ("FR",): {
            "emit": ["A"],
            "baseline": "A",
            "coefs": {},
            "last_vector": pd.Series({"A": 1.0}),
            "all_fitted": True,
        }
    }
    assert share_fit_containers_mod.share_fit_map_or_none(invalid) is None


def test_share_projection_uses_cached_fit_map_and_skip_paths(tmp_path: Path) -> None:
    state = _state(runtime_proj_base=tmp_path)
    cache_key = core_mod.fit_cache_key(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        model_type="log_ratio_time",
        target_object="fd_share_sp|('r_f',)|s_p|('__ALL__',)|()",
        historical_years=[2018, 2019, 2020],
    )
    state.regression_fit_cache = {
        cache_key: {
            ("FR",): {
                "emit": [],
                "baseline": "A",
                "coefs": {},
                "structural_zero_categories": [],
                "last_vector": pd.Series({"A": 1.0}),
                "all_fitted": True,
            },
            ("US",): {
                "emit": ["B"],
                "baseline": None,
                "coefs": {"B": (0.0, 1.0, 0.9, 0.1, 3, 2019.0)},
                "structural_zero_categories": [],
                "last_vector": pd.Series({"B": 1.0}),
                "all_fitted": True,
            },
            ("DE",): {
                "emit": ["A", "B"],
                "baseline": "A",
                "coefs": {},
                "structural_zero_categories": ["B"],
                "last_vector": pd.Series({"A": 1.0, "B": 0.0}),
                "all_fitted": True,
            },
        }
    }
    projected = share_mod.project_share_from_time_logit(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_share_sp",
        historical_years=[2018, 2019, 2020],
        share_by_year={
            2018: pd.Series([0.3, 0.7], index=pd.Index(["A", "B"], name="s_p")),
            2019: pd.Series([0.4, 0.6], index=pd.Index(["A", "B"], name="s_p")),
            2020: pd.Series([0.5, 0.5], index=pd.Index(["A", "B"], name="s_p")),
        },
        target_year=2030,
        future_years=[2030],
        container_levels=["r_f"],
        category_level="s_p",
        selected_categories=None,
        selected_containers=None,
        state=state,
    )
    assert ("DE", "A") in projected.index
    assert ("DE", "B") in projected.index
    assert float(projected.loc[("DE", "A")]) == 1.0
    assert float(projected.loc[("DE", "B")]) == 0.0


def test_level_regression_requires_three_obs_and_clips_outputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        level_mod._fit_ols_map_for_domains(
            source="oecd_v2025",
            fu_code="L2.a.a",
            l2_method="UT(FD)",
            target_object="fd_rf",
            historical_years=[2019, 2020],
            history_by_year={
                2019: pd.Series([1.0], index=pd.Index(["FR"], name="r_p")),
                2020: pd.Series([2.0], index=pd.Index(["FR"], name="r_p")),
            },
            predictor_by_year={
                2019: pd.Series([10.0], index=pd.Index(["FR"], name="r_p")),
                2020: pd.Series([20.0], index=pd.Index(["FR"], name="r_p")),
            },
            selected_domains=None,
            state=_state(runtime_proj_base=tmp_path),
        )

    state = _state(runtime_proj_base=tmp_path)
    historical_years = [2018, 2019, 2020]
    history = {
        2018: pd.Series([8.0, 5.0], index=pd.Index(["FR", "US"], name="r_p")),
        2019: pd.Series([4.0, 5.0], index=pd.Index(["FR", "US"], name="r_p")),
        2020: pd.Series([0.0, 5.0], index=pd.Index(["FR", "US"], name="r_p")),
    }
    gdp = {
        2018: pd.Series([1.0, 1.0], index=pd.Index(["FR", "US"], name="r_p")),
        2019: pd.Series([2.0, 2.0], index=pd.Index(["FR", "US"], name="r_p")),
        2020: pd.Series([3.0, 3.0], index=pd.Index(["FR", "US"], name="r_p")),
    }
    projected = level_mod.project_series_from_gdp(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_rf",
        target_year=2030,
        historical_years=historical_years,
        history_by_year=history,
        gdp_by_year=gdp,
        gdp_target=pd.Series([4.0, 4.0], index=pd.Index(["FR", "US"], name="r_p")),
        selected_domains=None,
        state=state,
    )
    assert float(projected.loc["FR"]) == 0.0
    assert float(projected.loc["US"]) == 5.0
    assert state.regression_uncertainty_rows

    projected_cached = level_mod.project_series_from_gdp(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_rf",
        target_year=2031,
        historical_years=historical_years,
        history_by_year=history,
        gdp_by_year=gdp,
        gdp_target=pd.Series([4.0, 4.0], index=pd.Index(["FR", "US"], name="r_p")),
        selected_domains=None,
        state=state,
    )
    assert float(projected_cached.loc["US"]) == 5.0

    selected_state = _state(runtime_proj_base=tmp_path)
    selected_years = [2017, 2018, 2019, 2020]
    projected_selected = level_mod.project_series_from_gdp(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_rf",
        target_year=2030,
        historical_years=selected_years,
        history_by_year={
            2017: pd.Series([np.nan, 5.0], index=pd.Index(["FR", "US"], name="r_p")),
            **history,
        },
        gdp_by_year={
            2017: pd.Series([0.5, 0.5], index=pd.Index(["FR", "US"], name="r_p")),
            **gdp,
        },
        gdp_target=pd.Series([np.nan, 4.0], index=pd.Index(["FR", "US"], name="r_p")),
        selected_domains=["FR"],
        state=selected_state,
    )
    assert np.isnan(float(projected_selected.loc["FR"]))

    clip_path = projection_clipping_log_path(
        state=state,
    )
    clipped_rows = pd.read_csv(clip_path)
    assert clipped_rows["domain_key"].tolist() == ["FR", "FR"]


def test_share_projection_strict_zero_policy_and_stability(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    state = _state(runtime_proj_base=tmp_path)
    share_by_year = {
        2018: pd.Series([0.0, 1.0, 0.0], index=pd.Index(["A", "B", "C"], name="s_p")),
        2019: pd.Series([0.2, 0.8, 0.0], index=pd.Index(["A", "B", "C"], name="s_p")),
        2020: pd.Series([0.3, 0.7, 0.0], index=pd.Index(["A", "B", "C"], name="s_p")),
        2021: pd.Series([0.4, 0.6, 0.0], index=pd.Index(["A", "B", "C"], name="s_p")),
    }
    with caplog.at_level("WARNING"):
        projected = share_mod.project_share_from_time_logit(
            source="oecd_v2025",
            fu_code="L2.a.a",
            l2_method="UT(FD)",
            target_object="fd_share_sp",
            historical_years=[2018, 2019, 2020, 2021],
            share_by_year=share_by_year,
            target_year=2100,
            future_years=[2030, 2100],
            container_levels=[],
            category_level="s_p",
            selected_categories=None,
            selected_containers=None,
            state=state,
        )
    assert any(record.levelname == "WARNING" for record in caplog.records)
    assert np.isfinite(projected.to_numpy(dtype=float)).all()
    assert float(projected.loc["C"]) == 0.0
    assert abs(float(projected.sum()) - 1.0) <= 1.0e-12
    assert state.regression_uncertainty_rows

    fit_log = share_fit_window_log_path(
        state=state,
    )
    cases = set(pd.read_csv(fit_log)["case"].tolist())
    assert cases == {"all_zero_category", "subset_fit_window"}

from pathlib import Path
from types import SimpleNamespace

import importlib

import pandas as pd

reg_mod = importlib.import_module("pyaesa.asocc.orchestration.write.regression_stats.write")


def _context(tmp_path: Path) -> SimpleNamespace:
    published_source = "oecd_v2025"
    return SimpleNamespace(
        proj_base=tmp_path / "run_reg_aggreg_pre_L1",
        output_format="csv",
        l1_reg_aggreg="pre",
        source=published_source,
        output_source_label=None,
        output_source=published_source,
        agg_version=None,
        agg_reg=False,
        group_indices=False,
        projection_context=SimpleNamespace(reg_window=(2000, 2005)),
    )


def _stats_row(*, n_obs: int, slope: float) -> dict[str, object]:
    return {
        "projection_branch": "regression",
        "source": "oecd",
        "fu_code": "L2.a.a",
        "l2_method": "UT(FDa)",
        "model_type": "ols_level",
        "target_object": "x",
        "domain_key": "k",
        "fit_start_year": 2000,
        "fit_end_year": 2005,
        "n_obs": n_obs,
        "intercept": 1.0,
        "slope": slope,
        "r_squared": 0.9,
        "p_value_slope": 0.01,
        "x_object": "gdp_by_domain",
        "x_unit": "USD_2021/yr",
        "y_object": "x",
        "y_unit": "USD_2021/yr",
    }


def _unc_row(*, n_obs: int, sigma2: float) -> dict[str, object]:
    return {
        "projection_branch": "regression",
        "source": "oecd",
        "fu_code": "L2.a.a",
        "l2_method": "UT(FDa)",
        "model_type": "ols_level",
        "target_object": "x",
        "domain_key": "k",
        "fit_start_year": 2000,
        "fit_end_year": 2005,
        "n_obs": n_obs,
        "sigma2_hat": sigma2,
        "df_resid": 1,
        "x_mean": 1.0,
        "ssx": 2.0,
        "x_min": 0.0,
        "x_max": 2.0,
        "years_used": "2000-2002",
        "notes": "ols_mean_var_simple",
    }


def _fit_input_row(*, fit_year: int) -> dict[str, object]:
    return {
        "projection_branch": "regression",
        "source": "oecd",
        "fu_code": "L2.a.a",
        "l2_method": "UT(FDa)",
        "model_type": "ols_level",
        "target_object": "x",
        "domain_key": "k",
        "fit_start_year": 2000,
        "fit_end_year": 2005,
        "fit_year": fit_year,
        "x_object": "gdp_by_domain",
        "x_unit": "USD_2021/yr",
        "x_value": 1.0,
        "y_object": "x",
        "y_unit": "USD_2021/yr",
        "y_value": 2.0,
        "y_kind": "observed",
        "ratio_value": pd.NA,
        "numerator_object": pd.NA,
        "numerator_value": pd.NA,
        "denominator_object": pd.NA,
        "denominator_value": pd.NA,
    }


def _state(
    *,
    stats_rows: list[dict[str, object]],
    uncertainty_rows: list[dict[str, object]],
    fit_inputs_rows: list[dict[str, object]],
) -> SimpleNamespace:
    return SimpleNamespace(
        regression_stats_rows=stats_rows,
        regression_uncertainty_rows=uncertainty_rows,
        regression_fit_inputs_rows=fit_inputs_rows,
        write_progress_total=2,
        write_progress_current=0,
        write_progress_last_width=0,
    )


def test_write_regression_stats_writes_current_rows_only(tmp_path: Path) -> None:
    context = _context(tmp_path)
    path = reg_mod.write_regression_stats(
        context=context,
        state=_state(
            stats_rows=[_stats_row(n_obs=3, slope=1.0)],
            uncertainty_rows=[_unc_row(n_obs=3, sigma2=0.5)],
            fit_inputs_rows=[_fit_input_row(fit_year=2001)],
        ),
    )

    assert path is not None and path.exists()
    written = pd.read_csv(path)
    assert written.shape[0] == 1
    assert int(written.loc[0, "n_obs"]) == 3
    assert float(written.loc[0, "slope"]) == 1.0
    assert float(written.loc[0, "sigma2_hat"]) == 0.5
    assert path.with_name("regression_stats_columns_defs.txt").exists()

    fit_inputs_path = reg_mod.fit_inputs_path_for_format(
        proj_base=context.proj_base,
        output_format=context.output_format,
        source=context.source,
        agg_version=context.agg_version,
    )
    fit_inputs = pd.read_csv(fit_inputs_path)
    assert fit_inputs["fit_year"].tolist() == [2001]


def test_write_regression_stats_returns_none_without_current_rows(tmp_path: Path) -> None:
    assert (
        reg_mod.write_regression_stats(
            context=_context(tmp_path),
            state=_state(stats_rows=[], uncertainty_rows=[], fit_inputs_rows=[]),
        )
        is None
    )


def test_write_regression_stats_writes_fit_inputs_without_stats(tmp_path: Path) -> None:
    context = _context(tmp_path)
    returned_path = reg_mod.write_regression_stats(
        context=context,
        state=_state(
            stats_rows=[],
            uncertainty_rows=[],
            fit_inputs_rows=[_fit_input_row(fit_year=2001)],
        ),
    )

    assert returned_path is None
    fit_inputs_path = reg_mod.fit_inputs_path_for_format(
        proj_base=context.proj_base,
        output_format=context.output_format,
        source=context.source,
        agg_version=context.agg_version,
    )
    assert fit_inputs_path.exists()
    assert pd.read_csv(fit_inputs_path)["fit_year"].tolist() == [2001]

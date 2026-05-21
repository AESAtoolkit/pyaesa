import pandas as pd
import pytest

from pyaesa.asocc.orchestration.projection.regression.projection_clipping_log import (
    CLIP_KEY_COLUMNS,
)
from pyaesa.asocc.orchestration.write.regression_stats import models as mod
from pyaesa.asocc.orchestration.write.regression_stats.columns import REG_KEY


def _base_row(
    *,
    model_type: str,
    domain_key: str,
    target_object: str,
    x_center_value: object,
    baseline_object: object,
    category_object: object,
) -> dict[str, object]:
    row = {
        "projection_branch": "compute_asocc",
        "source": "oecd_v2025",
        "fu_code": "L2.a.a",
        "l2_method": "UT(FD)",
        "model_type": model_type,
        "target_object": target_object,
        "domain_key": domain_key,
        "fit_start_year": 2005,
        "fit_end_year": 2007,
        "x_object": "gdp",
        "x_unit": "USD_2021/yr",
        "x_transform": "",
        "x_center_value": x_center_value,
        "y_object": target_object,
        "y_unit": "dimensionless",
        "y_transform": "",
        "numerator_object": "numerator",
        "denominator_object": "denominator",
        "baseline_object": baseline_object,
        "category_object": category_object,
        "n_obs": 6,
        "df_resid": 4,
        "intercept": 1.2,
        "slope": 0.4,
        "r_squared": 0.9,
        "p_value_slope": 0.01,
        "sigma2_hat": 0.25,
        "x_mean": 1.5,
        "ssx": 5.0,
        "x_min": 0.0,
        "x_max": 2.0,
        "years_used": "2005-2007",
        "deterministic_clip_lower": "pre-filled",
        "deterministic_clip_applied_count_hint": "pre-filled",
        "notes": "",
    }
    for column in mod.REGRESSION_MODELS_COLUMNS:
        row.setdefault(column, "")
    return row


def test_raise_on_duplicate_keys_and_missing_uncertainty_cover_guards() -> None:
    empty = pd.DataFrame(columns=REG_KEY)
    mod._raise_on_duplicate_keys(frame=empty, label="regression_stats_rows")  # noqa: SLF001

    with pytest.raises(ValueError):
        mod._raise_on_duplicate_keys(  # noqa: SLF001
            frame=pd.DataFrame([{"projection_branch": "compute_asocc"}]),
            label="regression_stats_rows",
        )

    dup_row = {
        "projection_branch": "compute_asocc",
        "source": "oecd_v2025",
        "fu_code": "L2.a.a",
        "l2_method": "UT(FD)",
        "model_type": "ols_level",
        "target_object": "fd_rf",
        "domain_key": "FR",
        "fit_start_year": 2005,
        "fit_end_year": 2007,
    }
    with pytest.raises(ValueError):
        mod._raise_on_duplicate_keys(  # noqa: SLF001
            frame=pd.DataFrame([dup_row, dup_row]),
            label="regression_stats_rows",
        )

    no_error_frame = pd.DataFrame(
        [
            {
                "projection_branch": "compute_asocc",
                "source": "oecd_v2025",
                "fu_code": "L2.a.a",
                "l2_method": "UT(FD)",
                "model_type": "ols_level",
                "target_object": "fd_rf",
                "domain_key": "FR",
                "fit_start_year": 2005,
                "fit_end_year": 2007,
                "sigma2_hat": 0.25,
                "df_resid": 4,
            }
        ]
    )
    mod._raise_on_missing_uncertainty(  # noqa: SLF001
        frame=no_error_frame,
        required_columns=["sigma2_hat", "df_resid"],
    )

    with pytest.raises(ValueError):
        mod._raise_on_missing_uncertainty(  # noqa: SLF001
            frame=no_error_frame.assign(sigma2_hat=""),
            required_columns=["sigma2_hat", "df_resid"],
        )

    with pytest.raises(ValueError):
        mod._raise_on_missing_uncertainty(  # noqa: SLF001
            frame=no_error_frame.drop(columns=["df_resid"]),
            required_columns=["sigma2_hat", "df_resid"],
        )


def test_normalize_regression_models_frame_covers_ols_log_ratio_and_blank_rows() -> None:
    frame = pd.DataFrame(
        [
            _base_row(
                model_type="ols_level",
                domain_key="FR",
                target_object="fd_rf",
                x_center_value="",
                baseline_object="",
                category_object="",
            ),
            _base_row(
                model_type="ols_level",
                domain_key="DE",
                target_object="fd_rf",
                x_center_value="",
                baseline_object="",
                category_object="",
            ),
            _base_row(
                model_type="log_ratio_time",
                domain_key="FR|cat/base",
                target_object="share_log_ratio",
                x_center_value=2005,
                baseline_object="",
                category_object="",
            ),
            _base_row(
                model_type="spline",
                domain_key="US",
                target_object="fd_rf",
                x_center_value="",
                baseline_object="kept?",
                category_object="kept?",
            ),
        ]
    )
    clip_counts = {
        tuple(str(frame.loc[0, column]) for column in CLIP_KEY_COLUMNS): 3,
    }
    normalized = mod._normalize_regression_models_frame(  # noqa: SLF001
        frame,
        clip_counts=clip_counts,
    )

    ols_row = normalized.loc[
        (normalized["model_type"] == "ols_level") & normalized["domain_key"].eq("FR")
    ].iloc[0]
    assert ols_row["x_transform"] == "level"
    assert ols_row["y_transform"] == "level"
    assert ols_row["x_center_value"] == ""
    assert ols_row["deterministic_clip_lower"] == 0.0
    clip_hint = str(ols_row["deterministic_clip_applied_count_hint"])
    assert "3" in clip_hint
    assert "projection_clipping_log.csv" in clip_hint
    untouched_ols_row = normalized.loc[
        (normalized["model_type"] == "ols_level") & normalized["domain_key"].eq("DE")
    ].iloc[0]
    assert untouched_ols_row["deterministic_clip_applied_count_hint"] != ""
    normalized_without_clip_counts = mod._normalize_regression_models_frame(  # noqa: SLF001
        frame.loc[frame["domain_key"].eq("DE")].copy(),
        clip_counts={},
    )
    assert normalized_without_clip_counts.loc[0, "deterministic_clip_applied_count_hint"] != ""

    log_ratio_row = normalized.loc[normalized["model_type"] == "log_ratio_time"].iloc[0]
    assert log_ratio_row["x_transform"] == "centered"
    assert log_ratio_row["y_transform"] == "log_ratio"
    assert log_ratio_row["baseline_object"] == "denominator"
    assert log_ratio_row["category_object"] == "numerator"
    assert log_ratio_row["deterministic_clip_lower"] == ""
    assert log_ratio_row["deterministic_clip_applied_count_hint"] == ""

    other_row = normalized.loc[normalized["model_type"] == "spline"].iloc[0]
    assert other_row["deterministic_clip_lower"] == ""
    assert other_row["deterministic_clip_applied_count_hint"] == ""


def test_normalize_regression_models_frame_requires_log_ratio_center_value() -> None:
    frame = pd.DataFrame(
        [
            _base_row(
                model_type="log_ratio_time",
                domain_key="FR|cat/base",
                target_object="share_log_ratio",
                x_center_value="",
                baseline_object="",
                category_object="",
            ),
        ]
    )
    with pytest.raises(ValueError):
        mod._normalize_regression_models_frame(  # noqa: SLF001
            frame,
            clip_counts={},
        )


def test_merge_regression_models_frame_merges_and_normalizes() -> None:
    stats_frame = pd.DataFrame(
        [
            {
                "projection_branch": "compute_asocc",
                "source": "oecd_v2025",
                "fu_code": "L2.a.a",
                "l2_method": "UT(FD)",
                "model_type": "ols_level",
                "target_object": "fd_rf",
                "domain_key": "FR",
                "fit_start_year": 2005,
                "fit_end_year": 2007,
                "x_object": "gdp",
                "x_unit": "USD_2021/yr",
                "y_object": "fd_rf",
                "y_unit": "USD_2021/yr",
                "intercept": 1.2,
                "slope": 0.4,
                "r_squared": 0.9,
                "p_value_slope": 0.01,
                "n_obs": 6,
            }
        ]
    )
    uncertainty_frame = pd.DataFrame(
        [
            {
                "projection_branch": "compute_asocc",
                "source": "oecd_v2025",
                "fu_code": "L2.a.a",
                "l2_method": "UT(FD)",
                "model_type": "ols_level",
                "target_object": "fd_rf",
                "domain_key": "FR",
                "fit_start_year": 2005,
                "fit_end_year": 2007,
                "sigma2_hat": 0.25,
                "df_resid": 4,
                "x_mean": 1.5,
                "ssx": 5.0,
                "x_min": 0.0,
                "x_max": 2.0,
                "years_used": "2005-2007",
                "notes": "ols_mean_var_simple",
            }
        ]
    )
    merged = mod._merge_regression_models_frame(  # noqa: SLF001
        stats_frame=stats_frame,
        uncertainty_frame=uncertainty_frame,
        clip_counts={tuple(str(stats_frame.loc[0, column]) for column in CLIP_KEY_COLUMNS): 1},
        required_uncertainty_columns=[
            "sigma2_hat",
            "df_resid",
            "x_mean",
            "ssx",
            "x_min",
            "x_max",
            "years_used",
            "notes",
        ],
    )
    assert merged.shape[0] == 1
    assert float(merged.loc[0, "sigma2_hat"]) == pytest.approx(0.25)
    assert merged.loc[0, "deterministic_clip_lower"] == 0.0
    clip_hint = str(merged.loc[0, "deterministic_clip_applied_count_hint"])
    assert "1" in clip_hint
    assert "projection_clipping_log.csv" in clip_hint

"""Common utilities shared by projection regression kernels."""

from collections.abc import Iterable
from typing import Protocol, cast

import numpy as np
import pandas as pd
from scipy.stats import linregress

from ..config.types import (
    RegressionFitInputRow,
    RegressionStatsRow,
    RegressionUncertaintyRow,
)
from pyaesa.shared.runtime.text import print_user_text_line

SHARE_EPSILON = 1.0e-9
MIN_OLS_UNCERTAINTY_OBS = 3


class _LinregressResultLike(Protocol):
    """Attributes used from ``scipy.stats.linregress`` output."""

    intercept: float
    slope: float
    rvalue: float
    pvalue: float


def serialize_years(*, years: list[int]) -> str:
    """Serialize fit years as deterministic compact ranges.

    Continuous runs are rendered as ``start end``; isolated years are rendered
    as scalars. Example: ``[1995, 1996, 1998, 2001, 2002]`` ->
    ``"1995-1996, 1998, 2001-2002"``.
    """
    ordered = sorted({int(year) for year in years})
    if not ordered:
        return ""
    ranges: list[tuple[int, int]] = []
    start = ordered[0]
    end = ordered[0]
    for year in ordered[1:]:
        if year == end + 1:
            end = year
            continue
        ranges.append((start, end))
        start = year
        end = year
    ranges.append((start, end))
    return ", ".join(
        str(range_start) if range_start == range_end else f"{range_start}-{range_end}"
        for range_start, range_end in ranges
    )


def validate_min_ols_uncertainty_observations(
    *,
    n_obs: int,
    context: str,
    detail: str,
) -> None:
    """Validate the minimum sample size required for OLS uncertainty terms."""
    if int(n_obs) >= MIN_OLS_UNCERTAINTY_OBS:
        return
    raise ValueError(f"{context}: {detail} Got n_obs={int(n_obs)}.")


def fit_simple_ols(
    *,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit y = intercept + slope * x and return diagnostics."""
    if x.size != y.size:
        raise ValueError("OLS fit requires aligned x/y sizes.")
    if x.size < 2:
        raise ValueError("OLS fit requires at least two observations.")
    fit = cast(_LinregressResultLike, linregress(x, y))
    return (
        float(fit.intercept),
        float(fit.slope),
        float(fit.rvalue * fit.rvalue),
        float(fit.pvalue),
    )


def coerce_numeric_pairs(
    *,
    x_values: Iterable[object],
    y_values: Iterable[object],
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite aligned numeric x/y arrays."""
    x_series = cast(
        pd.Series,
        pd.to_numeric(pd.Series(list(x_values), dtype="object"), errors="raise"),
    )
    y_series = cast(
        pd.Series,
        pd.to_numeric(pd.Series(list(y_values), dtype="object"), errors="raise"),
    )
    x = x_series.to_numpy(dtype=np.float64, copy=False)
    y = y_series.to_numpy(dtype=np.float64, copy=False)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def coerce_numeric_scalar(value: object) -> float:
    """Return one numeric scalar or NaN when value is not coercible."""
    coerced_series = cast(
        pd.Series,
        pd.to_numeric(pd.Series([value], dtype="object"), errors="raise"),
    )
    coerced = coerced_series.iloc[0]
    return float(coerced)


def compute_ols_uncertainty_scalars(
    *,
    x: np.ndarray,
    y: np.ndarray,
    intercept: float,
    slope: float,
) -> tuple[float, float, float, int, float, float]:
    """Compute strict OLS uncertainty scalars for intercept+slope models."""
    if x.size != y.size:
        raise ValueError("OLS uncertainty requires aligned x/y sizes.")
    n_obs = int(x.size)
    validate_min_ols_uncertainty_observations(
        n_obs=n_obs,
        context="OLS uncertainty for intercept+slope",
        detail="Requires at least three observations.",
    )
    x_mean = float(np.mean(x))
    centered = x - x_mean
    ssx = float(np.sum(centered * centered))
    if not np.isfinite(ssx) or ssx <= 0.0:
        raise ValueError(
            f"OLS uncertainty requires a varying regressor (SSx > 0). Received SSx={ssx}."
        )
    fitted = float(intercept) + float(slope) * x
    resid = y - fitted
    df_resid = n_obs - 2
    sigma2_hat = float(np.sum(resid * resid) / float(df_resid))
    return (
        sigma2_hat,
        x_mean,
        ssx,
        int(df_resid),
        float(np.min(x)),
        float(np.max(x)),
    )


def fit_cache_key(
    *,
    source: str,
    fu_code: str,
    l2_method: str,
    model_type: str,
    target_object: str,
    historical_years: list[int],
) -> tuple:
    """Build deterministic cache key for one projected equation family."""
    return (
        str(source),
        str(fu_code),
        str(l2_method),
        str(model_type),
        str(target_object),
        tuple(int(year) for year in historical_years),
    )


def emit_regression_start_notice(
    *,
    source: str,
    fu_code: str,
    l2_method: str,
    model_type: str,
    target_object: str,
    state,
) -> None:
    """Print one deduplicated start message for one regression equation family."""
    key = (
        "projection-regression-start",
        str(source),
        str(fu_code),
        str(l2_method),
        str(model_type),
        str(target_object),
    )
    if key in state.notices_emitted:
        return
    state.notices_emitted.add(key)
    source_prefix = getattr(state, "runtime_source_prefix", None)
    if not isinstance(source_prefix, str) or not source_prefix.strip():
        source_prefix = f"[{source}]"
    line = (
        f"{source_prefix} [projection] computing regression "
        f"model={model_type}, target={target_object}, "
        f"fu={fu_code}, l2={l2_method}..."
    )
    progress = getattr(state, "runtime_progress", None)
    log_message = getattr(progress, "log_message", None)
    if callable(log_message):
        log_message(line, persistent=False)
        return
    print_user_text_line(line)


def future_year_range_label(*, future_years: list[int]) -> str:
    """Return compact future year range label."""
    if not future_years:
        return ""
    return f"{min(future_years)}-{max(future_years)}"


def mrio_level_unit_for_target(*, target_object: str, state) -> str:
    """Return MRIO unit for a regression target using setup loaded metadata."""
    metric = str(target_object).strip().lower()
    explicit = getattr(state, "mrio_units", {}).get(metric)
    if explicit is not None:
        return str(explicit)
    if metric.startswith(("fd_", "gva_", "fda_", "gvaa_", "x_")):
        default_unit = getattr(state, "mrio_default_monetary_unit", None)
        return cast(str, default_unit)
    raise ValueError(
        f"Unknown MRIO regression target for unit resolution: target_object='{target_object}'."
    )


def append_regression_row(*, state, row: dict[str, object]) -> None:
    """Append one regression diagnostics row in run state."""
    state.regression_stats_rows.append(row)


def append_regression_fit_input_row(*, state, row: dict[str, object]) -> None:
    """Append one per year fit input diagnostics row in run state."""
    state.regression_fit_inputs_rows.append(row)


def append_regression_uncertainty_row(*, state, row: dict[str, object]) -> None:
    """Append one OLS uncertainty metadata row in run state."""
    state.regression_uncertainty_rows.append(row)


def build_regression_row(
    *,
    source: str,
    fu_code: str,
    l2_method: str,
    model_type: str,
    target_object: str,
    domain_key: str,
    fit_start_year: int,
    fit_end_year: int,
    n_obs: int,
    intercept: float,
    slope: float,
    r_squared: float,
    p_value_slope: float,
    x_object: str = "",
    x_unit: str = "",
    x_transform: str = "",
    x_center_value: float | str = "",
    y_object: str = "",
    y_unit: str = "",
    y_transform: str = "",
    numerator_object: str = "",
    denominator_object: str = "",
    baseline_object: str = "",
    category_object: str = "",
    deterministic_clip_lower: float | str = "",
    deterministic_clip_applied_count_hint: str = "",
) -> dict[str, object]:
    """Build one regression diagnostics row as dictionary."""
    return RegressionStatsRow(
        projection_branch="regression",
        source=source,
        fu_code=fu_code,
        l2_method=l2_method,
        model_type=model_type,
        target_object=target_object,
        domain_key=domain_key,
        fit_start_year=fit_start_year,
        fit_end_year=fit_end_year,
        n_obs=n_obs,
        intercept=intercept,
        slope=slope,
        r_squared=r_squared,
        p_value_slope=p_value_slope,
        x_object=x_object,
        x_unit=x_unit,
        x_transform=x_transform,
        x_center_value=x_center_value,
        y_object=y_object,
        y_unit=y_unit,
        y_transform=y_transform,
        numerator_object=numerator_object,
        denominator_object=denominator_object,
        baseline_object=baseline_object,
        category_object=category_object,
        deterministic_clip_lower=deterministic_clip_lower,
        deterministic_clip_applied_count_hint=deterministic_clip_applied_count_hint,
    ).as_dict()


def build_regression_fit_input_row(
    *,
    source: str,
    fu_code: str,
    l2_method: str,
    model_type: str,
    target_object: str,
    domain_key: str,
    fit_start_year: int,
    fit_end_year: int,
    fit_year: int,
    x_value: float,
    y_value: float,
    y_kind: str,
    ratio_value: float = float("nan"),
    numerator_value: float = float("nan"),
    denominator_value: float = float("nan"),
    x_object: str = "",
    x_unit: str = "",
    y_object: str = "",
    y_unit: str = "",
    numerator_object: str = "",
    denominator_object: str = "",
) -> dict[str, object]:
    """Build one per year fit input row as dictionary."""
    return RegressionFitInputRow(
        projection_branch="regression",
        source=source,
        fu_code=fu_code,
        l2_method=l2_method,
        model_type=model_type,
        target_object=target_object,
        domain_key=domain_key,
        fit_start_year=fit_start_year,
        fit_end_year=fit_end_year,
        fit_year=fit_year,
        x_value=x_value,
        y_value=y_value,
        y_kind=y_kind,
        ratio_value=ratio_value,
        numerator_value=numerator_value,
        denominator_value=denominator_value,
        x_object=x_object,
        x_unit=x_unit,
        y_object=y_object,
        y_unit=y_unit,
        numerator_object=numerator_object,
        denominator_object=denominator_object,
    ).as_dict()


def build_regression_uncertainty_row(
    *,
    source: str,
    fu_code: str,
    l2_method: str,
    model_type: str,
    target_object: str,
    domain_key: str,
    fit_start_year: int,
    fit_end_year: int,
    n_obs: int,
    sigma2_hat: float,
    df_resid: int,
    x_mean: float,
    ssx: float,
    x_min: float,
    x_max: float,
    years_used: str,
    notes: str,
) -> dict[str, object]:
    """Build one regression uncertainty metadata row as dictionary."""
    return RegressionUncertaintyRow(
        projection_branch="regression",
        source=source,
        fu_code=fu_code,
        l2_method=l2_method,
        model_type=model_type,
        target_object=target_object,
        domain_key=domain_key,
        fit_start_year=fit_start_year,
        fit_end_year=fit_end_year,
        n_obs=n_obs,
        sigma2_hat=sigma2_hat,
        df_resid=df_resid,
        x_mean=x_mean,
        ssx=ssx,
        x_min=x_min,
        x_max=x_max,
        years_used=years_used,
        notes=notes,
    ).as_dict()


def clip_share_values(values: pd.Series) -> pd.Series:
    """Clip probability values away from exact 0/1 bounds."""
    out = cast(pd.Series, pd.to_numeric(values, errors="raise"))
    return out.clip(lower=SHARE_EPSILON, upper=1.0 - SHARE_EPSILON)

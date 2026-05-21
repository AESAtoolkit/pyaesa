"""Regression kernels for post historical UT projection (GDP scaled levels)."""

from collections.abc import Hashable
from typing import cast

import numpy as np
import pandas as pd

from ...method_scope import _max_historical_mrio_year
from .projection_clipping_log import write_projection_clipping_log
from .regression_core_utils import (
    append_regression_fit_input_row,
    append_regression_row,
    append_regression_uncertainty_row,
    build_regression_fit_input_row,
    build_regression_row,
    build_regression_uncertainty_row,
    coerce_numeric_pairs,
    coerce_numeric_scalar,
    compute_ols_uncertainty_scalars,
    emit_regression_start_notice,
    fit_cache_key,
    fit_simple_ols,
    mrio_level_unit_for_target,
    serialize_years,
    validate_min_ols_uncertainty_observations,
)


def _fit_ols_map_for_domains(
    *,
    source: str,
    fu_code: str,
    l2_method: str,
    target_object: str,
    historical_years: list[int],
    history_by_year: dict[int, pd.Series],
    predictor_by_year: dict[int, pd.Series],
    selected_domains: list[str] | None,
    state,
) -> dict[object, tuple[float, float, float, float, int]]:
    """Fit and cache per domain OLS parameters for one equation family."""
    selected_signature = (
        tuple(sorted(str(domain) for domain in selected_domains))
        if selected_domains
        else ("__ALL__",)
    )
    key = fit_cache_key(
        source=source,
        fu_code=fu_code,
        l2_method=l2_method,
        model_type="ols_level",
        target_object=f"{target_object}|domains={selected_signature}",
        historical_years=historical_years,
    )
    cached = state.regression_fit_cache.get(key)
    if isinstance(cached, dict):
        return cached
    emit_regression_start_notice(
        source=source,
        fu_code=fu_code,
        l2_method=l2_method,
        model_type="ols_level",
        target_object=target_object,
        state=state,
    )

    domains: set[object] = set()
    for year in historical_years:
        domains.update(idx for idx in predictor_by_year[int(year)].index)
        domains.update(idx for idx in history_by_year[int(year)].index)
    predictor_value_maps = {
        int(year): predictor_by_year[int(year)].to_dict() for year in historical_years
    }
    history_value_maps = {
        int(year): history_by_year[int(year)].to_dict() for year in historical_years
    }
    if selected_domains:
        selected = {str(domain) for domain in selected_domains}
        domains = {domain for domain in domains if str(domain) in selected}
    fit_start = int(min(historical_years))
    fit_end = cast(int, _max_historical_mrio_year(historical_years=historical_years))
    level_unit = mrio_level_unit_for_target(target_object=target_object, state=state)
    fit_map: dict[object, tuple[float, float, float, float, int]] = {}
    ordered_domains = sorted(domains, key=str)
    for domain in ordered_domains:
        x_hist = [predictor_value_maps[int(year)].get(domain, np.nan) for year in historical_years]
        y_hist = [history_value_maps[int(year)].get(domain, np.nan) for year in historical_years]
        fit_points: list[tuple[int, float, float]] = []
        for year, x_raw, y_raw in zip(historical_years, x_hist, y_hist):
            x_value = coerce_numeric_scalar(x_raw)
            y_value = coerce_numeric_scalar(y_raw)
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            fit_points.append((int(year), float(x_value), float(y_value)))
        x_fit, y_fit = coerce_numeric_pairs(x_values=x_hist, y_values=y_hist)
        n_obs = int(x_fit.size)
        validate_min_ols_uncertainty_observations(
            n_obs=n_obs,
            context=(
                "Cannot fit regression with uncertainty for level projection: "
                f"target_object='{target_object}', domain_key='{domain}', "
                f"fit_start_year={fit_start}, fit_end_year={fit_end}."
            ),
            detail="Simple OLS uncertainty requires n_obs >= 3 (df_resid = n_obs - 2).",
        )
        intercept, slope, r_squared, p_value = fit_simple_ols(x=x_fit, y=y_fit)
        (
            sigma2_hat,
            x_mean,
            ssx,
            df_resid,
            x_min,
            x_max,
        ) = compute_ols_uncertainty_scalars(
            x=x_fit,
            y=y_fit,
            intercept=intercept,
            slope=slope,
        )
        fit_map[domain] = (intercept, slope, r_squared, p_value, n_obs)
        row = build_regression_row(
            source=source,
            fu_code=fu_code,
            l2_method=l2_method,
            model_type="ols_level",
            target_object=target_object,
            domain_key=str(domain),
            fit_start_year=int(fit_start),
            fit_end_year=int(fit_end),
            n_obs=int(n_obs),
            intercept=float(intercept),
            slope=float(slope),
            r_squared=float(r_squared),
            p_value_slope=float(p_value),
            x_object="gdp_by_domain",
            x_unit="USD_2021/yr",
            y_object=target_object,
            y_unit=level_unit,
        )
        append_regression_row(state=state, row=row)
        uncertainty_row = build_regression_uncertainty_row(
            source=source,
            fu_code=fu_code,
            l2_method=l2_method,
            model_type="ols_level",
            target_object=target_object,
            domain_key=str(domain),
            fit_start_year=int(fit_start),
            fit_end_year=int(fit_end),
            n_obs=int(n_obs),
            sigma2_hat=float(sigma2_hat),
            df_resid=int(df_resid),
            x_mean=float(x_mean),
            ssx=float(ssx),
            x_min=float(x_min),
            x_max=float(x_max),
            years_used=serialize_years(years=[fit_year for fit_year, _x, _y in fit_points]),
            notes="ols_mean_var_simple",
        )
        append_regression_uncertainty_row(state=state, row=uncertainty_row)
        for fit_year, x_value, y_value in fit_points:
            fit_input = build_regression_fit_input_row(
                source=source,
                fu_code=fu_code,
                l2_method=l2_method,
                model_type="ols_level",
                target_object=target_object,
                domain_key=str(domain),
                fit_start_year=int(fit_start),
                fit_end_year=int(fit_end),
                fit_year=int(fit_year),
                x_value=float(x_value),
                y_value=float(y_value),
                y_kind="level",
                x_object="gdp_by_domain",
                x_unit="USD_2021/yr",
                y_object=target_object,
                y_unit=level_unit,
            )
            append_regression_fit_input_row(state=state, row=fit_input)
    state.regression_fit_cache[key] = fit_map
    return fit_map


def project_series_from_gdp(
    *,
    source: str,
    fu_code: str,
    l2_method: str,
    target_object: str,
    target_year: int,
    historical_years: list[int],
    history_by_year: dict[int, pd.Series],
    gdp_by_year: dict[int, pd.Series],
    gdp_target: pd.Series,
    selected_domains: list[str] | None,
    state,
) -> pd.Series:
    """Project one region indexed level series using strict per domain OLS.

    The OLS fit requires at least three observations because residual variance
    estimation uses df_resid = n_obs - 2. Fewer points cannot provide the
    uncertainty scalars required by projection diagnostics.
    """
    reference_index_name: Hashable | None = history_by_year[int(historical_years[0])].index.name
    if selected_domains is None:
        proj_index = gdp_target.index
    else:
        selected_ordered = list(dict.fromkeys(str(domain) for domain in selected_domains))
        index_by_label: dict[str, object] = {}
        for domain in gdp_target.index:
            index_by_label.setdefault(str(domain), domain)
        proj_index = pd.Index(
            [index_by_label[domain] for domain in selected_ordered],
            name=gdp_target.index.name,
        )
    fit_map = _fit_ols_map_for_domains(
        source=source,
        fu_code=fu_code,
        l2_method=l2_method,
        target_object=target_object,
        historical_years=historical_years,
        history_by_year=history_by_year,
        predictor_by_year=gdp_by_year,
        selected_domains=selected_domains,
        state=state,
    )
    fit_end_year = cast(int, _max_historical_mrio_year(historical_years=historical_years))
    raw_pred = pd.Series(index=proj_index, dtype=float)
    fit_map_by_label = {str(domain): fit for domain, fit in fit_map.items()}
    for domain in proj_index:
        fit = fit_map_by_label[str(domain)]
        intercept, slope, _r2, _p, _n_obs = fit
        x_target = coerce_numeric_scalar(gdp_target.get(domain, np.nan))
        if not np.isfinite(x_target):
            raw_pred.loc[domain] = np.nan
            continue
        raw_pred.loc[domain] = float(intercept + slope * x_target)
    raw_pred.index = raw_pred.index.rename(reference_index_name)

    # Monetary level targets are nonnegative by construction. We therefore clip
    # negatives to zero and log every clipped value for reproducibility.
    write_projection_clipping_log(
        before=raw_pred,
        source=source,
        projection_branch="regression",
        fu_code=fu_code,
        l2_method=l2_method,
        target_object=target_object,
        year=int(target_year),
        unit=mrio_level_unit_for_target(target_object=target_object, state=state),
        fit_start_year=int(min(historical_years)),
        fit_end_year=int(fit_end_year),
        state=state,
    )
    return raw_pred.clip(lower=0.0)

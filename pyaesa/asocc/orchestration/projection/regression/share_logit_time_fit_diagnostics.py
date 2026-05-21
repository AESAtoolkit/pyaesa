"""Diagnostics writers used by strict share logit time fit builder."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from .regression_core_utils import (
    append_regression_fit_input_row,
    append_regression_row,
    append_regression_uncertainty_row,
    build_regression_fit_input_row,
    build_regression_row,
    build_regression_uncertainty_row,
    compute_ols_uncertainty_scalars,
    serialize_years,
)
from .share_logit_time_fit_types import ShareFitBuildConfig, _FitPoint

LOG_UNITLESS = "log_unitless"


@dataclass(frozen=True)
class ShareFitDiagnosticsContext:
    """Stable diagnostics context shared across one fit map build."""

    config: ShareFitBuildConfig
    fit_start: int
    fit_end: int
    state: object


class ShareFitDiagnosticsPayload(NamedTuple):
    """Typed payload for one fitted category diagnostics write."""

    domain_key: str
    container_name: str
    category: object
    baseline: object
    n_obs: int
    intercept: float
    slope: float
    r_squared: float
    p_value: float
    year_center: float
    x_centered: np.ndarray
    y_values: np.ndarray
    fit_points: list[_FitPoint]
    valid_years: list[int]


def persist_share_fit_diagnostics(
    *,
    context: ShareFitDiagnosticsContext,
    payload: ShareFitDiagnosticsPayload,
) -> None:
    """Append regression stats, uncertainty, and per year fit input rows."""
    domain_key = payload.domain_key
    container_name = payload.container_name
    n_obs = int(payload.n_obs)
    intercept = float(payload.intercept)
    slope = float(payload.slope)
    r_squared = float(payload.r_squared)
    p_value = float(payload.p_value)
    fit_points = payload.fit_points
    valid_years = payload.valid_years

    append_regression_row(
        state=context.state,
        row=build_regression_row(
            source=context.config.source,
            fu_code=context.config.fu_code,
            l2_method=context.config.l2_method,
            model_type="log_ratio_time",
            target_object=context.config.target_object,
            domain_key=domain_key,
            fit_start_year=int(context.fit_start),
            fit_end_year=int(context.fit_end),
            n_obs=n_obs,
            intercept=intercept,
            slope=slope,
            r_squared=r_squared,
            p_value_slope=p_value,
            x_object="year",
            x_unit="year",
            x_center_value=float(payload.year_center),
            y_object="log(share_c/share_b)",
            y_unit="dimensionless",
            numerator_object=f"share({payload.category}) in {container_name}",
            denominator_object=f"share({payload.baseline}) in {container_name}",
        ),
    )
    uncertainty = compute_ols_uncertainty_scalars(
        x=np.asarray(payload.x_centered, dtype=float),
        y=np.asarray(payload.y_values, dtype=float),
        intercept=intercept,
        slope=slope,
    )
    append_regression_uncertainty_row(
        state=context.state,
        row=build_regression_uncertainty_row(
            source=context.config.source,
            fu_code=context.config.fu_code,
            l2_method=context.config.l2_method,
            model_type="log_ratio_time",
            target_object=context.config.target_object,
            domain_key=domain_key,
            fit_start_year=int(context.fit_start),
            fit_end_year=int(context.fit_end),
            n_obs=n_obs,
            sigma2_hat=float(uncertainty[0]),
            df_resid=int(uncertainty[3]),
            x_mean=float(uncertainty[1]),
            ssx=float(uncertainty[2]),
            x_min=float(uncertainty[4]),
            x_max=float(uncertainty[5]),
            years_used=serialize_years(years=valid_years),
            notes="ols_mean_var_simple",
        ),
    )
    for fit_point in fit_points:
        append_regression_fit_input_row(
            state=context.state,
            row=build_regression_fit_input_row(
                source=context.config.source,
                fu_code=context.config.fu_code,
                l2_method=context.config.l2_method,
                model_type="log_ratio_time",
                target_object=context.config.target_object,
                domain_key=domain_key,
                fit_start_year=int(context.fit_start),
                fit_end_year=int(context.fit_end),
                fit_year=int(fit_point[0]),
                x_value=float(fit_point[1]),
                y_value=float(fit_point[2]),
                y_kind="log_ratio",
                ratio_value=float(fit_point[3]),
                numerator_value=float(fit_point[4]),
                denominator_value=float(fit_point[5]),
                x_object="fit_year_centered",
                x_unit="year",
                y_object="log(numerator_share/denominator_share)",
                y_unit=LOG_UNITLESS,
                numerator_object=f"share({payload.category}) in {container_name}",
                denominator_object=f"share({payload.baseline}) in {container_name}",
            ),
        )


def last_modeled_vector(
    *,
    historical_years: list[int],
    modeled_by_year: dict[int, pd.Series],
) -> pd.Series:
    """Return the latest modeled vector after baseline coverage has been selected."""
    return next(
        vector
        for year in sorted(historical_years, reverse=True)
        for vector in [modeled_by_year[int(year)]]
        if float(vector.sum(min_count=1)) > 0.0
    )

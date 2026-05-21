"""Column dictionary builder for regression stats outputs."""

from pyaesa.shared.runtime.text import join_user_text_lines

from ...projection.config.types import MODEL_REGRESSION_KEY

REG_KEY = list(MODEL_REGRESSION_KEY)

REGRESSION_MODELS_COLUMNS = [
    "projection_branch",
    "source",
    "fu_code",
    "l2_method",
    "model_type",
    "target_object",
    "domain_key",
    "fit_start_year",
    "fit_end_year",
    "x_object",
    "x_unit",
    "x_transform",
    "x_center_value",
    "y_object",
    "y_unit",
    "y_transform",
    "numerator_object",
    "denominator_object",
    "baseline_object",
    "category_object",
    "n_obs",
    "df_resid",
    "intercept",
    "slope",
    "r_squared",
    "p_value_slope",
    "sigma2_hat",
    "x_mean",
    "ssx",
    "x_min",
    "x_max",
    "years_used",
    "deterministic_clip_lower",
    "deterministic_clip_applied_count_hint",
    "notes",
]

_REGRESSION_COLUMNS_DEFINITIONS = {
    "projection_branch": (
        "Name of the projection run branch (e.g., project name or internal branch label).",
    ),
    "source": ("MRIO source identifier (e.g., `exiobase_3102_ixi`).",),
    "fu_code": ("Functional unit code (e.g., `L2.c.b`).",),
    "l2_method": ("L2 method label.",),
    "model_type": (
        "Regression model family used for projection.",
        "`ols_level`: OLS of a level target vs GDP predictor.",
        "`log_ratio_time`: OLS of log(share_category / share_baseline) vs centered year.",
    ),
    "target_object": (
        "Name of the target being projected (e.g., `FD_total`, `GVA_total`, `share_log_ratio`).",
    ),
    "domain_key": (
        "Domain identifier for the regression (e.g., region code like `US` for levels; "
        "or `container|category/baseline` for shares).",
    ),
    "fit_start_year": (
        "Inclusive year bounds for the historical fit window (before filtering invalid points).",
    ),
    "fit_end_year": (
        "Inclusive year bounds for the historical fit window (before filtering invalid points).",
    ),
    "x_object": (
        "Name of the predictor (e.g., `GDP_PPP` for levels, `year` for time-based share fits).",
    ),
    "x_unit": ("Unit of x (e.g., `constant_2017USD` or `year`).",),
    "x_transform": ("Transform applied to x before fitting (e.g., `level`, `centered`).",),
    "x_center_value": (
        'Centering constant used when `x_transform="centered"`.',
        "For `log_ratio_time`: this is the `year_center` so that `x = year - x_center_value`.",
        "Blank for uncentered/level x.",
    ),
    "y_object": (
        "Name/description of response variable.",
        "For `log_ratio_time`: `log(share_c/share_b)`.",
    ),
    "y_unit": ("Unit of y (e.g., `EUR` for levels, `dimensionless` for log-ratios).",),
    "y_transform": ("Transform applied to y before fitting (`level` or `log_ratio`).",),
    "numerator_object": (
        "For `log_ratio_time` only: category name for numerator in `log(share_num/share_den)`.",
        "Blank for `ols_level`.",
    ),
    "denominator_object": (
        "For `log_ratio_time` only: baseline name for denominator in `log(share_num/share_den)`.",
        "Blank for `ols_level`.",
    ),
    "baseline_object": ("baseline_object == denominator_object",),
    "category_object": ("category_object == numerator_object",),
    "n_obs": (
        "Number of observations used in the fitted regression (after filtering invalid points).",
    ),
    "df_resid": ("Residual degrees of freedom (simple OLS): `n_obs - 2`.",),
    "intercept": ("Fitted OLS coefficients for the transformed model.",),
    "slope": ("Fitted OLS coefficients for the transformed model.",),
    "r_squared": ("Coefficient of determination for the fit (on transformed scale).",),
    "p_value_slope": ("Two-sided p-value for slope coefficient (on transformed scale).",),
    "sigma2_hat": (
        "Estimated residual variance sigma2_hat on the transformed scale: "
        "sum(residual^2) / df_resid.",
        "Used by projection uncertainty as the fitted variance estimate for hierarchical "
        "multi-point regression sampling with one sampled latent variance, conditional "
        "coefficient runs, and year-specific future residual shocks "
        "(Wooldridge, 2009; Greene, 2003; Seber and Lee, 2003).",
    ),
    "x_mean": (
        "Mean of x used in the fit (after transforms).",
        "For centered year in shares, this should be ~0.",
        "Used with `ssx` in the conditional coefficient covariance for hierarchical "
        "multi-point regression sampling "
        "(Wooldridge, 2009; Greene, 2003; Seber and Lee, 2003).",
    ),
    "ssx": (
        "Sum of squares of x about its mean: SSx = sum((x - x_mean)^2).",
        "Required for conditional coefficient uncertainty in hierarchical multi-point "
        "regression sampling "
        "(Wooldridge, 2009; Greene, 2003; Seber and Lee, 2003).",
    ),
    "x_min": (
        "Min of x values used in the fit (after transforms). Useful for "
        "detecting extrapolation later.",
    ),
    "x_max": (
        "Max of x values used in the fit (after transforms). Useful for "
        "detecting extrapolation later.",
    ),
    "years_used": (
        "Deterministic compact fit-year ranges used after filtering.",
        "Examples: `1995-2019` (continuous), `1995-1998, 2001, 2003-2005` (discontinuous).",
    ),
    "deterministic_clip_lower": (
        "If set (e.g., 0.0), deterministic projections of this model's target "
        "are clipped at this lower bound.",
        "The same lower bound is applied after sampled `ols_level` future evaluations.",
        "This clipping step is package-specific rather than a direct regression-text result.",
        "Blank if not clipped.",
    ),
    "deterministic_clip_applied_count_hint": (
        "Clipping status for deterministic level projection rows.",
        "`no clipping` means no negative values were clipped for that model key.",
        "When clipping occurred, this stores the count and points to "
        "`projection_clipping_log.csv`.",
    ),
    "notes": (
        "Free text notes (e.g., regression-evaluation metadata, subset-fit-window info, "
        "numerical stability notes like stable softmax shift).",
    ),
}


def render_regression_columns_defs(*, columns: list[str]) -> str:
    """Return deterministic text dictionary for regression_stats.csv columns."""
    blocks: list[str] = []
    for column in columns:
        try:
            lines = _REGRESSION_COLUMNS_DEFINITIONS[column]
        except KeyError as exc:
            raise ValueError(f"Missing regression column definition for '{column}'.") from exc
        block_lines = [column, *(f"- {line}" for line in lines)]
        blocks.append("\n".join(block_lines))
    return join_user_text_lines("\n\n".join(blocks).splitlines(), trailing_newline=True)

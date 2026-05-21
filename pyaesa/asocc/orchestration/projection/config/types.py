"""Core data structures for L2 UT projection runtime."""

from dataclasses import dataclass

RegWindowSelector = list[int] | range | None
RegWindowBounds = tuple[int, int] | None

PROJECTION_MODES = {"regression", "historical_reuse"}
UT_ADJUSTED_METHODS = {"UT(FDa)", "UT(GVAa)"}
BASE_REGRESSION_KEY = (
    "projection_branch",
    "source",
    "fu_code",
    "l2_method",
    "target_object",
    "domain_key",
)
MODEL_REGRESSION_KEY = (
    *BASE_REGRESSION_KEY[:4],
    "model_type",
    *BASE_REGRESSION_KEY[4:],
    "fit_start_year",
    "fit_end_year",
)
FIT_INPUT_REGRESSION_KEY = (*MODEL_REGRESSION_KEY, "fit_year")


@dataclass(frozen=True)
class ProjectionContext:
    """Resolved projection controls for one run branch."""

    enabled: bool
    mode: str | None
    max_historical_year: int
    future_years: tuple[int, ...]
    reg_window: RegWindowBounds
    l2_reuse_years: tuple[int, ...]
    ut_methods_in_scope: tuple[str, ...]
    l2_method_route_by_name: dict[str, str]

    def is_future_year(self, year: int) -> bool:
        """Return whether a studied year is beyond historical MRIO range."""
        return int(year) > int(self.max_historical_year)

    def route_for_l2_method(self, l2_method: str) -> str | None:
        """Return projection route for one L2 method."""
        return self.l2_method_route_by_name.get(l2_method)

    def l2_reuse_years_for(self) -> tuple[int, ...]:
        """Return all configured L2 reuse years for UT historical reuse routes."""
        return tuple(sorted({int(year) for year in self.l2_reuse_years}))


@dataclass(frozen=True)
class RegressionStatsRow:
    """One persisted diagnostics row for a fitted projection equation."""

    projection_branch: str
    source: str
    fu_code: str
    l2_method: str
    model_type: str
    target_object: str
    domain_key: str
    fit_start_year: int
    fit_end_year: int
    n_obs: int
    intercept: float
    slope: float
    r_squared: float
    p_value_slope: float
    x_object: str = ""
    x_unit: str = ""
    x_transform: str = ""
    x_center_value: float | str = ""
    y_object: str = ""
    y_unit: str = ""
    y_transform: str = ""
    numerator_object: str = ""
    denominator_object: str = ""
    baseline_object: str = ""
    category_object: str = ""
    deterministic_clip_lower: float | str = ""
    deterministic_clip_applied_count_hint: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return row as a deterministic plain dictionary."""
        return {
            "projection_branch": self.projection_branch,
            "source": self.source,
            "fu_code": self.fu_code,
            "l2_method": self.l2_method,
            "model_type": self.model_type,
            "target_object": self.target_object,
            "domain_key": self.domain_key,
            "fit_start_year": int(self.fit_start_year),
            "fit_end_year": int(self.fit_end_year),
            "n_obs": int(self.n_obs),
            "intercept": float(self.intercept),
            "slope": float(self.slope),
            "r_squared": float(self.r_squared),
            "p_value_slope": float(self.p_value_slope),
            "x_object": self.x_object,
            "x_unit": self.x_unit,
            "x_transform": self.x_transform,
            "x_center_value": self.x_center_value,
            "y_object": self.y_object,
            "y_unit": self.y_unit,
            "y_transform": self.y_transform,
            "numerator_object": self.numerator_object,
            "denominator_object": self.denominator_object,
            "baseline_object": self.baseline_object,
            "category_object": self.category_object,
            "deterministic_clip_lower": self.deterministic_clip_lower,
            "deterministic_clip_applied_count_hint": (self.deterministic_clip_applied_count_hint),
        }


@dataclass(frozen=True)
class RegressionFitInputRow:
    """One persisted per year fit input row for a projection equation."""

    projection_branch: str
    source: str
    fu_code: str
    l2_method: str
    model_type: str
    target_object: str
    domain_key: str
    fit_start_year: int
    fit_end_year: int
    fit_year: int
    x_value: float
    y_value: float
    y_kind: str
    ratio_value: float
    numerator_value: float
    denominator_value: float
    x_object: str = ""
    x_unit: str = ""
    y_object: str = ""
    y_unit: str = ""
    numerator_object: str = ""
    denominator_object: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return row as a deterministic plain dictionary."""
        return {
            "projection_branch": self.projection_branch,
            "source": self.source,
            "fu_code": self.fu_code,
            "l2_method": self.l2_method,
            "model_type": self.model_type,
            "target_object": self.target_object,
            "domain_key": self.domain_key,
            "fit_start_year": int(self.fit_start_year),
            "fit_end_year": int(self.fit_end_year),
            "fit_year": int(self.fit_year),
            "x_value": float(self.x_value),
            "y_value": float(self.y_value),
            "y_kind": self.y_kind,
            "ratio_value": float(self.ratio_value),
            "numerator_value": float(self.numerator_value),
            "denominator_value": float(self.denominator_value),
            "x_object": self.x_object,
            "x_unit": self.x_unit,
            "y_object": self.y_object,
            "y_unit": self.y_unit,
            "numerator_object": self.numerator_object,
            "denominator_object": self.denominator_object,
        }


@dataclass(frozen=True)
class RegressionUncertaintyRow:
    """One persisted row containing OLS uncertainty metadata scalars."""

    projection_branch: str
    source: str
    fu_code: str
    l2_method: str
    model_type: str
    target_object: str
    domain_key: str
    fit_start_year: int
    fit_end_year: int
    n_obs: int
    sigma2_hat: float
    df_resid: int
    x_mean: float
    ssx: float
    x_min: float
    x_max: float
    years_used: str
    notes: str

    def as_dict(self) -> dict[str, object]:
        """Return row as a deterministic plain dictionary."""
        return {
            "projection_branch": self.projection_branch,
            "source": self.source,
            "fu_code": self.fu_code,
            "l2_method": self.l2_method,
            "model_type": self.model_type,
            "target_object": self.target_object,
            "domain_key": self.domain_key,
            "fit_start_year": int(self.fit_start_year),
            "fit_end_year": int(self.fit_end_year),
            "n_obs": int(self.n_obs),
            "sigma2_hat": float(self.sigma2_hat),
            "df_resid": int(self.df_resid),
            "x_mean": float(self.x_mean),
            "ssx": float(self.ssx),
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "years_used": self.years_used,
            "notes": self.notes,
        }

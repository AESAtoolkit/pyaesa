"""Family neutral Sobol estimator diagnostics and convergence checks."""

import numpy as np
import pandas as pd

from pyaesa.shared.uncertainty_assessment.sobol.plan import SobolPlan
from pyaesa.shared.uncertainty_assessment.sobol.accumulator import sobol_confidence_converged

SOBOL_DIAGNOSTIC_TOLERANCE = 1e-8


def sobol_source_summary_confidence_converged(
    *,
    source_summary: pd.DataFrame,
    plan: SobolPlan,
) -> bool:
    """Return whether source summary S1 and ST confidence widths converged."""
    active = source_summary["variance_weight_sum"].astype(float).gt(0.0)
    s1_converged = sobol_confidence_converged(
        values=source_summary.loc[active, "variance_weighted_S1"].to_numpy(dtype=float),
        half_widths=source_summary.loc[
            active,
            "variance_weighted_S1_confidence_half_width",
        ].to_numpy(dtype=float),
        rtol=plan.rtol,
        abs_tol=plan.abs_tol,
        scale_floor=plan.scale_floor,
    )
    st_converged = sobol_confidence_converged(
        values=source_summary.loc[active, "variance_weighted_ST"].to_numpy(dtype=float),
        half_widths=source_summary.loc[
            active,
            "variance_weighted_ST_confidence_half_width",
        ].to_numpy(dtype=float),
        rtol=plan.rtol,
        abs_tol=plan.abs_tol,
        scale_floor=plan.scale_floor,
    )
    return s1_converged and st_converged


def max_sobol_confidence_half_width(*, source_summary: pd.DataFrame, column: str) -> float:
    """Return the maximum finite confidence half width in a source summary column."""
    values = source_summary[column].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.shape[0] == 0:
        return float("nan")
    return float(finite.max())


def sobol_source_summary_estimator_range_pass(
    *,
    source_summary: pd.DataFrame,
    plan: SobolPlan,
) -> bool:
    """Return whether source summary estimates stay in tolerance of Sobol bounds."""
    active = source_summary["variance_weight_sum"].astype(float).gt(0.0)
    if not bool(active.any()):
        return True
    checks = []
    for value_column, precision_column in (
        ("variance_weighted_S1", "variance_weighted_S1_confidence_half_width"),
        ("variance_weighted_ST", "variance_weighted_ST_confidence_half_width"),
    ):
        values = source_summary.loc[active, value_column].to_numpy(dtype=float)
        precision = source_summary.loc[active, precision_column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        margin = _finite_margin(precision[finite])
        tolerance = np.maximum(
            margin,
            plan.abs_tol + plan.rtol * np.maximum(np.abs(values[finite]), plan.scale_floor),
        )
        checks.append(bool(np.all(values[finite] >= -tolerance)))
        checks.append(bool(np.all(values[finite] <= 1.0 + tolerance)))
    s1_values = source_summary.loc[active, "variance_weighted_S1"].to_numpy(dtype=float)
    st_values = source_summary.loc[active, "variance_weighted_ST"].to_numpy(dtype=float)
    st_precision = source_summary.loc[
        active,
        "variance_weighted_ST_confidence_half_width",
    ].to_numpy(dtype=float)
    finite_pair = np.isfinite(s1_values) & np.isfinite(st_values)
    st_margin = _finite_margin(st_precision[finite_pair])
    pair_tolerance = np.maximum(
        st_margin,
        plan.abs_tol + plan.rtol * np.maximum(np.abs(st_values[finite_pair]), plan.scale_floor),
    )
    checks.append(bool(np.all(st_values[finite_pair] + pair_tolerance >= s1_values[finite_pair])))
    return bool(all(checks))


def sobol_diagnostic_counts(
    *,
    s1: np.ndarray,
    st: np.ndarray,
    s1_confidence_half_width: np.ndarray,
    st_confidence_half_width: np.ndarray,
    variance: np.ndarray,
) -> dict[str, int]:
    """Return confidence aware finite sample Sobol diagnostic counts."""
    defined = (
        np.isfinite(variance)[None, :]
        & (variance[None, :] > 0.0)
        & np.isfinite(s1)
        & np.isfinite(st)
    )
    s1_margin = _finite_margin(s1_confidence_half_width)
    st_margin = _finite_margin(st_confidence_half_width)
    negative_s1 = defined & ((s1 + s1_margin) < -SOBOL_DIAGNOSTIC_TOLERANCE)
    st_below_s1 = defined & ((st + st_margin + SOBOL_DIAGNOSTIC_TOLERANCE) < (s1 - s1_margin))
    above_one = defined & (
        ((s1 - s1_margin) > 1.0 + SOBOL_DIAGNOSTIC_TOLERANCE)
        | ((st - st_margin) > 1.0 + SOBOL_DIAGNOSTIC_TOLERANCE)
    )
    diagnostic = negative_s1 | st_below_s1 | above_one
    return {
        "diagnostic_output_count": int(diagnostic.sum()),
        "negative_S1_count": int(negative_s1.sum()),
        "ST_below_S1_count": int(st_below_s1.sum()),
        "above_one_count": int(above_one.sum()),
    }


def sobol_diagnostic_label(
    *,
    s1: float,
    st: float,
    s1_confidence_half_width: float,
    st_confidence_half_width: float,
    variance: float,
) -> str:
    """Return a confidence aware row level finite sample Sobol diagnostic label."""
    if not np.isfinite(variance) or variance <= 0.0 or not np.isfinite(s1) or not np.isfinite(st):
        return "undefined_zero_or_nonfinite_variance"
    labels = []
    s1_margin = float(s1_confidence_half_width) if np.isfinite(s1_confidence_half_width) else 0.0
    st_margin = float(st_confidence_half_width) if np.isfinite(st_confidence_half_width) else 0.0
    if s1 + s1_margin < -SOBOL_DIAGNOSTIC_TOLERANCE:
        labels.append("negative_S1")
    if st + st_margin + SOBOL_DIAGNOSTIC_TOLERANCE < s1 - s1_margin:
        labels.append("ST_below_S1")
    if (
        s1 - s1_margin > 1.0 + SOBOL_DIAGNOSTIC_TOLERANCE
        or st - st_margin > 1.0 + SOBOL_DIAGNOSTIC_TOLERANCE
    ):
        labels.append("above_one")
    return ";".join(labels) if labels else "ok"


def _finite_margin(values: np.ndarray) -> np.ndarray:
    margins = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(margins), margins, 0.0)

"""Family neutral Sobol result table construction."""

from statistics import NormalDist

import numpy as np
import pandas as pd

from pyaesa.shared.uncertainty_assessment.sobol.accumulator import SobolIndexEstimate
from pyaesa.shared.uncertainty_assessment.sobol.diagnostics import (
    sobol_diagnostic_counts,
    sobol_diagnostic_label,
)

SOBOL_SUMMARY_METRIC_COLUMNS: tuple[str, ...] = (
    "output_count",
    "defined_output_count",
    "undefined_output_count",
    "variance_weight_sum",
    "variance_weighted_S1",
    "variance_weighted_S1_confidence_half_width",
    "variance_weighted_ST",
    "variance_weighted_ST_confidence_half_width",
    "variance_weighted_ST_minus_S1",
    "confidence_level",
    "estimator_diagnostics_pass",
    "diagnostic_output_count",
    "negative_S1_count",
    "ST_below_S1_count",
    "above_one_count",
)


def sobol_index_table(
    *,
    identity: pd.DataFrame,
    dimension_names: tuple[str, ...],
    estimates: SobolIndexEstimate,
) -> pd.DataFrame:
    """Return row level Sobol indices by output identity and source."""
    rows = []
    identity_rows = identity.to_dict(orient="records")
    for row_index, identity_row in enumerate(identity_rows):
        for dimension_index, dimension_name in enumerate(dimension_names):
            s1 = estimates.s1[dimension_index, row_index]
            st = estimates.st[dimension_index, row_index]
            rows.append(
                {
                    **identity_row,
                    "source_name": dimension_name,
                    "sobol_output_variance": estimates.variance[row_index],
                    "S1": s1,
                    "S1_confidence_half_width": estimates.s1_confidence_half_width[
                        dimension_index, row_index
                    ],
                    "ST": st,
                    "ST_confidence_half_width": estimates.st_confidence_half_width[
                        dimension_index, row_index
                    ],
                    "ST_minus_S1": st - s1,
                    "estimator_diagnostic": sobol_diagnostic_label(
                        s1=s1,
                        st=st,
                        s1_confidence_half_width=estimates.s1_confidence_half_width[
                            dimension_index, row_index
                        ],
                        st_confidence_half_width=estimates.st_confidence_half_width[
                            dimension_index, row_index
                        ],
                        variance=estimates.variance[row_index],
                    ),
                }
            )
    return pd.DataFrame(rows)


def sobol_global_source_summary(
    *,
    dimension_names: tuple[str, ...],
    estimates: SobolIndexEstimate,
    confidence_level: float,
) -> pd.DataFrame:
    """Return one variance weighted source summary across all outputs."""
    rows = _sobol_source_summary_rows(
        group_values={},
        positions=np.arange(estimates.variance.shape[0], dtype=np.int64),
        dimension_names=dimension_names,
        estimates=estimates,
        confidence_level=confidence_level,
    )
    return pd.DataFrame(rows).sort_values(
        by=["variance_weighted_ST", "source_name"],
        ascending=[False, True],
        na_position="last",
        ignore_index=True,
    )


def sobol_source_summary_by_group(
    *,
    identity: pd.DataFrame,
    group_columns: tuple[str, ...],
    dimension_names: tuple[str, ...],
    estimates: SobolIndexEstimate,
    confidence_level: float,
) -> pd.DataFrame:
    """Return variance weighted Sobol source summaries by output groups."""
    if not group_columns:
        rows = _sobol_source_summary_rows(
            group_values={},
            positions=np.arange(estimates.variance.shape[0], dtype=np.int64),
            dimension_names=dimension_names,
            estimates=estimates,
            confidence_level=confidence_level,
        )
        return pd.DataFrame(rows)
    grouped = (
        identity.loc[:, list(group_columns)]
        .assign(_sobol_output_position=np.arange(len(identity), dtype=np.int64))
        .groupby(list(group_columns), dropna=False, sort=False)["_sobol_output_position"]
        .agg(tuple)
    )
    rows = []
    for group_key, positions in grouped.items():
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        rows.extend(
            _sobol_source_summary_rows(
                group_values=dict(zip(group_columns, key_values, strict=True)),
                positions=np.array(positions, dtype=np.int64),
                dimension_names=dimension_names,
                estimates=estimates,
                confidence_level=confidence_level,
            )
        )
    return pd.DataFrame(rows)


def sobol_source_summary_columns(
    *,
    selector_columns: tuple[str, ...],
    invariant_columns: tuple[str, ...] = (),
) -> list[str]:
    """Return canonical source summary column order."""
    return [
        "summary_level",
        *selector_columns,
        "source_name",
        *invariant_columns,
        *SOBOL_SUMMARY_METRIC_COLUMNS,
    ]


def _sobol_source_summary_rows(
    *,
    group_values: dict[str, object],
    positions: np.ndarray,
    dimension_names: tuple[str, ...],
    estimates: SobolIndexEstimate,
    confidence_level: float,
) -> list[dict[str, object]]:
    variance = estimates.variance
    weights = np.where(
        np.isfinite(variance[positions]) & (variance[positions] > 0.0), variance[positions], 0.0
    )
    rows: list[dict[str, object]] = []
    output_count = int(len(positions))
    for dimension_index, dimension_name in enumerate(dimension_names):
        s1_values = estimates.s1[dimension_index, positions]
        st_values = estimates.st[dimension_index, positions]
        defined = (weights > 0.0) & np.isfinite(s1_values) & np.isfinite(st_values)
        denominator = float(weights[defined].sum())
        if denominator > 0.0:
            variance_weighted_s1 = (
                float((s1_values[defined] * weights[defined]).sum()) / denominator
            )
            variance_weighted_st = (
                float((st_values[defined] * weights[defined]).sum()) / denominator
            )
            s1_confidence_half_width = _weighted_resample_half_width(
                resamples=estimates.s1_resamples[:, dimension_index, positions],
                weights=weights,
                defined=defined,
                denominator=denominator,
                confidence_level=confidence_level,
            )
            st_confidence_half_width = _weighted_resample_half_width(
                resamples=estimates.st_resamples[:, dimension_index, positions],
                weights=weights,
                defined=defined,
                denominator=denominator,
                confidence_level=confidence_level,
            )
        else:
            variance_weighted_s1 = np.nan
            variance_weighted_st = np.nan
            s1_confidence_half_width = np.nan
            st_confidence_half_width = np.nan
        diagnostic_counts = sobol_diagnostic_counts(
            s1=s1_values[None, :],
            st=st_values[None, :],
            s1_confidence_half_width=estimates.s1_confidence_half_width[
                dimension_index,
                positions,
            ][None, :],
            st_confidence_half_width=estimates.st_confidence_half_width[
                dimension_index,
                positions,
            ][None, :],
            variance=variance[positions],
        )
        rows.append(
            {
                **group_values,
                "source_name": dimension_name,
                "output_count": output_count,
                "defined_output_count": int(defined.sum()),
                "undefined_output_count": output_count - int(defined.sum()),
                "variance_weight_sum": denominator,
                "variance_weighted_S1": variance_weighted_s1,
                "variance_weighted_S1_confidence_half_width": s1_confidence_half_width,
                "variance_weighted_ST": variance_weighted_st,
                "variance_weighted_ST_confidence_half_width": st_confidence_half_width,
                "variance_weighted_ST_minus_S1": variance_weighted_st - variance_weighted_s1,
                "confidence_level": confidence_level,
                "estimator_diagnostics_pass": int(diagnostic_counts["diagnostic_output_count"])
                == 0,
                **diagnostic_counts,
            }
        )
    return rows


def _weighted_resample_half_width(
    *,
    resamples: np.ndarray,
    weights: np.ndarray,
    defined: np.ndarray,
    denominator: float,
    confidence_level: float,
) -> float:
    values = resamples[:, defined]
    weighted = (values * weights[defined][None, :]).sum(axis=1) / denominator
    return float(_confidence_half_width(values=weighted, confidence_level=confidence_level))


def _confidence_half_width(*, values: np.ndarray, confidence_level: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.shape[0] < 2:
        return float("nan")
    z_value = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    return float(z_value * np.nanstd(finite, ddof=1))

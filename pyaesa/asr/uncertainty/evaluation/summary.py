"""ASR uncertainty summary identity and metric collapse."""

from pathlib import Path

import numpy as np
import pandas as pd

from pyaesa.acc.uncertainty.evaluation.summary import (
    acc_summary_excluded_columns,
    acc_summary_identity_groups,
)
from pyaesa.acc.uncertainty.sources.source_keys import ASOCC_INTER_METHOD_SOURCE
from pyaesa.shared.uncertainty_assessment.evaluation.scenario_groups import (
    scenario_identity_groups_from_excluded_columns,
)
from pyaesa.asr.uncertainty.runtime.models import ASRUncertaintyPlan
from pyaesa.shared.runtime.scenario.columns import ASOCC_TIME_ROUTE_PUBLIC_COLUMN
from pyaesa.shared.runtime.scenario.time_routes import collapse_asocc_time_route
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    collapse_values_to_summary_groups,
    identity_groups_from_excluded_columns,
    sparse_public_row_group_membership_index,
)
from pyaesa.shared.uncertainty_assessment.io.public_summary import (
    exact_summary_and_frequency_from_public_runs,
)
from pyaesa.shared.uncertainty_assessment.io.summary_kernels import SUMMARY_STATISTICS
from pyaesa.shared.uncertainty_assessment.io.tables import write_uncertainty_table

ASR_SUMMARY_METRIC_COLUMN = "asr_metric"
ASR_VALUE_METRIC = "asr"
ASR_FREQUENCY_OF_NO_TRANSGRESSION_METRIC = "frequency_of_no_transgression"
ASR_CUMULATIVE_VALUE_METRIC = "cumulative_asr"
ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC = "cumulative_frequency_of_no_transgression"
ASR_FREQUENCY_VALUE_COLUMN = "frequency_of_no_transgression"
ASR_CUMULATIVE_FREQUENCY_VALUE_COLUMN = "cumulative_frequency_of_no_transgression"
ASR_SUMMARY_SCOPE_COLUMN = "asr_summary_scope"
ASR_SUMMARY_SCOPE_PER_METHOD = "per_method"
ASR_SUMMARY_SCOPE_INTER_METHOD = "inter_method"


def asr_summary_identity_groups(
    *,
    identity: pd.DataFrame,
    active_sources: tuple[str, ...],
    dynamic_category_uncertainty_active: bool = False,
) -> tuple[pd.DataFrame, tuple[tuple[str, ...], ...]]:
    """Return ASR summary identity rows for public and figure scoped summaries."""
    if ASOCC_INTER_METHOD_SOURCE not in set(active_sources):
        summary_identity, public_row_groups = acc_summary_identity_groups(
            identity=identity,
            active_sources=active_sources,
            dynamic_category_uncertainty_active=dynamic_category_uncertainty_active,
        )
        summary_identity = summary_identity.copy()
        summary_identity[ASR_SUMMARY_SCOPE_COLUMN] = ASR_SUMMARY_SCOPE_PER_METHOD
        return summary_identity, public_row_groups

    method_excluded = acc_summary_excluded_columns(
        active_sources=active_sources,
        dynamic_category_uncertainty_active=dynamic_category_uncertainty_active,
    )
    method_excluded.difference_update(
        {"l1_l2_method", "l1_method", "l2_method", ASOCC_TIME_ROUTE_PUBLIC_COLUMN}
    )
    method_identity, method_groups = identity_groups_from_excluded_columns(
        identity=identity,
        excluded_columns=method_excluded,
    )
    inter_excluded = acc_summary_excluded_columns(
        active_sources=active_sources,
        dynamic_category_uncertainty_active=dynamic_category_uncertainty_active,
    )
    inter_excluded.add("l2_reuse_year")
    inter_identity, inter_groups = scenario_identity_groups_from_excluded_columns(
        identity=identity,
        excluded_columns=inter_excluded,
    )
    method_identity = method_identity.copy()
    inter_identity = inter_identity.copy()
    if ASOCC_TIME_ROUTE_PUBLIC_COLUMN in identity.columns:
        inter_identity[ASOCC_TIME_ROUTE_PUBLIC_COLUMN] = _collapsed_time_routes_for_groups(
            identity=identity,
            public_row_groups=inter_groups,
        )
    method_identity[ASR_SUMMARY_SCOPE_COLUMN] = ASR_SUMMARY_SCOPE_PER_METHOD
    inter_identity[ASR_SUMMARY_SCOPE_COLUMN] = ASR_SUMMARY_SCOPE_INTER_METHOD
    return (
        pd.concat([method_identity, inter_identity], ignore_index=True, sort=False),
        (*method_groups, *inter_groups),
    )


def _collapsed_time_routes_for_groups(
    *,
    identity: pd.DataFrame,
    public_row_groups: tuple[tuple[str, ...], ...],
) -> list[object]:
    route_by_public_id = identity.set_index("public_row_id")[ASOCC_TIME_ROUTE_PUBLIC_COLUMN]
    return [
        collapse_asocc_time_route(route_by_public_id.loc[[int(value) for value in group]].tolist())
        for group in public_row_groups
    ]


def collapse_asr_values_to_summary(
    *,
    values: np.ndarray,
    plan: ASRUncertaintyPlan,
) -> np.ndarray:
    """Collapse dense ASR rows to source aware summary groups."""
    return collapse_values_to_summary_groups(
        values=values,
        public_row_groups=plan.summary_public_row_groups,
    )


def collapse_asr_cumulative_values_to_summary(
    *,
    values: np.ndarray,
    plan: ASRUncertaintyPlan,
) -> np.ndarray:
    """Collapse cumulative ASR public rows to summary metric groups."""
    return collapse_values_to_summary_groups(
        values=values,
        public_row_groups=plan.cumulative_summary_public_row_groups,
    )


def asr_sparse_public_row_group_membership_index(*, plan: ASRUncertaintyPlan) -> np.ndarray:
    """Return public row id to ASR summary group memberships for sparse runs."""
    return sparse_public_row_group_membership_index(
        public_row_groups=plan.summary_public_row_groups
    )


def summary_identity_with_metrics(base: pd.DataFrame) -> pd.DataFrame:
    """Return yearly ASR and frequency of no-transgression summary identities."""
    asr = base.copy()
    frequency = base.copy()
    asr[ASR_SUMMARY_METRIC_COLUMN] = ASR_VALUE_METRIC
    frequency[ASR_SUMMARY_METRIC_COLUMN] = ASR_FREQUENCY_OF_NO_TRANSGRESSION_METRIC
    return pd.concat([asr, frequency], ignore_index=True)


def cumulative_summary_identity_with_metrics(base: pd.DataFrame) -> pd.DataFrame:
    """Return cumulative ASR and frequency of no-transgression summary identities."""
    asr = base.copy()
    frequency = base.copy()
    asr[ASR_SUMMARY_METRIC_COLUMN] = ASR_CUMULATIVE_VALUE_METRIC
    frequency[ASR_SUMMARY_METRIC_COLUMN] = ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC
    return pd.concat([asr, frequency], ignore_index=True)


def write_asr_summary_table(
    *,
    path: Path,
    summary_identity: pd.DataFrame,
    runs_path: Path,
    run_count: int,
    output_format: str,
    public_row_groups: tuple[tuple[str, ...], ...],
    sparse: bool,
) -> None:
    """Write ASR summary rows with metric specific public statistics."""
    summary = asr_summary_table_from_public_runs(
        identity_frame=summary_identity,
        runs_path=runs_path,
        run_count=run_count,
        output_format=output_format,
        public_row_groups=public_row_groups,
        sparse=sparse,
    )
    write_uncertainty_table(
        path=path,
        frame=summary,
        output_format=output_format,
    )


def asr_summary_table_from_public_runs(
    *,
    identity_frame: pd.DataFrame,
    runs_path: Path,
    run_count: int,
    output_format: str,
    public_row_groups: tuple[tuple[str, ...], ...],
    sparse: bool,
) -> pd.DataFrame:
    """Return ASR value statistics and fNT means from public run artifacts."""
    metric = pd.Series(identity_frame[ASR_SUMMARY_METRIC_COLUMN], copy=False).astype(str)
    frequency_mask = metric.isin(
        [
            ASR_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
            ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
        ]
    ).to_numpy(dtype=bool)
    value_positions = np.flatnonzero(~frequency_mask)
    frequency_positions = np.flatnonzero(frequency_mask)
    summary, mean = exact_summary_and_frequency_from_public_runs(
        identity_frame=identity_frame.iloc[value_positions].reset_index(drop=True),
        runs_path=runs_path,
        output_format=output_format,
        run_count=run_count,
        public_row_groups=public_row_groups,
        sparse=sparse,
    )
    frequency = identity_frame.iloc[frequency_positions].reset_index(drop=True).copy()
    for statistic in SUMMARY_STATISTICS:
        frequency[statistic] = np.nan
    frequency["mean"] = mean
    combined = pd.concat([summary, frequency], ignore_index=True).loc[
        :, [*identity_frame.columns, *SUMMARY_STATISTICS]
    ]
    return asr_summary_output_table(combined)


def asr_summary_output_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return public ASR summary rows with fNT exposed as a frequency column."""
    out = summary.copy()
    metric = pd.Series(out[ASR_SUMMARY_METRIC_COLUMN], copy=False).astype(str)
    _move_frequency_mean(
        frame=out,
        metric=metric,
        metric_name=ASR_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
        output_column=ASR_FREQUENCY_VALUE_COLUMN,
    )
    _move_frequency_mean(
        frame=out,
        metric=metric,
        metric_name=ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
        output_column=ASR_CUMULATIVE_FREQUENCY_VALUE_COLUMN,
    )
    return out


def _move_frequency_mean(
    *,
    frame: pd.DataFrame,
    metric: pd.Series,
    metric_name: str,
    output_column: str,
) -> None:
    mask = metric.eq(metric_name)
    if not bool(mask.any()):
        return
    frame[output_column] = np.nan
    frame.loc[mask, output_column] = pd.to_numeric(frame.loc[mask, "mean"], errors="raise")
    frame.loc[mask, list(SUMMARY_STATISTICS)] = np.nan

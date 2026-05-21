"""Family neutral Sobol source summary selector levels."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyaesa.shared.uncertainty_assessment.sobol.accumulator import SobolIndexEstimate
from pyaesa.shared.uncertainty_assessment.sobol.summary import (
    sobol_source_summary_by_group,
    sobol_source_summary_columns,
)


@dataclass(frozen=True)
class SobolSummaryLevel:
    """One source summary grouping level."""

    summary_level: str
    group_columns: tuple[str, ...]


@dataclass(frozen=True)
class SobolInvariantAxisExpansion:
    """Configuration for repeating invariant outputs under active axis values."""

    axis_column: str
    axis_values: tuple[str, ...]
    contains_column: str
    count_column: str


@dataclass(frozen=True)
class _ExpandedSummaryInputs:
    identity: pd.DataFrame
    estimates: SobolIndexEstimate
    invariant_output: np.ndarray


_MISSING_GROUP_VALUE = object()


def sobol_source_summary_by_levels(
    *,
    identity: pd.DataFrame,
    dimension_names: tuple[str, ...],
    estimates: SobolIndexEstimate,
    confidence_level: float,
    levels: tuple[SobolSummaryLevel, ...],
    selector_columns: tuple[str, ...],
    invariant_axis: SobolInvariantAxisExpansion,
) -> pd.DataFrame:
    """Return grouped source summaries for configured public selector levels."""
    expanded = _expand_invariant_axis(
        identity=identity,
        estimates=estimates,
        invariant_axis=invariant_axis,
    )
    frames = [
        _source_summary_for_level(
            level=level,
            identity=expanded.identity,
            invariant_output=expanded.invariant_output,
            invariant_axis=invariant_axis,
            dimension_names=dimension_names,
            estimates=expanded.estimates,
            confidence_level=confidence_level,
        )
        for level in levels
    ]
    summary = pd.concat(frames, ignore_index=True)
    columns = sobol_source_summary_columns(
        selector_columns=selector_columns,
        invariant_columns=(invariant_axis.contains_column, invariant_axis.count_column),
    )
    return summary.loc[:, columns]


def _source_summary_for_level(
    *,
    level: SobolSummaryLevel,
    identity: pd.DataFrame,
    invariant_output: np.ndarray,
    invariant_axis: SobolInvariantAxisExpansion,
    dimension_names: tuple[str, ...],
    estimates: SobolIndexEstimate,
    confidence_level: float,
) -> pd.DataFrame:
    summary = sobol_source_summary_by_group(
        identity=identity,
        group_columns=level.group_columns,
        dimension_names=dimension_names,
        estimates=estimates,
        confidence_level=confidence_level,
    )
    metadata = _invariant_axis_metadata(
        identity=identity,
        invariant_output=invariant_output,
        group_columns=level.group_columns,
        invariant_axis=invariant_axis,
    )
    summary = summary.merge(
        metadata,
        on=list(level.group_columns),
        how="left",
        validate="many_to_one",
    )
    summary[invariant_axis.contains_column] = summary[invariant_axis.contains_column].fillna(False)
    summary[invariant_axis.count_column] = (
        summary[invariant_axis.count_column].fillna(0).astype(int)
    )
    summary.insert(0, "summary_level", level.summary_level)
    return summary


def _expand_invariant_axis(
    *,
    identity: pd.DataFrame,
    estimates: SobolIndexEstimate,
    invariant_axis: SobolInvariantAxisExpansion,
) -> _ExpandedSummaryInputs:
    if invariant_axis.axis_column not in identity.columns or not invariant_axis.axis_values:
        return _ExpandedSummaryInputs(
            identity=identity.copy(),
            estimates=estimates,
            invariant_output=np.zeros(len(identity), dtype=bool),
        )
    comparable_columns = tuple(
        column
        for column in identity.columns
        if column not in {"public_row_id", invariant_axis.axis_column}
    )
    expansion_values = _invariant_axis_values_by_group(
        identity=identity,
        comparable_columns=comparable_columns,
        invariant_axis=invariant_axis,
    )
    positions: list[int] = []
    invariant: list[bool] = []
    rows: list[dict[str, object]] = []
    for position, row in enumerate(identity.to_dict(orient="records")):
        if _is_missing_scalar(row.get(invariant_axis.axis_column)):
            values = expansion_values.get(
                _group_key(row=row, comparable_columns=comparable_columns),
                (),
            )
            if values:
                for axis_value in values:
                    repeated = dict(row)
                    repeated[invariant_axis.axis_column] = axis_value
                    rows.append(repeated)
                    positions.append(position)
                    invariant.append(True)
            else:
                rows.append(dict(row))
                positions.append(position)
                invariant.append(False)
        else:
            rows.append(dict(row))
            positions.append(position)
            invariant.append(False)
    output_positions = np.array(positions, dtype=np.int64)
    expanded_estimates = SobolIndexEstimate(
        s1=estimates.s1[:, output_positions],
        st=estimates.st[:, output_positions],
        variance=estimates.variance[output_positions],
        s1_confidence_half_width=estimates.s1_confidence_half_width[:, output_positions],
        st_confidence_half_width=estimates.st_confidence_half_width[:, output_positions],
        s1_resamples=estimates.s1_resamples[:, :, output_positions],
        st_resamples=estimates.st_resamples[:, :, output_positions],
    )
    return _ExpandedSummaryInputs(
        identity=pd.DataFrame(rows),
        estimates=expanded_estimates,
        invariant_output=np.array(invariant, dtype=bool),
    )


def _invariant_axis_values_by_group(
    *,
    identity: pd.DataFrame,
    comparable_columns: tuple[str, ...],
    invariant_axis: SobolInvariantAxisExpansion,
) -> dict[tuple[object, ...], tuple[str, ...]]:
    values_by_group: dict[tuple[object, ...], list[str]] = {}
    for row in identity.to_dict(orient="records"):
        axis_value = row.get(invariant_axis.axis_column)
        if _is_missing_scalar(axis_value):
            continue
        key = _group_key(row=row, comparable_columns=comparable_columns)
        values = values_by_group.setdefault(key, [])
        axis_text = str(axis_value)
        if axis_text not in values:
            values.append(axis_text)
    return {
        key: _ordered_axis_values(actual=tuple(actual), requested=invariant_axis.axis_values)
        for key, actual in values_by_group.items()
    }


def _group_key(
    *,
    row: dict[str, object],
    comparable_columns: tuple[str, ...],
) -> tuple[object, ...]:
    return tuple(
        _MISSING_GROUP_VALUE if _is_missing_scalar(row.get(column)) else row.get(column)
        for column in comparable_columns
    )


def _ordered_axis_values(*, actual: tuple[str, ...], requested: tuple[str, ...]) -> tuple[str, ...]:
    ordered = [value for value in requested if value in actual]
    ordered.extend(value for value in actual if value not in ordered)
    return tuple(ordered)


def _invariant_axis_metadata(
    *,
    identity: pd.DataFrame,
    invariant_output: np.ndarray,
    group_columns: tuple[str, ...],
    invariant_axis: SobolInvariantAxisExpansion,
) -> pd.DataFrame:
    marker = "_sobol_invariant_output"
    return (
        identity.loc[:, list(group_columns)]
        .assign(**{marker: invariant_output})
        .groupby(list(group_columns), dropna=False, sort=False)[marker]
        .agg(["any", "sum"])
        .reset_index()
        .rename(
            columns={
                "any": invariant_axis.contains_column,
                "sum": invariant_axis.count_column,
            }
        )
    )


def _is_missing_scalar(value: object) -> bool:
    return bool(pd.isna(value))

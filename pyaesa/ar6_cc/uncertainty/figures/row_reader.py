"""Row readers for AR6 CC uncertainty figure products."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from pyaesa.ar6_cc.uncertainty.figures.scope_planner import (
    FigureContext,
    SUMMARY_STAT_COLUMNS,
    TRAJECTORY_BAND_COLUMNS,
)
from pyaesa.shared.figures.uncertainty_run_values import collect_selected_compact_run_values
from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table


@dataclass(frozen=True)
class FigureTables:
    """Public AR6 CC uncertainty tables required by figure rendering."""

    summary: pd.DataFrame
    post_study_summary: pd.DataFrame | None
    budget_rows: pd.DataFrame
    source_methods: pd.DataFrame


def read_figure_tables(*, context: FigureContext) -> FigureTables:
    """Read public AR6 CC uncertainty artifacts needed by figures."""
    summary = read_uncertainty_table(
        path=context.paths.summary_stats_runs,
        output_format=context.output_format,
        csv_dtypes={
            "cc_category": "string",
            "ssp_scenario": "string",
            "cc_flow": "string",
            "cc_variable": "string",
            "impact_unit": "string",
        },
    )
    source_methods = pd.read_csv(context.paths.source_methods)
    post_study_summary = (
        read_uncertainty_table(
            path=context.paths.post_study_summary_stats_runs,
            output_format=context.output_format,
            csv_dtypes={
                "cc_category": "string",
                "ssp_scenario": "string",
                "cc_flow": "string",
                "cc_variable": "string",
                "impact_unit": "string",
            },
        )
        if context.has_post_study_period
        else None
    )
    budget_identity = read_uncertainty_table(
        path=context.paths.budget_row_identity,
        output_format=context.output_format,
        csv_dtypes={
            "cc_category": "string",
            "ssp_scenario": "string",
            "cc_flow": "string",
            "cc_variable": "string",
            "impact_unit": "string",
            "period_segment": "string",
        },
    )
    return FigureTables(
        summary=_prepare_summary(summary),
        post_study_summary=(
            None if post_study_summary is None else _prepare_summary(post_study_summary)
        ),
        budget_rows=_prepare_budget_rows(
            identity=budget_identity,
            values_by_id=collect_selected_compact_run_values(
                path=context.paths.budget_runs,
                output_format=context.output_format,
                public_row_ids=budget_identity["budget_row_id"].astype(int).tolist(),
                stop_run_index=int(context.manifest.completed_runs),
            ),
        ),
        source_methods=_prepare_source_methods(source_methods),
    )


def summary_rows_by_category(*, tables: FigureTables) -> pd.DataFrame:
    """Return summary rows for inactive category uncertainty figures."""
    columns = [
        "cc_category",
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        "year",
        *TRAJECTORY_BAND_COLUMNS,
    ]
    return (
        _combined_summary(tables=tables)
        .loc[:, columns]
        .sort_values(
            ["ssp_scenario", "impact_unit", "cc_category", "year"],
            kind="stable",
        )
    )


def summary_rows_global(*, tables: FigureTables) -> pd.DataFrame:
    """Return integrated summary rows for active category uncertainty figures."""
    columns = [
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        "year",
        *TRAJECTORY_BAND_COLUMNS,
    ]
    return (
        _combined_summary(tables=tables)
        .loc[:, columns]
        .sort_values(
            ["ssp_scenario", "impact_unit", "year"],
            kind="stable",
        )
    )


def budget_rows_by_category(*, tables: FigureTables) -> pd.DataFrame:
    """Return budget run arrays for inactive category uncertainty figures."""
    columns = [
        "cc_category",
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        "period_segment",
        "__budget_values",
    ]
    return tables.budget_rows.loc[:, columns].sort_values(
        ["ssp_scenario", "impact_unit", "cc_category", "period_segment"],
        kind="stable",
    )


def budget_rows_global(*, tables: FigureTables) -> pd.DataFrame:
    """Return budget run arrays for active category uncertainty figures."""
    columns = [
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        "period_segment",
        "__budget_values",
    ]
    return tables.budget_rows.loc[:, columns].sort_values(
        ["ssp_scenario", "impact_unit", "period_segment"],
        kind="stable",
    )


def category_pair_counts(*, tables: FigureTables) -> dict[tuple[str, str], int]:
    """Return retained model-scenario pair counts by category scope."""
    pairs = tables.source_methods.loc[
        :, ["ssp_scenario", "cc_category", "cc_model", "cc_scenario"]
    ].drop_duplicates(ignore_index=True)
    counts = pairs.groupby(
        ["ssp_scenario", "cc_category"],
        dropna=False,
        sort=True,
    ).size()
    return {_two_part_key(key): int(value) for key, value in counts.items()}


def common_pair_counts(*, tables: FigureTables) -> dict[str, int]:
    """Return retained model-scenario pair counts by active category ensemble scope."""
    pairs = tables.source_methods.loc[
        :, ["ssp_scenario", "cc_category", "cc_model", "cc_scenario"]
    ].drop_duplicates(ignore_index=True)
    counts = pairs.groupby("ssp_scenario", dropna=False, sort=True).size()
    return {str(key): int(value) for key, value in counts.items()}


def categories_by_common_scope(*, tables: FigureTables) -> dict[str, list[str]]:
    """Return retained category labels by active category ensemble scope."""
    categories = tables.source_methods.groupby(
        "ssp_scenario",
        dropna=False,
        sort=True,
    )["cc_category"].unique()
    return {str(key): sorted(str(value) for value in values) for key, values in categories.items()}


def _prepare_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "public_row_id" in work.columns:
        work["public_row_id"] = _numeric_series(frame=work, column="public_row_id").astype(int)
    work["year"] = _numeric_series(frame=work, column="year").astype(int)
    for column in ("cc_category", "ssp_scenario", "cc_flow", "cc_variable", "impact_unit"):
        if column in work.columns:
            work[column] = _text_series(frame=work, column=column)
    for column in SUMMARY_STAT_COLUMNS:
        work[column] = _numeric_series(frame=work, column=column)
    return work.reset_index(drop=True)


def _combined_summary(*, tables: FigureTables) -> pd.DataFrame:
    parts = [tables.summary]
    if tables.post_study_summary is not None:
        parts.append(tables.post_study_summary)
    return pd.concat(parts, ignore_index=True)


def _prepare_budget_rows(
    *,
    identity: pd.DataFrame,
    values_by_id: dict[int, np.ndarray],
) -> pd.DataFrame:
    work = identity.copy()
    work["budget_row_id"] = _numeric_series(frame=work, column="budget_row_id").astype(int)
    for column in (
        "cc_category",
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        "period_segment",
    ):
        if column in work.columns:
            work[column] = _text_series(frame=work, column=column)
    work["__budget_values"] = [
        values_by_id[int(row_id)] for row_id in work["budget_row_id"].tolist()
    ]
    return work.reset_index(drop=True)


def _prepare_source_methods(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for column in (
        "cc_category",
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        "cc_model",
        "cc_scenario",
    ):
        work[column] = _text_series(frame=work, column=column)
    return work.reset_index(drop=True)


def _summary_identity_columns(*, context: FigureContext) -> tuple[str, ...]:
    if context.category_uncertainty:
        return ("ssp_scenario", "cc_flow", "cc_variable", "impact_unit", "year")
    return ("cc_category", "ssp_scenario", "cc_flow", "cc_variable", "impact_unit", "year")


def _budget_identity_columns(*, context: FigureContext) -> tuple[str, ...]:
    common = ("budget_row_id", "ssp_scenario", "cc_flow", "cc_variable", "impact_unit")
    if context.category_uncertainty:
        return (*common, "period_segment")
    return ("budget_row_id", "cc_category", *common[1:], "period_segment")


def _numeric_series(*, frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one numeric frame column as a pandas Series."""
    return pd.Series(
        pd.to_numeric(pd.Series(frame.loc[:, column], copy=False), errors="raise"),
        index=frame.index,
    )


def _text_series(*, frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one text frame column as a pandas Series."""
    return pd.Series(frame.loc[:, column], copy=False).astype(str)


def _two_part_key(key: object) -> tuple[str, str]:
    values = cast(tuple[object, object], key)
    return str(values[0]), str(values[1])

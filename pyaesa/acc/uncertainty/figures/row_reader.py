"""Public aCC uncertainty table readers for figures."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyaesa.acc.figures.common import (
    AR6_CATEGORY_SCOPE_COLUMN,
    BUDGET_VALUES_COLUMN,
    DYNAMIC_CC_TYPE,
    PAIR_COUNT_COLUMN,
    VALUE_ARRAY_COLUMN,
    attach_common_columns,
)
from pyaesa.acc.uncertainty.figures.scope_planner import FigureContext
from pyaesa.acc.uncertainty.sources.source_keys import (
    AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE,
    ASOCC_PROJECTION_SOURCE,
    ASOCC_REFERENCE_YEAR_SOURCE,
)
from pyaesa.shared.figures.dynamic_ar6 import (
    MODEL_SCENARIO_SAMPLING_METHOD_COLUMN,
    category_scope_label,
)
from pyaesa.shared.figures.uncertainty_run_values import (
    RUN_INDEX_ARRAY_COLUMN,
    collect_selected_compact_run_values,
    collect_selected_sparse_run_indexed_values,
    sum_values_by_run_index,
)
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.runtime.scenario.time_routes import collapse_asocc_time_route
from pyaesa.shared.tabular.scalars import is_display_missing
from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table


@dataclass(frozen=True)
class FigureTables:
    """Public tables needed by aCC uncertainty figures."""

    identity: pd.DataFrame
    summary: pd.DataFrame


def read_figure_tables(*, context: FigureContext, include_summary: bool = True) -> FigureTables:
    """Read public identity and summary tables once for aCC figures."""
    identity = read_uncertainty_table(
        path=context.paths.public_row_identity,
        output_format=context.output_format,
    )
    summary = (
        read_uncertainty_table(
            path=context.paths.summary_stats_runs,
            output_format=context.output_format,
        )
        if include_summary
        else pd.DataFrame()
    )
    if context.dynamic_cc_sampling_method is not None:
        identity[MODEL_SCENARIO_SAMPLING_METHOD_COLUMN] = context.dynamic_cc_sampling_method
        if include_summary:
            summary[MODEL_SCENARIO_SAMPLING_METHOD_COLUMN] = context.dynamic_cc_sampling_method
    return FigureTables(identity=identity, summary=summary)


def prepared_summary_rows(*, context: FigureContext, summary: pd.DataFrame) -> pd.DataFrame:
    """Return summary rows with common aCC plot columns."""
    rows = _filter_requested_years(frame=summary, context=context)
    rows["fu_code"] = context.fu_code
    return attach_common_columns(rows)


def prepared_identity_rows(*, context: FigureContext, identity: pd.DataFrame) -> pd.DataFrame:
    """Return public row identity with common aCC plot columns."""
    rows = _filter_requested_years(frame=identity, context=context)
    rows["fu_code"] = context.fu_code
    return attach_common_columns(rows)


def value_rows_from_runs(
    *,
    context: FigureContext,
    identity_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Attach selected run distributions to public aCC identity rows."""
    public_ids = _public_ids(identity_rows)
    out = identity_rows.copy()
    if context.run_layout == "sparse_selected_rows":
        indexed_values = collect_selected_sparse_run_indexed_values(
            path=context.paths.public_runs,
            output_format=context.output_format,
            public_row_ids=public_ids,
            stop_run_index=int(context.manifest.completed_runs),
        )
        out[RUN_INDEX_ARRAY_COLUMN] = [
            indexed_values.get(
                int(public_id),
                (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)),
            )[0]
            for public_id in out["public_row_id"].tolist()
        ]
        out[VALUE_ARRAY_COLUMN] = [
            indexed_values.get(
                int(public_id),
                (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)),
            )[1]
            for public_id in out["public_row_id"].tolist()
        ]
        return drop_empty_value_rows(rows=out)
    values = collect_selected_compact_run_values(
        path=context.paths.public_runs,
        output_format=context.output_format,
        public_row_ids=public_ids,
        stop_run_index=int(context.manifest.completed_runs),
    )
    out[VALUE_ARRAY_COLUMN] = [
        values.get(int(public_id), np.empty(0, dtype=np.float64))
        for public_id in out["public_row_id"].tolist()
    ]
    return drop_empty_value_rows(rows=out)


def collapsed_value_rows(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    include_method_axis: bool,
) -> pd.DataFrame:
    """Collapse run arrays across active sampled axes for visible figure identities."""
    drop_columns = _sampled_axis_drop_columns(
        context=context, include_method_axis=include_method_axis
    )
    row_owned_ssp = ASOCC_SSP_SCENARIO_COLUMN in rows.columns and (
        "__figure_ssp_scope" in rows.columns or AR6_CC_SSP_SCENARIO_COLUMN in rows.columns
    )
    if row_owned_ssp:
        drop_columns.add(ASOCC_SSP_SCENARIO_COLUMN)
    key_columns = [
        column
        for column in rows.columns
        if column
        not in {*drop_columns, "public_row_id", VALUE_ARRAY_COLUMN, RUN_INDEX_ARRAY_COLUMN}
    ]
    has_dynamic_cc = _contains_dynamic_cc(rows)
    only_dynamic_cc = _contains_only_dynamic_cc(rows) if has_dynamic_cc else False
    dynamic_pair_columns = (
        _dynamic_pair_count_columns(frame=rows, context=context) if has_dynamic_cc else []
    )
    records = []
    for _key, group in rows.groupby(key_columns, dropna=False, sort=True):
        arrays = [np.asarray(values, dtype=np.float64) for values in group[VALUE_ARRAY_COLUMN]]
        merged = np.concatenate(arrays)
        first_row = group.iloc[0]
        payload = {column: first_row[column] for column in key_columns}
        if row_owned_ssp:
            payload[ASOCC_SSP_SCENARIO_COLUMN] = _collapsed_row_owned_ssp(group=group)
        payload[ASOCC_TIME_ROUTE_PUBLIC_COLUMN] = collapse_asocc_time_route(
            group[ASOCC_TIME_ROUTE_PUBLIC_COLUMN].tolist()
        )
        if has_dynamic_cc:
            pair_count = _dynamic_model_scenario_pair_count(
                group=group,
                pair_columns=dynamic_pair_columns,
                group_is_dynamic=only_dynamic_cc,
            )
            if pair_count is not None:
                payload[PAIR_COUNT_COLUMN] = pair_count
            category_scope = _dynamic_category_scope(group=group, context=context)
            if category_scope:
                payload[AR6_CATEGORY_SCOPE_COLUMN] = category_scope
        payload[VALUE_ARRAY_COLUMN] = merged
        if RUN_INDEX_ARRAY_COLUMN in group.columns:
            payload[RUN_INDEX_ARRAY_COLUMN] = np.concatenate(
                [np.asarray(values, dtype=np.int64) for values in group[RUN_INDEX_ARRAY_COLUMN]]
            )
        records.append(payload)
    return pd.DataFrame.from_records(records)


def attach_dynamic_budget_values(
    *,
    summary_rows: pd.DataFrame,
    value_rows: pd.DataFrame,
    context: FigureContext,
    include_method_axis: bool,
) -> pd.DataFrame:
    """Attach dynamic cumulative budget run arrays to rows with selected values."""
    collapsed = collapsed_value_rows(
        rows=value_rows,
        context=context,
        include_method_axis=include_method_axis,
    )
    budget_columns = _dynamic_budget_key_columns(collapsed)
    budget_values = _dynamic_budget_values(collapsed, key_columns=budget_columns)
    pair_counts = _dynamic_pair_counts_by_key(collapsed, key_columns=budget_columns)
    out = _summary_rows_with_selected_values(summary_rows)
    category_scope = _single_dynamic_category_scope(collapsed)
    if category_scope:
        out[AR6_CATEGORY_SCOPE_COLUMN] = category_scope
    out[BUDGET_VALUES_COLUMN] = [
        budget_values[_row_key(row=row, columns=budget_columns)] for _index, row in out.iterrows()
    ]
    out[PAIR_COUNT_COLUMN] = [
        pair_counts[_row_key(row=row, columns=budget_columns)] for _index, row in out.iterrows()
    ]
    return out


def _summary_rows_with_selected_values(rows: pd.DataFrame) -> pd.DataFrame:
    if "mean" not in rows.columns:
        return rows.copy()
    return rows.loc[~pd.Series(rows["mean"], copy=False).isna()].copy()


def drop_empty_value_rows(*, rows: pd.DataFrame) -> pd.DataFrame:
    """Return run based rows with at least one selected value."""
    mask = rows[VALUE_ARRAY_COLUMN].map(lambda values: len(values) > 0)
    return rows.loc[mask].reset_index(drop=True)


def _sampled_axis_drop_columns(
    *,
    context: FigureContext,
    include_method_axis: bool,
) -> set[str]:
    active = set(context.active_sources)
    dropped = set()
    if ASOCC_REFERENCE_YEAR_SOURCE in active:
        dropped.add("reference_year")
    if ASOCC_PROJECTION_SOURCE in active:
        dropped.add("l2_reuse_year")
    if AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE in active:
        dropped.update({"cc_model", "cc_scenario"})
    if context.dynamic_category_uncertainty_active:
        dropped.add("cc_category")
    if not include_method_axis:
        dropped.update({"__method", "l1_l2_method", "l1_method", "l2_method"})
        dropped.add(ASOCC_TIME_ROUTE_PUBLIC_COLUMN)
    return dropped


def _filter_requested_years(*, frame: pd.DataFrame, context: FigureContext) -> pd.DataFrame:
    years = pd.Series(pd.to_numeric(frame["year"], errors="raise"), copy=False).astype(int)
    return frame.loc[years.isin(list(context.requested_years))].copy()


def _public_ids(frame: pd.DataFrame) -> list[int]:
    return [int(value) for value in frame["public_row_id"].tolist()]


def _contains_dynamic_cc(frame: pd.DataFrame) -> bool:
    values = pd.Series(frame["cc_type"], copy=False).astype("string").str.strip()
    return bool(values.eq(DYNAMIC_CC_TYPE).any())


def _contains_only_dynamic_cc(frame: pd.DataFrame) -> bool:
    values = pd.Series(frame["cc_type"], copy=False).dropna().astype(str).str.strip()
    visible = {value for value in values.tolist() if value}
    return visible == {DYNAMIC_CC_TYPE}


def _collapsed_row_owned_ssp(*, group: pd.DataFrame) -> object:
    values = pd.Series(group[ASOCC_SSP_SCENARIO_COLUMN], copy=False)
    visible = [
        str(value).strip().upper()
        for value in values.tolist()
        if not is_display_missing(value) and str(value).strip()
    ]
    return next(iter(sorted(set(visible))), pd.NA)


def _dynamic_model_scenario_pair_count(
    *,
    group: pd.DataFrame,
    pair_columns: list[str],
    group_is_dynamic: bool,
) -> int | None:
    if not pair_columns:
        return None
    if not group_is_dynamic:
        group_cc_type = str(group.iloc[0]["cc_type"]).strip()
        if group_cc_type != DYNAMIC_CC_TYPE:
            return None
    return int(group.loc[:, pair_columns].drop_duplicates(ignore_index=True).shape[0])


def _dynamic_pair_count_columns(*, frame: pd.DataFrame, context: FigureContext) -> list[str]:
    pair_columns = ["cc_model", "cc_scenario"]
    if context.dynamic_category_uncertainty_active:
        pair_columns.insert(0, "cc_category")
    return [column for column in pair_columns if column in frame.columns]


def _dynamic_budget_key_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        VALUE_ARRAY_COLUMN,
        RUN_INDEX_ARRAY_COLUMN,
        AR6_CATEGORY_SCOPE_COLUMN,
        "year",
        PAIR_COUNT_COLUMN,
        MODEL_SCENARIO_SAMPLING_METHOD_COLUMN,
        ASOCC_SSP_SCENARIO_COLUMN,
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
        "reference_year",
        "l2_reuse_year",
        "asocc_ssp_start_year",
    }
    return [column for column in frame.columns if column not in excluded]


def _dynamic_budget_values(
    frame: pd.DataFrame,
    *,
    key_columns: list[str],
) -> dict[tuple[str, ...], np.ndarray]:
    budgets: dict[tuple[str, ...], np.ndarray] = {}
    for _key, group in frame.groupby(key_columns, dropna=False, sort=True):
        ordered = group.sort_values("year", kind="stable")
        row = pd.Series(ordered.iloc[0], copy=False)
        budgets[_row_key(row=row, columns=key_columns)] = _sum_dynamic_budget_values(ordered)
    return budgets


def _sum_dynamic_budget_values(group: pd.DataFrame) -> np.ndarray:
    if RUN_INDEX_ARRAY_COLUMN in group.columns:
        run_indices = np.concatenate(
            [np.asarray(values, dtype=np.int64) for values in group[RUN_INDEX_ARRAY_COLUMN]]
        )
        values = np.concatenate(
            [np.asarray(values, dtype=np.float64) for values in group[VALUE_ARRAY_COLUMN]]
        )
        _run_indices, summed = sum_values_by_run_index(run_indices=run_indices, values=values)
        return summed
    total: np.ndarray | None = None
    for values in group[VALUE_ARRAY_COLUMN].tolist():
        numeric = np.asarray(values, dtype=np.float64)
        total = numeric.copy() if total is None else total + numeric
    return np.empty(0, dtype=np.float64) if total is None else total


def _dynamic_pair_counts_by_key(
    frame: pd.DataFrame,
    *,
    key_columns: list[str],
) -> dict[tuple[str, ...], int]:
    return {
        _row_key(row=pd.Series(group.iloc[0], copy=False), columns=key_columns): int(
            pd.Series(group[PAIR_COUNT_COLUMN], copy=False).iloc[0]
        )
        for _key, group in frame.groupby(key_columns, dropna=False, sort=True)
        if PAIR_COUNT_COLUMN in group.columns
    }


def _row_key(*, row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple(_key_value(row[column]) for column in columns)


def _key_value(value: object) -> str:
    return {True: "<missing>", False: str(value)}[bool(is_display_missing(value))]


def _dynamic_category_scope(*, group: pd.DataFrame, context: FigureContext) -> str:
    if not context.dynamic_category_uncertainty_active or "cc_category" not in group.columns:
        return ""
    values = [
        str(value).strip()
        for value in group["cc_category"].tolist()
        if not is_display_missing(value) and str(value).strip()
    ]
    return category_scope_label(values)


def _single_dynamic_category_scope(frame: pd.DataFrame) -> str:
    if AR6_CATEGORY_SCOPE_COLUMN not in frame.columns:
        return ""
    values = [
        str(value).strip()
        for value in frame[AR6_CATEGORY_SCOPE_COLUMN].tolist()
        if not is_display_missing(value) and str(value).strip()
    ]
    return next(iter(dict.fromkeys(values)), "")

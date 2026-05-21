"""Public ASR uncertainty table readers for figures."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyaesa.acc.uncertainty.sources.source_keys import (
    AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE,
    ASOCC_PROJECTION_SOURCE,
    ASOCC_REFERENCE_YEAR_SOURCE,
)
from pyaesa.asr.figures.common import VALUE_ARRAY_COLUMN, attach_common_columns
from pyaesa.asr.uncertainty.evaluation.summary import (
    ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
    ASR_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
    ASR_SUMMARY_SCOPE_COLUMN,
    ASR_SUMMARY_SCOPE_INTER_METHOD,
    ASR_SUMMARY_METRIC_COLUMN,
    ASR_VALUE_METRIC,
)
from pyaesa.asr.uncertainty.figures.scope_planner import FigureContext
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.runtime.scenario.time_routes import collapse_asocc_time_route
from pyaesa.shared.figures.dynamic_ar6 import (
    AR6_CATEGORY_SCOPE_COLUMN,
    category_scope_label,
    DYNAMIC_AR6_CC_TYPE,
    MODEL_SCENARIO_PAIR_COUNT_COLUMN,
    MODEL_SCENARIO_SAMPLING_METHOD_COLUMN,
)
from pyaesa.shared.figures.lcia_metadata import LCIAMetadata, load_lcia_metadata
from pyaesa.shared.figures.uncertainty_run_values import (
    collect_selected_compact_run_values,
    collect_selected_sparse_run_values,
)
from pyaesa.shared.tabular.scalars import is_display_missing
from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table

SUMMARY_STAT_COLUMNS = ("mean", "std", "min", "p5", "p25", "median", "p75", "p95", "max")


@dataclass(frozen=True)
class FigureTables:
    """Public tables needed by ASR uncertainty figures."""

    identity: pd.DataFrame
    summary: pd.DataFrame
    cumulative_identity: pd.DataFrame
    cumulative_summary: pd.DataFrame


def read_figure_tables(*, context: FigureContext, include_cumulative: bool = True) -> FigureTables:
    """Read public identity and summary tables once for ASR figures."""
    identity = read_uncertainty_table(
        path=context.paths.public_row_identity,
        output_format=context.output_format,
    )
    summary = read_uncertainty_table(
        path=context.paths.summary_stats_runs,
        output_format=context.output_format,
    )
    if _identity_is_dynamic_ar6(identity):
        identity[MODEL_SCENARIO_SAMPLING_METHOD_COLUMN] = context.dynamic_cc_sampling_method
        summary[MODEL_SCENARIO_SAMPLING_METHOD_COLUMN] = context.dynamic_cc_sampling_method
        cumulative_identity = (
            read_uncertainty_table(
                path=context.paths.cumulative_row_identity,
                output_format=context.output_format,
            )
            if include_cumulative
            else identity.iloc[0:0].copy()
        )
        cumulative_summary = (
            read_uncertainty_table(
                path=context.paths.cumulative_summary_stats_runs,
                output_format=context.output_format,
            )
            if include_cumulative
            else summary.iloc[0:0].copy()
        )
        if include_cumulative:
            cumulative_identity[MODEL_SCENARIO_SAMPLING_METHOD_COLUMN] = (
                context.dynamic_cc_sampling_method
            )
            cumulative_summary[MODEL_SCENARIO_SAMPLING_METHOD_COLUMN] = (
                context.dynamic_cc_sampling_method
            )
    else:
        cumulative_identity = identity.iloc[0:0].copy()
        cumulative_summary = summary.iloc[0:0].copy()
    return FigureTables(
        identity=identity,
        summary=summary,
        cumulative_identity=cumulative_identity,
        cumulative_summary=cumulative_summary,
    )


def _identity_is_dynamic_ar6(identity: pd.DataFrame) -> bool:
    values = {
        str(value).strip()
        for value in identity["cc_type"].dropna().astype(str).tolist()
        if str(value).strip()
    }
    return values == {DYNAMIC_AR6_CC_TYPE}


def prepared_summary_rows(*, context: FigureContext, summary: pd.DataFrame) -> pd.DataFrame:
    """Return ASR summary rows with common plot columns."""
    rows = _filter_requested_years(frame=_metric_rows(summary, ASR_VALUE_METRIC), context=context)
    rows = _visible_asr_summary_rows(rows)
    rows = _asr_value_bound_rows(rows)
    rows["fu_code"] = context.fu_code
    return _attach_summary_common_columns(rows)


def prepared_frequency_rows(*, context: FigureContext, summary: pd.DataFrame) -> pd.DataFrame:
    """Return frequency of no-transgression summary rows."""
    rows = _filter_requested_years(
        frame=_metric_rows(summary, ASR_FREQUENCY_OF_NO_TRANSGRESSION_METRIC),
        context=context,
    )
    rows = _asr_value_bound_rows(rows)
    rows["fu_code"] = context.fu_code
    return _attach_summary_common_columns(rows)


def prepared_cumulative_frequency_rows(
    *,
    context: FigureContext,
    cumulative_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return cumulative frequency of no-transgression summary rows."""
    rows = _metric_rows(
        cumulative_summary,
        ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
    )
    rows = _asr_value_bound_rows(rows)
    rows["fu_code"] = context.fu_code
    return _attach_summary_common_columns(rows)


def prepared_identity_rows(*, context: FigureContext, identity: pd.DataFrame) -> pd.DataFrame:
    """Return public row identity with common ASR plot columns."""
    rows = _filter_requested_years(frame=identity, context=context)
    rows = _asr_value_bound_rows(rows)
    rows["fu_code"] = context.fu_code
    return attach_common_columns(rows)


def prepared_cumulative_identity_rows(
    *,
    context: FigureContext,
    cumulative_identity: pd.DataFrame,
) -> pd.DataFrame:
    """Return cumulative ASR identity rows with common plot columns."""
    rows = _asr_value_bound_rows(cumulative_identity)
    rows["fu_code"] = context.fu_code
    return attach_common_columns(rows)


def attach_dynamic_pair_counts(
    *,
    summary_rows: pd.DataFrame,
    identity_rows: pd.DataFrame,
    context: FigureContext,
    include_method_axis: bool,
) -> pd.DataFrame:
    """Attach dynamic AR6 model-scenario pair counts to public summary rows."""
    key_columns = _dynamic_pair_key_columns(
        rows=identity_rows,
        context=context,
        include_method_axis=include_method_axis,
    )
    pair_counts = _dynamic_pair_counts(
        rows=identity_rows,
        context=context,
        key_columns=key_columns,
    )
    out = summary_rows.copy()
    category_scopes = _dynamic_category_scopes_by_key(
        rows=identity_rows,
        context=context,
        key_columns=key_columns,
    )
    if category_scopes:
        out[AR6_CATEGORY_SCOPE_COLUMN] = [
            category_scopes[_row_key(row=row, columns=key_columns)]
            for _index, row in out.iterrows()
        ]
    out[MODEL_SCENARIO_PAIR_COUNT_COLUMN] = [
        pair_counts[_row_key(row=row, columns=key_columns)] for _index, row in out.iterrows()
    ]
    return out


def value_rows_from_runs(
    *,
    context: FigureContext,
    identity_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Attach selected ASR run distributions to public identity rows."""
    public_ids = _public_ids(identity_rows)
    out = identity_rows.copy()
    if context.run_layout == "sparse_selected_rows":
        values = collect_sparse_run_values(context=context, public_row_ids=public_ids)
        out[VALUE_ARRAY_COLUMN] = [
            values.get(int(public_id), np.empty(0, dtype=np.float64))
            for public_id in out["public_row_id"].tolist()
        ]
        return drop_empty_value_rows(rows=out)
    values = collect_compact_run_values(context=context, public_row_ids=public_ids)
    out[VALUE_ARRAY_COLUMN] = [
        values[int(public_id)] for public_id in out["public_row_id"].tolist()
    ]
    return out


def cumulative_value_rows_from_runs(
    *,
    context: FigureContext,
    cumulative_identity_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Attach selected cumulative ASR run distributions to period identity rows."""
    public_ids = _public_ids(cumulative_identity_rows)
    values = collect_selected_compact_run_values(
        path=context.paths.cumulative_runs,
        output_format=context.output_format,
        public_row_ids=public_ids,
        stop_run_index=int(context.manifest.completed_runs),
    )
    out = cumulative_identity_rows.copy()
    out[VALUE_ARRAY_COLUMN] = [values[int(public_id)] for public_id in out["public_row_id"]]
    return out


def collapsed_value_rows(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    include_method_axis: bool,
) -> pd.DataFrame:
    """Collapse run arrays across active sampled axes for visible identities."""
    drop_columns = _sampled_axis_drop_columns(
        context=context,
        include_method_axis=include_method_axis,
    )
    row_owned_ssp = ASOCC_SSP_SCENARIO_COLUMN in rows.columns and (
        "__figure_ssp_scope" in rows.columns or "ar6_cc_ssp_scenario" in rows.columns
    )
    if row_owned_ssp:
        drop_columns.add(ASOCC_SSP_SCENARIO_COLUMN)
    key_columns = [
        column
        for column in rows.columns
        if column not in {*drop_columns, "public_row_id", VALUE_ARRAY_COLUMN}
    ]
    threshold_values = _max_threshold_series(
        rows,
        threshold_lookup=_max_threshold_lookup(rows),
    )
    has_dynamic_ar6 = _contains_dynamic_ar6(rows)
    only_dynamic_ar6 = _contains_only_dynamic_ar6(rows) if has_dynamic_ar6 else False
    dynamic_pair_columns = (
        _dynamic_pair_count_columns(frame=rows, context=context) if has_dynamic_ar6 else []
    )
    records = []
    for _key, group in rows.groupby(key_columns, dropna=False, sort=True):
        arrays = [np.asarray(values, dtype=np.float64) for values in group[VALUE_ARRAY_COLUMN]]
        first_row = group.iloc[0]
        payload = {column: first_row[column] for column in key_columns}
        if row_owned_ssp:
            payload[ASOCC_SSP_SCENARIO_COLUMN] = _collapsed_row_owned_ssp(group=group)
        if ASOCC_TIME_ROUTE_PUBLIC_COLUMN in group.columns:
            payload[ASOCC_TIME_ROUTE_PUBLIC_COLUMN] = collapse_asocc_time_route(
                group[ASOCC_TIME_ROUTE_PUBLIC_COLUMN].tolist()
            )
        if has_dynamic_ar6:
            pair_count = _dynamic_model_scenario_pair_count(
                group=group,
                pair_columns=dynamic_pair_columns,
                group_is_dynamic=only_dynamic_ar6,
            )
            if pair_count is not None:
                payload[MODEL_SCENARIO_PAIR_COUNT_COLUMN] = pair_count
        threshold = _group_max_threshold(group, threshold_values=threshold_values)
        if threshold is not None:
            payload["__asr_max_threshold"] = threshold
            category_scope = _dynamic_category_scope(group=group, context=context)
            if category_scope:
                payload[AR6_CATEGORY_SCOPE_COLUMN] = category_scope
        payload[VALUE_ARRAY_COLUMN] = np.concatenate(arrays)
        records.append(payload)
    return pd.DataFrame.from_records(records)


def summary_rows_from_collapsed_values(rows: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics from rows that already carry collapsed run arrays."""
    records = []
    for _index, row in rows.iterrows():
        values = np.asarray(row[VALUE_ARRAY_COLUMN], dtype=np.float64)
        payload = row.drop(labels=[VALUE_ARRAY_COLUMN]).to_dict()
        payload.update(summary_stats(values))
        records.append(payload)
    return pd.DataFrame.from_records(records)


def drop_empty_value_rows(*, rows: pd.DataFrame) -> pd.DataFrame:
    """Return run based rows with at least one selected value."""
    mask = rows[VALUE_ARRAY_COLUMN].map(lambda values: len(values) > 0)
    return rows.loc[mask].reset_index(drop=True)


def collect_compact_run_values(
    *,
    context: FigureContext,
    public_row_ids: Iterable[int],
) -> dict[int, np.ndarray]:
    """Collect selected compact ASR run matrix columns by public row id."""
    return collect_selected_compact_run_values(
        path=context.paths.public_runs,
        output_format=context.output_format,
        public_row_ids=public_row_ids,
        stop_run_index=int(context.manifest.completed_runs),
    )


def collect_sparse_run_values(
    *,
    context: FigureContext,
    public_row_ids: Iterable[int],
) -> dict[int, np.ndarray]:
    """Collect selected sparse ASR run rows by public row id."""
    return collect_selected_sparse_run_values(
        path=context.paths.public_runs,
        output_format=context.output_format,
        public_row_ids=public_row_ids,
        stop_run_index=int(context.manifest.completed_runs),
    )


def summary_stats(values: np.ndarray) -> dict[str, float]:
    """Return figure summary statistics for one ASR run distribution."""
    numeric = np.asarray(values, dtype=np.float64)
    p5, p25, median, p75, p95 = np.nanpercentile(numeric, [5, 25, 50, 75, 95])
    return {
        "mean": float(np.nanmean(numeric)),
        "std": float(np.nanstd(numeric)),
        "min": float(np.nanmin(numeric)),
        "p5": float(p5),
        "p25": float(p25),
        "median": float(median),
        "p75": float(p75),
        "p95": float(p95),
        "max": float(np.nanmax(numeric)),
    }


def _visible_asr_summary_rows(rows: pd.DataFrame) -> pd.DataFrame:
    numeric = rows.loc[:, list(SUMMARY_STAT_COLUMNS)].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    return rows.loc[np.isfinite(values).any(axis=1)].copy()


def _metric_rows(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    return summary.loc[summary[ASR_SUMMARY_METRIC_COLUMN].astype(str).eq(metric)].copy()


def _attach_summary_common_columns(rows: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for scope, group in rows.groupby(ASR_SUMMARY_SCOPE_COLUMN, dropna=False, sort=False):
        scoped = group.copy()
        if str(scope) == ASR_SUMMARY_SCOPE_INTER_METHOD:
            scoped = scoped.drop(
                columns=[
                    column
                    for column in ("l1_l2_method", "l1_method", "l2_method")
                    if column in scoped.columns
                ]
            )
        pieces.append(attach_common_columns(scoped))
    return pd.concat(pieces, ignore_index=True, sort=False)


def _sampled_axis_drop_columns(
    *,
    context: FigureContext,
    include_method_axis: bool,
) -> set[str]:
    active = set(context.active_sources)
    dropped = set()
    if _has_source(active, ASOCC_REFERENCE_YEAR_SOURCE):
        dropped.add("reference_year")
    if _has_source(active, ASOCC_PROJECTION_SOURCE):
        dropped.add("l2_reuse_year")
    if _has_source(active, AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE):
        dropped.update({"cc_model", "cc_scenario"})
    if context.dynamic_category_uncertainty_active:
        dropped.add("cc_category")
    if not include_method_axis:
        dropped.update({"__method", "l1_l2_method", "l1_method", "l2_method"})
        dropped.add(ASOCC_TIME_ROUTE_PUBLIC_COLUMN)
    return dropped


def _has_source(active: set[str], source: str) -> bool:
    return source in active or any(name.endswith(source) for name in active)


def _filter_requested_years(*, frame: pd.DataFrame, context: FigureContext) -> pd.DataFrame:
    years = pd.Series(pd.to_numeric(frame["year"], errors="raise"), copy=False).astype(int)
    return frame.loc[years.isin(list(context.requested_years))].copy()


def _asr_value_bound_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if "cc_bound" not in rows.columns:
        out = rows.copy()
        out["__asr_max_threshold"] = np.nan
        return out
    bounds = {str(value).strip() for value in rows["cc_bound"].dropna().astype(str)}
    out = rows.copy()
    if {"min_cc", "max_cc"}.issubset(bounds):
        out = out.loc[out["cc_bound"].astype(str).eq("min_cc")].copy()
        out["cc_bound"] = "both"
    out["__asr_max_threshold"] = _max_threshold_series(
        out,
        threshold_lookup=_max_threshold_lookup(out),
    )
    return out


def _group_max_threshold(
    group: pd.DataFrame,
    *,
    threshold_values: pd.Series,
) -> float | None:
    values = pd.Series(threshold_values.loc[group.index], copy=False).dropna()
    if values.empty:
        return None
    return float(values.max())


def _max_threshold_series(
    rows: pd.DataFrame,
    *,
    threshold_lookup: dict[tuple[str, str], float],
) -> pd.Series:
    if not threshold_lookup:
        return pd.Series(np.nan, index=rows.index, dtype="float64")
    bounds = pd.Series(rows["cc_bound"], copy=False).astype("string").str.strip()
    values: list[float] = []
    for bound, method, impact in zip(
        bounds.tolist(),
        rows["lcia_method"].tolist(),
        rows["impact"].tolist(),
        strict=True,
    ):
        threshold = (
            threshold_lookup.get((str(method).strip(), str(impact).strip()))
            if str(bound).strip() in {"both", "max_cc"}
            else None
        )
        values.append(float("nan") if threshold is None else threshold)
    return pd.Series(values, index=rows.index, dtype="float64")


def _max_threshold_lookup(rows: pd.DataFrame) -> dict[tuple[str, str], float]:
    if "cc_bound" not in rows.columns:
        return {}
    bounds = pd.Series(rows["cc_bound"], copy=False).astype("string").str.strip()
    candidate_mask = bounds.isin(["both", "max_cc"])
    if not bool(candidate_mask.any()):
        return {}
    candidates = rows.loc[candidate_mask, ["lcia_method", "impact"]].drop_duplicates()
    metadata_by_method: dict[str, LCIAMetadata] = {}
    lookup: dict[tuple[str, str], float] = {}
    for row in candidates.itertuples(index=False, name=None):
        method = str(row[0]).strip()
        impact = str(row[1]).strip()
        metadata = metadata_by_method.get(method)
        if metadata is None:
            metadata = load_lcia_metadata(method)
            metadata_by_method[method] = metadata
        ratio = metadata.ratios.get(impact)
        if ratio is not None and float(ratio) > 1.0:
            lookup[(method, impact)] = float(ratio)
    return lookup


def _public_ids(frame: pd.DataFrame) -> list[int]:
    return [int(value) for value in frame["public_row_id"].tolist()]


def _contains_dynamic_ar6(frame: pd.DataFrame) -> bool:
    values = pd.Series(frame["cc_type"], copy=False).astype("string").str.strip()
    return bool(values.eq(DYNAMIC_AR6_CC_TYPE).any())


def _contains_only_dynamic_ar6(frame: pd.DataFrame) -> bool:
    values = pd.Series(frame["cc_type"], copy=False).dropna().astype(str).str.strip()
    visible = {value for value in values.tolist() if value}
    return visible == {DYNAMIC_AR6_CC_TYPE}


def _collapsed_row_owned_ssp(*, group: pd.DataFrame) -> object:
    values = pd.Series(group[ASOCC_SSP_SCENARIO_COLUMN], copy=False)
    visible = [
        str(value).strip().upper()
        for value in values.tolist()
        if not is_display_missing(value) and str(value).strip()
    ]
    return next(iter(sorted(set(visible))), pd.NA)


def _dynamic_pair_key_columns(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    include_method_axis: bool,
) -> list[str]:
    drop_columns = _sampled_axis_drop_columns(
        context=context,
        include_method_axis=include_method_axis,
    )
    excluded = {
        *drop_columns,
        "public_row_id",
        "year",
        MODEL_SCENARIO_PAIR_COUNT_COLUMN,
        MODEL_SCENARIO_SAMPLING_METHOD_COLUMN,
    }
    return [column for column in rows.columns if column not in excluded]


def _dynamic_pair_counts(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    key_columns: list[str],
) -> dict[tuple[str, ...], int]:
    pair_columns = _dynamic_pair_count_columns(frame=rows, context=context)
    group_is_dynamic = _contains_only_dynamic_ar6(rows)
    return {
        _row_key(row=pd.Series(group.iloc[0], copy=False), columns=key_columns): int(
            _dynamic_model_scenario_pair_count(
                group=group,
                pair_columns=pair_columns,
                group_is_dynamic=group_is_dynamic,
            )
            or 0
        )
        for _key, group in rows.groupby(key_columns, dropna=False, sort=True)
    }


def _dynamic_category_scopes_by_key(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    key_columns: list[str],
) -> dict[tuple[str, ...], str]:
    if not context.dynamic_category_uncertainty_active:
        return {}
    return {
        _row_key(row=pd.Series(group.iloc[0], copy=False), columns=key_columns): scope
        for _key, group in rows.groupby(key_columns, dropna=False, sort=True)
        if (scope := _dynamic_category_scope(group=group, context=context))
    }


def _dynamic_model_scenario_pair_count(
    *,
    group: pd.DataFrame,
    pair_columns: list[str],
    group_is_dynamic: bool,
) -> int | None:
    if not group_is_dynamic and not _contains_only_dynamic_ar6(group):
        return None
    if not pair_columns:
        return None
    return int(group.loc[:, pair_columns].drop_duplicates(ignore_index=True).shape[0])


def _dynamic_pair_count_columns(*, frame: pd.DataFrame, context: FigureContext) -> list[str]:
    pair_columns = ["cc_model", "cc_scenario"]
    if context.dynamic_category_uncertainty_active:
        pair_columns.insert(0, "cc_category")
    return [column for column in pair_columns if column in frame.columns]


def _dynamic_category_scope(*, group: pd.DataFrame, context: FigureContext) -> str:
    if not context.dynamic_category_uncertainty_active or "cc_category" not in group.columns:
        return ""
    values = pd.Series(group["cc_category"], copy=False).dropna().astype(str).str.strip()
    values = values.loc[values.ne("") & ~values.str.lower().isin(["nan", "none", "nat"])].tolist()
    return category_scope_label(values)


def _row_key(*, row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple(_key_value(row[column]) for column in columns)


def _key_value(value: object) -> str:
    if is_display_missing(value):
        return "<missing>"
    return str(value)

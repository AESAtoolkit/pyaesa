"""Public table readers for aSoCC uncertainty figures."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from collections.abc import Iterable

from pyaesa.asocc.uncertainty.figures.scope_planner import (
    METHOD_COLUMNS,
    VALUE_ARRAY_COLUMN,
    FigureContext,
    attach_common_plot_columns,
)
from pyaesa.asocc.uncertainty.schema.public_rows import ASOCC_UNCERTAINTY_CSV_DTYPES
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.runtime.scenario.columns import ASOCC_TIME_ROUTE_PUBLIC_COLUMN
from pyaesa.shared.runtime.scenario.time_routes import collapse_asocc_time_route
from pyaesa.shared.tabular.scalars import is_display_missing
from pyaesa.shared.figures.uncertainty_run_values import (
    collect_selected_compact_run_values,
    collect_selected_sparse_run_values,
)
from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table


@dataclass(frozen=True)
class FigureTables:
    """Public tables needed by the aSoCC uncertainty figure owner."""

    identity: pd.DataFrame
    summary: pd.DataFrame


def read_figure_tables(*, context: FigureContext, include_summary: bool = True) -> FigureTables:
    """Read public identity and summary tables once for figure planning."""
    identity = read_uncertainty_table(
        path=context.paths.public_row_identity,
        output_format=context.output_format,
        csv_dtypes=ASOCC_UNCERTAINTY_CSV_DTYPES,
    )
    summary = (
        read_uncertainty_table(
            path=context.paths.summary_stats_runs,
            output_format=context.output_format,
            csv_dtypes=ASOCC_UNCERTAINTY_CSV_DTYPES,
        )
        if include_summary
        else pd.DataFrame()
    )
    return FigureTables(identity=identity, summary=summary)


def prepared_summary_rows(*, context: FigureContext, summary: pd.DataFrame) -> pd.DataFrame:
    """Return summary rows with plot owned helper columns."""
    rows = summary.copy()
    rows = _filter_requested_years(frame=rows, context=context)
    return attach_common_plot_columns(frame=rows, context=context)


def deterministic_rows_from_summary(
    *,
    context: FigureContext,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return deterministic compatible rows from no source uncertainty summary."""
    rows = prepared_summary_rows(context=context, summary=summary)
    return deterministic_mean_rows(rows=rows)


def deterministic_mean_rows(*, rows: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic compatible rows whose visible value is the mean."""
    out = rows.copy()
    out["value"] = pd.to_numeric(out["mean"], errors="raise")
    return out


def prepared_identity_rows(*, context: FigureContext, identity: pd.DataFrame) -> pd.DataFrame:
    """Return public row identity with plot owned helper columns."""
    rows = _filter_requested_years(frame=identity, context=context)
    return attach_common_plot_columns(frame=rows, context=context)


def violin_rows_from_compact_runs(
    *,
    context: FigureContext,
    identity_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Attach compact run distributions to one public identity frame."""
    public_ids = _public_ids(identity_rows)
    values = collect_compact_run_values(context=context, public_row_ids=public_ids)
    return _attach_value_arrays(frame=identity_rows, values=values)


def violin_rows_from_sparse_runs(
    *,
    context: FigureContext,
    identity_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Attach sparse selected run distributions to one public identity frame."""
    public_ids = _public_ids(identity_rows)
    values = collect_sparse_run_values(context=context, public_row_ids=public_ids)
    return _attach_value_arrays(frame=identity_rows, values=values)


def collapsed_violin_rows(
    *,
    rows: pd.DataFrame,
    drop_columns: tuple[str, ...] = (
        *METHOD_COLUMNS,
        "public_row_id",
        "__method",
        "reference_year",
        "l2_reuse_year",
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
    ),
) -> pd.DataFrame:
    """Collapse inter-method run arrays into one method invariant identity."""
    dropped = set(drop_columns)
    row_owned_ssp = ASOCC_SSP_SCENARIO_COLUMN in rows.columns
    if row_owned_ssp:
        dropped.add(ASOCC_SSP_SCENARIO_COLUMN)
    value_columns = [column for column in rows.columns if column not in dropped]
    key_columns = [column for column in value_columns if column != VALUE_ARRAY_COLUMN]
    records = []
    for _key, group in rows.groupby(key_columns, dropna=False, sort=True):
        arrays = [np.asarray(values, dtype=np.float64) for values in group[VALUE_ARRAY_COLUMN]]
        merged = np.concatenate(arrays)
        payload = {column: group.iloc[0][column] for column in key_columns}
        if row_owned_ssp:
            payload[ASOCC_SSP_SCENARIO_COLUMN] = _collapsed_row_owned_ssp(group=group)
        payload[ASOCC_TIME_ROUTE_PUBLIC_COLUMN] = collapse_asocc_time_route(
            group[ASOCC_TIME_ROUTE_PUBLIC_COLUMN].tolist()
        )
        payload[VALUE_ARRAY_COLUMN] = merged
        records.append(payload)
    return pd.DataFrame.from_records(records)


def drop_empty_value_rows(*, rows: pd.DataFrame) -> pd.DataFrame:
    """Return run based figure rows that have at least one selected value."""
    mask = rows[VALUE_ARRAY_COLUMN].map(lambda values: len(values) > 0)
    return rows.loc[mask].reset_index(drop=True)


def _collapsed_row_owned_ssp(*, group: pd.DataFrame) -> object:
    values = pd.Series(group[ASOCC_SSP_SCENARIO_COLUMN], copy=False)
    visible = [
        str(value).strip().upper()
        for value in values.tolist()
        if not is_display_missing(value) and str(value).strip()
    ]
    return next(iter(sorted(set(visible))), pd.NA)


def collect_compact_run_values(
    *,
    context: FigureContext,
    public_row_ids: Iterable[int],
) -> dict[int, np.ndarray]:
    """Collect selected compact run matrix columns by public row id."""
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
    """Collect selected sparse run rows by public row id."""
    return collect_selected_sparse_run_values(
        path=context.paths.public_runs,
        output_format=context.output_format,
        public_row_ids=public_row_ids,
        stop_run_index=int(context.manifest.completed_runs),
    )


def _filter_requested_years(*, frame: pd.DataFrame, context: FigureContext) -> pd.DataFrame:
    years = pd.Series(pd.to_numeric(frame["year"], errors="raise"), copy=False).astype(int)
    return frame.loc[years.isin(list(context.requested_years))].copy()


def _public_ids(frame: pd.DataFrame) -> list[int]:
    return [int(value) for value in frame["public_row_id"].tolist()]


def _attach_value_arrays(*, frame: pd.DataFrame, values: dict[int, np.ndarray]) -> pd.DataFrame:
    out = frame.copy()
    out[VALUE_ARRAY_COLUMN] = [
        values.get(int(public_id), np.empty(0, dtype=np.float64))
        for public_id in out["public_row_id"].tolist()
    ]
    return out

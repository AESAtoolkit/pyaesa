"""Public table readers for IO-LCA uncertainty figures."""

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import pandas as pd

from pyaesa.io_lca.uncertainty.figures.scope_planner import (
    VALUE_ARRAY_COLUMN,
    FigureContext,
)
from pyaesa.shared.figures.uncertainty_run_values import collect_selected_compact_run_values
from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table


@dataclass(frozen=True)
class FigureTables:
    """Public tables needed by the IO-LCA uncertainty figure owner."""

    identity: pd.DataFrame
    summary: pd.DataFrame


def read_figure_tables(*, context: FigureContext, include_summary: bool = True) -> FigureTables:
    """Read public identity and summary tables once for figure planning."""
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
    return FigureTables(identity=identity, summary=summary)


def prepared_summary_rows(*, context: FigureContext, summary: pd.DataFrame) -> pd.DataFrame:
    """Return summary rows filtered to requested figure years."""
    return _filter_requested_years(frame=summary, context=context)


def prepared_identity_rows(*, context: FigureContext, identity: pd.DataFrame) -> pd.DataFrame:
    """Return identity rows filtered to requested figure years."""
    return _filter_requested_years(frame=identity, context=context)


def violin_rows_from_compact_runs(
    *,
    context: FigureContext,
    identity_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Attach compact run distributions to one public identity frame."""
    public_ids = _public_ids(identity_rows)
    values = collect_compact_run_values(context=context, public_row_ids=public_ids)
    out = identity_rows.copy()
    out[VALUE_ARRAY_COLUMN] = [
        values.get(int(public_id), np.empty(0, dtype=np.float64))
        for public_id in out["public_row_id"].tolist()
    ]
    return out


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


def _filter_requested_years(*, frame: pd.DataFrame, context: FigureContext) -> pd.DataFrame:
    years = pd.Series(pd.to_numeric(frame["year"], errors="raise"), copy=False).astype(int)
    return frame.loc[years.isin(list(context.requested_years))].copy()


def _public_ids(frame: pd.DataFrame) -> list[int]:
    return [int(value) for value in frame["public_row_id"].tolist()]

"""Projection uncertainty for final aSoCC rows."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyaesa.asocc.uncertainty.inputs.deterministic_rows import (
    ASOCC_VALUE_COLUMN,
    LoadedAsoccFinalRows,
)
from pyaesa.asocc.uncertainty.inputs.external_rows import external_method_row_mask
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch
from pyaesa.shared.uncertainty_assessment.monte_carlo.random_streams import uniform_by_run_index
from pyaesa.asocc.uncertainty.io.source_methods import SourceMethodRow
from pyaesa.asocc.uncertainty.sources.names import PROJECTION_SOURCE

PROJECTION_RANDOM_STREAM = "asocc.projection.l2_reuse_year"
_REUSE_YEAR_COLUMN = "l2_reuse_year"


@dataclass(frozen=True)
class ProjectionPlan:
    """Projection sampled final row plan."""

    public_columns: tuple[str, ...]
    passthrough_rows: pd.DataFrame
    sampled_rows: pd.DataFrame
    l2_reuse_years: tuple[int, ...]
    candidate_values: np.ndarray
    source_method_row: SourceMethodRow


def build_projection_plan(
    *,
    loaded: LoadedAsoccFinalRows,
    external_method_labels: tuple[str, ...] = (),
) -> ProjectionPlan:
    """Build projection sampling plan from final deterministic rows."""
    rows = loaded.rows.reset_index(drop=True)
    reuse = pd.Series(rows.loc[:, _REUSE_YEAR_COLUMN], copy=False)
    external_rows = external_method_row_mask(frame=rows, method_labels=external_method_labels)
    # External methods carry user supplied aSoCC values. Projection uncertainty
    # never changes them; only inter-method and reference year uncertainty can
    # sample external method rows.
    projected = rows.loc[reuse.notna() & ~external_rows].reset_index(drop=True)
    passthrough = rows.loc[reuse.isna() | external_rows].reset_index(drop=True)
    group_columns = [
        column for column in rows.columns if column not in {ASOCC_VALUE_COLUMN, _REUSE_YEAR_COLUMN}
    ]
    sampled_rows = projected.drop_duplicates(group_columns, ignore_index=True)
    sampled_rows = sampled_rows.loc[:, [*group_columns, ASOCC_VALUE_COLUMN]]
    indexed = projected.merge(
        sampled_rows.loc[:, group_columns].assign(
            _projection_group=np.arange(len(sampled_rows), dtype=np.int64)
        ),
        on=group_columns,
        how="left",
        sort=False,
    )
    indexed["_l2_reuse_year_int"] = pd.Series(
        pd.to_numeric(pd.Series(indexed.loc[:, _REUSE_YEAR_COLUMN], copy=False), errors="raise"),
        index=indexed.index,
    ).astype("int64")
    l2_reuse_years = tuple(sorted(int(value) for value in indexed["_l2_reuse_year_int"].unique()))
    pivot = indexed.pivot(
        index="_projection_group",
        columns="_l2_reuse_year_int",
        values=ASOCC_VALUE_COLUMN,
    ).sort_index()
    candidate_values = pivot.loc[:, list(l2_reuse_years)].to_numpy(dtype=np.float64).T
    return ProjectionPlan(
        public_columns=tuple(sampled_rows.columns),
        passthrough_rows=passthrough,
        sampled_rows=sampled_rows.reset_index(drop=True),
        l2_reuse_years=l2_reuse_years,
        candidate_values=candidate_values,
        source_method_row=projection_source_method_row(
            loaded=loaded,
            l2_reuse_years=l2_reuse_years,
        ),
    )


def projection_uncertainty_has_targets(
    *,
    loaded: LoadedAsoccFinalRows,
    external_method_labels: tuple[str, ...] = (),
) -> bool:
    """Return whether selected rows expose at least two pyaesa owned L2 reuse year candidates."""
    rows = loaded.rows
    if _REUSE_YEAR_COLUMN not in rows.columns:
        return False
    reuse = pd.Series(rows.loc[:, _REUSE_YEAR_COLUMN], copy=False)
    external_rows = external_method_row_mask(frame=rows, method_labels=external_method_labels)
    candidates: pd.Series = pd.Series(
        pd.to_numeric(reuse.loc[reuse.notna() & ~external_rows], errors="raise"),
        copy=False,
    )
    return bool(candidates.nunique(dropna=True) >= 2)


def projection_public_row_template(*, plan: ProjectionPlan) -> pd.DataFrame:
    """Return the stable final public row template for projection uncertainty."""
    pieces = [
        plan.passthrough_rows.loc[:, plan.public_columns],
        plan.sampled_rows.loc[:, plan.public_columns],
    ]
    return pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)


def collapse_projection_public_template(*, template: pd.DataFrame) -> pd.DataFrame:
    """Return the public template after collapsing the sampled L2 reuse year axis."""
    if _REUSE_YEAR_COLUMN not in template.columns:
        return template
    rows = template.reset_index(drop=True).copy()
    reuse = pd.Series(rows.loc[:, _REUSE_YEAR_COLUMN], copy=False)
    passthrough = rows.loc[reuse.isna()].copy()
    projected = rows.loc[reuse.notna()].copy()
    group_columns = [
        column for column in rows.columns if column not in {ASOCC_VALUE_COLUMN, _REUSE_YEAR_COLUMN}
    ]
    sampled_rows = projected.drop_duplicates(group_columns, ignore_index=True)
    sampled_rows = sampled_rows.loc[:, [*group_columns, ASOCC_VALUE_COLUMN]]
    return pd.concat(
        [passthrough.drop(columns=[_REUSE_YEAR_COLUMN]), sampled_rows],
        ignore_index=True,
    )


def sample_projection_indices(*, plan: ProjectionPlan, batch: RunBatch) -> np.ndarray:
    """Return sampled L2 reuse year candidate positions for one run batch."""
    return projection_indices_for_l2_reuse_years(
        plan=plan,
        l2_reuse_years=sample_projection_l2_reuse_years(plan=plan, batch=batch),
    )


def sample_projection_l2_reuse_years(
    *,
    plan: ProjectionPlan,
    batch: RunBatch,
    unit_values: np.ndarray | None = None,
) -> np.ndarray:
    """Return sampled L2 reuse year values for one run batch."""
    if not plan.l2_reuse_years:
        return np.zeros(batch.n_runs, dtype=np.int64)
    uniform = (
        np.asarray(unit_values, dtype=np.float64)
        if unit_values is not None
        else uniform_by_run_index(
            stream_name=PROJECTION_RANDOM_STREAM,
            run_indices=batch.run_indices(),
        )
    )
    indices = np.floor(uniform * len(plan.l2_reuse_years)).astype(np.int64)
    return np.array(plan.l2_reuse_years, dtype=np.int64)[indices]


def projection_indices_for_l2_reuse_years(
    *,
    plan: ProjectionPlan,
    l2_reuse_years: np.ndarray,
) -> np.ndarray:
    """Return plan candidate positions for sampled L2 reuse year values."""
    if not plan.l2_reuse_years:
        return np.zeros(len(l2_reuse_years), dtype=np.int64)
    return np.searchsorted(
        np.array(plan.l2_reuse_years, dtype=np.int64),
        l2_reuse_years,
    ).astype(np.int64)


def projection_value_matrix_for_indices(
    *,
    plan: ProjectionPlan,
    batch: RunBatch,
    selected_indices: np.ndarray,
) -> np.ndarray:
    """Return projection value matrix for already selected L2 reuse year positions."""
    parts: list[np.ndarray] = []
    if not plan.passthrough_rows.empty:
        values = plan.passthrough_rows[ASOCC_VALUE_COLUMN].to_numpy(dtype="float64")
        parts.append(np.broadcast_to(values, (batch.n_runs, len(values))))
    if not plan.sampled_rows.empty:
        parts.append(plan.candidate_values[selected_indices, :])
    return np.concatenate(parts, axis=1)


def apply_projection_uncertainty_to_matrix(
    *,
    template: pd.DataFrame,
    values: np.ndarray,
    plan: ProjectionPlan,
    batch: RunBatch,
    selected_indices: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Collapse the sampled L2 reuse year axis on an existing compact value matrix."""
    rows = template.reset_index(drop=True).copy()
    rows["_input_position"] = np.arange(len(rows), dtype=np.int64)
    reuse = pd.Series(rows.loc[:, _REUSE_YEAR_COLUMN], copy=False)
    passthrough = rows.loc[reuse.isna()].copy()
    projected = rows.loc[reuse.notna()].copy()
    group_columns = [
        column
        for column in rows.columns
        if column not in {ASOCC_VALUE_COLUMN, _REUSE_YEAR_COLUMN, "_input_position"}
    ]
    output_template = collapse_projection_public_template(
        template=rows.drop(columns=["_input_position"])
    )
    sampled_rows = output_template.iloc[len(passthrough) :].reset_index(drop=True)
    output_positions = sampled_rows.loc[:, group_columns].assign(
        _output_position=np.arange(len(passthrough), len(output_template), dtype=np.int64)
    )
    indexed = projected.merge(output_positions, on=group_columns, how="left", sort=False)
    indexed["_l2_reuse_year_int"] = pd.Series(
        pd.to_numeric(pd.Series(indexed.loc[:, _REUSE_YEAR_COLUMN], copy=False), errors="raise"),
        index=indexed.index,
    ).astype("int64")
    output = np.empty((batch.n_runs, len(output_template)), dtype=np.float64)
    if not passthrough.empty:
        passthrough_positions = passthrough["_input_position"].to_numpy(dtype=np.int64)
        output[:, : len(passthrough_positions)] = values[:, passthrough_positions]
    if sampled_rows.empty:
        return output_template, output
    selected_l2_reuse_years = np.array(plan.l2_reuse_years, dtype=np.int64)[selected_indices]
    for l2_reuse_year in np.unique(selected_l2_reuse_years):
        run_positions = np.flatnonzero(selected_l2_reuse_years == int(l2_reuse_year))
        reuse_rows = indexed.loc[indexed["_l2_reuse_year_int"].eq(int(l2_reuse_year))]
        output_positions = reuse_rows["_output_position"].to_numpy(dtype=np.int64)
        input_positions = reuse_rows["_input_position"].to_numpy(dtype=np.int64)
        output[np.ix_(run_positions, output_positions)] = values[
            np.ix_(run_positions, input_positions)
        ]
    return output_template, output


def projection_source_method_row(
    *,
    loaded: LoadedAsoccFinalRows,
    l2_reuse_years: tuple[int, ...],
) -> SourceMethodRow:
    """Return the compact scientific log row for projection uncertainty."""
    return SourceMethodRow(
        source_component="asocc",
        source_name=PROJECTION_SOURCE,
        scope=str(loaded.base_asocc_args["fu_code"]),
        applied_bucket=loaded.final_bucket,
        year_min=min(loaded.requested_years),
        year_max=max(loaded.requested_years),
        distribution="discrete uniform over deterministic L2 reuse years",
        shared_random_variable="run_index",
        formula="sampled row = deterministic final aSoCC candidate selected by l2_reuse_year",
        notes=f"Candidate l2_reuse_year values: {';'.join(str(year) for year in l2_reuse_years)}",
    )

"""Period segmentation for one-pass AR6 CC uncertainty run dispatch."""

from typing import Any

import numpy as np
import pandas as pd

from pyaesa.ar6_cc.uncertainty.evaluation.summary_identity import (
    ar6_cc_summary_identity_groups,
)
from pyaesa.ar6_cc.uncertainty.runtime.models import AR6CCUncertaintyPlan
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    run_positions_in_window,
    sparse_public_row_group_index,
)
from pyaesa.shared.uncertainty_assessment.io.run_writers import SparseRunRows


def trajectory_segment(
    *,
    plan: AR6CCUncertaintyPlan,
    years: list[int],
    category_uncertainty: bool,
) -> dict[str, Any]:
    """Return one public row identity view over the full sampled plan."""
    requested_years = {int(year) for year in years}
    full_identity = plan.identity.copy()
    year_values = pd.Series(
        pd.to_numeric(pd.Series(full_identity.loc[:, "year"], copy=False), errors="raise"),
        index=full_identity.index,
    ).astype(int)
    mask = year_values.isin(sorted(requested_years))
    selected = full_identity.loc[mask].reset_index(drop=True)
    full_ids = pd.Series(
        pd.to_numeric(pd.Series(selected.loc[:, "public_row_id"], copy=False), errors="raise"),
        index=selected.index,
    ).to_numpy(dtype=np.int64, copy=False)
    selected["public_row_id"] = np.arange(len(selected), dtype=np.int64)
    summary_identity, public_row_groups = ar6_cc_summary_identity_groups(
        identity=selected,
        category_uncertainty=category_uncertainty,
    )
    plan_public_ids = pd.Series(
        pd.to_numeric(pd.Series(plan.identity.loc[:, "public_row_id"], copy=False), errors="raise")
    )
    full_to_segment = np.full(int(plan_public_ids.max()) + 1, -1, dtype=np.int64)
    full_to_segment[full_ids] = selected["public_row_id"].to_numpy(dtype=np.int64, copy=False)
    return {
        "identity": selected,
        "summary_identity": summary_identity,
        "public_row_groups": public_row_groups,
        "summary_group_index": sparse_public_row_group_index(public_row_groups=public_row_groups),
        "full_to_segment": full_to_segment,
        "period_segment": "study_period"
        if min(requested_years) == int(year_values.min())
        else "post_study_period",
    }


def budget_identity_and_segments(
    *,
    study: dict[str, Any],
    post: dict[str, Any] | None,
    category_uncertainty: bool,
) -> tuple[pd.DataFrame, tuple[dict[str, Any], ...]]:
    """Return cumulative budget identities and group mappings for all segments."""
    segments = [budget_segment(segment=study, category_uncertainty=category_uncertainty)]
    if post is not None:
        segments.append(budget_segment(segment=post, category_uncertainty=category_uncertainty))
    identity = pd.concat([segment["identity"] for segment in segments], ignore_index=True)
    identity.insert(0, "budget_row_id", np.arange(len(identity), dtype=np.int64))
    return identity.reset_index(drop=True), tuple(segments)


def budget_segment(
    *,
    segment: dict[str, Any],
    category_uncertainty: bool,
) -> dict[str, Any]:
    """Return cumulative budget grouping for one period segment."""
    identity = segment["identity"].copy()
    identity["period_segment"] = str(segment["period_segment"])
    excluded = {"public_row_id", "cc_model", "cc_scenario", "year"}
    if category_uncertainty:
        excluded.add("cc_category")
    columns = [column for column in identity.columns if column not in excluded]
    grouped = identity.groupby(columns, dropna=False, sort=False)["public_row_id"].agg(tuple)
    group_identity = grouped.index.to_frame(index=False).reset_index(drop=True)
    groups = tuple(tuple(int(value) for value in values) for values in grouped.tolist())
    return {
        "identity": group_identity,
        "group_index": _budget_group_index(public_row_groups=groups),
        "group_count": len(group_identity),
        "full_to_segment": segment["full_to_segment"],
    }


def remap_sparse_rows(*, rows: SparseRunRows, full_to_segment: np.ndarray) -> SparseRunRows:
    """Return sparse run rows for one period segment."""
    mapped = full_to_segment[rows.public_row_id]
    mask = mapped >= 0
    return SparseRunRows(
        run_index=rows.run_index[mask],
        public_row_id=mapped[mask],
        values=rows.values[mask],
        value_column=rows.value_column,
    )


def budget_matrix_from_full_sparse_rows(
    *,
    rows: SparseRunRows,
    run_indices: np.ndarray,
    segments: tuple[dict[str, Any], ...],
) -> np.ndarray:
    """Return cumulative budget run values for all period segments."""
    matrices = []
    for segment in segments:
        remapped = remap_sparse_rows(rows=rows, full_to_segment=segment["full_to_segment"])
        matrices.append(
            budget_matrix_from_sparse_rows(
                rows=remapped,
                run_indices=run_indices,
                public_row_group_index=segment["group_index"],
                group_count=int(segment["group_count"]),
            )
        )
    return np.concatenate(matrices, axis=1)


def budget_matrix_from_sparse_rows(
    *,
    rows: SparseRunRows,
    run_indices: np.ndarray,
    public_row_group_index: np.ndarray,
    group_count: int,
) -> np.ndarray:
    """Return cumulative budget runs from sparse yearly rows."""
    row_runs = run_positions_in_window(run_indices=run_indices, row_run_index=rows.run_index)
    row_groups = public_row_group_index[rows.public_row_id]
    values = np.zeros((len(run_indices), int(group_count)), dtype=np.float64)
    np.add.at(values, (row_runs, row_groups), rows.values)
    return values


def _budget_group_index(*, public_row_groups: tuple[tuple[int, ...], ...]) -> np.ndarray:
    max_public_row_id = max(public_row_id for group in public_row_groups for public_row_id in group)
    index = np.empty(max_public_row_id + 1, dtype=np.int64)
    for group_index, group in enumerate(public_row_groups):
        index[list(group)] = group_index
    return index

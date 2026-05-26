"""Inter-method branch value sampling for aSoCC uncertainty."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from pyaesa.asocc.uncertainty.engine.inter_method.execution import (
    InterMethodBranchExecution,
    InterMethodExecutionPlan,
)
from pyaesa.asocc.uncertainty.engine.inter_method.identity import (
    public_row_ids_for_branch,
)
from pyaesa.asocc.uncertainty.engine.monte_carlo.sampling import (
    compact_batch_inter_mrio_matches,
    sample_compact_batch,
)
from pyaesa.asocc.uncertainty.engine.evaluation.source_unit_intervals import (
    SourceUnitIntervalSamples,
)
from pyaesa.asocc.uncertainty.engine.evaluation.summary_identity import (
    summary_identity_groups,
)
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    sparse_public_row_group_index,
)
from pyaesa.asocc.uncertainty.lcia_support.sampling import LCIASharedUMatrix
from pyaesa.asocc.uncertainty.sources.inter_method import (
    INTER_METHOD_SOURCE,
    InterMethodPlan,
    sample_inter_method_labels,
)
from pyaesa.asocc.uncertainty.sources.inter_mrio import InterMrioPlan
from pyaesa.asocc.uncertainty.sources.inter_mrio import InterMrioInterpolationMatches
from pyaesa.asocc.uncertainty.sources.lcia import (
    LCIA_SOURCE,
    lcia_shared_u_for_plan,
)
from pyaesa.asocc.uncertainty.sources.projection import (
    PROJECTION_SOURCE,
    projection_indices_for_l2_reuse_years,
    sample_projection_l2_reuse_years,
)
from pyaesa.asocc.uncertainty.schema.public_rows import (
    ASOCC_PUBLIC_VALUE_COLUMN,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch
from pyaesa.shared.runtime.memory import memory_bounded_rows
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan
from pyaesa.shared.uncertainty_assessment.io.run_writers import SparseRunRows


@dataclass(frozen=True)
class SparseInterMethodBatch:
    """Selected inter-method run rows for one batch."""

    identity: pd.DataFrame
    run_indices: np.ndarray
    sparse_rows: SparseRunRows
    external_run_counts: dict[str, int]


@dataclass(frozen=True)
class _InterMethodSampleState:
    selected: np.ndarray
    projection_l2_reuse_years: np.ndarray | None
    lcia_shared_u: LCIASharedUMatrix | None


def sample_sparse_inter_method_batch(
    *,
    loaded,
    inter_method_plan: InterMethodPlan,
    execution_plan: InterMethodExecutionPlan,
    inter_mrio_plan: InterMrioPlan | None,
    batch,
    sources: SourceActivationPlan,
    identity: pd.DataFrame | None,
    external_render_offsets: dict[str, int] | None = None,
    inter_mrio_matches_by_branch: dict[str, InterMrioInterpolationMatches] | None = None,
) -> SparseInterMethodBatch:
    """Return selected inter-method rows for Monte Carlo sparse persistence."""
    run_indices = batch.run_indices()
    row_universe = execution_plan.row_universe
    public_identity = row_universe.identity if identity is None else identity
    sample_state = _inter_method_sample_state(
        inter_method_plan=inter_method_plan,
        execution_plan=execution_plan,
        batch=batch,
        source_units=None,
    )
    public_lcia_axis = row_universe.public_lcia_axis
    public_row_ids_by_branch = {
        branch.label: public_row_ids_for_branch(row_universe=row_universe, label=branch.label)
        for branch in execution_plan.branches
    }
    row_offsets, sparse_row_count = _sparse_row_offsets(
        selected=sample_state.selected,
        row_counts_by_branch={
            label: len(public_row_ids) for label, public_row_ids in public_row_ids_by_branch.items()
        },
    )
    sparse_run_index = np.empty(sparse_row_count, dtype=np.int64)
    sparse_public_row_id = np.empty(sparse_row_count, dtype=np.int64)
    sparse_values = np.empty(sparse_row_count, dtype=np.float64)
    external_counts: dict[str, int] = {}
    offsets = external_render_offsets or {}
    for branch in execution_plan.branches:
        run_positions = np.flatnonzero(sample_state.selected == branch.label)
        if run_positions.size == 0:
            continue
        external_run_indices_by_label = _external_run_indices_for_branch(
            branch=branch,
            offsets=offsets,
            run_count=int(run_positions.size),
        )
        if external_run_indices_by_label is not None:
            external_counts[branch.label] = int(run_positions.size)
        branch_batch, branch_projection_selection, branch_lcia_shared_u = _branch_state(
            branch=branch,
            batch=batch,
            run_positions=run_positions,
            sample_state=sample_state,
        )
        _branch_identity, _run_indices, branch_values = sample_compact_batch(
            loaded=branch.loaded,
            inter_mrio_plan=None if branch.external_plan.method_labels else inter_mrio_plan,
            lcia_plan=branch.lcia_plan,
            projection_plan=branch.projection_plan,
            batch=branch_batch,
            sources=sources,
            external_plan=branch.external_plan,
            projection_selection=branch_projection_selection,
            lcia_shared_u=branch_lcia_shared_u,
            source_units=None,
            external_run_indices_by_label=external_run_indices_by_label,
            lcia_public_axis=public_lcia_axis,
            inter_mrio_matches=_branch_inter_mrio_matches(
                branch=branch,
                matches_by_branch=inter_mrio_matches_by_branch,
            ),
        )
        _assign_sparse_branch_rows(
            run_index=sparse_run_index,
            public_row_id=sparse_public_row_id,
            values=sparse_values,
            row_offsets=row_offsets,
            run_positions=run_positions,
            branch_run_indices=branch_batch.run_indices(),
            branch_public_row_id=public_row_ids_by_branch[branch.label],
            branch_values=branch_values,
        )
    return SparseInterMethodBatch(
        identity=public_identity,
        run_indices=run_indices,
        sparse_rows=SparseRunRows(
            run_index=sparse_run_index,
            public_row_id=sparse_public_row_id,
            values=sparse_values,
            value_column=ASOCC_PUBLIC_VALUE_COLUMN,
        ),
        external_run_counts=external_counts,
    )


def sample_inter_method_summary_matrix_batch(
    *,
    loaded,
    inter_method_plan: InterMethodPlan,
    execution_plan: InterMethodExecutionPlan,
    inter_mrio_plan: InterMrioPlan | None,
    batch,
    sources: SourceActivationPlan,
    source_units: SourceUnitIntervalSamples,
    inter_mrio_matches_by_branch: dict[str, InterMrioInterpolationMatches] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return method collapsed inter-method values for Sobol analysis."""
    row_universe = execution_plan.row_universe
    full_identity = row_universe.identity
    summary_identity, public_row_groups = summary_identity_groups(
        identity=full_identity,
        sources=sources,
        inter_method_only=True,
    )
    public_lcia_axis = row_universe.public_lcia_axis
    public_group_index = sparse_public_row_group_index(public_row_groups=public_row_groups)
    output = np.full((batch.n_runs, len(public_row_groups)), np.nan, dtype=np.float64)
    sample_state = _inter_method_sample_state(
        inter_method_plan=inter_method_plan,
        execution_plan=execution_plan,
        batch=batch,
        source_units=source_units,
    )
    branches_by_label = {branch.label: branch for branch in execution_plan.branches}
    for label in dict.fromkeys(str(value) for value in sample_state.selected.tolist()):
        branch = branches_by_label[label]
        run_positions = np.flatnonzero(sample_state.selected == label)
        branch_batch, branch_projection_selection, branch_lcia_shared_u = _branch_state(
            branch=branch,
            batch=batch,
            run_positions=run_positions,
            sample_state=sample_state,
        )
        _branch_identity, _run_indices, branch_values = sample_compact_batch(
            loaded=branch.loaded,
            inter_mrio_plan=None if branch.external_plan.method_labels else inter_mrio_plan,
            lcia_plan=branch.lcia_plan,
            projection_plan=branch.projection_plan,
            batch=branch_batch,
            sources=sources,
            external_plan=branch.external_plan,
            projection_selection=branch_projection_selection,
            lcia_shared_u=branch_lcia_shared_u,
            source_units=_source_units_for_positions(
                source_units=source_units,
                positions=run_positions,
            ),
            lcia_public_axis=public_lcia_axis,
            inter_mrio_matches=_branch_inter_mrio_matches(
                branch=branch,
                matches_by_branch=inter_mrio_matches_by_branch,
            ),
        )
        _assign_branch_summary_values(
            output=output,
            run_positions=run_positions,
            public_group_ids=public_group_index[
                public_row_ids_for_branch(row_universe=row_universe, label=branch.label)
            ],
            branch_values=branch_values,
        )
    return summary_identity, output


def inter_method_inter_mrio_matches_by_branch(
    *,
    execution_plan: InterMethodExecutionPlan,
    inter_mrio_plan: InterMrioPlan | None,
) -> dict[str, InterMrioInterpolationMatches]:
    """Return reusable inter-MRIO row matches for inter-method branches."""
    if inter_mrio_plan is None:
        return {}
    matches: dict[str, InterMrioInterpolationMatches] = {}
    for branch in execution_plan.branches:
        if branch.external_plan.method_labels:
            continue
        branch_matches = compact_batch_inter_mrio_matches(
            loaded=branch.loaded,
            inter_mrio_plan=inter_mrio_plan,
            lcia_plan=branch.lcia_plan,
            projection_plan=branch.projection_plan,
        )
        matches[branch.label] = cast(InterMrioInterpolationMatches, branch_matches)
    return matches


def _inter_method_sample_state(
    *,
    inter_method_plan: InterMethodPlan,
    execution_plan: InterMethodExecutionPlan,
    batch,
    source_units: SourceUnitIntervalSamples | None,
) -> _InterMethodSampleState:
    selected = sample_inter_method_labels(
        plan=inter_method_plan,
        batch=batch,
        unit_values=None if source_units is None else source_units.values_for(INTER_METHOD_SOURCE),
    )
    projection_l2_reuse_years = (
        sample_projection_l2_reuse_years(
            plan=execution_plan.projection_plan,
            batch=batch,
            unit_values=None
            if source_units is None
            else source_units.values_for(PROJECTION_SOURCE),
        )
        if execution_plan.projection_plan is not None
        else None
    )
    lcia_shared_u = (
        lcia_shared_u_for_plan(
            plan=execution_plan.lcia_plan,
            batch=batch,
            unit_values=None if source_units is None else source_units.values_for(LCIA_SOURCE),
        )
        if execution_plan.lcia_plan is not None
        else None
    )
    return _InterMethodSampleState(
        selected=selected,
        projection_l2_reuse_years=projection_l2_reuse_years,
        lcia_shared_u=lcia_shared_u,
    )


def _branch_state(
    *,
    branch: InterMethodBranchExecution,
    batch,
    run_positions: np.ndarray,
    sample_state: _InterMethodSampleState,
) -> tuple[RunBatch, np.ndarray | None, LCIASharedUMatrix | None]:
    branch_batch = _selected_run_batch(batch=batch, run_positions=run_positions)
    projection_selection = (
        projection_indices_for_l2_reuse_years(
            plan=branch.projection_plan,
            l2_reuse_years=cast(np.ndarray, sample_state.projection_l2_reuse_years)[run_positions],
        )
        if branch.projection_plan is not None
        else None
    )
    lcia_shared_u = (
        LCIASharedUMatrix(
            key_positions=cast(LCIASharedUMatrix, sample_state.lcia_shared_u).key_positions,
            values=cast(LCIASharedUMatrix, sample_state.lcia_shared_u).values[
                run_positions,
                :,
            ],
        )
        if branch.lcia_plan is not None
        else None
    )
    return branch_batch, projection_selection, lcia_shared_u


def _branch_inter_mrio_matches(
    *,
    branch: InterMethodBranchExecution,
    matches_by_branch: dict[str, InterMrioInterpolationMatches] | None,
) -> InterMrioInterpolationMatches | None:
    if branch.external_plan.method_labels or matches_by_branch is None:
        return None
    return matches_by_branch.get(branch.label)


def _assign_branch_summary_values(
    *,
    output: np.ndarray,
    run_positions: np.ndarray,
    public_group_ids: np.ndarray,
    branch_values: np.ndarray,
) -> None:
    for group_id in np.unique(public_group_ids):
        value_positions = np.flatnonzero(public_group_ids == group_id)
        block = branch_values[:, value_positions]
        counts = np.sum(np.isfinite(block), axis=1)
        sums = np.nansum(block, axis=1)
        output[run_positions, int(group_id)] = np.divide(
            sums,
            counts,
            out=np.full(branch_values.shape[0], np.nan, dtype=np.float64),
            where=counts > 0,
        )


def _sparse_row_offsets(
    *,
    selected: np.ndarray,
    row_counts_by_branch: dict[str, int],
) -> tuple[np.ndarray, int]:
    row_counts = np.fromiter(
        (row_counts_by_branch[str(label)] for label in selected.tolist()),
        dtype=np.int64,
        count=selected.size,
    )
    offsets = np.empty_like(row_counts)
    if row_counts.size == 0:
        return offsets, 0
    offsets[0] = 0
    if row_counts.size > 1:
        np.cumsum(row_counts[:-1], out=offsets[1:])
    return offsets, int(offsets[-1] + row_counts[-1])


def _assign_sparse_branch_rows(
    *,
    run_index: np.ndarray,
    public_row_id: np.ndarray,
    values: np.ndarray,
    row_offsets: np.ndarray,
    run_positions: np.ndarray,
    branch_run_indices: np.ndarray,
    branch_public_row_id: np.ndarray,
    branch_values: np.ndarray,
) -> None:
    row_count = int(branch_public_row_id.size)
    if row_count == 0 or run_positions.size == 0:
        return
    destination = (
        row_offsets[run_positions, np.newaxis] + np.arange(row_count, dtype=np.int64)
    ).reshape(-1)
    run_index[destination] = np.repeat(branch_run_indices, row_count)
    public_row_id[destination] = np.tile(branch_public_row_id, run_positions.size)
    values[destination] = branch_values.reshape(-1)


def _source_units_for_positions(
    *,
    source_units: SourceUnitIntervalSamples,
    positions: np.ndarray,
) -> SourceUnitIntervalSamples:
    return SourceUnitIntervalSamples(
        values_by_source={
            source: np.asarray(values, dtype=np.float64)[positions]
            for source, values in source_units.values_by_source.items()
        }
    )


def external_run_offsets_for_start(
    *,
    inter_method_plan: InterMethodPlan,
    start_run_index: int,
    external_labels: tuple[str, ...],
) -> dict[str, int]:
    """Return external run counts used before one package run index."""
    labels = tuple(str(label) for label in external_labels)
    counts = {label: 0 for label in labels}
    if start_run_index <= 0 or not labels:
        return counts
    label_set = set(labels)
    chunk_size = _inter_method_offset_chunk_size()
    for start in range(0, int(start_run_index), chunk_size):
        stop = min(start + chunk_size, int(start_run_index))
        batch = RunBatch(
            batch_index=0,
            start_run_index=start,
            stop_run_index=stop,
            rng_seed=0,
        )
        selected = sample_inter_method_labels(plan=inter_method_plan, batch=batch)
        for label, count in zip(*np.unique(selected, return_counts=True), strict=True):
            text = str(label)
            if text in label_set:
                counts[text] += int(count)
    return counts


def _inter_method_offset_chunk_size() -> int:
    row_bytes = np.dtype(object).itemsize * len(("selected_label", "unique_workspace"))
    row_bytes += np.dtype(np.int64).itemsize * len(("unique_counts",))
    return memory_bounded_rows(bytes_per_row=row_bytes)


def _external_run_indices_for_branch(
    *,
    branch: InterMethodBranchExecution,
    offsets: dict[str, int],
    run_count: int,
) -> dict[str, np.ndarray] | None:
    if not branch.external_plan.monte_carlo_sources:
        return None
    start = int(offsets.get(branch.label, 0))
    return {branch.label: np.arange(start, start + int(run_count), dtype=np.int64)}


def _selected_run_batch(*, batch, run_positions: np.ndarray) -> RunBatch:
    run_indices = batch.run_indices()[run_positions]
    return RunBatch(
        batch_index=batch.batch_index,
        start_run_index=int(run_indices[0]),
        stop_run_index=int(run_indices[-1]) + 1,
        rng_seed=batch.rng_seed,
        run_index_values=tuple(int(value) for value in run_indices),
    )

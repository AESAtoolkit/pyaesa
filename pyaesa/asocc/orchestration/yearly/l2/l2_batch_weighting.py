"""Vectorized L2 preweight weighting kernels for impact batch execution."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

from .l2_reuse_frames import _l2_reuse_year_frame_from_values


def _numeric_series_to_numpy(*, values: pd.Series) -> np.ndarray:
    """Return one numeric Series as a NumPy array."""
    return np.asarray(values.to_numpy(dtype=np.float64, copy=False), dtype=np.float64)


def _single_column_frame_to_numpy(*, frame: pd.DataFrame) -> np.ndarray:
    """Return the numeric value column from a canonical one year frame."""
    return np.asarray(frame.to_numpy(dtype=np.float64, copy=False), dtype=np.float64)[:, 0]


@dataclass(frozen=True)
class _WeightAlignmentPlan:
    """Canonical numeric weighting plan for one batch kernel call."""

    axis_labels: pd.Index
    axis_codes: np.ndarray
    axis_weight_positions: np.ndarray
    group_codes: np.ndarray
    grouped_index: pd.MultiIndex
    pre_index: pd.MultiIndex
    pre_values: np.ndarray
    grouped_index_with_impact_cache: dict[tuple[str, ...], pd.MultiIndex] = field(
        default_factory=dict,
        compare=False,
    )
    pre_index_with_impact_cache: dict[tuple[str, ...], pd.MultiIndex] = field(
        default_factory=dict,
        compare=False,
    )


_PlanCacheKey = tuple[int, str, tuple[str, ...], tuple[object, ...]]
_PlanCacheValue = tuple[pd.DataFrame, _WeightAlignmentPlan]


def _require_single_year_series(*, pre_weighted: pd.DataFrame, where: str) -> pd.Series:
    """Return the canonical preweight Series from a one year DataFrame."""
    del where
    return pre_weighted.iloc[:, 0]


def _require_multi_index(*, index: pd.Index, where: str) -> pd.MultiIndex:
    """Return index as MultiIndex."""
    del where
    return cast(pd.MultiIndex, index)


def _factorize_required_index(
    *,
    index: pd.Index,
    required_indices: tuple[str, ...],
    where: str,
) -> tuple[np.ndarray, pd.MultiIndex]:
    """Return grouping codes/index for required output levels."""
    multi_index = _require_multi_index(index=index, where=where)
    index_names = [str(name) for name in multi_index.names]
    required_positions = [index_names.index(name) for name in required_indices]
    required_codes = [
        np.asarray(multi_index.codes[position], dtype="int64") for position in required_positions
    ]
    if len(required_positions) == 1:
        row_codes, unique_codes = pd.factorize(required_codes[0], sort=False)
        labels = pd.Index(multi_index.levels[required_positions[0]], copy=False).take(unique_codes)
        grouped_index = pd.MultiIndex.from_arrays([labels], names=list(required_indices))
        return row_codes.astype("int64", copy=False), grouped_index
    dims = tuple(len(multi_index.levels[position]) for position in required_positions)
    flat_codes = np.ravel_multi_index(required_codes, dims=dims, mode="raise")
    row_codes, unique_flat_codes = pd.factorize(flat_codes, sort=False)
    unique_level_codes = np.unravel_index(np.asarray(unique_flat_codes, dtype="int64"), shape=dims)
    grouped_index = pd.MultiIndex.from_arrays(
        [
            pd.Index(multi_index.levels[position], copy=False).take(unique_level_codes[level_pos])
            for level_pos, position in enumerate(required_positions)
        ],
        names=list(required_indices),
    )
    return row_codes.astype("int64", copy=False), grouped_index


def _axis_labels_and_codes(
    *,
    index: pd.Index,
    axis_name: str,
    where: str,
) -> tuple[pd.Index, np.ndarray]:
    """Return unique labels and row level codes for one weighting axis."""
    multi_index = _require_multi_index(index=index, where=where)
    index_names = [str(name) for name in multi_index.names]
    axis_position = index_names.index(axis_name)
    original_codes = np.asarray(multi_index.codes[axis_position], dtype="int64")
    axis_codes, unique_codes = pd.factorize(original_codes, sort=False)
    axis_labels = pd.Index(multi_index.levels[axis_position], copy=False).take(unique_codes)
    return axis_labels, axis_codes.astype("int64", copy=False)


def _sum_min_count_one(
    *,
    values: np.ndarray,
    group_codes: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Aggregate values by integer group code with min count one semantics."""
    valid = ~np.isnan(values)
    if not bool(valid.any()):
        return np.full(group_size, np.nan, dtype="float64")
    sums = np.bincount(
        group_codes[valid],
        weights=values[valid],
        minlength=group_size,
    ).astype("float64", copy=False)
    counts = np.bincount(group_codes[valid], minlength=group_size)
    sums[counts == 0] = np.nan
    return sums


def _sum_min_count_one_matrix(
    *,
    values: np.ndarray,
    group_codes: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Aggregate one impact by source row matrix into grouped output rows."""
    valid = ~np.isnan(values)
    if not bool(valid.any()):
        return np.full((values.shape[0], group_size), np.nan, dtype=np.float64)
    safe_values = np.where(valid, values, 0.0)
    aggregated_by_group = np.zeros((group_size, values.shape[0]), dtype=np.float64)
    counts_by_group = np.zeros((group_size, values.shape[0]), dtype=np.int64)
    np.add.at(aggregated_by_group, group_codes, safe_values.T)
    np.add.at(counts_by_group, group_codes, valid.T.astype(np.int64, copy=False))
    aggregated = aggregated_by_group.T
    aggregated[counts_by_group.T == 0] = np.nan
    return aggregated


def _axis_weight_positions(*, weight_index: pd.Index, axis_labels: pd.Index) -> np.ndarray:
    """Return L1 weight positions for the L2 route weighting axis."""
    if len(weight_index) == len(axis_labels) and weight_index.equals(axis_labels):
        return np.arange(len(axis_labels), dtype=np.int64)
    return weight_index.get_indexer(axis_labels).astype("int64", copy=False)


def _axis_weight_values_for_plan(
    *,
    weight_values: np.ndarray,
    plan: _WeightAlignmentPlan,
) -> np.ndarray:
    """Return impact weights aligned to the route axis labels."""
    positions = plan.axis_weight_positions
    if bool((positions < 0).any()):
        present_mask = positions >= 0
        axis_values = np.full((weight_values.shape[0], len(positions)), np.nan, dtype=np.float64)
        axis_values[:, present_mask] = weight_values[:, positions[present_mask]]
        return axis_values
    return weight_values[:, positions]


def _source_weight_matrix_for_plan(
    *,
    weight_values: np.ndarray,
    plan: _WeightAlignmentPlan,
) -> np.ndarray:
    """Return source row aligned weights for all impacts."""
    return _axis_weight_values_for_plan(weight_values=weight_values, plan=plan)[:, plan.axis_codes]


def _build_weight_alignment_plan(
    *,
    pre_weighted: pd.DataFrame,
    weight_index: pd.Index,
    weight_axis: str,
    required_indices: tuple[str, ...],
    where: str,
    plan_cache: dict[_PlanCacheKey, _PlanCacheValue] | None = None,
) -> _WeightAlignmentPlan:
    """Build one canonical numeric plan for a batch weighting call."""
    weight_labels = tuple(weight_index.tolist())
    cache_key = (id(pre_weighted), str(weight_axis), tuple(required_indices), weight_labels)
    if plan_cache is not None and cache_key in plan_cache:
        return plan_cache[cache_key][1]
    pre_series = _require_single_year_series(pre_weighted=pre_weighted, where=where)
    pre_index = _require_multi_index(index=pre_series.index, where=where)
    axis_labels, axis_codes = _axis_labels_and_codes(
        index=pre_index,
        axis_name=weight_axis,
        where=where,
    )
    group_codes, grouped_index = _factorize_required_index(
        index=pre_index,
        required_indices=required_indices,
        where=where,
    )
    plan = _WeightAlignmentPlan(
        axis_labels=axis_labels,
        axis_codes=axis_codes,
        axis_weight_positions=_axis_weight_positions(
            weight_index=weight_index,
            axis_labels=axis_labels,
        ),
        group_codes=group_codes,
        grouped_index=grouped_index,
        pre_index=pre_index,
        pre_values=_numeric_series_to_numpy(values=pre_series),
    )
    if plan_cache is not None:
        plan_cache[cache_key] = (pre_weighted, plan)
    return plan


def _prepend_impact_level(
    *,
    impact_names: Sequence[str],
    base_index: pd.MultiIndex,
    index_cache: dict[tuple[str, ...], pd.MultiIndex],
) -> pd.MultiIndex:
    """Return one MultiIndex with a leading impact level."""
    impact_key = tuple(str(name) for name in impact_names)
    if impact_key in index_cache:
        return index_cache[impact_key]
    impact_count = len(impact_names)
    base_size = len(base_index)
    impact_codes = np.repeat(np.arange(impact_count, dtype=np.intp), base_size)
    repeated_codes = [
        np.tile(np.asarray(code, dtype=np.intp), impact_count) for code in base_index.codes
    ]
    prefixed = pd.MultiIndex(
        levels=[pd.Index(list(impact_names), name="impact"), *base_index.levels],
        codes=[impact_codes, *repeated_codes],
        names=["impact", *base_index.names],
        verify_integrity=False,
    )
    index_cache[impact_key] = prefixed
    return prefixed


def batch_weight_preweighted_ut_matrix(
    *,
    pre_weighted: pd.DataFrame,
    impact_names: Sequence[str],
    weight_index: pd.Index,
    weight_values: np.ndarray,
    weight_axis: str,
    required_indices: tuple[str, ...],
    year: int,
    include_contribution: bool,
    plan_cache: dict[_PlanCacheKey, _PlanCacheValue] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Batch weight UT preweights from a numeric impact weight matrix."""
    plan = _build_weight_alignment_plan(
        pre_weighted=pre_weighted,
        weight_index=weight_index,
        weight_axis=weight_axis,
        required_indices=required_indices,
        where="batch_weight_preweighted_ut",
        plan_cache=plan_cache,
    )
    impacts = tuple(str(impact_name) for impact_name in impact_names)
    impact_count = len(impacts)
    pre_size = len(plan.pre_index)
    group_size = len(plan.grouped_index)
    weight_matrix = _source_weight_matrix_for_plan(
        weight_values=weight_values,
        plan=plan,
    )
    weighted_matrix = weight_matrix * plan.pre_values[np.newaxis, :]
    aggregated_matrix = _sum_min_count_one_matrix(
        values=weighted_matrix,
        group_codes=plan.group_codes,
        group_size=group_size,
    )
    contribution_matrix = weighted_matrix if include_contribution else None

    aggregated_frame = pd.DataFrame(
        aggregated_matrix.reshape(impact_count * group_size, 1),
        index=_prepend_impact_level(
            impact_names=impacts,
            base_index=plan.grouped_index,
            index_cache=plan.grouped_index_with_impact_cache,
        ),
        columns=pd.Index([int(year)]),
    )
    if contribution_matrix is None:
        return aggregated_frame, None
    contribution_frame = pd.DataFrame(
        contribution_matrix.reshape(impact_count * pre_size, 1),
        index=_prepend_impact_level(
            impact_names=impacts,
            base_index=plan.pre_index,
            index_cache=plan.pre_index_with_impact_cache,
        ),
        columns=pd.Index([int(year)]),
    )
    return aggregated_frame, contribution_frame


def batch_weight_reuse_preweighted_ut_matrix(
    *,
    preweights_by_l2_reuse_year: Sequence[tuple[int, pd.DataFrame]],
    impact_names: Sequence[str],
    weight_index: pd.Index,
    weight_values: np.ndarray,
    weight_axis: str,
    required_indices: tuple[str, ...],
    year: int,
    include_contribution: bool,
    reference_year: int | None,
    plan_cache: dict[_PlanCacheKey, _PlanCacheValue] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Batch weight one historical reuse slice across all L2 reuse years."""
    first_preweight = preweights_by_l2_reuse_year[0][1]
    plan = _build_weight_alignment_plan(
        pre_weighted=first_preweight,
        weight_index=weight_index,
        weight_axis=weight_axis,
        required_indices=required_indices,
        where="batch_weight_reuse_preweighted_ut",
        plan_cache=plan_cache,
    )
    impacts = tuple(str(impact_name) for impact_name in impact_names)
    impact_count = len(impacts)
    pre_size = len(plan.pre_index)
    group_size = len(plan.grouped_index)
    weight_matrix = _source_weight_matrix_for_plan(
        weight_values=weight_values,
        plan=plan,
    )
    l2_reuse_years = [int(reuse_entry[0]) for reuse_entry in preweights_by_l2_reuse_year]
    aggregated_blocks: list[np.ndarray] = []
    contribution_blocks: list[np.ndarray] = []
    for reuse_entry in preweights_by_l2_reuse_year:
        pre_values = _single_column_frame_to_numpy(frame=reuse_entry[1])
        weighted_matrix = weight_matrix * pre_values[np.newaxis, :]
        aggregated_matrix = _sum_min_count_one_matrix(
            values=weighted_matrix,
            group_codes=plan.group_codes,
            group_size=group_size,
        )
        aggregated_blocks.append(aggregated_matrix.reshape(impact_count * group_size))
        if include_contribution:
            contribution_blocks.append(weighted_matrix.reshape(impact_count * pre_size))
    base_aggregated_index = _prepend_impact_level(
        impact_names=impacts,
        base_index=plan.grouped_index,
        index_cache=plan.grouped_index_with_impact_cache,
    )
    aggregated_frame = _l2_reuse_year_frame_from_values(
        base_index=base_aggregated_index,
        l2_reuse_years=l2_reuse_years,
        values=np.concatenate(aggregated_blocks)[:, np.newaxis],
        columns=pd.Index([int(year)]),
        reference_year=reference_year,
    )
    if not include_contribution:
        return aggregated_frame, None
    base_contribution_index = _prepend_impact_level(
        impact_names=impacts,
        base_index=plan.pre_index,
        index_cache=plan.pre_index_with_impact_cache,
    )
    contribution_frame = _l2_reuse_year_frame_from_values(
        base_index=base_contribution_index,
        l2_reuse_years=l2_reuse_years,
        values=np.concatenate(contribution_blocks)[:, np.newaxis],
        columns=pd.Index([int(year)]),
        reference_year=reference_year,
    )
    return aggregated_frame, contribution_frame


def batch_weight_preweighted_ar_matrix(
    *,
    pre_weighted: pd.DataFrame,
    impact_names: Sequence[str],
    weight_index: pd.Index,
    weight_values: np.ndarray,
    impact_level: str,
    weight_axis: str,
    required_indices: tuple[str, ...],
    year: int,
    plan_cache: dict[_PlanCacheKey, _PlanCacheValue] | None = None,
) -> pd.DataFrame:
    """Batch weight AR preweights from a numeric impact weight matrix."""
    pre_series = _require_single_year_series(
        pre_weighted=pre_weighted,
        where="batch_weight_preweighted_ar",
    )
    index = pre_series.index
    impact_order = tuple(str(impact_name) for impact_name in impact_names)
    impact_values = pd.Index(index.get_level_values(impact_level)).astype(str)
    impact_codes = pd.Index(impact_order, dtype="object").get_indexer(impact_values)
    unknown_mask = impact_codes < 0
    if bool(unknown_mask.any()):
        unknown = impact_values[unknown_mask].unique().tolist()
        sample = [str(value) for value in unknown[:10]]
        raise ValueError(
            "Some impacts in L2 intermediate AR data do not have matching "
            "L1 impact weights. "
            f"Unknown impacts (sample): {sample}."
        )
    impact_codes = impact_codes.astype("int64", copy=False)
    plan = _build_weight_alignment_plan(
        pre_weighted=pre_weighted,
        weight_index=weight_index,
        weight_axis=weight_axis,
        required_indices=required_indices,
        where="batch_weight_preweighted_ar",
        plan_cache=plan_cache,
    )
    weights_by_axis = _axis_weight_values_for_plan(
        weight_values=weight_values,
        plan=plan,
    )
    row_weights = weights_by_axis[impact_codes, plan.axis_codes]
    weighted_values = plan.pre_values * row_weights
    aggregated = _sum_min_count_one(
        values=weighted_values,
        group_codes=plan.group_codes,
        group_size=len(plan.grouped_index),
    )
    return pd.DataFrame(
        aggregated.reshape(len(plan.grouped_index), 1),
        index=plan.grouped_index,
        columns=pd.Index([int(year)]),
    )

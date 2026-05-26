"""Vectorized AR6 CC Monte Carlo source planning and evaluation."""

from typing import cast

import numpy as np
import pandas as pd

from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch
from pyaesa.shared.uncertainty_assessment.io.run_writers import SparseRunRows
from pyaesa.ar6_cc.deterministic.request.contracts import CC_FLOW_NEGATIVE
from pyaesa.ar6_cc.uncertainty.request.normalization import AR6_DYNAMIC_CC_SOURCE

from pyaesa.ar6_cc.uncertainty.runtime.models import (
    AR6CCCategoryPool,
    AR6CCSamplingGroup,
    AR6CCUncertaintyPlan,
    AR6CCUncertaintyRequest,
)

_TRAJECTORY_COLUMNS = ("cc_model", "cc_scenario", "cc_category", "ssp_scenario")
_IDENTITY_COLUMNS = (*_TRAJECTORY_COLUMNS, "cc_flow", "cc_variable", "impact_unit")


def build_ar6_cc_sampling_plan(
    *,
    request: AR6CCUncertaintyRequest,
    deterministic_rows: pd.DataFrame,
) -> AR6CCUncertaintyPlan:
    """Return the compact trajectory sampling plan for one AR6 CC request."""
    rows, year_columns = _canonical_rows(request=request, rows=deterministic_rows)
    groups_frame = rows.loc[:, ["cc_category", "ssp_scenario"]].drop_duplicates()
    groups_frame = groups_frame.sort_values(
        ["cc_category", "ssp_scenario"],
        kind="mergesort",
    ).reset_index(drop=True)
    identity = _public_identity(rows=rows, years=year_columns)
    group_identity, group_spans = _group_identity(
        rows=rows,
        groups=groups_frame,
        years=year_columns,
    )
    trajectory_values = rows.loc[:, year_columns].to_numpy(dtype=np.float64)
    groups = tuple(
        _sampling_group(
            rows=rows,
            category=str(category),
            ssp_scenario=str(ssp_scenario),
            output_start=group_spans[group_index][0],
            output_stop=group_spans[group_index][1],
        )
        for group_index, (category, ssp_scenario) in enumerate(
            groups_frame.loc[:, ["cc_category", "ssp_scenario"]].itertuples(
                index=False,
                name=None,
            )
        )
    )
    category_pools = _category_pools(
        groups=groups,
        category_uncertainty=bool(request.source_parameters["category_uncertainty"]),
    )
    return AR6CCUncertaintyPlan(
        identity=identity,
        group_identity=group_identity,
        trajectory_values=trajectory_values,
        groups=groups,
        category_pools=category_pools,
        source_method_rows=_source_method_rows(
            rows=rows,
            request=request,
            years=year_columns,
        ),
        source_parameters=dict(request.source_parameters),
        availability_messages=_availability_messages(request=request, groups=groups),
    )


def deterministic_ar6_cc_identity_and_values(
    *,
    request: AR6CCUncertaintyRequest,
    deterministic_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return deterministic AR6 CC row identity and values for downstream formulas."""
    rows, year_columns = _canonical_rows(request=request, rows=deterministic_rows)
    rows = rows.loc[rows["cc_flow"].astype(str) != CC_FLOW_NEGATIVE].reset_index(drop=True)
    year_count = len(year_columns)
    row_count = len(rows)
    identity = rows.loc[
        rows.index.repeat(year_count),
        ["cc_model", "cc_scenario", "cc_category", "ssp_scenario", "impact_unit"],
    ].reset_index(drop=True)
    identity["year"] = np.tile(np.asarray(year_columns, dtype=np.int64), row_count)
    identity.insert(0, "public_row_id", np.arange(len(identity), dtype=np.int64))
    values = rows.loc[:, year_columns].to_numpy(dtype=np.float64).reshape(-1)
    return identity, values


def sample_ar6_cc_sparse_rows(
    *,
    plan: AR6CCUncertaintyPlan,
    batch: RunBatch,
) -> SparseRunRows:
    """Return sampled AR6 CC selected trajectory rows for one run batch."""
    rng = batch.rng()
    run_indices = batch.run_indices()
    pieces: list[SparseRunRows] = []
    if bool(plan.source_parameters["category_uncertainty"]):
        for pool in plan.category_pools:
            category_runs = rng.integers(0, len(pool.group_indices), size=batch.n_runs)
            for local_position, group_index in enumerate(pool.group_indices):
                run_positions = np.flatnonzero(category_runs == local_position)
                if run_positions.size == 0:
                    continue
                group = plan.groups[group_index]
                selected = _sample_candidate_positions(
                    rng=rng,
                    group=group,
                    run_count=int(run_positions.size),
                    sampling_method=str(plan.source_parameters["sampling_method"]),
                )
                pieces.append(
                    _sparse_rows_for_selected(
                        plan=plan,
                        group=group,
                        run_indices=run_indices[run_positions],
                        selected_positions=selected,
                    )
                )
        return _concat_sparse_rows(pieces=pieces)
    for group in plan.groups:
        selected = _sample_candidate_positions(
            rng=rng,
            group=group,
            run_count=batch.n_runs,
            sampling_method=str(plan.source_parameters["sampling_method"]),
        )
        pieces.append(
            _sparse_rows_for_selected(
                plan=plan,
                group=group,
                run_indices=run_indices,
                selected_positions=selected,
            )
        )
    return _concat_sparse_rows(pieces=pieces)


def _canonical_rows(
    *,
    request: AR6CCUncertaintyRequest,
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    renamed = rows.copy()
    renamed.columns = [
        int(column) if isinstance(column, str) and column.isdigit() else column
        for column in renamed.columns
    ]
    year_columns = list(request.years)
    ordered = renamed.loc[:, [*_IDENTITY_COLUMNS, *year_columns]].copy()
    for column in _IDENTITY_COLUMNS:
        ordered[column] = ordered[column].astype(str)
    flow_rank = ordered["cc_flow"].map({"net_emissions": 0, "positive_emissions": 0}).fillna(1)
    ordered["_flow_rank"] = flow_rank.astype(int)
    ordered = (
        ordered.sort_values(
            [
                "cc_category",
                "ssp_scenario",
                "impact_unit",
                "cc_model",
                "cc_scenario",
                "_flow_rank",
                "cc_flow",
            ],
            kind="mergesort",
        )
        .drop(columns=["_flow_rank"])
        .reset_index(drop=True)
    )
    return ordered, year_columns


def _public_identity(*, rows: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    row_count = len(rows)
    year_count = len(years)
    repeated = rows.loc[rows.index.repeat(year_count), list(_IDENTITY_COLUMNS)].reset_index(
        drop=True
    )
    repeated["year"] = np.tile(np.asarray(years, dtype=np.int64), row_count)
    repeated.insert(0, "public_row_id", np.arange(len(repeated), dtype=np.int64))
    return repeated.loc[
        :,
        [
            "public_row_id",
            "cc_category",
            "ssp_scenario",
            "cc_flow",
            "cc_variable",
            "impact_unit",
            "cc_model",
            "cc_scenario",
            "year",
        ],
    ]


def _group_identity(
    *,
    rows: pd.DataFrame,
    groups: pd.DataFrame,
    years: list[int],
) -> tuple[pd.DataFrame, tuple[tuple[int, int], ...]]:
    identity_parts: list[pd.DataFrame] = []
    spans: list[tuple[int, int]] = []
    start = 0
    year_count = len(years)
    for category, ssp_scenario in groups.loc[:, ["cc_category", "ssp_scenario"]].itertuples(
        index=False,
        name=None,
    ):
        mask = (rows["cc_category"] == str(category)) & (rows["ssp_scenario"] == str(ssp_scenario))
        positions = np.flatnonzero(mask.to_numpy(dtype=bool))
        first_trajectory = int(positions[0])
        flow_count = _flow_count_for_positions(rows=rows, positions=positions)
        flow_rows = rows.iloc[first_trajectory : first_trajectory + flow_count]
        repeated = flow_rows.loc[
            flow_rows.index.repeat(year_count),
            ["cc_category", "ssp_scenario", "cc_flow", "cc_variable", "impact_unit"],
        ].reset_index(drop=True)
        repeated["year"] = np.tile(np.asarray(years, dtype=np.int64), flow_count)
        identity_parts.append(repeated)
        stop = start + len(repeated)
        spans.append((start, stop))
        start = stop
    identity = pd.concat(identity_parts, ignore_index=True) if identity_parts else pd.DataFrame()
    identity.insert(0, "public_row_id", np.arange(len(identity), dtype=np.int64))
    return identity, tuple(spans)


def _sparse_rows_for_selected(
    *,
    plan: AR6CCUncertaintyPlan,
    group: AR6CCSamplingGroup,
    run_indices: np.ndarray,
    selected_positions: np.ndarray,
) -> SparseRunRows:
    year_count = int(plan.trajectory_values.shape[1])
    flow_count = int(group.flow_count)
    year_offsets = np.arange(year_count, dtype=np.int64)
    flow_offsets = np.arange(flow_count, dtype=np.int64)
    row_positions = selected_positions[:, None] + flow_offsets[None, :]
    flat_positions = row_positions.reshape(-1)
    public_row_id = np.repeat(flat_positions.astype(np.int64, copy=False), year_count)
    public_row_id = public_row_id * year_count + np.tile(year_offsets, len(flat_positions))
    return SparseRunRows(
        run_index=np.repeat(run_indices.astype(np.int64, copy=False), year_count * flow_count),
        public_row_id=public_row_id,
        values=plan.trajectory_values[flat_positions, :].reshape(-1),
        value_column="cc",
    )


def _concat_sparse_rows(*, pieces: list[SparseRunRows]) -> SparseRunRows:
    run_index = np.concatenate([piece.run_index for piece in pieces])
    public_row_id = np.concatenate([piece.public_row_id for piece in pieces])
    values = np.concatenate([piece.values for piece in pieces])
    order = np.lexsort((public_row_id, run_index))
    return SparseRunRows(
        run_index=run_index[order],
        public_row_id=public_row_id[order],
        values=values[order],
        value_column="cc",
    )


def _sampling_group(
    *,
    rows: pd.DataFrame,
    category: str,
    ssp_scenario: str,
    output_start: int,
    output_stop: int,
) -> AR6CCSamplingGroup:
    mask = (rows["cc_category"] == category) & (rows["ssp_scenario"] == ssp_scenario)
    positions = np.flatnonzero(mask.to_numpy(dtype=bool))
    trajectory_frame = rows.iloc[positions].drop_duplicates(subset=list(_TRAJECTORY_COLUMNS))
    start_positions = trajectory_frame.index.to_numpy(dtype=np.int64, copy=False)
    flow_count = _flow_count_for_positions(rows=rows, positions=positions)
    model_positions = tuple(
        start_positions[rows.iloc[start_positions]["cc_model"].astype(str).to_numpy() == str(model)]
        for model in sorted(set(rows.iloc[start_positions]["cc_model"].astype(str)))
    )
    return AR6CCSamplingGroup(
        category=category,
        ssp_scenario=ssp_scenario,
        flow_count=flow_count,
        candidate_positions=start_positions.astype(np.int64, copy=False),
        model_candidate_positions=model_positions,
        output_start=output_start,
        output_stop=output_stop,
    )


def _flow_count_for_positions(*, rows: pd.DataFrame, positions: np.ndarray) -> int:
    """Return the number of flow rows materialized for each retained trajectory."""
    trajectory_count = len(
        rows.iloc[positions].drop_duplicates(subset=list(_TRAJECTORY_COLUMNS)).index
    )
    return int(len(positions) / trajectory_count)


def _category_pools(
    *,
    groups: tuple[AR6CCSamplingGroup, ...],
    category_uncertainty: bool,
) -> tuple[AR6CCCategoryPool, ...]:
    if not category_uncertainty:
        return ()
    pool_keys = sorted({group.ssp_scenario for group in groups})
    pools = tuple(
        AR6CCCategoryPool(
            ssp_scenario=ssp,
            group_indices=tuple(
                index for index, group in enumerate(groups) if group.ssp_scenario == ssp
            ),
        )
        for ssp in pool_keys
    )
    if max(len(pool.group_indices) for pool in pools) < 2:
        pool_counts = {pool.ssp_scenario: len(pool.group_indices) for pool in pools}
        raise ValueError(
            f"{AR6_DYNAMIC_CC_SOURCE}.category_uncertainty requires at least two retained "
            "AR6 CC categories in at least one SSP pool. "
            f"Retained pool counts={pool_counts}."
        )
    return pools


def _sample_candidate_positions(
    *,
    rng: np.random.Generator,
    group: AR6CCSamplingGroup,
    run_count: int,
    sampling_method: str,
) -> np.ndarray:
    if sampling_method == "srs":
        local = rng.integers(0, len(group.candidate_positions), size=run_count)
        return cast(np.ndarray, group.candidate_positions[local])
    model_choice = rng.integers(0, len(group.model_candidate_positions), size=run_count)
    selected = np.empty(run_count, dtype=np.int64)
    for model_index, model_positions in enumerate(group.model_candidate_positions):
        run_positions = np.flatnonzero(model_choice == model_index)
        if run_positions.size == 0:
            continue
        local = rng.integers(0, len(model_positions), size=int(run_positions.size))
        selected[run_positions] = model_positions[local]
    return selected


def _source_method_rows(
    *,
    rows: pd.DataFrame,
    request: AR6CCUncertaintyRequest,
    years: list[int],
) -> pd.DataFrame:
    source_columns = ["cc_category", "ssp_scenario", "impact_unit"]
    trajectory_group_columns = ["cc_category", "ssp_scenario"]
    method_rows = rows.loc[
        :,
        [*source_columns, "cc_model", "cc_scenario", "cc_flow", "cc_variable"],
    ].copy()
    trajectory_rows = method_rows.drop_duplicates(
        subset=[*trajectory_group_columns, "cc_model", "cc_scenario"],
        keep="first",
    ).reset_index(drop=True)
    sampling_method = str(request.source_parameters["sampling_method"])
    group_sizes = trajectory_rows.groupby(trajectory_group_columns, sort=False)[
        "cc_scenario"
    ].transform("size")
    if sampling_method == "srs":
        trajectory_probability = 1.0 / group_sizes.astype(np.float64)
    else:
        model_counts = trajectory_rows.groupby(trajectory_group_columns, sort=False)[
            "cc_model"
        ].transform("nunique")
        scenarios_for_model = trajectory_rows.groupby(
            [*trajectory_group_columns, "cc_model"],
            sort=False,
        )["cc_scenario"].transform("size")
        trajectory_probability = 1.0 / (
            model_counts.astype(np.float64) * scenarios_for_model.astype(np.float64)
        )
    if bool(request.source_parameters["category_uncertainty"]):
        pool_category_counts = trajectory_rows.groupby(["ssp_scenario"], sort=False)[
            "cc_category"
        ].transform("nunique")
        category_probability = 1.0 / pool_category_counts.astype(np.float64)
    else:
        category_probability = 1.0
    trajectory_probabilities = trajectory_rows.loc[
        :, [*trajectory_group_columns, "cc_model", "cc_scenario"]
    ].copy()
    trajectory_probabilities["trajectory_probability"] = trajectory_probability
    trajectory_probabilities["category_probability"] = category_probability
    method_rows = method_rows.merge(
        trajectory_probabilities,
        on=[*trajectory_group_columns, "cc_model", "cc_scenario"],
        how="left",
        sort=False,
    )
    source = pd.DataFrame(index=method_rows.index)
    source["source_component"] = "ar6_cc"
    source["source_name"] = AR6_DYNAMIC_CC_SOURCE
    for column in [*source_columns, "cc_model", "cc_scenario", "cc_flow", "cc_variable"]:
        source[column] = method_rows[column].astype(str)
    source["year_min"] = int(min(years))
    source["year_max"] = int(max(years))
    source["emissions_mode"] = request.emissions_mode
    source["sampling_method"] = sampling_method
    source["category_uncertainty"] = bool(request.source_parameters["category_uncertainty"])
    source["category_probability"] = method_rows["category_probability"].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    source["trajectory_probability"] = method_rows["trajectory_probability"].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    source["joint_probability"] = source["category_probability"] * source["trajectory_probability"]
    source["formula"] = "sample one retained deterministic AR6 CC trajectory"
    return source.reset_index(drop=True)


def _availability_messages(
    *,
    request: AR6CCUncertaintyRequest,
    groups: tuple[AR6CCSamplingGroup, ...],
) -> tuple[str, ...]:
    messages: list[str] = []
    retained_categories = sorted({group.category for group in groups})
    retained_ssps = sorted({group.ssp_scenario for group in groups})
    missing_categories = [
        category for category in request.category if category not in retained_categories
    ]
    if missing_categories:
        messages.append(
            "Requested AR6 CC categories have no retained model-scenario pair "
            "for the requested SSP, emissions mode, gas coverage, AFOLU setting, "
            f"and study window: {', '.join(missing_categories)}."
        )
    missing_ssps = [ssp for ssp in request.ssp_scenario if ssp not in retained_ssps]
    if missing_ssps:
        messages.append(
            "Requested AR6 CC SSP scenarios have no retained model-scenario pair "
            "for the requested categories, emissions mode, gas coverage, AFOLU setting, "
            f"and study window: {', '.join(missing_ssps)}."
        )
    categories = list(request.category)
    pool_keys = sorted({group.ssp_scenario for group in groups})
    for ssp in pool_keys:
        retained_in_pool = {group.category for group in groups if group.ssp_scenario == ssp}
        missing_in_pool = [category for category in categories if category not in retained_in_pool]
        if missing_in_pool:
            messages.append(
                "Requested AR6 CC categories have no retained model-scenario pair for "
                f"ssp_scenario={ssp}: {', '.join(missing_in_pool)}."
            )
    return tuple(messages)

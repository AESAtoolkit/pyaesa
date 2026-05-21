"""LCIA uncertainty owner for deterministic aSoCC public rows."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.asocc.methods.registry.registry import REGISTRY
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import (
    ASOCC_VALUE_COLUMN,
    LoadedAsoccFinalRows,
)
from pyaesa.asocc.uncertainty.inputs.external_rows import external_method_row_mask
from pyaesa.asocc.uncertainty.io.source_methods import SourceMethodRow
from pyaesa.shared.lcia.cov_inputs import (
    load_lcia_cov_inputs,
    normalize_lcia_uncertainty_parameters,
)
from pyaesa.asocc.uncertainty.lcia_support.sampling import (
    LCIASampleBlock,
    LCIASharedUMatrix,
    attach_lcia_sampling_columns,
    lcia_sample_block,
    lcia_source_method_rows,
    lcia_source_method_template,
    sample_lcia_block_matrix,
    shared_u_matrix_for_lcia_blocks,
)
from pyaesa.asocc.uncertainty.sources.lcia_combined import (
    CombinedLCIARoutePlan,
    build_combined_routes,
    combined_route_value_matrix,
)
from pyaesa.asocc.uncertainty.sources.lcia_support import (
    LCIASupportRowCache,
    combined_final_rows,
)
from pyaesa.asocc.uncertainty.sources.names import LCIA_SOURCE
from pyaesa.shared.selectors.aggregate_labels import aggregate_selector_label_or_none
from pyaesa.shared.tabular.scalars import is_display_missing
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch

_REGION_SELECTOR_COLUMNS = ("r_f", "r_p", "r_c", "r_u")


@dataclass(frozen=True)
class LCIAPlan:
    """Resolved LCIA owner plan for one aSoCC uncertainty run."""

    public_columns: tuple[str, ...]
    passthrough_rows: pd.DataFrame
    direct_rows: pd.DataFrame
    direct_block: LCIASampleBlock | None
    combined_routes: tuple[CombinedLCIARoutePlan, ...]
    source_method_rows: tuple[SourceMethodRow, ...]


def build_lcia_plan(
    *,
    loaded: LoadedAsoccFinalRows,
    parameters: dict[str, Any],
    support_cache: LCIASupportRowCache,
    include_source_methods: bool = True,
    external_method_labels: tuple[str, ...] = (),
) -> LCIAPlan:
    """Resolve LCIA sampled rows from deterministic aSoCC row ownership."""
    normalized_parameters = normalize_lcia_uncertainty_parameters(parameters=parameters)
    covs = load_lcia_cov_inputs(
        sector_cov_mapping=cast(dict[str, str], normalized_parameters["sector_cov_mapping"]),
        group_reg=bool(loaded.base_asocc_args["group_reg"]),
        group_version=cast(str | None, loaded.base_asocc_args["group_version"]),
        aggregate_region_covs=_uses_aggregate_region_covs(loaded=loaded),
    )
    rows = loaded.rows.reset_index(drop=True)
    external_rows = external_method_row_mask(frame=rows, method_labels=external_method_labels)
    affected = pd.Series(False, index=rows.index, dtype=bool)
    # External aSoCC methods are already supplied as user values. The LCIA
    # source samples pyaesa owned LCIA factors only.
    direct_mask = _direct_lcia_mask(rows=rows, loaded=loaded) & ~external_rows
    direct_template = rows.loc[direct_mask].copy()
    direct_rows = (
        attach_lcia_sampling_columns(
            rows=direct_template,
            loaded=loaded,
            covs=covs,
            applied_bucket=loaded.final_bucket,
            allocation_column="l1_l2_method",
            weight_axis=None,
        )
        if not direct_template.empty
        else direct_template
    )
    affected.loc[direct_mask] = True
    routes, support_templates = build_combined_routes(
        loaded=loaded,
        covs=covs,
        rows=rows,
        affected=affected,
        support_cache=support_cache,
        external_method_labels=external_method_labels,
    )
    method_rows = (
        lcia_source_method_rows(
            loaded=loaded,
            source_name=LCIA_SOURCE,
            templates=[
                lcia_source_method_template(rows=direct_rows),
                *support_templates,
            ],
        )
        if include_source_methods
        else []
    )
    return LCIAPlan(
        public_columns=tuple(rows.columns),
        passthrough_rows=rows.loc[~affected].reset_index(drop=True),
        direct_rows=direct_rows.reset_index(drop=True),
        direct_block=lcia_sample_block(template=direct_rows) if not direct_rows.empty else None,
        combined_routes=tuple(routes),
        source_method_rows=tuple(method_rows),
    )


def lcia_uncertainty_has_targets(
    *,
    loaded: LoadedAsoccFinalRows,
    external_method_labels: tuple[str, ...] = (),
) -> bool:
    """Return whether deterministic aSoCC rows expose LCIA sampled owners."""
    rows = loaded.rows.reset_index(drop=True)
    external_rows = external_method_row_mask(frame=rows, method_labels=external_method_labels)
    eligible_rows = rows.loc[~external_rows].reset_index(drop=True)
    if bool(_direct_lcia_mask(rows=eligible_rows, loaded=loaded).any()):
        return True
    for l2_method, l1_method in loaded.asocc_scope.combined:
        final_rows = combined_final_rows(
            rows=eligible_rows,
            l1_method=l1_method,
            l2_method=l2_method,
        )
        if final_rows.empty:
            continue
        if REGISTRY.method_requires_lcia(l1_method, None):
            return True
        if REGISTRY.method_requires_lcia(l2_method, str(loaded.base_asocc_args["fu_code"])):
            return True
    return False


def lcia_public_row_template(*, plan: LCIAPlan) -> pd.DataFrame:
    """Return the stable final public row template for one LCIA plan."""
    pieces = [
        plan.passthrough_rows.loc[:, plan.public_columns],
        plan.direct_rows.loc[:, plan.public_columns],
        *(route.final_rows.loc[:, plan.public_columns] for route in plan.combined_routes),
    ]
    return pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)


def sample_lcia_public_value_matrix(
    *,
    plan: LCIAPlan,
    batch: RunBatch,
    shared_u: LCIASharedUMatrix | None = None,
    unit_values: np.ndarray | None = None,
) -> np.ndarray:
    """Return LCIA sampled public values as run by public row matrix."""
    parts: list[np.ndarray] = []
    blocks = _lcia_sample_blocks(plan=plan)
    shared_u = (
        lcia_shared_u_for_plan(plan=plan, batch=batch, unit_values=unit_values)
        if blocks and shared_u is None
        else shared_u
    )
    if not plan.passthrough_rows.empty:
        values = plan.passthrough_rows[ASOCC_VALUE_COLUMN].to_numpy(dtype="float64")
        parts.append(np.broadcast_to(values, (batch.n_runs, len(values))))
    if not plan.direct_rows.empty:
        parts.append(
            sample_lcia_block_matrix(
                block=cast(LCIASampleBlock, plan.direct_block),
                shared_u=cast(LCIASharedUMatrix, shared_u),
            )
        )
    parts.extend(
        combined_route_value_matrix(
            route=route,
            batch=batch,
            shared_u=cast(LCIASharedUMatrix, shared_u),
        )
        for route in plan.combined_routes
    )
    return np.concatenate(parts, axis=1)


def lcia_shared_u_for_plan(
    *,
    plan: LCIAPlan,
    batch: RunBatch,
    unit_values: np.ndarray | None = None,
) -> LCIASharedUMatrix:
    """Return one shared LCIA random matrix for a complete run scoped plan."""
    blocks = _lcia_sample_blocks(plan=plan)
    if unit_values is None:
        return shared_u_matrix_for_lcia_blocks(blocks=blocks, batch=batch)
    keys = np.unique(np.concatenate([block.unique_shared_u_keys for block in blocks]))
    values = np.broadcast_to(
        np.asarray(unit_values, dtype=np.float64)[:, None],
        (batch.n_runs, len(keys)),
    )
    return LCIASharedUMatrix(
        key_positions={str(key): index for index, key in enumerate(keys.tolist())},
        values=values,
    )


def lcia_sampling_memory_row_counts(*, plan: LCIAPlan | None) -> tuple[int, int]:
    """Return shared U key and largest sampled support row counts for planning."""
    if plan is None:
        return 0, 0
    blocks = _lcia_sample_blocks(plan=plan)
    if not blocks:
        return 0, 0
    shared_keys = np.unique(np.concatenate([block.unique_shared_u_keys for block in blocks]))
    sampled_rows = max(len(block.lower_bound) for block in blocks)
    return len(shared_keys), sampled_rows


def _lcia_sample_blocks(*, plan: LCIAPlan) -> tuple[LCIASampleBlock, ...]:
    blocks: list[LCIASampleBlock] = []
    if plan.direct_block is not None:
        blocks.append(plan.direct_block)
    for route in plan.combined_routes:
        if route.l1_block is not None:
            blocks.append(route.l1_block)
        if route.l2_block is not None:
            blocks.append(route.l2_block)
    return tuple(blocks)


def _direct_lcia_mask(*, rows: pd.DataFrame, loaded: LoadedAsoccFinalRows) -> pd.Series:
    combined = _has_text(rows, "l1_method") & _has_text(rows, "l2_method")
    l1_final = loaded.final_bucket == "level_1"
    l1_mask = _has_text(rows, "l1_method") & ~_has_text(rows, "l2_method")
    l2_mask = _has_text(rows, "l2_method") & ~combined
    fu_code = str(loaded.base_asocc_args["fu_code"])
    l1_required = _method_lcia_flags(rows=rows, column="l1_method", fu_code=None)
    l2_required = _method_lcia_flags(rows=rows, column="l2_method", fu_code=fu_code)
    return (l1_final & l1_mask & l1_required) | (l2_mask & l2_required)


def _uses_aggregate_region_covs(*, loaded: LoadedAsoccFinalRows) -> bool:
    if not bool(loaded.base_asocc_args.get("aggreg_indices")):
        return False
    return any(
        aggregate_selector_label_or_none(loaded.base_asocc_args.get(column)) is not None
        for column in _REGION_SELECTOR_COLUMNS
    )


def _has_text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    series = pd.Series(frame.loc[:, column], copy=False)
    return series.notna()


def _method_lcia_flags(
    *,
    rows: pd.DataFrame,
    column: str,
    fu_code: str | None,
) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(False, index=rows.index, dtype=bool)
    series = pd.Series(rows.loc[:, column], copy=False)
    present = series.notna()
    text = series.loc[present].astype(str)
    flags = {
        label: (not is_display_missing(label)) and REGISTRY.method_requires_lcia(label, fu_code)
        for label in text.unique()
    }
    out = pd.Series(False, index=rows.index, dtype=bool)
    out.loc[text.index] = text.map(flags).to_numpy(dtype=bool)
    return out

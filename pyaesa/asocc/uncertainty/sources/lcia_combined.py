"""Combined route evaluation for aSoCC LCIA uncertainty."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from pyaesa.asocc.methods.compute_l1 import resolve_l1_region_label
from pyaesa.asocc.methods.registry.registry import REGISTRY
from pyaesa.asocc.runtime.output.contracts import join_file_owned_tokens
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import LoadedAsoccFinalRows
from pyaesa.asocc.uncertainty.inputs.external_rows import external_method_row_mask
from pyaesa.shared.lcia.cov_inputs import LCIACoVInputs
from pyaesa.asocc.uncertainty.lcia_support.sampling import (
    LCIASampleBlock,
    LCIASharedUMatrix,
    attach_lcia_sampling_columns,
    lcia_sample_block,
    lcia_source_method_template,
    sample_lcia_block_matrix,
)
from pyaesa.asocc.uncertainty.sources.lcia_support import (
    LCIASupportRowCache,
    combined_final_rows,
    combined_route_coefficients,
    final_years,
    l2_support_years,
    support_rows,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch


@dataclass(frozen=True)
class CombinedLCIARoutePlan:
    """One deterministic two step LCIA composition route."""

    method_label: str
    final_rows: pd.DataFrame
    l1_block: LCIASampleBlock | None
    l2_block: LCIASampleBlock | None
    l1_sampled: bool
    l2_sampled: bool
    l1_coefficients: csr_matrix | None
    l2_coefficients: csr_matrix | None


def build_combined_routes(
    *,
    loaded: LoadedAsoccFinalRows,
    covs: LCIACoVInputs,
    rows: pd.DataFrame,
    affected: pd.Series,
    support_cache: LCIASupportRowCache,
    external_method_labels: tuple[str, ...],
) -> tuple[list[CombinedLCIARoutePlan], list[pd.DataFrame]]:
    """Resolve combined deterministic routes that require LCIA source sampling."""
    routes: list[CombinedLCIARoutePlan] = []
    source_templates: list[pd.DataFrame] = []
    external_rows = external_method_row_mask(frame=rows, method_labels=external_method_labels)
    eligible_rows = rows.loc[~external_rows]
    for l2_method, l1_method in loaded.asocc_scope.combined:
        final_rows = combined_final_rows(
            rows=eligible_rows,
            l1_method=l1_method,
            l2_method=l2_method,
        )
        l1_sampled = REGISTRY.method_requires_lcia(l1_method, None)
        l2_sampled = REGISTRY.method_requires_lcia(
            l2_method,
            str(loaded.base_asocc_args["fu_code"]),
        )
        if not l1_sampled and not l2_sampled:
            continue
        affected.loc[final_rows.index] = True
        weight_axis = REGISTRY.l2_weight_axis_for_method(
            l2_method,
            str(loaded.base_asocc_args["fu_code"]),
        )
        l1_axis = resolve_l1_region_label(
            l1_method=l1_method,
            fu_code=str(loaded.base_asocc_args["fu_code"]),
        )
        for raw_lcia_method, route_final in final_rows.groupby(
            "lcia_method",
            dropna=True,
            sort=False,
        ):
            lcia_method = str(raw_lcia_method)
            l1_rows = support_rows(
                loaded=loaded,
                bucket="level_1",
                stem=join_file_owned_tokens(f"l1_{l1_method}", lcia_method if l1_sampled else None),
                requested_years=final_years(final_rows=route_final),
                support_cache=support_cache,
            ).reset_index(drop=True)
            l2_rows = support_rows(
                loaded=loaded,
                bucket="l2_in_l1",
                stem=join_file_owned_tokens(f"l2_{l2_method}", lcia_method if l2_sampled else None),
                requested_years=l2_support_years(final_rows=route_final),
                support_cache=support_cache,
            ).reset_index(drop=True)
            l1_coefficients, l2_coefficients = combined_route_coefficients(
                final_rows=route_final,
                l1_rows=l1_rows,
                l2_rows=l2_rows,
                weight_axis=weight_axis,
                l1_axis=l1_axis,
                l1_sampled=l1_sampled,
            )
            if l1_sampled:
                l1_rows = attach_lcia_sampling_columns(
                    rows=l1_rows,
                    loaded=loaded,
                    covs=covs,
                    applied_bucket="level_1",
                    allocation_column="l1_method",
                    weight_axis=None,
                )
                source_templates.append(lcia_source_method_template(rows=l1_rows))
            if l2_sampled:
                l2_rows = attach_lcia_sampling_columns(
                    rows=l2_rows,
                    loaded=loaded,
                    covs=covs,
                    applied_bucket="l2_in_l1",
                    allocation_column="l2_method",
                    weight_axis=weight_axis,
                )
                source_templates.append(lcia_source_method_template(rows=l2_rows))
            routes.append(
                CombinedLCIARoutePlan(
                    method_label=str(route_final["l1_l2_method"].iloc[0]),
                    final_rows=route_final.reset_index(drop=True),
                    l1_block=lcia_sample_block(template=l1_rows) if l1_sampled else None,
                    l2_block=lcia_sample_block(template=l2_rows) if l2_sampled else None,
                    l1_sampled=l1_sampled,
                    l2_sampled=l2_sampled,
                    l1_coefficients=l1_coefficients,
                    l2_coefficients=l2_coefficients,
                )
            )
    return routes, source_templates


def combined_route_value_matrix(
    *,
    route: CombinedLCIARoutePlan,
    batch: RunBatch,
    shared_u: LCIASharedUMatrix,
) -> np.ndarray:
    """Evaluate one combined route as a run by public row value matrix."""
    if route.l1_sampled:
        l1 = sample_lcia_block_matrix(
            block=cast(LCIASampleBlock, route.l1_block),
            shared_u=shared_u,
        )
        return _support_matrix_product(
            values=l1,
            coefficients=cast(csr_matrix, route.l1_coefficients),
        )
    l2 = sample_lcia_block_matrix(
        block=cast(LCIASampleBlock, route.l2_block),
        shared_u=shared_u,
    )
    return _support_matrix_product(
        values=l2,
        coefficients=cast(csr_matrix, route.l2_coefficients),
    )


def _support_matrix_product(*, values: np.ndarray, coefficients: csr_matrix) -> np.ndarray:
    return np.asarray(coefficients.T.dot(values.T)).T

"""aSoCC uncertainty summary identity grouping."""

import pandas as pd

from pyaesa.asocc.uncertainty.sources.names import (
    INTER_METHOD_SOURCE,
    PROJECTION_SOURCE,
    REFERENCE_YEAR_SOURCE,
)
from pyaesa.shared.runtime.scenario.columns import ASOCC_TIME_ROUTE_PUBLIC_COLUMN
from pyaesa.shared.runtime.scenario.time_routes import collapse_asocc_time_route
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    identity_groups_from_excluded_columns,
)
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan

ASOCC_SUMMARY_SCOPE_COLUMN = "asocc_summary_scope"
ASOCC_SUMMARY_SCOPE_PER_METHOD = "per_method"
ASOCC_SUMMARY_SCOPE_INTER_METHOD = "inter_method"


def summary_identity_groups(
    *,
    identity: pd.DataFrame,
    sources: SourceActivationPlan,
    inter_method_only: bool = False,
) -> tuple[pd.DataFrame, tuple[tuple[str, ...], ...]]:
    """Return exact summary identity rows and backing public row id groups."""
    sampled_axes = set()
    if sources.is_active(PROJECTION_SOURCE):
        sampled_axes.add("l2_reuse_year")
    if sources.is_active(REFERENCE_YEAR_SOURCE):
        sampled_axes.add("reference_year")
    if not sources.is_active(INTER_METHOD_SOURCE):
        return _scoped_summary_groups(
            identity=identity,
            excluded_columns=sampled_axes,
            scope=ASOCC_SUMMARY_SCOPE_PER_METHOD,
        )
    pieces: list[pd.DataFrame] = []
    groups: list[tuple[str, ...]] = []
    if not inter_method_only:
        scope_identity, scope_groups = _scoped_summary_groups(
            identity=identity,
            excluded_columns=sampled_axes,
            scope=ASOCC_SUMMARY_SCOPE_PER_METHOD,
        )
        pieces.append(scope_identity)
        groups.extend(scope_groups)
    scope_identity, scope_groups = _scoped_summary_groups(
        identity=identity,
        excluded_columns={
            *sampled_axes,
            "l1_l2_method",
            "l1_method",
            "l2_method",
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
        },
        scope=ASOCC_SUMMARY_SCOPE_INTER_METHOD,
    )
    pieces.append(scope_identity)
    groups.extend(scope_groups)
    return pd.concat(pieces, ignore_index=True, sort=False), tuple(groups)


def _scoped_summary_groups(
    *,
    identity: pd.DataFrame,
    excluded_columns: set[str],
    scope: str,
) -> tuple[pd.DataFrame, tuple[tuple[str, ...], ...]]:
    summary_identity, public_row_groups = identity_groups_from_excluded_columns(
        identity=identity,
        excluded_columns=excluded_columns,
    )
    if (
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN in excluded_columns
        and ASOCC_TIME_ROUTE_PUBLIC_COLUMN in identity.columns
    ):
        summary_identity = summary_identity.copy()
        summary_identity[ASOCC_TIME_ROUTE_PUBLIC_COLUMN] = _collapsed_time_routes_for_groups(
            identity=identity,
            public_row_groups=public_row_groups,
        )
    summary_identity = summary_identity.copy()
    summary_identity[ASOCC_SUMMARY_SCOPE_COLUMN] = scope
    return summary_identity, public_row_groups


def _collapsed_time_routes_for_groups(
    *,
    identity: pd.DataFrame,
    public_row_groups: tuple[tuple[str, ...], ...],
) -> list[object]:
    route_by_public_id = identity.set_index("public_row_id")[ASOCC_TIME_ROUTE_PUBLIC_COLUMN]
    return [
        collapse_asocc_time_route(route_by_public_id.loc[[int(value) for value in group]].tolist())
        for group in public_row_groups
    ]

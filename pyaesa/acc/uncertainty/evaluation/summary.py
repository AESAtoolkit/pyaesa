"""aCC uncertainty summary identity grouping."""

import pandas as pd

from pyaesa.acc.uncertainty.sources.source_keys import (
    AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE,
    ASOCC_INTER_METHOD_SOURCE,
    ASOCC_PROJECTION_SOURCE,
    ASOCC_REFERENCE_YEAR_SOURCE,
)
from pyaesa.shared.runtime.scenario.columns import ASOCC_TIME_ROUTE_PUBLIC_COLUMN
from pyaesa.shared.runtime.scenario.time_routes import collapse_asocc_time_route
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    identity_groups_from_excluded_columns,
)

ACC_SUMMARY_SCOPE_COLUMN = "acc_summary_scope"
ACC_SUMMARY_SCOPE_PER_METHOD = "per_method"
ACC_SUMMARY_SCOPE_INTER_METHOD = "inter_method"


def acc_summary_identity_groups(
    *,
    identity: pd.DataFrame,
    active_sources: tuple[str, ...],
    dynamic_category_uncertainty_active: bool = False,
) -> tuple[pd.DataFrame, tuple[tuple[str, ...], ...]]:
    """Return ACC summary rows and backing public row groups for sampled axes."""
    excluded_columns = acc_summary_excluded_columns(
        active_sources=active_sources,
        dynamic_category_uncertainty_active=dynamic_category_uncertainty_active,
    )
    if ASOCC_INTER_METHOD_SOURCE not in set(active_sources):
        return _scoped_summary_groups(
            identity=identity,
            excluded_columns=excluded_columns,
            scope=ACC_SUMMARY_SCOPE_PER_METHOD,
        )
    method_excluded = set(excluded_columns)
    method_excluded.difference_update(
        {"l1_l2_method", "l1_method", "l2_method", ASOCC_TIME_ROUTE_PUBLIC_COLUMN}
    )
    method_identity, method_groups = _scoped_summary_groups(
        identity=identity,
        excluded_columns=method_excluded,
        scope=ACC_SUMMARY_SCOPE_PER_METHOD,
    )
    inter_identity, inter_groups = _scoped_summary_groups(
        identity=identity,
        excluded_columns=excluded_columns,
        scope=ACC_SUMMARY_SCOPE_INTER_METHOD,
    )
    return (
        pd.concat([method_identity, inter_identity], ignore_index=True, sort=False),
        (*method_groups, *inter_groups),
    )


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
    summary_identity[ACC_SUMMARY_SCOPE_COLUMN] = scope
    return summary_identity, public_row_groups


def acc_summary_excluded_columns(
    *,
    active_sources: tuple[str, ...],
    dynamic_category_uncertainty_active: bool = False,
) -> set[str]:
    """Return public identity columns represented by sampled ACC source axes."""
    excluded = set()
    if ASOCC_PROJECTION_SOURCE in active_sources:
        excluded.add("l2_reuse_year")
    if ASOCC_REFERENCE_YEAR_SOURCE in active_sources:
        excluded.add("reference_year")
    if ASOCC_INTER_METHOD_SOURCE in active_sources:
        excluded.update({"l1_l2_method", "l1_method", "l2_method"})
        excluded.add(ASOCC_TIME_ROUTE_PUBLIC_COLUMN)
    if AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE in active_sources:
        excluded.update({"cc_model", "cc_scenario"})
    if dynamic_category_uncertainty_active:
        excluded.add("cc_category")
    return excluded


def acc_dynamic_category_uncertainty_active_from_manifest(
    *,
    manifest: UncertaintyManifest,
) -> bool:
    """Return whether one ACC manifest used active AR6 category uncertainty."""
    source_parameters = manifest.source_parameters or {}
    return bool(source_parameters.get("dynamic_cc_category_uncertainty", False))


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

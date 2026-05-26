"""Canonical figure-scope coverage ownership for internal IO-LCA rendering."""

import pandas as pd

from pyaesa.shared.runtime.reuse.contracts import normalize_selector_payload

from ..data.metadata import (
    compatible_scope,
    get_lcia_method_done_and_skipped_years,
    scope_complete_and_existing,
)


def done_and_skipped_lcia_years(*, scope: dict, lcia_method: str) -> tuple[set[int], set[int]]:
    """Return done/skipped years for one LCIA method from deterministic_io_lca metadata."""
    return get_lcia_method_done_and_skipped_years(
        scope=scope,
        section="main",
        lcia_method=lcia_method,
    )


def validate_lcia_method_coverage(
    *,
    io_scope: dict,
    lcia_method: str,
    years: list[int],
) -> None:
    """Raise when requested figure years are not covered by deterministic_io_lca outputs."""
    done, skipped = done_and_skipped_lcia_years(scope=io_scope, lcia_method=lcia_method)
    requested = {int(year) for year in years}
    missing = sorted(requested - done - skipped)
    if missing:
        raise ValueError(
            "Requested figure years are missing from deterministic_io_lca outputs for LCIA "
            f"method '{lcia_method}'. Missing years: {missing}. "
            "Re-run deterministic_io_lca(...) first."
        )


def require_main_result_columns(
    *,
    frame: pd.DataFrame,
    lcia_method: str,
    selector_axes: tuple[str, ...],
) -> None:
    """Validate required columns before generating figures."""
    required = {"year", "impact", "lca_value", "impact_unit"}
    required.update(selector_axes)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "deterministic_io_lca results for LCIA method "
            f"'{lcia_method}' are missing required columns: {missing}."
        )


def resolve_io_scope(
    *,
    io_log_payload: dict,
    project_name: str,
    source: str,
    agg_reg: bool,
    agg_sec: bool,
    agg_version: str | None,
    years: list[int],
    lcia_methods: list[str],
    fu_code: str,
    filters: dict[str, list[str] | None],
    group_indices: bool,
) -> tuple[str, dict]:
    """Resolve compatible deterministic_io_lca run scope and output format for figures."""
    requested_years = {int(year) for year in years}
    requested_lcia_methods = {str(lcia_method) for lcia_method in lcia_methods}
    requested_selectors = normalize_selector_payload(
        filters,
        context="deterministic_io_lca figure filters",
    )
    found_incomplete = False
    for io_output_format in ("csv", "pickle", "parquet"):
        scope = compatible_scope(
            payload=io_log_payload,
            project_name=project_name,
            source=source,
            agg_reg=agg_reg,
            agg_sec=agg_sec,
            agg_version=agg_version,
            fu_code=fu_code,
            group_indices=group_indices,
            output_format=io_output_format,
            requested_years=requested_years,
            requested_methods=requested_lcia_methods,
            requested_selectors=requested_selectors,
        )
        if scope is None:
            continue
        if scope_complete_and_existing(scope):
            return io_output_format, scope
        found_incomplete = True
    if found_incomplete:
        raise ValueError(
            "A matching deterministic_io_lca run exists but is incomplete or has missing outputs. "
            "Re-run deterministic_io_lca(...) first."
        )
    raise ValueError(
        "No compatible deterministic_io_lca outputs were found for this figure request. "
        "Run deterministic_io_lca(...) first."
    )

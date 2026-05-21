"""Deterministic prerequisite orchestration for aSoCC uncertainty."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pyaesa.asocc.deterministic_asocc import deterministic_asocc
from pyaesa.asocc.runtime.scope.branch_resolution import (
    AsoccDeterministicPathScope,
    allocate_run_metadata_path,
)
from pyaesa.asocc.runtime.paths.family_roots import is_native_asocc_source
from pyaesa.asocc.runtime.request.defaults import DETERMINISTIC_ASOCC_OPTIONAL_DEFAULTS
from pyaesa.asocc.runtime.request.scope import AsoccScope, build_asocc_scope
from pyaesa.asocc.runtime.request.normalization import normalize_base_allocate_args
from pyaesa.asocc.runtime.scope.persisted_scope import (
    AsoccPersistedRunCatalog,
    AsoccPersistedRunScope,
    load_asocc_persisted_run_catalog,
)
from pyaesa.asocc.uncertainty.sources.names import REFERENCE_YEAR_SOURCE
from pyaesa.asocc.io.metadata import _load_run_metadata
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
from pyaesa.shared.runtime.reuse.contracts import asocc_signature_matches_request
from pyaesa.shared.tabular.table_io import read_table

_ROW_COVERED_SIGNATURE_FIELDS = {
    "reference_years_input": "reference_year",
    "l2_reuse_years": "l2_reuse_year",
}
_UNCERTAINTY_BASE_FORBIDDEN_KEYS = {
    "figures",
    "figure_options",
    "figure_format",
    "figure_external_method",
}


@dataclass(frozen=True)
class AsoccDeterministicPrerequisite:
    """Resolved deterministic aSoCC prerequisite request."""

    base_asocc_args: dict[str, Any]
    asocc_scope: AsoccScope
    path_scope: AsoccDeterministicPathScope
    deterministic_manifest_path: Path
    persisted_scope_matches: tuple[tuple[AsoccPersistedRunScope, list[int]], ...]
    reuse_status: str


def prepare_asocc_deterministic_prerequisite(
    *,
    base_asocc_args: dict[str, Any],
    refresh: bool,
    reference_year_uncertainty_active: bool = False,
    figures: bool = False,
    figure_format: dict[str, Any] | None = None,
    figure_options: dict[str, bool] | None = None,
    figure_external_method: dict[str, Any] | None = None,
    phase: PhasePrinter | NullPhasePrinter | None = None,
) -> AsoccDeterministicPrerequisite:
    """Materialize and resolve the deterministic prerequisite for aSoCC uncertainty."""
    normalized = normalize_base_allocate_args(
        base_asocc_args,
        additional_forbidden_keys=_UNCERTAINTY_BASE_FORBIDDEN_KEYS,
    )
    _validate_reference_year_request(
        base_asocc_args=normalized,
        reference_year_uncertainty_active=reference_year_uncertainty_active,
    )
    if is_native_asocc_source(source=str(normalized["source"])):
        asocc_scope = build_asocc_scope(base_allocate_args=normalized)
        path_scope = asocc_scope.resolve_path_scope()
        metadata_path = allocate_run_metadata_path(scope=path_scope)
        reuse_status = "reused_exact"
        scope_matches = _load_matching_persisted_scopes(
            metadata_path=metadata_path,
            asocc_scope=asocc_scope,
            requested_years=normalized["years"],
        )
        if (
            figures
            or refresh
            or not _scope_matches_cover_request(
                matches=scope_matches,
                requested_years=normalized["years"],
            )
        ):
            report = deterministic_asocc(
                **_deterministic_call_args(
                    base_asocc_args=normalized,
                    refresh=refresh,
                    figures=figures,
                    figure_format=figure_format,
                    figure_options=figure_options,
                    figure_external_method=figure_external_method,
                ),
                _phase=phase,
            )
            reuse_status = report.reuse_status
            scope_matches = _load_matching_persisted_scopes(
                metadata_path=metadata_path,
                asocc_scope=asocc_scope,
                requested_years=normalized["years"],
            )
    else:
        asocc_scope = build_asocc_scope(base_allocate_args=normalized)
        path_scope, metadata_path = asocc_scope.resolve_disaggregation_scope(
            source_label=str(normalized["source"])
        )
        catalog = load_asocc_persisted_run_catalog(payload=_load_run_metadata(metadata_path))
        normalized, asocc_scope, scope_matches = _disaggregated_scope_matches(
            base_asocc_args=normalized,
            asocc_scope=asocc_scope,
            catalog=catalog,
        )
        reuse_status = "reused_exact"
    return AsoccDeterministicPrerequisite(
        base_asocc_args=normalized,
        asocc_scope=asocc_scope,
        path_scope=path_scope,
        deterministic_manifest_path=metadata_path,
        persisted_scope_matches=scope_matches,
        reuse_status=reuse_status,
    )


def _load_matching_persisted_scopes(
    *,
    metadata_path: Path,
    asocc_scope: AsoccScope,
    requested_years: list[int],
) -> tuple[tuple[AsoccPersistedRunScope, list[int]], ...]:
    if not metadata_path.exists():
        return ()
    catalog = load_asocc_persisted_run_catalog(payload=_load_run_metadata(metadata_path))
    return _load_matching_persisted_scopes_from_catalog(
        catalog=catalog,
        asocc_scope=asocc_scope,
        requested_years=requested_years,
    )


def _load_matching_persisted_scopes_from_catalog(
    *,
    catalog: AsoccPersistedRunCatalog,
    asocc_scope: AsoccScope,
    requested_years: list[int],
) -> tuple[tuple[AsoccPersistedRunScope, list[int]], ...]:
    """Return persisted deterministic scopes matching one aSoCC request."""
    return tuple(
        _matching_persisted_scopes(
            asocc_scope=asocc_scope,
            scopes=list(catalog.scopes),
            requested_years=requested_years,
            run_ssp_scenarios=catalog.run_ssp_scenarios,
        )
    )


def _scope_matches_cover_request(
    *,
    matches: tuple[tuple[AsoccPersistedRunScope, list[int]], ...],
    requested_years: list[int],
) -> bool:
    matched_years = set().union(*(set(years) for _scope, years in matches)) if matches else set()
    return set(requested_years).issubset(matched_years)


def _matching_persisted_scopes(
    *,
    asocc_scope: AsoccScope,
    scopes: list[AsoccPersistedRunScope],
    requested_years: list[int],
    run_ssp_scenarios: list[str] | None,
) -> list[tuple[AsoccPersistedRunScope, list[int]]]:
    matches: list[tuple[AsoccPersistedRunScope, list[int]]] = []
    requested_set = {int(year) for year in requested_years}
    for scope in scopes:
        completed = {int(year) for year in scope.completed_years}
        years = sorted(requested_set & completed)
        if not years:
            continue
        signature = asocc_scope.compute_signature(
            years=years,
            output_format=scope.output_format,
            intermediate_outputs=scope.intermediate_outputs,
            historical_year_cap=scope.compute_signature.as_dict().get("historical_year_cap"),
            variant_tag=scope.variant_tag,
        )
        signature = _row_coverage_canonical_signature(
            requested_signature=signature,
            scope=scope,
            years=years,
        )
        if signature is None:
            continue
        if asocc_signature_matches_request(
            requested_signature=signature,
            scope=scope,
            run_ssp_scenarios=run_ssp_scenarios,
        ):
            matches.append((scope, years))
    return sorted(
        matches,
        key=lambda match: (
            len(set(match[0].completed_years) - set(match[1])),
            len(match[0].completed_years),
            str(match[0].timestamp or ""),
        ),
    )


def _row_coverage_canonical_signature(
    *,
    requested_signature: dict[str, Any],
    scope: AsoccPersistedRunScope,
    years: list[int],
) -> dict[str, Any] | None:
    candidate_signature = scope.compute_signature.as_dict()
    canonical_signature = dict(requested_signature)
    needed_axes: list[tuple[str, str, set[int]]] = []
    for signature_field, row_column in _ROW_COVERED_SIGNATURE_FIELDS.items():
        requested_values = _optional_int_set(canonical_signature.get(signature_field))
        if requested_values is None or candidate_signature.get(signature_field) is not None:
            continue
        needed_axes.append((signature_field, row_column, requested_values))
    if not needed_axes:
        return canonical_signature
    row_coverage = _persisted_output_axis_values(
        scope=scope,
        years=years,
        axis_columns={row_column for _signature_field, row_column, _requested in needed_axes},
    )
    for signature_field, row_column, requested_values in needed_axes:
        persisted_values = row_coverage[row_column]
        if persisted_values and not requested_values.issubset(persisted_values):
            return None
        canonical_signature[signature_field] = None
    return canonical_signature


def _persisted_output_axis_values(
    *,
    scope: AsoccPersistedRunScope,
    years: list[int],
    axis_columns: set[str],
) -> dict[str, set[int]]:
    values = {column: set() for column in axis_columns}
    for output in scope.outputs:
        frame = read_table(path=Path(output))
        year_columns = [str(year) for year in years if str(year) in frame.columns]
        live_rows = frame.loc[:, year_columns].notna().any(axis=1)
        for row_column in axis_columns & set(frame.columns):
            raw_values = frame.loc[live_rows, row_column]
            meaningful = raw_values.loc[raw_values.notna()]
            numeric_values = pd.Series(pd.to_numeric(meaningful, errors="raise"), copy=False)
            values[row_column].update(int(value) for value in numeric_values.tolist())
    return values


def _optional_int_set(value: Any) -> set[int] | None:
    if value is None:
        return None
    return {int(item) for item in value}


def _deterministic_call_args(
    *,
    base_asocc_args: dict[str, Any],
    refresh: bool,
    figures: bool,
    figure_format: dict[str, Any] | None,
    figure_options: dict[str, bool] | None,
    figure_external_method: dict[str, Any] | None,
) -> dict[str, Any]:
    keys = (
        set(DETERMINISTIC_ASOCC_OPTIONAL_DEFAULTS)
        - {"output_format", "intermediate_outputs", "refresh"}
    ) | {"project_name", "source", "fu_code"}
    call_args = {key: value for key, value in base_asocc_args.items() if key in keys}
    call_args.update(
        {
            "figures": figures,
            "figure_format": figure_format,
            "figure_options": figure_options or {"per_method": True, "multi_method": True},
            "figure_external_method": figure_external_method,
            "refresh": refresh,
        }
    )
    return call_args


def _disaggregated_scope_matches(
    *,
    base_asocc_args: dict[str, Any],
    asocc_scope: AsoccScope,
    catalog: AsoccPersistedRunCatalog,
) -> tuple[dict[str, Any], AsoccScope, tuple[tuple[AsoccPersistedRunScope, list[int]], ...]]:
    """Match a disaggregated source using projection fields persisted in its manifest."""
    requested_years = base_asocc_args["years"]
    matches = _load_matching_persisted_scopes_from_catalog(
        catalog=catalog,
        asocc_scope=asocc_scope,
        requested_years=requested_years,
    )
    return base_asocc_args, asocc_scope, matches


def _validate_reference_year_request(
    *,
    base_asocc_args: dict[str, Any],
    reference_year_uncertainty_active: bool,
) -> None:
    reference_years = base_asocc_args.get("reference_years")
    years = base_asocc_args.get("years")
    if reference_years is None or years is None:
        return
    if reference_year_uncertainty_active:
        blocked_years = [
            int(year)
            for year in years
            if not any(int(reference_year) <= int(year) for reference_year in reference_years)
        ]
        if blocked_years:
            raise ValueError(
                f"{REFERENCE_YEAR_SOURCE} requires at least one requested reference_year "
                f"less than or equal to studied year {blocked_years[0]}."
            )
        return
    invalid = [
        (int(reference_year), int(year))
        for reference_year in reference_years
        for year in years
        if int(reference_year) > int(year)
    ]
    if invalid:
        reference_year, year = invalid[0]
        raise ValueError(
            "reference_years cannot be greater than studied years unless "
            f"{REFERENCE_YEAR_SOURCE} is active. "
            f"Found reference_year={reference_year} for year={year}."
        )

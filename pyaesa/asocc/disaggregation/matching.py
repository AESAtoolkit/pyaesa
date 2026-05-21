"""Prerequisite scope matching for published-output disaggregation."""

from pathlib import Path

from pyaesa.shared.runtime.reuse.contracts import (
    asocc_signature_matches_request,
)

from pyaesa.asocc.io.metadata import _load_run_metadata
from ..orchestration.common_formatting import format_year_ranges
from ..orchestration.setup.run_setup import _prepare_context
from pyaesa.asocc.runtime.scope.branch_resolution import (
    allocate_run_metadata_path,
    path_scope_from_signature,
)
from pyaesa.asocc.runtime.scope.persisted_scope import load_asocc_persisted_run_catalog
from .models import MatchedRun


class _MissingYearsError(ValueError):
    """Internal carrier for explicit missing-year payloads."""

    def __init__(self, missing_years: list[int]) -> None:
        super().__init__(_format_years_with_count(missing_years))
        self.missing_years = list(missing_years)


def _format_years_with_count(years: list[int]) -> str:
    """Return compact requested-year wording."""
    ordered = sorted({int(year) for year in years})
    count = len(ordered)
    label = "year" if count == 1 else "years"
    return f"{format_year_ranges(ordered)} ({count} {label})"


def _build_user_rerun_message(
    *,
    selector_name: str,
    request,
    missing_years: list[int],
) -> str:
    """Return the user-facing rerun guidance for one missing prerequisite scope."""
    lines = [f"Missing prerequisite deterministic_asocc outputs for selector '{selector_name}'."]
    scope_parts = [
        f"source='{request.source}'",
        f"group_reg={bool(request.group_reg)}",
        f"group_sec={bool(request.group_sec)}",
    ]
    if request.group_version:
        scope_parts.append(f"group_version='{request.group_version}'")
    lines.append("Selector scope: " + ", ".join(scope_parts))
    lines.append(f"Sectors (s_p): {request.s_p}")
    mode = "grouped" if bool(request.aggreg_indices) else "ungrouped"
    lines.append(f"Branch mode: l1_reg_aggreg='{request.l1_reg_aggreg}', aggreg_indices='{mode}'")
    lines.append(f"Missing years to run: {_format_years_with_count(missing_years)}.")
    lines.append("Reason: missing required published output coverage.")
    lines.append(
        "Action: re-run deterministic_asocc for this selector with the same "
        "base_asocc_args and the missing years above."
    )
    return "\n".join(lines)


def _run_metadata_path_for_request(*, selector_name: str, context) -> Path:
    """Return the scope manifest path for one prepared selector context."""
    scope = path_scope_from_signature(
        proj_base=context.proj_base,
        source_label=context.output_source,
        run_signature=context.run_signature,
        context_label=f"Disaggregation selector '{selector_name}' metadata lookup",
    )
    return allocate_run_metadata_path(scope=scope)


def _pick_scope(
    *,
    request_signature: dict,
    catalog,
    requested_years: list[int],
) -> tuple[str, dict, list[int]]:
    """Return the best reusable persisted scope for one selector request."""
    candidates = [
        scope
        for scope in catalog.scopes
        if asocc_signature_matches_request(
            requested_signature=request_signature,
            scope=scope,
            run_ssp_scenarios=catalog.run_ssp_scenarios,
        )
    ]
    exact = [scope for scope in candidates if scope.covers_years(requested_years)]
    if exact:
        exact.sort(
            key=lambda scope: (
                len(set(scope.completed_years) - set(requested_years)),
                len(scope.completed_years),
                scope.scope_key,
            )
        )
        picked = exact[0]
        return picked.scope_key, picked.compute_signature.as_dict(), list(picked.completed_years)
    covered_years = sorted({year for scope in candidates for year in scope.completed_years})
    missing = sorted(set(int(year) for year in requested_years) - set(covered_years))
    raise _MissingYearsError(missing)


def match_selector_scope(
    *,
    selector_name: str,
    request,
    requested_years: list[int],
) -> MatchedRun:
    """Resolve one prerequisite deterministic scope for the exact requested years."""
    context, _state, _skipped = _prepare_context(request=request)
    run_meta_path = _run_metadata_path_for_request(selector_name=selector_name, context=context)
    if not run_meta_path.exists():
        raise ValueError(
            f"No deterministic_asocc run metadata found for selector '{selector_name}'. "
            f"Expected: {run_meta_path}"
        )
    catalog = load_asocc_persisted_run_catalog(payload=_load_run_metadata(run_meta_path))
    requested_signature = context.run_signature
    try:
        scope_key, scope_signature, completed_years = _pick_scope(
            request_signature=requested_signature,
            catalog=catalog,
            requested_years=requested_years,
        )
    except _MissingYearsError as exc:
        raise ValueError(
            _build_user_rerun_message(
                selector_name=selector_name,
                request=request,
                missing_years=exc.missing_years,
            )
        ) from None
    return MatchedRun(
        selector_name=selector_name,
        proj_base=context.proj_base,
        run_metadata_path=run_meta_path,
        scope_key=scope_key,
        scope_signature=scope_signature,
        completed_years=completed_years,
        output_source_label=context.output_source,
    )

"""Projection argument normalization and validation."""

from ....data.source_schema import default_regression_window_for_source
from ...method_scope import _unique_l2_methods_in_scope
from ...method_scope import _max_historical_mrio_year
from ....methods.registry.registry import REGISTRY
from .types import (
    PROJECTION_MODES,
    ProjectionContext,
    RegWindowBounds,
    RegWindowSelector,
    UT_ADJUSTED_METHODS,
)


def list_ut_l2_methods_in_scope(
    *,
    fu_code: str,
    selected_l2_one_step: list[str],
    combined: list[tuple[str, str]],
) -> list[str]:
    """Return deterministic unique UT L2-method names selected for the run."""
    return [
        l2_method
        for l2_method in _unique_l2_methods_in_scope(
            selected_l2_one_step=selected_l2_one_step,
            combined=combined,
        )
        if REGISTRY.method_is_ut(l2_method, level="L2", fu_code=fu_code)
    ]


def build_l2_method_route_by_name(
    *,
    ut_methods: list[str],
    mode: str,
) -> dict[str, str]:
    """Return L2-method routing map for post historical UT years."""
    route: dict[str, str] = {}
    for l2_method in ut_methods:
        if l2_method in UT_ADJUSTED_METHODS:
            route[l2_method] = "historical_reuse"
            continue
        route[l2_method] = mode
    return route


def split_historical_future_years(
    *,
    years: list[int],
    max_historical_year: int,
) -> tuple[list[int], list[int]]:
    """Split resolved years into historical and future partitions."""
    historical = [int(y) for y in years if int(y) <= int(max_historical_year)]
    future = [int(y) for y in years if int(y) > int(max_historical_year)]
    return historical, future


def _normalize_projection_mode(mode: str | None) -> str | None:
    """Validate projection mode selector."""
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if normalized not in PROJECTION_MODES:
        raise ValueError(
            f"Unsupported projection_mode '{mode}'. Use one of: {sorted(PROJECTION_MODES)}."
        )
    return normalized


def _normalize_year_selector(
    *,
    value: int | list[int] | range | None,
    name: str,
) -> list[int]:
    """Convert a year selector API value into a sorted unique list."""
    if value is None:
        return []
    years: list[int] = []
    if isinstance(value, int):
        years = [int(value)]
    elif isinstance(value, range):
        years = [int(year) for year in value]
    elif isinstance(value, list):
        years = [int(year) for year in value]
    else:
        raise ValueError(f"Unsupported {name} selector type '{type(value).__name__}'.")
    return sorted(set(years))


def _require_years_available(
    *,
    years: list[int],
    historical_years: list[int],
    label: str,
) -> None:
    """Fail fast when requested years are outside historical coverage."""
    if not years:
        return
    historical = set(int(y) for y in historical_years)
    missing = sorted(int(y) for y in years if int(y) not in historical)
    if missing:
        max_historical = _max_historical_mrio_year(historical_years=historical_years)
        if historical_years and max_historical is not None:
            coverage_label = f"{int(min(historical_years))}-{int(max_historical)}"
        else:
            coverage_label = "unknown"
        raise ValueError(
            f"{label} must be fully available in historical MRIO years. "
            f"Coverage: {coverage_label}. Missing years: {missing}."
        )


def resolve_projection_context(
    *,
    source: str,
    fu_code: str,
    resolved_years: list[int],
    historical_years: list[int],
    selected_l2_one_step: list[str],
    combined: list[tuple[str, str]],
    projection_mode: str | None,
    reg_window: RegWindowSelector,
    l2_reuse_years: int | list[int] | range | None,
) -> ProjectionContext:
    """Resolve projection context for one run branch."""
    max_historical = max(int(year) for year in historical_years)
    _, future_years = split_historical_future_years(
        years=resolved_years,
        max_historical_year=max_historical,
    )
    ut_l2_methods = list_ut_l2_methods_in_scope(
        fu_code=fu_code,
        selected_l2_one_step=selected_l2_one_step,
        combined=combined,
    )
    if not ut_l2_methods or not future_years:
        return ProjectionContext(
            enabled=False,
            mode=None,
            max_historical_year=max_historical,
            future_years=tuple(sorted(set(int(y) for y in future_years))),
            reg_window=None,
            l2_reuse_years=tuple(),
            ut_methods_in_scope=tuple(ut_l2_methods),
            l2_method_route_by_name={},
        )

    # Default route for future UT years is regression unless explicitly set.
    mode = _normalize_projection_mode(projection_mode) or "regression"
    reg_window_norm: RegWindowBounds = None
    if reg_window is None:
        reg_window_norm = default_regression_window_for_source(source)
    else:
        reg_years_input = [int(year) for year in reg_window]
        if not reg_years_input:
            raise ValueError("reg_window must contain at least one year.")
        expected_reg_years = list(range(int(reg_years_input[0]), int(reg_years_input[-1]) + 1))
        if reg_years_input != expected_reg_years:
            raise ValueError(
                f"reg_window must define consecutive years with step 1. Got {reg_years_input}."
            )
        reg_window_norm = (int(reg_years_input[0]), int(reg_years_input[-1]))
    if reg_window_norm is None:
        reg_window_norm = (max_historical, max_historical)
    reg_years = list(range(reg_window_norm[0], reg_window_norm[1] + 1))
    _require_years_available(
        years=reg_years,
        historical_years=historical_years,
        label="reg_window",
    )

    adjusted_ut_in_scope = any(l2_method in UT_ADJUSTED_METHODS for l2_method in ut_l2_methods)
    reuse_selector_required = (mode == "historical_reuse") or adjusted_ut_in_scope
    if not reuse_selector_required:
        if l2_reuse_years is not None:
            raise ValueError(
                "Argument 'l2_reuse_years' is not applicable for this configuration: "
                "projection_mode='regression' with no adjusted UT methods in scope."
            )
        l2_reuse_years_norm: list[int] = []
    else:
        l2_reuse_years_norm = _normalize_year_selector(
            value=l2_reuse_years,
            name="l2_reuse_years",
        )
        # When reuse routing is active and selector is omitted, default to reg_window.
        if not l2_reuse_years_norm:
            l2_reuse_years_norm = list(reg_years)
        _require_years_available(
            years=l2_reuse_years_norm,
            historical_years=historical_years,
            label="l2_reuse_years",
        )

    l2_method_route_by_name = build_l2_method_route_by_name(
        ut_methods=ut_l2_methods,
        mode=mode,
    )
    return ProjectionContext(
        enabled=True,
        mode=mode,
        max_historical_year=max_historical,
        future_years=tuple(sorted(set(int(y) for y in future_years))),
        reg_window=reg_window_norm,
        l2_reuse_years=tuple(sorted(set(l2_reuse_years_norm))),
        ut_methods_in_scope=tuple(ut_l2_methods),
        l2_method_route_by_name=l2_method_route_by_name,
    )


def required_projection_years(
    *,
    projection_context: ProjectionContext,
) -> list[int]:
    """Return historical studied years required by projection configuration."""
    if not projection_context.enabled:
        return []
    required: set[int] = set()
    if projection_context.mode == "regression" and projection_context.reg_window is not None:
        start, end = projection_context.reg_window
        required.update(range(int(start), int(end) + 1))
    required.update(int(year) for year in projection_context.l2_reuse_years)
    return sorted(required)

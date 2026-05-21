"""Scenario-to-file routing for yearly outputs."""

from collections.abc import Callable

from pyaesa.asocc.runtime.reporting.family import emit_deduplicated_family_warning
from ....io.metadata import RunContext, RunState
from ....methods.registry.registry import REGISTRY


def resolve_output_ssp_scenario(
    *,
    context,
    year: int,
    ssp_scenario: str | None,
    scenario_dependent: bool,
) -> str | None:
    """Route WB backed years to base files and SSP backed years to scenario files."""
    if not scenario_dependent:
        return None
    wb_year_columns = {str(col) for col in context.wb_df.columns if str(col).isdigit()}
    return None if str(int(year)) in wb_year_columns else ssp_scenario


def emit_notice(
    *,
    context: RunContext,
    state: RunState,
    key: str,
    message: str,
) -> None:
    """Emit one deduplicated runtime notice to the aSoCC runtime reporter.

    The key is used as a once only guard so long year loops do not spam
    repeated warnings with the same root cause.
    """
    emit_deduplicated_family_warning(
        context=context,
        state=state,
        key=key,
        message=message,
    )


def emit_pr_hr_rp1_zero_fallback_notice(
    *,
    context: RunContext,
    state: RunState,
    l1_method: str,
    boundaries: list[str],
    impacts: list[str],
    target_year: int,
    fallback_year: int,
    use_original_domain: bool,
    ssp_scenario: str | None,
) -> None:
    """Emit one deduplicated notice for the PR-HR RP=1 zero year fallback."""
    impacts_text = ", ".join(sorted(str(value) for value in impacts))
    boundary_order = {"CBA_FD": 0, "PBA": 1}
    boundary_values = sorted(
        {str(value) for value in boundaries},
        key=lambda value: (boundary_order.get(value, len(boundary_order)), value),
    )
    boundary_text = "/".join(boundary_values)
    key = (
        "pr-hr-rp1-zero-fallback:"
        f"{l1_method}:{impacts_text}:{target_year}:{fallback_year}:"
        f"{use_original_domain}:{ssp_scenario}"
    )
    message = (
        "Historical responsibility (PR-HR) cumulative "
        "impact per capita reused the last non-zero LCIA year. "
        f"Method={l1_method}; boundary={boundary_text}; impacts={impacts_text}; "
        f"requested year={target_year}; fallback year={fallback_year}. Reason: "
        "EXIOBASE data error, impact value 0 for requested year."
    )
    emit_notice(
        context=context,
        state=state,
        key=key,
        message=message,
    )


def record_pr_hr_rp1_zero_fallback_notice(
    *,
    state: RunState,
    l1_method: str,
    lcia_kind: str,
    impacts: list[str],
    target_year: int,
    fallback_year: int,
    use_original_domain: bool,
    ssp_scenario: str | None,
) -> None:
    """Record one PR-HR RP=1 fallback event for later merged emission."""
    notice_key = (
        str(l1_method),
        tuple(sorted(str(value) for value in impacts)),
        int(target_year),
        int(fallback_year),
        bool(use_original_domain),
        ssp_scenario,
    )
    state.pr_hr_rp1_zero_fallback_pending.setdefault(notice_key, set()).add(str(lcia_kind))


def build_pr_hr_rp1_zero_fallback_recorder(
    *,
    state: RunState,
    l1_method: str,
    lcia_kind: str,
    use_original_domain: bool,
    ssp_scenario: str | None,
) -> Callable[[list[str], int, int], None]:
    """Return the reporting callback used by PR-HR scientific fallback logic."""

    def _record(
        impacts: list[str],
        target_year: int,
        fallback_year: int,
    ) -> None:
        record_pr_hr_rp1_zero_fallback_notice(
            state=state,
            l1_method=str(l1_method),
            lcia_kind=str(lcia_kind),
            impacts=impacts,
            target_year=target_year,
            fallback_year=fallback_year,
            use_original_domain=bool(use_original_domain),
            ssp_scenario=ssp_scenario,
        )

    return _record


def flush_pr_hr_rp1_zero_fallback_notices(
    *,
    context: RunContext,
    state: RunState,
) -> None:
    """Flush merged PR-HR RP=1 fallback notices collected for the current pass."""
    pending = state.pr_hr_rp1_zero_fallback_pending
    if not pending:
        return
    for notice_key, boundaries in sorted(pending.items()):
        l1_method, impacts, target_year, fallback_year, use_original_domain, ssp_scenario = (
            notice_key
        )
        emit_pr_hr_rp1_zero_fallback_notice(
            context=context,
            state=state,
            l1_method=str(l1_method),
            boundaries=sorted(boundaries),
            impacts=list(impacts),
            target_year=int(target_year),
            fallback_year=int(fallback_year),
            use_original_domain=bool(use_original_domain),
            ssp_scenario=ssp_scenario,
        )
    pending.clear()


def is_scenario_dependent_l1(l1_method: str | None) -> bool:
    """Return whether an L1 method depends on scenario driven inputs.

    Scenario dependent methods consume population and/or GDP, so their outputs
    are routed to SSP specific files for SSP backed years.
    """
    if l1_method is None:
        return False
    specs = REGISTRY.get_method(l1_method, level="L1")
    return any(spec.needs_pop or spec.needs_gdp for spec in specs)


def is_scenario_dependent_l2_projection(
    *,
    context,
    year: int,
    l2_method: str,
) -> bool:
    """Return whether L2 projection makes this method scenario dependent."""
    projection_context = context.projection_context
    if projection_context is None or not projection_context.enabled:
        return False
    if projection_context.mode != "regression":
        return False
    if not projection_context.is_future_year(int(year)):
        return False
    return projection_context.route_for_l2_method(l2_method) == "regression"


def is_historical_reuse_l2_projection(
    *,
    context,
    year: int,
    l2_method: str,
) -> bool:
    """Return whether L2 projection routes this method through historical reuse."""
    projection_context = context.projection_context
    if projection_context is None or not projection_context.enabled:
        return False
    if not projection_context.is_future_year(int(year)):
        return False
    return projection_context.route_for_l2_method(l2_method) == "historical_reuse"


def l2_projection_subfolder(
    *,
    context,
    year: int,
    l2_method: str,
    bucket: str,
) -> str | None:
    """Return projection subfolder for one L2 output bucket."""
    projection_context = context.projection_context
    if projection_context is None or not projection_context.enabled:
        return None
    if not projection_context.is_future_year(int(year)):
        return None
    route = projection_context.route_for_l2_method(l2_method)
    if route is None:
        return None
    regression_subfolder = "regression_proj"
    if bucket == "l2_vs_global":
        return regression_subfolder if route == "regression" else "historical_reuse"
    if bucket == "utility_propagation_contrib":
        if route == "regression":
            return regression_subfolder
        return "historical_reuse" if route == "historical_reuse" else None
    if bucket == "l2_in_l1":
        return regression_subfolder if route == "regression" else None
    return None


def is_regression_projection_year(*, context, year: int) -> bool:
    """Return whether a year is a regression projected future year."""
    projection_context = context.projection_context
    if projection_context is None or not projection_context.enabled:
        return False
    if projection_context.mode != "regression":
        return False
    return projection_context.is_future_year(int(year))


def regression_projection_subfolder_for_context(*, context) -> str:
    """Return regression projection subfolder name for active context."""
    projection_context = context.projection_context
    del projection_context
    return "regression_proj"

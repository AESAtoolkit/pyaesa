"""Structured deterministic aSoCC summary records."""

from typing import Any

from pyaesa.asocc.data.source_schema import default_historical_cutoff_for_source
from pyaesa.asocc.methods.registry.registry import REGISTRY
from pyaesa.asocc.runtime.methods.labels import l1_l2_method_label
from pyaesa.shared.runtime.reporting.labels import labelled_values_line

from .common_formatting import format_year_ranges

_AR_FAMILIES = {"AR_E", "AR_ECAP"}
_TWO_STEP_L2_ROUTE_LABEL = "two step L2 routes (final L2 vs global = L1 * L2 in L1)"
_ONE_STEP_L2_ROUTE_LABEL = "one step L2 routes (direct L2 vs global)"


def deterministic_asocc_info_messages(*, context) -> list[str]:
    """Return scientific INFO messages owned by deterministic aSoCC."""
    messages: list[str] = []
    projection_context = getattr(context, "projection_context", None)
    if projection_context is not None and bool(getattr(projection_context, "enabled", False)):
        messages.extend(
            _projection_route_messages(
                context=context,
                projection_context=projection_context,
            )
        )
    messages.extend(_reference_year_messages(context=context))
    messages.extend(_ssp_route_messages(context=context))
    return _ordered_unique(messages)


def _projection_route_messages(*, context, projection_context) -> list[str]:
    future_years = [int(year) for year in getattr(projection_context, "future_years", ())]
    if not future_years:
        return []
    future_years_text = format_year_ranges(future_years)
    future_year_label = "future year" if len(set(future_years)) == 1 else "future years"
    messages: list[str] = []
    regression_methods = _methods_for_projection_route(
        projection_context=projection_context,
        route="regression",
    )
    if regression_methods:
        window = getattr(projection_context, "reg_window", None)
        window_text = "unknown"
        if window is not None:
            window_text = f"{int(window[0])}-{int(window[1])}"
        messages.append(
            "Regression projection uses fit window "
            f"{window_text} for {future_year_label} {future_years_text}; affected routes: "
            f"{_format_l2_route_groups(context=context, l2_methods=regression_methods)}."
        )
    reuse_methods = _methods_for_projection_route(
        projection_context=projection_context,
        route="historical_reuse",
    )
    if reuse_methods:
        reuse_values = [int(year) for year in getattr(projection_context, "l2_reuse_years", ())]
        reuse_years = format_year_ranges(reuse_values)
        reuse_label = "year" if len(set(reuse_values)) == 1 else "years"
        messages.append(
            "L2 historical reuse uses L2 reuse "
            f"{reuse_label} "
            f"{reuse_years} for {future_year_label} {future_years_text}; affected routes: "
            f"{_format_l2_route_groups(context=context, l2_methods=reuse_methods)}."
        )
    return messages


def _methods_for_projection_route(*, projection_context, route: str) -> list[str]:
    return [
        str(method)
        for method, method_route in getattr(
            projection_context,
            "l2_method_route_by_name",
            {},
        ).items()
        if str(method_route) == route
    ]


def _reference_year_messages(*, context) -> list[str]:
    groups = _ar_reference_route_groups(context=context)
    if not any(groups.values()):
        return []
    refs = _reference_years_for_summary(context=context)
    if not refs:
        return []
    reference_label = "Reference year" if len(set(refs)) == 1 else "Reference years"
    return [
        f"{reference_label} "
        f"{format_year_ranges(refs)} apply to AR routes: {_format_route_groups(groups)}."
    ]


def _reference_years_for_summary(*, context) -> list[int]:
    explicit_refs = getattr(context, "reference_years", None)
    if explicit_refs:
        return sorted({int(year) for year in explicit_refs})
    historical_years = sorted({int(year) for year in getattr(context, "historical_years", [])})
    cutoff = default_historical_cutoff_for_source(str(getattr(context, "source")))
    if cutoff is None:
        return historical_years
    return [year for year in historical_years if int(year) <= int(cutoff)]


def _ar_reference_route_groups(*, context) -> dict[str, list[str]]:
    l1 = [
        method for method in getattr(context, "selected_l1", []) if _l1_method_is_ar(method=method)
    ]
    l2_in_l1 = [
        l1_l2_method_label(l1_method=l1_method, l2_method=l2_method)
        for l2_method, l1_method in getattr(context, "combined", [])
        if _l2_method_is_ar(context=context, method=l2_method, l1_weighting=True)
        or _l1_method_is_ar(method=l1_method)
    ]
    l2_vs_global = [
        method
        for method in getattr(context, "selected_l2_one_step", [])
        if _l2_method_is_ar(context=context, method=method, l1_weighting=False)
    ]
    return {
        "Level 1 methods": l1,
        _TWO_STEP_L2_ROUTE_LABEL: l2_in_l1,
        _ONE_STEP_L2_ROUTE_LABEL: l2_vs_global,
    }


def _l1_method_is_ar(*, method: str) -> bool:
    return REGISTRY.method_family(method, level="L1") in _AR_FAMILIES


def _l2_method_is_ar(*, context, method: str, l1_weighting: bool) -> bool:
    return (
        REGISTRY.method_family(
            method,
            level="L2",
            fu_code=str(getattr(context, "fu_code")),
            l1_weighting=l1_weighting,
        )
        in _AR_FAMILIES
    )


def _ssp_route_messages(*, context) -> list[str]:
    scenarios = [
        str(value)
        for value in getattr(context, "ssp_scenario_options", [])
        if value is not None and str(value).strip()
    ]
    if not scenarios:
        return []
    wb_years = {
        int(str(column))
        for column in getattr(getattr(context, "wb_df"), "columns", [])
        if str(column).isdigit()
    }
    ssp_years = sorted(
        int(year) for year in getattr(context, "requested_years", []) if int(year) not in wb_years
    )
    if not ssp_years:
        return []
    scenario_line = labelled_values_line(
        "SSP scenario",
        "SSP scenarios",
        tuple(scenarios),
        _format_values(scenarios),
    )
    return [
        f"{scenario_line} apply from year {min(ssp_years)}; affected weights: "
        f"{_ssp_affected_weight_groups(context=context)}."
    ]


def _ssp_affected_weight_groups(*, context) -> str:
    groups: list[str] = []
    if getattr(context, "selected_l1", []):
        groups.append("Level 1 weights")
    if getattr(context, "combined", []):
        groups.append(
            "two step L2 in L1 weights; final two step L2 vs global shares can "
            "change when Level 1 and/or L2 in L1 weights change"
        )
    if getattr(context, "selected_l2_one_step", []):
        groups.append("direct one step L2 vs global weights")
    return "; ".join(groups) if groups else "none"


def _format_l2_route_groups(*, context, l2_methods: list[str]) -> str:
    selected = set(l2_methods)
    groups = {
        _TWO_STEP_L2_ROUTE_LABEL: [
            l1_l2_method_label(l1_method=l1_method, l2_method=l2_method)
            for l2_method, l1_method in getattr(context, "combined", [])
            if l2_method in selected
        ],
        _ONE_STEP_L2_ROUTE_LABEL: [
            method for method in getattr(context, "selected_l2_one_step", []) if method in selected
        ],
    }
    return _format_route_groups(groups)


def _format_route_groups(groups: dict[str, list[str]]) -> str:
    parts = [
        f"{group_name} {_format_values(values)}" for group_name, values in groups.items() if values
    ]
    return "; ".join(parts) if parts else "none"


def _format_values(values: list[Any]) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(cleaned) if cleaned else "none"


def _ordered_unique(messages: list[str]) -> list[str]:
    return list(dict.fromkeys(text for message in messages if (text := str(message).strip())))

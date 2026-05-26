"""Resolved run-plan setup for disaggregate_asocc branch orchestration."""

from ..entrypoints.argument_contracts import ensure_list_str
from ..methods.registry.registry import REGISTRY, resolve_required_indices
from pyaesa.asocc.orchestration.setup.request.selection import _resolve_filters
from ..runtime.selection.normalize import normalize_l1_reg_mode, normalize_output_mode
from ..runtime.selection.resolve import resolve_method_selection
from .models import DisaggregationRunPlan, ParsedArgs


def _coerce_optional_filter(value: object | None, *, name: str) -> str | list[str] | None:
    """Coerce optional filter args to deterministic_asocc accepted input types."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"base_asocc_args.{name} must be None, a string, or a list of strings.")


def _validate_l2_region_filters(
    *,
    fu_code: str,
    s_p: list[str],
    r_p: list[str] | None,
    r_c: list[str] | None,
    r_f: list[str] | None,
    combined_non_lcia: list[tuple[str, str]],
    one_step_non_lcia: list[str],
) -> None:
    """Fail fast using deterministic_asocc selector guards for L2 disaggregation scope."""
    required_indices = resolve_required_indices(
        fu_code=fu_code,
        selected_l1=[],
        combined=combined_non_lcia,
        selected_l2_one_step=one_step_non_lcia,
        l1_kinds_needed=set(),
    )
    try:
        _resolve_filters(
            required_indices=required_indices,
            r_p=r_p,
            s_p=s_p,
            r_c=r_c,
            r_f=r_f,
        )
    except ValueError as exc:
        raise ValueError(
            "Invalid region selector for disaggregate_asocc L2 scope "
            f"(fu_code='{fu_code}', required_indices={sorted(required_indices)}): "
            f"{exc}"
        ) from exc


def _filter_non_lcia_l2_selection(
    *,
    fu_code: str,
    combined: list[tuple[str, str]] | None,
    one_step: list[str] | None,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Return the non-LCIA only L2 method selections for disaggregation."""
    combined_non_lcia = [
        (l2_name, l1_name)
        for l2_name, l1_name in (combined or [])
        if (not REGISTRY.method_requires_lcia(l2_name, fu_code))
        and (not REGISTRY.method_requires_lcia(l1_name, None))
    ]
    one_step_non_lcia = [
        l2_method
        for l2_method in (one_step or [])
        if not REGISTRY.method_requires_lcia(l2_method, fu_code)
    ]
    return combined_non_lcia, one_step_non_lcia


def build_disaggregation_run_plan(parsed: ParsedArgs) -> DisaggregationRunPlan:
    """Resolve deterministic scope planning shared across all disaggregation branches."""
    base_args = parsed.base_allocate_args
    r_p = ensure_list_str(_coerce_optional_filter(base_args.get("r_p"), name="r_p"))
    r_c = ensure_list_str(_coerce_optional_filter(base_args.get("r_c"), name="r_c"))
    r_f = ensure_list_str(_coerce_optional_filter(base_args.get("r_f"), name="r_f"))
    l1_full, combined_full, one_step_full = resolve_method_selection(
        fu_code=base_args["fu_code"],
        method_plan=base_args["method_plan"],
        l1_methods=base_args["l1_methods"],
        one_step_methods=base_args["one_step_methods"],
        two_step_methods=base_args["two_step_methods"],
        l1_l2_pairs=base_args["l1_l2_pairs"],
    )
    combined_non_lcia, one_step_non_lcia = _filter_non_lcia_l2_selection(
        fu_code=base_args["fu_code"],
        combined=combined_full,
        one_step=one_step_full,
    )
    if not combined_non_lcia and not one_step_non_lcia:
        raise ValueError(
            "No non-LCIA L2 methods are available for disaggregation under "
            "the requested method selection."
        )
    _validate_l2_region_filters(
        fu_code=base_args["fu_code"],
        s_p=parsed.disaggregation.ref_disagg_run.s_p,
        r_p=r_p,
        r_c=r_c,
        r_f=r_f,
        combined_non_lcia=combined_non_lcia,
        one_step_non_lcia=one_step_non_lcia,
    )
    return DisaggregationRunPlan(
        r_p=r_p,
        r_c=r_c,
        r_f=r_f,
        l1_methods=[] if l1_full is None else l1_full,
        combined_non_lcia=combined_non_lcia,
        one_step_non_lcia=one_step_non_lcia,
        selected_l2_methods=[
            *one_step_non_lcia,
            *(l2_method for l2_method, _ in combined_non_lcia),
        ],
        ssp_scenarios=ensure_list_str(base_args["ssp_scenario"]),
        group_indices=normalize_output_mode(base_args["group_indices"]),
        l1_reg_aggreg=normalize_l1_reg_mode(base_args["l1_reg_aggreg"]),
    )

"""Shared deterministic allocation context reconstruction."""

from typing import Any, cast

from pyaesa.asocc.io.logging import close_loggers_for_scope
from pyaesa.asocc.orchestration.setup.run_setup import _prepare_context
from pyaesa.asocc.orchestration.setup.pipeline.builders import _build_initial_state
from pyaesa.asocc.orchestration.setup.request.types import PrepareContextRequest
from pyaesa.asocc.runtime.scope.branch_resolution import outputs_project_root
from pyaesa.asocc.runtime.selection.resolve import resolve_method_selection


def rebuild_allocation_context(
    *,
    base_allocate_args: dict[str, Any],
    years: list[int],
    source: str,
    output_source_label: str,
    variant_tag: str | None = None,
    aggreg_indices: bool = False,
    output_format: str = "csv",
    intermediate_outputs: bool = False,
    historical_year_cap: int | None = None,
):
    """Rebuild deterministic allocation context and initial state from public args."""
    project_root = outputs_project_root(project_name=str(base_allocate_args["project_name"]))
    l1_methods, combined_pairs, one_step_methods = resolve_method_selection(
        fu_code=base_allocate_args["fu_code"],
        method_plan=base_allocate_args["method_plan"],
        l1_methods=base_allocate_args["l1_methods"],
        one_step_methods=base_allocate_args["one_step_methods"],
        two_step_methods=base_allocate_args["two_step_methods"],
        l1_l2_pairs=base_allocate_args["l1_l2_pairs"],
    )
    try:
        prepared_context, _prepared_state, _is_complete = _prepare_context(
            request=PrepareContextRequest(
                project_name=base_allocate_args["project_name"],
                source=str(source),
                group_version=base_allocate_args["group_version"],
                group_reg=base_allocate_args["group_reg"],
                group_sec=base_allocate_args["group_sec"],
                years=list(years),
                historical_year_cap=historical_year_cap,
                refresh=False,
                lcia_method=base_allocate_args["lcia_method"],
                fu_code=base_allocate_args["fu_code"],
                r_p=base_allocate_args["r_p"],
                s_p=base_allocate_args["s_p"],
                r_c=base_allocate_args["r_c"],
                r_f=base_allocate_args["r_f"],
                l_1=l1_methods,
                l_2_combined_with_l_1=combined_pairs,
                l_2_one_step=one_step_methods,
                reference_years=base_allocate_args["reference_years"],
                ssp_scenario=base_allocate_args["ssp_scenario"],
                projection_mode=base_allocate_args["projection_mode"],
                reg_window=base_allocate_args["reg_window"],
                l2_reuse_years=base_allocate_args["l2_reuse_years"],
                l1_reg_aggreg=base_allocate_args["l1_reg_aggreg"],
                variant_tag=variant_tag,
                aggreg_indices=bool(aggreg_indices),
                output_format=str(output_format),
                intermediate_outputs=bool(intermediate_outputs),
                output_source_label=str(output_source_label),
            )
        )
    finally:
        close_loggers_for_scope(project_root)
    return prepared_context, _build_initial_state(
        ssp_scenario_options=prepared_context.ssp_scenario_options
    )


def resolve_external_ssp_scenario_options_by_year(
    *,
    base_allocate_args: dict[str, Any],
    years: list[int],
    output_source_label: str,
) -> dict[int, list[str | None]]:
    """Return the canonical per-year scenario routing for external input reuse."""
    allocation_context, _allocation_state = rebuild_allocation_context(
        base_allocate_args=base_allocate_args,
        source=str(base_allocate_args["source"]),
        output_source_label=str(output_source_label),
        years=years,
    )
    return cast(dict[int, list[str | None]], allocation_context.ssp_scenario_options_by_year)

"""Output specification and storage for L1 orchestration."""

from ....runtime.output.contracts import (
    OutputRoute,
    OutputSpec,
    identifier_columns_from_frame,
    join_file_owned_tokens,
)
from ..shared.output_spec_cache import (
    get_cached_output_spec,
    set_cached_output_spec,
)
from ..shared.scenario_routing import is_scenario_dependent_l1, resolve_output_ssp_scenario
from .l1_slicing import _slice_l1_frame_for_compute
from .l1_types import _L1RunContext, _L1StorePayload


def _build_l1_output_spec(
    *,
    l1_method: str,
    lcia_method: str | None,
    frame,
    ssp_scenario: str | None,
    grouped_mode: bool,
    state=None,
) -> OutputSpec:
    """Build one typed L1 output spec."""
    identifier_columns = list(identifier_columns_from_frame(frame))
    if "reference_year" in frame.columns and "reference_year" not in identifier_columns:
        identifier_columns.append("reference_year")
    cache_key = (
        "L1",
        l1_method,
        lcia_method,
        ssp_scenario,
        bool(grouped_mode),
        tuple(identifier_columns),
    )
    cached = get_cached_output_spec(state=state, key=cache_key)
    if isinstance(cached, OutputSpec):
        return cached
    scenario_dependent = is_scenario_dependent_l1(l1_method)
    route = OutputRoute(
        level="L1",
        bucket=None,
        source=None,
        grouped_mode=grouped_mode,
        variant_tag=None,
        ssp_scenario=ssp_scenario,
        lcia_method=lcia_method,
        projection_subfolder=None,
    )
    output_spec = OutputSpec(
        l1_l2_method=l1_method,
        l2_method=None,
        l1_method=l1_method,
        file_stem=join_file_owned_tokens(f"l1_{l1_method}", lcia_method),
        route=route,
        scenario_dependent=scenario_dependent,
        identifier_columns=tuple(identifier_columns),
    )
    set_cached_output_spec(state=state, key=cache_key, spec=output_spec)
    return output_spec


def _store_l1_frame(
    *,
    run: _L1RunContext,
    payload: _L1StorePayload,
) -> None:
    """Store one L1 frame to scenario outputs and year map."""
    publish_year = int(run.year) in run.context.persisted_years
    if payload.value_frame is None:
        frame_sliced = _slice_l1_frame_for_compute(run=run, frame=payload.frame)
        value_frame_sliced = frame_sliced
    else:
        frame_sliced = (
            _slice_l1_frame_for_compute(run=run, frame=payload.frame) if publish_year else None
        )
        value_frame_sliced = _slice_l1_frame_for_compute(run=run, frame=payload.value_frame)
    if publish_year and frame_sliced is not None:
        spec = _build_l1_output_spec(
            l1_method=payload.resolved_name,
            lcia_method=payload.lcia_method,
            frame=frame_sliced,
            ssp_scenario=resolve_output_ssp_scenario(
                context=run.context,
                year=run.year,
                ssp_scenario=run.ssp_scenario,
                scenario_dependent=is_scenario_dependent_l1(payload.resolved_name),
            ),
            grouped_mode=bool(run.context.group_indices),
            state=run.state,
        )
        run.state.l1_results_by_ssp_scenario[run.ssp_scenario].setdefault(spec, []).append(
            frame_sliced
        )
    run.l1_results_year[payload.year_key] = value_frame_sliced

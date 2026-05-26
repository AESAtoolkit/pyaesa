"""Per year processing for allocation runs."""

import gc
from typing import TYPE_CHECKING, cast

from pathlib import Path

from ..method_scope import _max_historical_mrio_year, _unique_l2_methods_in_scope
from ...data.paths import _get_mrio_year_dir
from ...io.metadata import RunContext, RunState
from ...data.run_lcia import _load_lcia_for_year
from ...methods.registry.registry import REGISTRY
from .l2.l2_slicing import _slice_l2_inputs_for_compute
from .l2.l2_types import _L2ComputeInputs
from .enacting_metric.enacting_metric_base import (
    record_adjusted_ut_preweights,
    record_lcia_enacting_metrics,
)
from .shared.year_inputs import (
    _MrioPayload,
    _ScenarioRunContext,
    _load_year_mrio_payloads_required,
    _load_reg_agg_map,
)
from .shared.scenario_routing import emit_notice
from .shared.scenario_processing import _process_scenario_for_year

_GC_COLLECT_EVERY_N_YEARS = 24

if TYPE_CHECKING:
    from pyaesa.shared.runtime.reporting.progress import StatusProgressPrinter, YearProgressPrinter


def _latest_existing_year_dir(
    *,
    source: str,
    historical_years: list[int],
    year: int,
    agg_version: str | None,
) -> tuple[int | None, Path | None]:
    """Return latest existing MRIO year directory not after the requested year."""
    for hist_year in sorted((value for value in historical_years if value <= year), reverse=True):
        candidate = _get_mrio_year_dir(
            source=source,
            year=hist_year,
            agg_version=agg_version,
        )
        if candidate.exists():
            return hist_year, candidate
    return None, None


def _process_year(
    *,
    context: RunContext,
    state: RunState,
    year: int,
    progress: "YearProgressPrinter | StatusProgressPrinter | None" = None,
) -> bool:
    """Process one MRIO year for allocation outputs."""
    l1_needed = bool(context.selected_l1 or context.combined)
    selected_l2_names = _unique_l2_methods_in_scope(
        selected_l2_one_step=context.selected_l2_one_step,
        combined=context.combined,
    )
    has_ut = any(
        REGISTRY.method_is_ut(name, level="L2", fu_code=context.fu_code)
        for name in selected_l2_names
    )
    has_ar = any(
        REGISTRY.method_is_ar(name, level="L2", fu_code=context.fu_code)
        for name in selected_l2_names
    )
    max_historical_year = _max_historical_mrio_year(historical_years=context.historical_years)
    saved_dir = None
    has_year_dir = False
    # ISO3/L1-only branches intentionally avoid MRIO payload loading.
    if not context.l1_only_no_mrio:
        saved_dir = _get_mrio_year_dir(
            source=context.source,
            year=year,
            agg_version=context.agg_version,
        )
        has_year_dir = saved_dir.exists()
    projection_active = (
        context.projection_context is not None
        and context.projection_context.enabled
        and context.projection_context.is_future_year(year)
    )
    projection_can_supply_mrio = bool(projection_active and has_ut)
    ar_can_run_without_studied_year_mrio = bool(
        has_ar and max_historical_year is not None and int(year) > int(max_historical_year)
    )
    if (
        not has_year_dir
        and not l1_needed
        and not projection_can_supply_mrio
        and not ar_can_run_without_studied_year_mrio
    ):
        state.skipped_years[year] = "MRIO year directory missing"
        return False

    if progress is not None:
        progress.begin_year(year)

    reg_agg_map = _load_reg_agg_map(context=context, state=state)

    needs_mrio = bool(selected_l2_names)
    mrio_payload: _MrioPayload | None = None
    if needs_mrio and has_year_dir and saved_dir is not None:
        try:
            mrio_payload = _load_year_mrio_payloads_required(
                saved_dir=saved_dir,
                context=context,
                needs_mrio=True,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            emit_notice(
                context=context,
                state=state,
                key="l2-mrio-enacting-metric-metrics-missing",
                message=(
                    "L2 MRIO enacting metrics unavailable for year "
                    f"{year}; L2 computation skipped. Reason: {exc}"
                ),
            )
            mrio_payload = None

    lcia_saved_year: int | None = year
    lcia_saved_dir = saved_dir
    if context.needs_lcia and not has_year_dir:
        # When studied year is beyond available MRIO years, LCIA can still be
        # sourced from the latest historical year for AR/PR-HR continuity rules.
        lcia_saved_year, lcia_saved_dir = _latest_existing_year_dir(
            source=context.source,
            historical_years=context.historical_years,
            year=year,
            agg_version=context.agg_version,
        )

    lcia_by_method = None
    lcia_effective_year_by_method: dict[str, int] | None = None
    if context.needs_lcia and lcia_saved_dir is not None and lcia_saved_year is not None:
        lcia_effective_year_by_method = {}
        lcia_by_method = _load_lcia_for_year(
            context=context,
            state=state,
            year=lcia_saved_year,
            saved_dir=lcia_saved_dir,
            allow_method_year_fallback=True,
            method_year_out=lcia_effective_year_by_method,
        )
    lcia_by_method_original = None
    lcia_effective_year_by_method_original: dict[str, int] | None = None
    if context.use_original_l1_post_domain and context.needs_lcia:
        original_saved_year = year
        original_saved_dir = _get_mrio_year_dir(
            source=context.source,
            year=original_saved_year,
            agg_version=None,
        )
        if not original_saved_dir.exists():
            original_saved_year, original_saved_dir = _latest_existing_year_dir(
                source=context.source,
                historical_years=context.historical_years,
                year=year,
                agg_version=None,
            )
        lcia_effective_year_by_method_original = {}
        lcia_by_method_original = _load_lcia_for_year(
            context=context,
            state=state,
            year=(cast(int, original_saved_year) if original_saved_year is not None else year),
            saved_dir=cast(Path, original_saved_dir),
            agg_version_override=None,
            allow_method_year_fallback=True,
            method_year_out=lcia_effective_year_by_method_original,
        )

    if bool(getattr(context, "intermediate_outputs", True)):
        record_lcia_enacting_metrics(
            context=context,
            state=state,
            year=year,
            lcia_by_method=lcia_by_method,
            lcia_effective_year_by_method=lcia_effective_year_by_method,
        )

    if mrio_payload is not None and bool(getattr(context, "intermediate_outputs", True)):
        record_adjusted_ut_preweights(
            context=context,
            state=state,
            year=year,
            enacting_metric_l1=mrio_payload.enacting_metric_l1,
            enacting_metric_l2=mrio_payload.enacting_metric_l2,
            utility=mrio_payload.utility,
        )

    scenario_plan = context.ssp_scenario_options_by_year or {}
    ssp_scenarios_for_year = scenario_plan[year]
    l2_inputs_sliced: _L2ComputeInputs | None = None
    if mrio_payload is not None:
        l2_inputs_sliced = _slice_l2_inputs_for_compute(
            context=context,
            inputs=mrio_payload.l2_inputs,
        )
    primary_ssp_scenario = ssp_scenarios_for_year[0]
    for ssp_scenario in ssp_scenarios_for_year:
        run_ctx = _ScenarioRunContext(
            context=context,
            state=state,
            year=year,
            ssp_scenario=ssp_scenario,
        )
        _process_scenario_for_year(
            run_ctx=run_ctx,
            lcia_by_method=lcia_by_method,
            lcia_by_method_original=lcia_by_method_original,
            lcia_effective_year_by_method=lcia_effective_year_by_method,
            lcia_effective_year_by_method_original=lcia_effective_year_by_method_original,
            reg_agg_map=reg_agg_map,
            mrio_payload=mrio_payload,
            l2_inputs_sliced=l2_inputs_sliced,
            process_invariant_methods=(ssp_scenario == primary_ssp_scenario),
        )

    if mrio_payload is not None:
        del mrio_payload
    if int(year) in {int(value) for value in context.resolved_years}:
        state.processed_years.append(year)
    if len(state.processed_years) % _GC_COLLECT_EVERY_N_YEARS == 0:
        gc.collect()
    if progress is not None:
        progress.complete_year(year)
    return True

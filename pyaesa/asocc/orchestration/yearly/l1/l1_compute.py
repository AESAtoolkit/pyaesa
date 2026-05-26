"""Compute orchestration for L1 methods."""

import pandas as pd

from ....methods.compute_l1 import compute_l1_method
from ....methods.registry.registry import REGISTRY
from .l1_lcia_compute import _compute_l1_lcia_method
from .l1_slicing import (
    _l2_axes_for_l1_method,
    _single_axis_or_none,
    _slice_l1_frame_for_compute,
)
from .l1_store import _build_l1_output_spec
from .l1_types import _L1RunContext


def _compute_l1_non_lcia_result(
    *,
    run: _L1RunContext,
    l1_method: str,
    region_label_override: str | None,
) -> pd.DataFrame:
    """Compute one non LCIA L1 result, optionally forcing region axis name."""
    return compute_l1_method(
        l1_method=l1_method,
        fu_code=run.context.fu_code,
        year=run.year,
        population=run.pop_series,
        population_by_year=run.state.pop_series_by_ssp_scenario[run.ssp_scenario],
        population_ref=None,
        pr_pop=run.pr_pop,
        pr_gdp=run.pr_gdp,
        pr_to_mrio=run.pr_to_mrio,
        lcia_reg=None,
        lcia_reg_by_year=None,
        rps_df=None,
        impact_parent_map=None,
        available_years=run.context.historical_years,
        reference_year=(run.context.reference_years[0] if run.context.reference_years else None),
        source_key=run.context.source,
        agg_version_reg=run.context.agg_version_reg,
        l1_reg_aggreg=run.context.l1_reg_aggreg,
        region_label_override=region_label_override,
    )


def _compute_l1_non_lcia_method(
    *,
    run: _L1RunContext,
    l1_method: str,
) -> None:
    """Compute and store one non LCIA L1 method."""
    axis_by_l2 = _l2_axes_for_l1_method(run=run, l1_method=l1_method)
    family = REGISTRY.method_family(l1_method, level="L1")
    if run.context.fu_code.startswith("L2.") and family in {"EG_POP", "PR_GDPCAP"}:
        l2_methods = list(axis_by_l2)
        if l2_methods:
            results_by_axis: dict[str, pd.DataFrame] = {}
            for axis in sorted(set(axis_by_l2.values())):
                results_by_axis[axis] = _compute_l1_non_lcia_result(
                    run=run,
                    l1_method=l1_method,
                    region_label_override=axis,
                )
                results_by_axis[axis] = _slice_l1_frame_for_compute(
                    run=run,
                    frame=results_by_axis[axis],
                )
            # Keep one canonical L1 output frame for logs/exports.
            selected_axis = axis_by_l2[l2_methods[0]]
            result = results_by_axis[selected_axis]
            for l2_method in l2_methods:
                run.l1_results_year[f"{l1_method}__for__{l2_method}"] = results_by_axis[
                    axis_by_l2[l2_method]
                ]
        else:
            result = _compute_l1_non_lcia_result(
                run=run,
                l1_method=l1_method,
                region_label_override=None,
            )
            result = _slice_l1_frame_for_compute(run=run, frame=result)
    else:
        result = _compute_l1_non_lcia_result(
            run=run,
            l1_method=l1_method,
            region_label_override=_single_axis_or_none(axis_by_l2),
        )
        result = _slice_l1_frame_for_compute(run=run, frame=result)

    spec = _build_l1_output_spec(
        l1_method=l1_method,
        lcia_method=None,
        frame=result,
        ssp_scenario=run.ssp_scenario,
        grouped_mode=bool(run.context.group_indices),
        state=run.state,
    )
    run.state.l1_results_by_ssp_scenario[run.ssp_scenario].setdefault(spec, []).append(result)
    run.l1_results_year[l1_method] = result


def _compute_l1_for_year(
    *,
    run: _L1RunContext,
    lcia_by_method: dict[str, dict] | None,
    lcia_by_method_original: dict[str, dict] | None,
) -> dict[str, pd.DataFrame]:
    """Compute L1 results for a single year."""
    for l1_method in run.context.selected_l1:
        if REGISTRY.method_requires_lcia(l1_method, None):
            if lcia_by_method is None and lcia_by_method_original is None:
                if run.context.needs_lcia and run.state.skipped_years.get(run.year) is None:
                    run.state.skipped_years[run.year] = "LCIA unavailable"
                continue
            _compute_l1_lcia_method(
                run=run,
                l1_method=l1_method,
                lcia_by_method={} if lcia_by_method is None else dict(lcia_by_method),
                lcia_by_method_original=(
                    None if lcia_by_method_original is None else dict(lcia_by_method_original)
                ),
            )
            continue

        _compute_l1_non_lcia_method(
            run=run,
            l1_method=l1_method,
        )
    return run.l1_results_year

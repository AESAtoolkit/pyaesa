"""Impact slice execution for combined (two step) L2 compute."""

import pandas as pd

from ....methods.compute_l2 import apply_l1_weights_to_preweighted
from ....methods.equations.ar_result_indexing import _add_reference_level
from ....methods.registry.registry import REGISTRY
from ....methods.run_ar import _compute_ar_l2_preweight
from ....methods.run_ut import (
    _compute_ut_weighted_contribution_from_preweight,
    _get_ut_l2_preweight,
    _weight_ut_contribution_from_preweight,
)
from ...projection.reuse.outputs import load_reuse_preweight
from .l2_batch_weighting import (
    batch_weight_reuse_preweighted_ut_matrix,
    batch_weight_preweighted_ar_matrix,
    batch_weight_preweighted_ut_matrix,
)
from .l2_compute_primitives import compute_ar_result, compute_non_ar_or_ut_result
from .l2_contracts import (
    require_frame,
    require_ref_year,
    require_required_indices,
    require_weight_axis,
    require_weights,
)
from .l2_impact_support import (
    _should_write_historical_reuse_utility_contrib,
    _ut_input_maps,
    _write_combined_outputs,
)
from .l2_reuse_frames import _combine_l2_reuse_year_frames
from .l2_slicing import (
    _impact_weight_matrix,
    _impact_weight_items,
    _normalize_l1_weights,
    _slice_l1_weight_frame_for_compute,
)
from .l2_types import (
    _L2RunContext,
    _L2SliceSpec,
    _L2WeightSpec,
    _is_ar_l1,
    _is_ar_l2,
    _is_ut_l2,
)
from .l2_ut_gvaa_closure import apply_ut_gvaa_identity_closure
from ..shared.scenario_routing import is_historical_reuse_l2_projection


def compute_combined_impact_results(
    *,
    run: _L2RunContext,
    slice_spec: _L2SliceSpec,
    l1_weights: pd.DataFrame | None,
) -> None:
    """Compute and store all impact slice results for one combined slice."""
    l2_method = slice_spec.l2_method
    fu_code = run.context.fu_code
    where = f"year={run.year}, fu_code='{fu_code}', l2_method='{l2_method}'"
    l2_is_ar = _is_ar_l2(l2_method=l2_method, fu_code=fu_code)
    l1_is_ar = bool(slice_spec.l1_name and _is_ar_l1(slice_spec.l1_name))
    has_utility_contrib = l2_method in {"UT(FDa)", "UT(GVAa)"}
    intermediate_outputs = bool(getattr(run.context, "intermediate_outputs", True))
    apply_gvaa_identity_closure = l2_method == "UT(GVAa)" and fu_code == "L2.a.b"
    add_reference_level = (
        slice_spec.l1_name is not None and l1_is_ar and slice_spec.ref_year is not None
    )
    reference_year_value: int | None = None
    if add_reference_level:
        reference_year_value = require_ref_year(
            ref_year=slice_spec.ref_year,
            where=where,
        )
    projection_context = run.context.projection_context
    historical_projection_context = (
        projection_context
        if projection_context is not None
        and _is_ut_l2(
            l2_method=l2_method,
            fu_code=fu_code,
        )
        and is_historical_reuse_l2_projection(
            context=run.context,
            year=run.year,
            l2_method=l2_method,
        )
        else None
    )
    use_historical_reuse = historical_projection_context is not None
    is_ar = l2_is_ar and slice_spec.ref_year is not None
    l2_reuse_years: tuple[int, ...] = tuple()
    l2_weight_axis: str | None = None
    l2_required_indices: tuple[str, ...] | None = None
    historical_preweights_by_year: dict[int, pd.DataFrame] = {}
    should_write_hist_utility_contrib = False
    if historical_projection_context is not None:
        l2_reuse_years = tuple(
            int(value) for value in historical_projection_context.l2_reuse_years_for()
        )
        l2_weight_axis = REGISTRY.l2_weight_axis_for_method(l2_method, fu_code)
        l2_required_indices = tuple(
            REGISTRY.required_indices(
                l2_method,
                fu_code,
                l1_weighting=True,
            )
        )
        historical_preweights_by_year = {
            int(l2_reuse_year): load_reuse_preweight(
                context=run.context,
                state=run.state,
                l2_method=l2_method,
                lcia_key=slice_spec.lcia_key,
                l2_reuse_year=int(l2_reuse_year),
            )
            for l2_reuse_year in l2_reuse_years
        }
        should_write_hist_utility_contrib = (
            intermediate_outputs
            and has_utility_contrib
            and _should_write_historical_reuse_utility_contrib(slice_spec=slice_spec)
        )

    sliced_l1_weights = (
        _slice_l1_weight_frame_for_compute(
            run=run,
            l2_method=l2_method,
            weights=l1_weights,
        )
        if isinstance(l1_weights, pd.DataFrame)
        else None
    )
    impact_matrix = _impact_weight_matrix(sliced_l1_weights)
    plan_cache = getattr(run.state, "l2_batch_weighting_plan_cache", None)

    if impact_matrix is not None and use_historical_reuse and not apply_gvaa_identity_closure:
        impact_names, weight_index, weight_values = impact_matrix
        weight_axis = require_weight_axis(weight_axis=l2_weight_axis, where=where)
        required_indices = require_required_indices(
            required_indices=l2_required_indices,
            where=where,
        )
        aggregated, contribution = batch_weight_reuse_preweighted_ut_matrix(
            preweights_by_l2_reuse_year=[
                (l2_reuse_year, historical_preweights_by_year[l2_reuse_year])
                for l2_reuse_year in l2_reuse_years
            ],
            impact_names=impact_names,
            weight_index=weight_index,
            weight_values=weight_values,
            weight_axis=weight_axis,
            required_indices=required_indices,
            year=run.year,
            include_contribution=should_write_hist_utility_contrib,
            reference_year=(reference_year_value if add_reference_level else None),
            plan_cache=plan_cache,
        )
        _write_combined_outputs(
            run=run,
            slice_spec=slice_spec,
            result=aggregated,
            impact=None,
            reference_year=None,
            contrib_result=contribution,
        )
        return

    if impact_matrix is not None and has_utility_contrib and not use_historical_reuse:
        if not apply_gvaa_identity_closure:
            impact_names, weight_index, weight_values = impact_matrix
            ut_enacting_metric_l1, ut_enacting_metric_l2, ut_utility = _ut_input_maps(run=run)
            weight_axis = REGISTRY.l2_weight_axis_for_method(l2_method, fu_code)
            required_indices = tuple(
                REGISTRY.required_indices(
                    l2_method,
                    fu_code,
                    l1_weighting=True,
                )
            )
            preweight = _get_ut_l2_preweight(
                context=run.context,
                state=run.state,
                ssp_scenario=run.ssp_scenario,
                l2_method=l2_method,
                year=run.year,
                lcia_data=slice_spec.lcia_data,
                lcia_key=slice_spec.lcia_key,
                ref_year=slice_spec.ref_year,
                enacting_metric_l1=ut_enacting_metric_l1,
                enacting_metric_l2=ut_enacting_metric_l2,
                utility=ut_utility,
            )
            aggregated, contribution = batch_weight_preweighted_ut_matrix(
                pre_weighted=preweight,
                impact_names=impact_names,
                weight_index=weight_index,
                weight_values=weight_values,
                weight_axis=weight_axis,
                required_indices=required_indices,
                year=run.year,
                include_contribution=intermediate_outputs,
                plan_cache=plan_cache,
            )
            _write_combined_outputs(
                run=run,
                slice_spec=slice_spec,
                result=aggregated,
                impact=None,
                reference_year=(reference_year_value if add_reference_level else None),
                contrib_result=contribution,
            )
            return

    if impact_matrix is not None and is_ar:
        impact_names, weight_index, weight_values = impact_matrix
        ref_year = require_ref_year(ref_year=slice_spec.ref_year, where=where)
        preweight = require_frame(
            frame=_compute_ar_l2_preweight(
                context=run.context,
                state=run.state,
                cache_key=(
                    slice_spec.l2_method,
                    None,
                    slice_spec.lcia_key,
                    ref_year,
                    True,
                ),
                l2_method=slice_spec.l2_method,
                year=run.year,
                ref_year=ref_year,
                lcia_data=slice_spec.lcia_data,
            ),
            where=where,
            subject="AR L2 preweight frame",
        )
        weighted = batch_weight_preweighted_ar_matrix(
            pre_weighted=preweight,
            impact_names=impact_names,
            weight_index=weight_index,
            weight_values=weight_values,
            impact_level="impact",
            weight_axis=REGISTRY.l2_weight_axis_for_method(l2_method, fu_code),
            required_indices=tuple(
                REGISTRY.required_indices(
                    l2_method,
                    fu_code,
                    l1_weighting=True,
                )
            ),
            year=run.year,
            plan_cache=plan_cache,
        )
        _write_combined_outputs(
            run=run,
            slice_spec=slice_spec,
            result=_add_reference_level(weighted, ref_year),
            impact=None,
            reference_year=None,
        )
        return

    split_source = sliced_l1_weights if sliced_l1_weights is not None else l1_weights
    impact_items: list[tuple[str | None, pd.Series | None]] = []
    for impact, weights in _impact_weight_items(split_source):
        impact_items.append((impact, None if weights is None else _normalize_l1_weights(weights)))

    for impact, weights in impact_items:
        if use_historical_reuse:
            weight_axis = require_weight_axis(weight_axis=l2_weight_axis, where=where)
            required_indices = require_required_indices(
                required_indices=l2_required_indices,
                where=where,
            )
            impact_weights = require_weights(weights=weights, where=where)
            results_by_l2_reuse_year = []
            contrib_results_by_l2_reuse_year = [] if should_write_hist_utility_contrib else None
            for l2_reuse_year in l2_reuse_years:
                l2_reuse_year_value = int(l2_reuse_year)
                reuse_preweight = historical_preweights_by_year[l2_reuse_year_value]
                contrib_result: pd.DataFrame | None = None
                if should_write_hist_utility_contrib:
                    contrib_result = _weight_ut_contribution_from_preweight(
                        context=run.context,
                        l2_method=l2_method,
                        year=run.year,
                        weights=impact_weights,
                        pre_weighted=reuse_preweight,
                        weight_axis=weight_axis,
                    )
                result = apply_l1_weights_to_preweighted(
                    l2_method=l2_method,
                    fu_code=fu_code,
                    year=run.year,
                    pre_weighted=reuse_preweight,
                    l1_weights=impact_weights,
                    weight_axis=weight_axis,
                    required_indices=required_indices,
                )
                if apply_gvaa_identity_closure:
                    result = apply_ut_gvaa_identity_closure(
                        run=run,
                        slice_spec=slice_spec,
                        weights=impact_weights,
                        impact=impact,
                        result=result,
                        l2_reuse_year=l2_reuse_year_value,
                    )
                results_by_l2_reuse_year.append((l2_reuse_year_value, result))
                if contrib_results_by_l2_reuse_year is not None and contrib_result is not None:
                    contrib_results_by_l2_reuse_year.append((l2_reuse_year_value, contrib_result))
            combined_reference_year = reference_year_value if add_reference_level else None
            result_with_reuse = _combine_l2_reuse_year_frames(
                frames_by_l2_reuse_year=results_by_l2_reuse_year,
                reference_year=combined_reference_year,
            )
            contrib_with_reuse = (
                None
                if contrib_results_by_l2_reuse_year is None
                else _combine_l2_reuse_year_frames(
                    frames_by_l2_reuse_year=contrib_results_by_l2_reuse_year,
                    reference_year=combined_reference_year,
                )
            )
            _write_combined_outputs(
                run=run,
                slice_spec=slice_spec,
                result=result_with_reuse,
                impact=impact,
                reference_year=None,
                contrib_result=contrib_with_reuse,
            )
            continue

        weight_spec = _L2WeightSpec(
            slice_spec=slice_spec,
            impact=impact,
            weights=weights,
        )
        contrib_result: pd.DataFrame | None = None
        if intermediate_outputs and has_utility_contrib and weights is not None:
            ut_enacting_metric_l1, ut_enacting_metric_l2, ut_utility = _ut_input_maps(run=run)
            contrib_result = _compute_ut_weighted_contribution_from_preweight(
                context=run.context,
                state=run.state,
                ssp_scenario=run.ssp_scenario,
                l2_method=l2_method,
                year=run.year,
                lcia_data=slice_spec.lcia_data,
                lcia_key=slice_spec.lcia_key,
                ref_year=slice_spec.ref_year,
                weights=weights,
                enacting_metric_l1=ut_enacting_metric_l1,
                enacting_metric_l2=ut_enacting_metric_l2,
                utility=ut_utility,
            )

        if is_ar:
            result = compute_ar_result(run=run, weight_spec=weight_spec)
        else:
            result = compute_non_ar_or_ut_result(
                run=run,
                weight_spec=weight_spec,
                precomputed_ut_contrib=contrib_result,
            )
            if apply_gvaa_identity_closure:
                result = apply_ut_gvaa_identity_closure(
                    run=run,
                    slice_spec=slice_spec,
                    weights=weights,
                    impact=impact,
                    result=result,
                )
        _write_combined_outputs(
            run=run,
            slice_spec=slice_spec,
            result=result,
            impact=impact,
            reference_year=(reference_year_value if add_reference_level else None),
            contrib_result=contrib_result,
        )

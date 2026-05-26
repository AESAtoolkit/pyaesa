"""One step L2 compute orchestration."""

from typing import Any, cast

import pandas as pd

from ....methods.compute_l2 import compute_l2_method
from ....methods.run_ar import _compute_ar_l2_result
from ...projection.reuse.outputs import (
    cache_historical_one_step_result,
    load_reuse_one_step_result,
)
from ..shared.scenario_routing import (
    is_historical_reuse_l2_projection,
    is_scenario_dependent_l2_projection,
    l2_projection_subfolder,
    resolve_output_ssp_scenario,
)
from .l2_compute_shared import _lcia_items_for_method, _reference_years_for
from .l2_outputs import (
    _append_l2_result,
    _build_l2_output_spec,
    _L2KeySpec,
)
from .l2_reuse_frames import _combine_l2_reuse_year_frames
from .l2_types import _is_ar_l2, _L2RunContext


def _compute_one_step_methods(run: _L2RunContext) -> None:
    """Compute one step L2 methods."""
    for l2_method in run.context.selected_l2_one_step:
        method_is_ar = _is_ar_l2(l2_method=l2_method, fu_code=run.context.fu_code)
        # Non LCIA methods expose one synthetic lcia_key=None item.
        lcia_items = _lcia_items_for_method(run=run, l2_method=l2_method)
        for lcia_key, lcia_data in lcia_items.items():
            refs = _reference_years_for(run=run, l2_method=l2_method, l1_name=None)
            for ref_year in refs:
                if method_is_ar and ref_year is not None:
                    # AR one step reuses AR cache flow with no L1 weights.
                    result = cast(
                        pd.DataFrame,
                        _compute_ar_l2_result(
                            context=run.context,
                            state=run.state,
                            cache_key=(l2_method, None, lcia_key, ref_year, False),
                            l2_method=l2_method,
                            year=run.year,
                            ref_year=ref_year,
                            lcia_data=lcia_data,
                            l1_weights=None,
                        ),
                    )
                    scenario_dependent = is_scenario_dependent_l2_projection(
                        context=run.context,
                        year=run.year,
                        l2_method=l2_method,
                    )
                    routed_ssp_scenario = resolve_output_ssp_scenario(
                        context=run.context,
                        year=run.year,
                        ssp_scenario=run.ssp_scenario,
                        scenario_dependent=scenario_dependent,
                    )
                    key = _build_l2_output_spec(
                        spec=_L2KeySpec(
                            route="l2_vs_global",
                            l2_method=l2_method,
                            l1_method=None,
                            lcia_method_name=lcia_key,
                            ssp_scenario=routed_ssp_scenario,
                            scenario_dependent=scenario_dependent,
                            grouped_mode=bool(run.context.group_indices),
                            projection_subfolder=l2_projection_subfolder(
                                context=run.context,
                                year=run.year,
                                l2_method=l2_method,
                                bucket="l2_vs_global",
                            ),
                        ),
                        frame=result,
                        state=run.state,
                    )
                    _append_l2_result(
                        state=run.state,
                        ssp_scenario=run.ssp_scenario,
                        key=key,
                        result=result,
                    )
                    continue

                if is_historical_reuse_l2_projection(
                    context=run.context,
                    year=run.year,
                    l2_method=l2_method,
                ):
                    projection_context = cast(Any, run.context.projection_context)
                    l2_reuse_years = projection_context.l2_reuse_years_for()
                    frames_by_l2_reuse_year: list[tuple[int, pd.DataFrame]] = []
                    for l2_reuse_year in l2_reuse_years:
                        l2_reuse_year_value = int(l2_reuse_year)
                        result = load_reuse_one_step_result(
                            context=run.context,
                            state=run.state,
                            l2_method=l2_method,
                            lcia_key=lcia_key,
                            l2_reuse_year=l2_reuse_year_value,
                            target_year=int(run.year),
                        )
                        frames_by_l2_reuse_year.append((l2_reuse_year_value, result))
                    result_with_reuse = _combine_l2_reuse_year_frames(
                        frames_by_l2_reuse_year=frames_by_l2_reuse_year,
                    )
                    scenario_dependent = is_scenario_dependent_l2_projection(
                        context=run.context,
                        year=run.year,
                        l2_method=l2_method,
                    )
                    routed_ssp_scenario = resolve_output_ssp_scenario(
                        context=run.context,
                        year=run.year,
                        ssp_scenario=run.ssp_scenario,
                        scenario_dependent=scenario_dependent,
                    )
                    key = _build_l2_output_spec(
                        spec=_L2KeySpec(
                            route="l2_vs_global",
                            l2_method=l2_method,
                            l1_method=None,
                            lcia_method_name=lcia_key,
                            ssp_scenario=routed_ssp_scenario,
                            scenario_dependent=scenario_dependent,
                            grouped_mode=bool(run.context.group_indices),
                            projection_subfolder=l2_projection_subfolder(
                                context=run.context,
                                year=run.year,
                                l2_method=l2_method,
                                bucket="l2_vs_global",
                            ),
                        ),
                        frame=result_with_reuse,
                        state=run.state,
                    )
                    _append_l2_result(
                        state=run.state,
                        ssp_scenario=run.ssp_scenario,
                        key=key,
                        result=result_with_reuse,
                    )
                    continue
                else:
                    result = compute_l2_method(
                        l2_method=l2_method,
                        fu_code=run.context.fu_code,
                        year=run.year,
                        l1_weights=None,
                        fd_rf=run.inputs.fd_rf,
                        gva_rp=run.inputs.gva_rp,
                        fd_rp_sp_rf=run.inputs.fd_rp_sp_rf,
                        fd_rp_sp=run.inputs.fd_rp_sp,
                        fd_rf_sp=run.inputs.fd_rf_sp,
                        gva_rp_sp=run.inputs.gva_rp_sp,
                        x_to_rc=run.inputs.x_to_rc,
                        kappa=run.inputs.kappa,
                        omega_reg=run.inputs.omega_reg,
                        lcia=lcia_data,
                        reference_year=ref_year,
                        pre_weighting=False,
                    )
                cache_historical_one_step_result(
                    context=run.context,
                    state=run.state,
                    year=run.year,
                    l2_method=l2_method,
                    lcia_key=lcia_key,
                    frame=result,
                )
                scenario_dependent = is_scenario_dependent_l2_projection(
                    context=run.context,
                    year=run.year,
                    l2_method=l2_method,
                )
                routed_ssp_scenario = resolve_output_ssp_scenario(
                    context=run.context,
                    year=run.year,
                    ssp_scenario=run.ssp_scenario,
                    scenario_dependent=scenario_dependent,
                )
                key = _build_l2_output_spec(
                    spec=_L2KeySpec(
                        route="l2_vs_global",
                        l2_method=l2_method,
                        l1_method=None,
                        lcia_method_name=lcia_key,
                        ssp_scenario=routed_ssp_scenario,
                        scenario_dependent=scenario_dependent,
                        grouped_mode=bool(run.context.group_indices),
                        projection_subfolder=l2_projection_subfolder(
                            context=run.context,
                            year=run.year,
                            l2_method=l2_method,
                            bucket="l2_vs_global",
                        ),
                    ),
                    frame=result,
                    state=run.state,
                )
                _append_l2_result(
                    state=run.state,
                    ssp_scenario=run.ssp_scenario,
                    key=key,
                    result=result,
                )

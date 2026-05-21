"""Output spec assembly and write hooks for L2 orchestration."""

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from ....methods.compute_l2 import compute_l2_method
from ....methods.equations.ar_result_indexing import (
    _add_reference_level,
)
from ....methods.run_ut import _get_ut_l2_preweight
from ....methods.run_ar import _compute_ar_l2_preweight
from ...projection.reuse.outputs import cache_historical_preweight
from ...projection.reuse.outputs import load_reuse_preweight
from ....runtime.methods.labels import l1_l2_method_label
from ....runtime.output.contracts import (
    OutputRoute,
    OutputSpec,
    identifier_columns_from_frame,
    join_file_owned_tokens,
)
from ..shared.scenario_routing import (
    is_scenario_dependent_l1,
    is_scenario_dependent_l2_projection,
    l2_projection_subfolder,
    resolve_output_ssp_scenario,
)
from ..shared.output_spec_cache import (
    get_cached_output_spec,
    set_cached_output_spec,
)
from .l2_types import (
    _L2RunContext,
    _L2SliceSpec,
    _is_ar_l2,
    _is_ut_l2,
)


@dataclass(frozen=True)
class _L2KeySpec:
    route: str
    l2_method: str
    l1_method: str | None
    lcia_method_name: str | None
    ssp_scenario: str | None
    scenario_dependent: bool
    grouped_mode: bool
    projection_subfolder: str | None = None
    file_suffix: str | None = None


def _build_l2_output_spec(
    *,
    spec: _L2KeySpec,
    frame: pd.DataFrame,
    state=None,
) -> OutputSpec:
    """Build one typed L2 output spec."""
    identifier_columns = list(identifier_columns_from_frame(frame))
    if "reference_year" in frame.columns and "reference_year" not in identifier_columns:
        identifier_columns.append("reference_year")
    cache_key = (
        "L2",
        spec.route,
        spec.l2_method,
        spec.l1_method,
        spec.lcia_method_name,
        spec.ssp_scenario,
        bool(spec.scenario_dependent),
        bool(spec.grouped_mode),
        spec.projection_subfolder,
        spec.file_suffix,
        tuple(identifier_columns),
    )
    cached = get_cached_output_spec(state=state, key=cache_key)
    if isinstance(cached, OutputSpec):
        return cached
    file_stem_tokens: tuple[str | None, ...]
    if spec.route == "l2_in_l1":
        file_stem_tokens = (f"l2_{spec.l2_method}", spec.lcia_method_name)
    elif spec.route == "l2_vs_global" and spec.l1_method:
        file_stem_tokens = (f"{spec.l1_method}_{spec.l2_method}", spec.lcia_method_name)
    else:
        file_stem_tokens = (spec.l2_method, spec.lcia_method_name)
    file_stem = join_file_owned_tokens(*file_stem_tokens)
    l1_l2_method = (
        l1_l2_method_label(l1_method=spec.l1_method, l2_method=spec.l2_method)
        if spec.l1_method
        else spec.l2_method
    )
    route = OutputRoute(
        level="L2",
        bucket=spec.route,
        source=None,
        grouped_mode=spec.grouped_mode,
        variant_tag=None,
        ssp_scenario=spec.ssp_scenario,
        lcia_method=spec.lcia_method_name,
        projection_subfolder=spec.projection_subfolder,
    )
    output_spec = OutputSpec(
        l1_l2_method=l1_l2_method,
        l2_method=spec.l2_method,
        l1_method=spec.l1_method,
        file_stem=file_stem,
        route=route,
        scenario_dependent=spec.scenario_dependent,
        identifier_columns=tuple(identifier_columns),
        terminal_suffix=spec.file_suffix,
    )
    set_cached_output_spec(state=state, key=cache_key, spec=output_spec)
    return output_spec


def _append_l2_result(
    *,
    state,
    ssp_scenario: str | None,
    key: OutputSpec,
    result: pd.DataFrame,
) -> None:
    """Append one L2 result to state."""
    state.l2_results_by_ssp_scenario[ssp_scenario].setdefault(key, []).append(result)


def _publishes_l2_year(run: _L2RunContext) -> bool:
    """Return whether the current compute year belongs to public outputs."""
    return int(run.year) in run.context.persisted_years


def _write_l2_vs_global(
    *,
    run: _L2RunContext,
    slice_spec: _L2SliceSpec,
    result: pd.DataFrame,
) -> None:
    """Store one L2-vs global result frame."""
    if not _publishes_l2_year(run):
        return
    projection_subfolder = l2_projection_subfolder(
        context=run.context,
        year=run.year,
        l2_method=slice_spec.l2_method,
        bucket="l2_vs_global",
    )
    scenario_dependent = is_scenario_dependent_l1(
        slice_spec.l1_name_resolved
    ) or is_scenario_dependent_l2_projection(
        context=run.context,
        year=run.year,
        l2_method=slice_spec.l2_method,
    )
    key = _build_l2_output_spec(
        spec=_L2KeySpec(
            route="l2_vs_global",
            l2_method=slice_spec.l2_method,
            l1_method=(None if slice_spec.treat_as_one_step else slice_spec.l1_name_resolved),
            lcia_method_name=slice_spec.lcia_key,
            ssp_scenario=resolve_output_ssp_scenario(
                context=run.context,
                year=run.year,
                ssp_scenario=run.ssp_scenario,
                scenario_dependent=scenario_dependent,
            ),
            scenario_dependent=scenario_dependent,
            grouped_mode=bool(run.context.aggreg_indices),
            projection_subfolder=projection_subfolder,
            file_suffix=None,
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


def _write_l2_utility_propagation_contrib(
    *,
    run: _L2RunContext,
    slice_spec: _L2SliceSpec,
    result: pd.DataFrame,
) -> None:
    """Store unsummed two step UT contribution by utility propagation axis."""
    if not bool(getattr(run.context, "intermediate_outputs", True)) or not _publishes_l2_year(run):
        return
    projection_subfolder = l2_projection_subfolder(
        context=run.context,
        year=run.year,
        l2_method=slice_spec.l2_method,
        bucket="utility_propagation_contrib",
    )
    suffix = "per_rf" if slice_spec.l2_method == "UT(FDa)" else "per_ru"
    scenario_dependent = is_scenario_dependent_l1(
        slice_spec.l1_name_resolved
    ) or is_scenario_dependent_l2_projection(
        context=run.context,
        year=run.year,
        l2_method=slice_spec.l2_method,
    )
    key = _build_l2_output_spec(
        spec=_L2KeySpec(
            route="utility_propagation_contrib",
            l2_method=slice_spec.l2_method,
            l1_method=slice_spec.l1_name_resolved,
            lcia_method_name=slice_spec.lcia_key,
            ssp_scenario=resolve_output_ssp_scenario(
                context=run.context,
                year=run.year,
                ssp_scenario=run.ssp_scenario,
                scenario_dependent=scenario_dependent,
            ),
            scenario_dependent=scenario_dependent,
            grouped_mode=bool(run.context.aggreg_indices),
            projection_subfolder=projection_subfolder,
            file_suffix=suffix,
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


def _write_l2_preweight(
    *,
    run: _L2RunContext,
    slice_spec: _L2SliceSpec,
) -> None:
    """Store one L2 preweight frame when needed."""
    l2_method = slice_spec.l2_method
    pre_lcia_key = (
        None if _is_ut_l2(l2_method=l2_method, fu_code=run.context.fu_code) else slice_spec.lcia_key
    )
    ref_key = (
        None if _is_ut_l2(l2_method=l2_method, fu_code=run.context.fu_code) else slice_spec.ref_year
    )
    written_key = (l2_method, pre_lcia_key, ref_key, run.year)
    if written_key in run.state.pre_weighting_written_by_ssp_scenario[run.ssp_scenario]:
        return
    is_ar_method = _is_ar_l2(l2_method=l2_method, fu_code=run.context.fu_code)
    is_ut_method = _is_ut_l2(l2_method=l2_method, fu_code=run.context.fu_code)
    is_adjusted_ut_method = l2_method in {"UT(FDa)", "UT(GVAa)"}
    if is_ar_method and slice_spec.ref_year is not None:
        pre = cast(
            pd.DataFrame,
            _compute_ar_l2_preweight(
                context=run.context,
                state=run.state,
                cache_key=(l2_method, None, pre_lcia_key, slice_spec.ref_year, True),
                l2_method=l2_method,
                year=run.year,
                ref_year=slice_spec.ref_year,
                lcia_data=slice_spec.lcia_data,
            ),
        )
        pre = _add_reference_level(pre, slice_spec.ref_year)
    elif is_adjusted_ut_method:
        pre = _get_ut_l2_preweight(
            context=run.context,
            state=run.state,
            ssp_scenario=run.ssp_scenario,
            l2_method=l2_method,
            year=run.year,
            lcia_data=None,
            lcia_key=None,
            ref_year=None,
            enacting_metric_l1={"fd_rf": run.inputs.fd_rf, "gva_rp": run.inputs.gva_rp},
            enacting_metric_l2={
                "fd_rp_sp_rf": run.inputs.fd_rp_sp_rf,
                "fd_rp_sp": run.inputs.fd_rp_sp,
                "fd_rf_sp": run.inputs.fd_rf_sp,
                "gva_rp_sp": run.inputs.gva_rp_sp,
            },
            utility={
                "x_to_rc": run.inputs.x_to_rc,
                "kappa": run.inputs.kappa,
                "omega_reg": run.inputs.omega_reg,
            },
        )
    else:
        pre = compute_l2_method(
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
            lcia=slice_spec.lcia_data,
            reference_year=slice_spec.ref_year,
            pre_weighting=True,
        )
    if is_ut_method:
        cache_historical_preweight(
            context=run.context,
            state=run.state,
            year=run.year,
            l2_method=l2_method,
            lcia_key=pre_lcia_key,
            frame=pre,
        )
    written_for_ssp_scenario = run.state.pre_weighting_written_by_ssp_scenario[run.ssp_scenario]
    written_for_ssp_scenario.add(written_key)
    if not _publishes_l2_year(run):
        return
    scenario_dependent = is_scenario_dependent_l2_projection(
        context=run.context,
        year=run.year,
        l2_method=l2_method,
    )
    key = _build_l2_output_spec(
        spec=_L2KeySpec(
            route="l2_in_l1",
            l2_method=l2_method,
            l1_method=None,
            lcia_method_name=pre_lcia_key,
            ssp_scenario=resolve_output_ssp_scenario(
                context=run.context,
                year=run.year,
                ssp_scenario=run.ssp_scenario,
                scenario_dependent=scenario_dependent,
            ),
            scenario_dependent=scenario_dependent,
            grouped_mode=bool(run.context.aggreg_indices),
            projection_subfolder=l2_projection_subfolder(
                context=run.context,
                year=run.year,
                l2_method=l2_method,
                bucket="l2_in_l1",
            ),
        ),
        frame=pre,
        state=run.state,
    )
    _append_l2_result(state=run.state, ssp_scenario=run.ssp_scenario, key=key, result=pre)


def _write_l2_historical_reuse_preweight(
    *,
    run: _L2RunContext,
    slice_spec: _L2SliceSpec,
) -> None:
    """Store source year L2 preweight frames used by historical reuse."""
    if not _publishes_l2_year(run):
        return
    projection_context = cast(Any, run.context.projection_context)
    l2_reuse_years = projection_context.l2_reuse_years_for()
    l2_method = slice_spec.l2_method
    pre_lcia_key = (
        None if _is_ut_l2(l2_method=l2_method, fu_code=run.context.fu_code) else slice_spec.lcia_key
    )
    written_key = (l2_method, pre_lcia_key, None, run.year)
    written_for_ssp_scenario = run.state.pre_weighting_written_by_ssp_scenario[run.ssp_scenario]
    if written_key in written_for_ssp_scenario:
        return
    frames = [
        load_reuse_preweight(
            context=run.context,
            state=run.state,
            l2_method=l2_method,
            lcia_key=pre_lcia_key,
            l2_reuse_year=int(l2_reuse_year),
        )
        for l2_reuse_year in l2_reuse_years
    ]
    written_for_ssp_scenario.add(written_key)
    key = _build_l2_output_spec(
        spec=_L2KeySpec(
            route="l2_in_l1",
            l2_method=l2_method,
            l1_method=None,
            lcia_method_name=pre_lcia_key,
            ssp_scenario=None,
            scenario_dependent=False,
            grouped_mode=bool(run.context.aggreg_indices),
            projection_subfolder=l2_projection_subfolder(
                context=run.context,
                year=run.year,
                l2_method=l2_method,
                bucket="l2_in_l1",
            ),
        ),
        frame=frames[0],
        state=run.state,
    )
    for frame in frames:
        _append_l2_result(state=run.state, ssp_scenario=run.ssp_scenario, key=key, result=frame)

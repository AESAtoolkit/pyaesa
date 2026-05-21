"""Compute only primitives for one L2 impact slice.

This module is the narrow execution boundary between yearly L2 orchestration
and method equations. It computes one result DataFrame for one (method, impact, optional
L1-weights) slice, with explicit routing for:
- AR flows (direct result or preweight + L1 weighting), and
- UT adjusted flows (weighted from UT preweights/utility inputs).

It does not orchestrate loops across impacts/years and does not write outputs.
Those responsibilities are in ``l2_compute_impacts.py`` and output modules.
"""

import pandas as pd

from ....methods.compute_l2 import apply_l1_weights_to_preweighted, compute_l2_method
from ....methods.equations.ar_result_indexing import _add_reference_level, _apply_impact_level
from ....methods.run_ar import (
    _compute_ar_l2_preweight,
    _compute_ar_l2_result as compute_ar_l2_result,
)
from ....methods.run_ut import _compute_ut_weighted_from_preweight
from .l2_contracts import require_frame, require_ref_year, require_weights
from .l2_types import _L2RunContext, _L2SliceSpec, _L2WeightSpec


def compute_non_ar_or_ut_result(
    *,
    run: _L2RunContext,
    weight_spec: _L2WeightSpec,
    precomputed_ut_contrib: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute non AR L2 result for one impact slice."""
    l2_method = weight_spec.slice_spec.l2_method
    where = f"year={run.year}, fu_code='{run.context.fu_code}', l2_method='{l2_method}'"
    if l2_method in {"UT(FDa)", "UT(GVAa)"}:
        ut_enacting_metric_l1 = {"fd_rf": run.inputs.fd_rf, "gva_rp": run.inputs.gva_rp}
        ut_enacting_metric_l2 = {
            "fd_rp_sp_rf": run.inputs.fd_rp_sp_rf,
            "fd_rp_sp": run.inputs.fd_rp_sp,
            "fd_rf_sp": run.inputs.fd_rf_sp,
            "gva_rp_sp": run.inputs.gva_rp_sp,
        }
        ut_utility = {
            "x_to_rc": run.inputs.x_to_rc,
            "kappa": run.inputs.kappa,
            "omega_reg": run.inputs.omega_reg,
        }
        return _compute_ut_weighted_from_preweight(
            context=run.context,
            state=run.state,
            ssp_scenario=run.ssp_scenario,
            l2_method=l2_method,
            year=run.year,
            lcia_data=weight_spec.slice_spec.lcia_data,
            lcia_key=None,
            ref_year=weight_spec.slice_spec.ref_year,
            weights=require_weights(weights=weight_spec.weights, where=where),
            enacting_metric_l1=ut_enacting_metric_l1,
            enacting_metric_l2=ut_enacting_metric_l2,
            utility=ut_utility,
            precomputed_contrib=precomputed_ut_contrib,
        )
    return compute_l2_method(
        l2_method=l2_method,
        fu_code=run.context.fu_code,
        year=run.year,
        l1_weights=(weight_spec.weights if not weight_spec.slice_spec.treat_as_one_step else None),
        fd_rf=run.inputs.fd_rf,
        gva_rp=run.inputs.gva_rp,
        fd_rp_sp_rf=run.inputs.fd_rp_sp_rf,
        fd_rp_sp=run.inputs.fd_rp_sp,
        fd_rf_sp=run.inputs.fd_rf_sp,
        gva_rp_sp=run.inputs.gva_rp_sp,
        x_to_rc=run.inputs.x_to_rc,
        kappa=run.inputs.kappa,
        omega_reg=run.inputs.omega_reg,
        lcia=weight_spec.slice_spec.lcia_data,
        reference_year=weight_spec.slice_spec.ref_year,
        pre_weighting=False,
    )


def compute_ar_result(
    *,
    run: _L2RunContext,
    weight_spec: _L2WeightSpec,
) -> pd.DataFrame:
    """Compute AR result for one impact slice."""
    slice_spec: _L2SliceSpec = weight_spec.slice_spec
    where = f"year={run.year}, fu_code='{run.context.fu_code}', l2_method='{slice_spec.l2_method}'"
    ref_year = require_ref_year(ref_year=slice_spec.ref_year, where=where)
    cache_key = (
        slice_spec.l2_method,
        None,
        slice_spec.lcia_key,
        ref_year,
        False,
    )
    if weight_spec.weights is None:
        result = require_frame(
            frame=compute_ar_l2_result(
                context=run.context,
                state=run.state,
                cache_key=cache_key,
                l2_method=slice_spec.l2_method,
                year=run.year,
                ref_year=ref_year,
                lcia_data=slice_spec.lcia_data,
                l1_weights=None,
            ),
            where=where,
            subject="AR L2 output",
        )
        if weight_spec.impact is not None:
            result = _apply_impact_level(result, weight_spec.impact)
        return result
    pre = require_frame(
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
    if weight_spec.impact is not None:
        pre = _apply_impact_level(pre, weight_spec.impact)
    weighted = apply_l1_weights_to_preweighted(
        l2_method=slice_spec.l2_method,
        fu_code=run.context.fu_code,
        year=run.year,
        pre_weighted=pre,
        l1_weights=weight_spec.weights,
    )
    return _add_reference_level(weighted, ref_year)

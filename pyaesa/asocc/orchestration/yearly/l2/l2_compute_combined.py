"""Combined (two step) L2 compute orchestration."""

import pandas as pd

from ....methods.registry.registry import REGISTRY
from . import l2_compute_impacts as _impacts
from .l2_compute_shared import (
    _l1_weights_key_for_pair,
    _lcia_items_for_method,
    _reference_years_for,
)
from .l2_outputs import _write_l2_historical_reuse_preweight, _write_l2_preweight
from .l2_types import (
    _CombinedSliceRequest,
    _L2RunContext,
    _L2SliceSpec,
    _is_ar_l1,
    _is_ar_l2,
)
from ..shared.scenario_routing import is_historical_reuse_l2_projection


def _compute_combined_impact_results(*, run, slice_spec, l1_weights):
    """Delegate combined impact slice compute to the impact execution flow."""
    return _impacts.compute_combined_impact_results(
        run=run,
        slice_spec=slice_spec,
        l1_weights=l1_weights,
    )


def _resolve_combined_slice(
    *,
    run: _L2RunContext,
    request: _CombinedSliceRequest,
) -> tuple[_L2SliceSpec, pd.DataFrame | None] | None:
    """Resolve one combined run slice and optional L1 weights."""
    l2_method = request.l2_method
    l1_name = request.l1_name
    l2_is_ar = _is_ar_l2(l2_method=l2_method, fu_code=run.context.fu_code)
    l1_is_ar = _is_ar_l1(l1_name)
    if (l2_is_ar or l1_is_ar) and request.lcia_data is None:
        return None
    l1_name_resolved = l1_name
    treat_as_one_step = l2_is_ar and l1_is_ar
    l1_key = l1_name_resolved
    if REGISTRY.method_requires_lcia(l1_name, None):
        if request.lcia_key is None:
            return None
        l1_key = f"{l1_key}_{request.lcia_key}"
    if request.ref_year is not None and l1_is_ar:
        l1_key = f"{l1_key}_ref_{request.ref_year}"
    l1_weights = None
    if not treat_as_one_step:
        pair_key = _l1_weights_key_for_pair(
            base_key=l1_key,
            l1_name=l1_name,
            l2_method=l2_method,
        )
        l1_weights = run.l1_results_year.get(pair_key)
        if l1_weights is None and pair_key != l1_key:
            l1_weights = run.l1_results_year.get(l1_key)
    if not treat_as_one_step and l1_weights is None:
        return None
    return (
        _L2SliceSpec(
            l2_method=l2_method,
            l1_name=l1_name,
            l1_name_resolved=l1_name_resolved,
            lcia_key=request.lcia_key,
            lcia_data=request.lcia_data,
            ref_year=request.ref_year,
            treat_as_one_step=treat_as_one_step,
        ),
        l1_weights,
    )


def _compute_combined_methods(run: _L2RunContext) -> None:
    """Compute L2 results weighted by L1 methods."""
    # Iteration order is method -> lcia key -> reference year so outputs are
    # deterministic and cache keys remain stable across runs.
    for l2_method, l1_name in run.context.combined:
        lcia_items = _lcia_items_for_method(run=run, l2_method=l2_method, l1_name=l1_name)
        for lcia_key, lcia_data in lcia_items.items():
            refs = _reference_years_for(run=run, l2_method=l2_method, l1_name=l1_name)
            for ref_year in refs:
                resolved = _resolve_combined_slice(
                    run=run,
                    request=_CombinedSliceRequest(
                        l2_method=l2_method,
                        l1_name=l1_name,
                        lcia_key=lcia_key,
                        lcia_data=lcia_data,
                        ref_year=ref_year,
                    ),
                )
                if resolved is None:
                    continue
                slice_spec, l1_weights = resolved
                _compute_combined_impact_results(
                    run=run,
                    slice_spec=slice_spec,
                    l1_weights=l1_weights,
                )
                if slice_spec.treat_as_one_step:
                    continue
                if is_historical_reuse_l2_projection(
                    context=run.context,
                    year=run.year,
                    l2_method=slice_spec.l2_method,
                ):
                    _write_l2_historical_reuse_preweight(run=run, slice_spec=slice_spec)
                else:
                    _write_l2_preweight(run=run, slice_spec=slice_spec)

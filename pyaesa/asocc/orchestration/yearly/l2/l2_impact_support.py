"""Support flow for combined L2 impact execution."""

import pandas as pd

from ....methods.equations.ar_result_indexing import _attach_impact_reference_levels
from ....methods.registry.registry import REGISTRY
from .l2_outputs import (
    _publishes_l2_year,
    _write_l2_utility_propagation_contrib,
    _write_l2_vs_global,
)
from .l2_types import _L2RunContext, _L2SliceSpec

_EnactingInputs = dict[str, pd.Series | pd.DataFrame]


def _should_write_historical_reuse_utility_contrib(
    *,
    slice_spec: _L2SliceSpec,
) -> bool:
    """Return whether historical reuse utility propagation outputs should be written."""
    l1_name = slice_spec.l1_name_resolved
    if not l1_name:
        return True
    return REGISTRY.method_family(l1_name, level="L1") != "AR_E"


def _ut_input_maps(
    *,
    run: _L2RunContext,
) -> tuple[_EnactingInputs, _EnactingInputs, _EnactingInputs]:
    """Return UT enacting metric inputs required by adjusted utility branches."""
    return (
        {"fd_rf": run.inputs.fd_rf, "gva_rp": run.inputs.gva_rp},
        {
            "fd_rp_sp_rf": run.inputs.fd_rp_sp_rf,
            "fd_rp_sp": run.inputs.fd_rp_sp,
            "fd_rf_sp": run.inputs.fd_rf_sp,
            "gva_rp_sp": run.inputs.gva_rp_sp,
        },
        {
            "x_to_rc": run.inputs.x_to_rc,
            "kappa": run.inputs.kappa,
            "omega_reg": run.inputs.omega_reg,
        },
    )


def _write_combined_outputs(
    *,
    run: _L2RunContext,
    slice_spec: _L2SliceSpec,
    result: pd.DataFrame,
    impact: str | None,
    reference_year: int | None,
    contrib_result: pd.DataFrame | None = None,
    l2_reuse_year: int | None = None,
) -> None:
    """Attach output levels and persist one combined slice result bundle."""
    if not _publishes_l2_year(run):
        return
    trailing_levels = () if l2_reuse_year is None else (("l2_reuse_year", int(l2_reuse_year)),)
    index_cache = getattr(run.state, "output_index_level_cache", None)
    result_to_write = _attach_impact_reference_levels(
        result=result,
        impact=impact,
        reference_year=reference_year,
        trailing_levels=trailing_levels,
        index_cache=index_cache,
    )
    if contrib_result is not None:
        contrib_to_write = _attach_impact_reference_levels(
            result=contrib_result,
            impact=impact,
            reference_year=reference_year,
            trailing_levels=trailing_levels,
            index_cache=index_cache,
        )
        _write_l2_utility_propagation_contrib(
            run=run,
            slice_spec=slice_spec,
            result=contrib_to_write,
        )
    _write_l2_vs_global(
        run=run,
        slice_spec=slice_spec,
        result=result_to_write,
    )

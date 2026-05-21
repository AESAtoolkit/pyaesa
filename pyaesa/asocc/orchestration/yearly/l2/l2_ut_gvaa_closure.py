"""UT(GVAa) identity closure for L2 yearly compute."""

from collections.abc import Hashable
from typing import cast

import pandas as pd

from ....methods.equations.ut_gva import compute_ut_gva_l2
from .l2_types import _L2RunContext, _L2SliceSpec

_UT_GVAA_IDENTITY_CLOSURE_TOL = 1e-18
_UT_GVAA_IDENTITY_CLOSURE_NOTE = (
    "UT(GVAa) was below UT(GVA) for this tuple. "
    "The final UT(GVAa) value was set equal to UT(GVA) because MRIO input-output "
    "identity disequilibrium (inputs < outputs) was detected for this sector-region pair "
    "(MRIO data quality issue)."
)


def _single_year_series(frame: pd.DataFrame, *, label: str) -> pd.Series:
    """Extract the only value column from a one year L2 output frame."""
    if frame.shape[1] != 1:
        raise ValueError(f"{label} must have exactly one year column.")
    return frame.iloc[:, 0]


def _align_floor_index_to_result(
    *,
    floor: pd.Series,
    result: pd.Series,
) -> pd.Series:
    """Align UT(GVA) floor index ordering to UT(GVAa) result ordering."""
    if not isinstance(floor.index, pd.MultiIndex) or not isinstance(result.index, pd.MultiIndex):
        return floor
    floor_names_raw = list(floor.index.names)
    result_names_raw = list(result.index.names)
    if (
        floor_names_raw == result_names_raw
        or any(name is None for name in floor_names_raw)
        or any(name is None for name in result_names_raw)
    ):
        return floor
    floor_names = [str(name) for name in floor_names_raw]
    result_names = [str(name) for name in result_names_raw]
    if set(floor_names) != set(result_names):
        return floor
    order = [floor_names.index(name) for name in result_names]
    return floor.reorder_levels(order)


def apply_ut_gvaa_identity_closure(
    *,
    run: _L2RunContext,
    slice_spec: _L2SliceSpec,
    weights: pd.Series | None,
    impact: str | None,
    result: pd.DataFrame,
    l2_reuse_year: int | None = None,
) -> pd.DataFrame:
    """Enforce UT(GVAa) >= UT(GVA) for L2.a.b and record correction audit rows."""
    if slice_spec.l2_method != "UT(GVAa)" or run.context.fu_code != "L2.a.b" or weights is None:
        return result

    raw_series = _single_year_series(result, label="UT(GVAa) result")
    floor_raw = _single_year_series(
        compute_ut_gva_l2(
            year=run.year,
            l1_weights=weights,
            gva_rp=run.inputs.gva_rp,
            gva_rp_sp=run.inputs.gva_rp_sp,
            pre_weighting=False,
        ),
        label="UT(GVA) floor",
    )
    floor_raw = _align_floor_index_to_result(floor=floor_raw, result=raw_series)
    floor_series = floor_raw.reindex(raw_series.index)
    floor_series = floor_series.where(floor_series.notna(), raw_series)

    delta = floor_series - raw_series
    mask = delta > _UT_GVAA_IDENTITY_CLOSURE_TOL
    if not bool(mask.any()):
        return result

    corrected = raw_series.copy()
    corrected.loc[mask] = floor_series.loc[mask]

    projection_branch = str(run.context.proj_base.name)
    ssp_scenario_label = "base" if run.ssp_scenario is None else str(run.ssp_scenario)
    impact_label = "" if impact is None else str(impact)
    lcia_key = "" if slice_spec.lcia_key is None else str(slice_spec.lcia_key)
    l1_method = "" if slice_spec.l1_name is None else str(slice_spec.l1_name)
    reference_year = int(slice_spec.ref_year) if slice_spec.ref_year is not None else pd.NA
    l2_reuse_year_value = int(l2_reuse_year) if l2_reuse_year is not None else pd.NA

    selected_keys = corrected.loc[mask].index.tolist()
    for index_key in selected_keys:
        if isinstance(corrected.index, pd.MultiIndex):
            index_tuple = index_key if isinstance(index_key, tuple) else (index_key,)
            key_by_level = {
                str(level_name): str(level_value)
                for level_name, level_value in zip(corrected.index.names, index_tuple)
            }
        else:
            key_by_level = {"r_p": str(index_key)}
        row = {
            "projection_branch": projection_branch,
            "source": run.context.source,
            "fu_code": run.context.fu_code,
            "year": int(run.year),
            "ssp_scenario": ssp_scenario_label,
            "l2_method": "UT(GVAa)",
            "comparator_method": "UT(GVA)",
            "l1_method": l1_method,
            "impact": impact_label,
            "lcia_key": lcia_key,
            "reference_year": reference_year,
            "l2_reuse_year": l2_reuse_year_value,
            "r_p": key_by_level.get("r_p", ""),
            "s_p": key_by_level.get("s_p", ""),
            "ut_gvaa_raw": float(raw_series.loc[index_key]),
            "ut_gva_floor": float(floor_series.loc[index_key]),
            "ut_gvaa_final": float(corrected.loc[index_key]),
            "delta_added": float(delta.loc[index_key]),
            "adjustment_note": _UT_GVAA_IDENTITY_CLOSURE_NOTE,
        }
        run.state.ut_gvaa_identity_closure_rows.append(row)

    column_name = result.columns[0]
    if isinstance(column_name, pd.Index):
        output_name: Hashable = str(column_name)
    else:
        output_name = cast(Hashable, column_name)
    return corrected.to_frame(output_name)

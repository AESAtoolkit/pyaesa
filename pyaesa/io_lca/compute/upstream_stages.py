"""Upstream stage diagnostic computations for IO-LCA."""

from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.asocc.runtime.scope.filtering import slice_frame_any_axis
from pyaesa.io_lca.contracts.fu_mapping import IOLCAFUSpec
from pyaesa.io_lca.data.loaders import UpstreamPayload, impact_unit_text_map
from pyaesa.io_lca.orchestration.io.method_support import (
    origin_long_columns,
    stage_public_columns,
)

from .stage_linkage import dominant_parent_link_map
from .upstream_long_rows import pair_labels, stage_rows_from_values, value_rows_from_values

_EPS = 1e-15


def _numeric_series(values: pd.Series) -> pd.Series:
    """Return one numeric Series with NA coerced to ``0.0``."""
    numeric = cast(pd.Series, pd.to_numeric(values, errors="raise"))
    return cast(pd.Series, numeric.fillna(0.0))


def _drop_zero_rows(
    *,
    frame: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    """Drop rows where all selected value columns are numerically zero."""
    if frame.empty:
        return frame
    work = frame.copy()
    for col in value_columns:
        work[col] = _numeric_series(cast(pd.Series, work[col]))
    mask = work[value_columns].abs().sum(axis=1) > _EPS
    return work.loc[mask].reset_index(drop=True)


def _stage_label(depth: int) -> str:
    """Return stage label using ``n`` as the first stage."""
    if int(depth) <= 0:
        return "n"
    return f"n-{int(depth)}"


def _aligned_structural_payload(
    payload: UpstreamPayload,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align A/L/S matrices to one shared product axis order."""
    product_index = payload.driver_matrix.index
    a_matrix = payload.a_matrix.reindex(index=product_index, columns=product_index, fill_value=0.0)
    l_matrix = payload.l_matrix.reindex(index=product_index, columns=product_index, fill_value=0.0)
    s_matrix = payload.s_matrix.reindex(columns=product_index, fill_value=0.0)
    return a_matrix, l_matrix, s_matrix


def _combo_records(
    *,
    combos: pd.DataFrame,
    selector_axes: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return selector combos as string keyed dictionaries."""
    records = [
        {str(key): value for key, value in zip(combos.columns, values, strict=True)}
        for values in combos.itertuples(index=False, name=None)
    ]
    if records:
        return records
    return [{axis: None for axis in selector_axes}]


def _combo_driver_vector(
    *,
    driver: pd.DataFrame,
    spec: IOLCAFUSpec,
    combo: dict[str, Any],
) -> pd.Series:
    """Build one selected driver vector ``d_sel`` for upstream diagnostics."""
    selected = driver.copy()
    for user_axis in spec.selector_axes:
        value = combo.get(user_axis)
        if value is None or str(value) == "nan":
            continue
        selected = slice_frame_any_axis(
            selected,
            axis_name=user_axis,
            allowed={str(value)},
        )
    if selected.empty:
        return pd.Series(0.0, index=driver.index, dtype=float)
    out = cast(pd.Series, selected.sum(axis=1))
    out = out.reindex(driver.index, fill_value=0.0)
    out.index = driver.index
    return out.astype(float)


def _stage_rows_for_combo(
    *,
    year: int,
    combo: dict[str, Any],
    selector_axes: tuple[str, ...],
    upstream_stages: int,
    a_matrix: pd.DataFrame,
    l_matrix: pd.DataFrame,
    s_matrix: pd.DataFrame,
    d_sel: pd.Series,
    unit_map: dict[str, str],
    use_leontief_multiplier: bool,
    emit_stage_rows: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute stage rows and exact additive origin totals for one combo."""
    impact_labels = s_matrix.index.astype(str).to_numpy()
    stage_r_labels, stage_s_labels = pair_labels(s_matrix.columns)
    a_values = a_matrix.to_numpy(dtype=float, copy=False)
    l_values = l_matrix.to_numpy(dtype=float, copy=False)
    s_values = s_matrix.to_numpy(dtype=float, copy=False)
    if use_leontief_multiplier:
        multiplier_total = s_values @ l_values
    else:
        multiplier_total = s_values
    d_vector = _numeric_series(cast(pd.Series, d_sel.reindex(a_matrix.index, fill_value=0.0)))
    q = d_vector.to_numpy(dtype=float, copy=False)
    stage_frames: list[pd.DataFrame] = []
    if emit_stage_rows:
        for depth in range(int(upstream_stages) + 1):
            q_prev = q.copy()
            if depth > 0:
                q = a_values @ q_prev
            direct_values = s_values * q[np.newaxis, :]
            total_values = multiplier_total * q[np.newaxis, :]
            embedded_values = total_values - direct_values
            merged = stage_rows_from_values(
                impact_labels=impact_labels,
                stage_r_labels=stage_r_labels,
                stage_s_labels=stage_s_labels,
                direct_values=direct_values,
                embedded_values=embedded_values,
                total_values=total_values,
                eps=_EPS,
            )
            merged["stage"] = _stage_label(depth)
            if depth == 0:
                merged["linked_from_stage"] = ""
                merged["linked_from_r_p"] = ""
                merged["linked_from_s_p"] = ""
            else:
                link_map = dominant_parent_link_map(
                    a_matrix=a_matrix,
                    q_prev=pd.Series(q_prev, index=a_matrix.columns),
                    eps=_EPS,
                )
                linked_pairs = [
                    link_map.get((str(r_p), str(s_p)), ("", ""))
                    for r_p, s_p in zip(merged["stage_r_p"], merged["stage_s_p"])
                ]
                merged["linked_from_stage"] = _stage_label(depth - 1)
                merged["linked_from_r_p"] = [pair[0] for pair in linked_pairs]
                merged["linked_from_s_p"] = [pair[1] for pair in linked_pairs]
            stage_frames.append(merged)
    if stage_frames:
        stage_rows = pd.concat(stage_frames, ignore_index=True)
    else:
        stage_rows = pd.DataFrame(
            columns=[
                "impact",
                "stage_r_p",
                "stage_s_p",
                "linked_from_stage",
                "linked_from_r_p",
                "linked_from_s_p",
                "direct_at_stage",
                "embedded_from_deeper_stages",
                "stage_total",
                "stage",
            ]
        )
    for axis in selector_axes:
        stage_rows[axis] = combo.get(axis)
    stage_rows["impact_unit"] = [
        unit_map.get(impact)
        for impact in cast(pd.Series, stage_rows["impact"]).astype(str).tolist()
    ]
    stage_rows = _drop_zero_rows(
        frame=stage_rows,
        value_columns=["direct_at_stage", "embedded_from_deeper_stages", "stage_total"],
    )
    stage_order = stage_public_columns(selector_axes)
    if use_leontief_multiplier:
        footprint_output = l_values @ d_vector.to_numpy(dtype=float, copy=False)
    else:
        footprint_output = d_vector.to_numpy(dtype=float, copy=False)
    origin_values = s_values * footprint_output[np.newaxis, :]
    origin_rows = value_rows_from_values(
        impact_labels=impact_labels,
        r_labels=stage_r_labels,
        s_labels=stage_s_labels,
        values=origin_values,
        eps=_EPS,
        value_column="lca_value",
        r_column="origin_r_p",
        s_column="origin_s_p",
    )
    origin_rows["year"] = int(year)
    for axis in selector_axes:
        origin_rows[axis] = combo.get(axis)
    origin_rows["impact_unit"] = [
        unit_map.get(impact)
        for impact in cast(pd.Series, origin_rows["impact"]).astype(str).tolist()
    ]
    origin_rows = _drop_zero_rows(frame=origin_rows, value_columns=["lca_value"])
    origin_order = origin_long_columns(selector_axes)
    return stage_rows.loc[:, stage_order], origin_rows.loc[:, origin_order]


def _fy_rows_for_combo(
    *,
    year: int,
    combo: dict[str, Any],
    spec: IOLCAFUSpec,
    fy_matrix: pd.DataFrame,
    unit_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return stage/origin rows for separate F_Y component."""
    # ``load_upstream_payload(...)`` normalizes F_Y columns onto the canonical
    # region axis before this helper is called.
    value_axis = "region"
    selected = fy_matrix.copy()
    region_axis = next(
        axis for axis in ("r_f", "r_p", "r_c", "r_u", "r_y") if axis in spec.selector_axes
    )
    region = combo.get(region_axis)
    if region is not None:
        selected = slice_frame_any_axis(selected, axis_name=value_axis, allowed={str(region)})
    fy_total = cast(pd.Series, selected.sum(axis=1))
    stage_rows = pd.DataFrame(
        {
            "impact": fy_total.index.astype(str),
            "stage_r_p": "F_Y",
            "stage_s_p": "",
            "linked_from_stage": "",
            "linked_from_r_p": "",
            "linked_from_s_p": "",
            "direct_at_stage": fy_total.to_numpy(dtype=float),
            "embedded_from_deeper_stages": 0.0,
            "stage_total": fy_total.to_numpy(dtype=float),
            "stage": "direct_final_demand_FY",
            "year": int(year),
        }
    )
    for axis in spec.selector_axes:
        stage_rows[axis] = combo.get(axis)
    stage_rows["impact_unit"] = [
        unit_map.get(impact)
        for impact in cast(pd.Series, stage_rows["impact"]).astype(str).tolist()
    ]
    stage_rows = _drop_zero_rows(
        frame=stage_rows,
        value_columns=["direct_at_stage", "embedded_from_deeper_stages", "stage_total"],
    )
    stage_order = stage_public_columns(spec.selector_axes)
    origin_rows = pd.DataFrame(
        {
            "year": int(year),
            "impact": fy_total.index.astype(str),
            "origin_r_p": "F_Y",
            # Keep origin PK stable across CSV round trips; empty strings are
            # read back as NaN and can create duplicate wide key rows.
            "origin_s_p": "F_Y",
            "lca_value": fy_total.to_numpy(dtype=float),
            "impact_unit": [
                unit_map.get(str(impact)) for impact in fy_total.index.astype(str).tolist()
            ],
        }
    )
    for axis in spec.selector_axes:
        origin_rows[axis] = combo.get(axis)
    origin_rows = _drop_zero_rows(frame=origin_rows, value_columns=["lca_value"])
    origin_order = origin_long_columns(spec.selector_axes)
    return stage_rows.loc[:, stage_order], origin_rows.loc[:, origin_order]


def compute_upstream_rows(
    *,
    year: int,
    spec: IOLCAFUSpec,
    combos: pd.DataFrame,
    payload: UpstreamPayload,
    upstream_stages: int,
    unit_by_impact: pd.Series,
    emit_stage_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute stage diagnostics and additive origin rows for one year/LCIA method."""
    a_matrix, l_matrix, s_matrix = _aligned_structural_payload(payload)
    unit_map = impact_unit_text_map(unit_by_impact=unit_by_impact)
    stage_rows_all: list[pd.DataFrame] = []
    origin_rows_all: list[pd.DataFrame] = []
    combo_records = _combo_records(combos=combos, selector_axes=spec.selector_axes)
    for combo in combo_records:
        d_sel = _combo_driver_vector(driver=payload.driver_matrix, spec=spec, combo=combo)
        use_leontief_multiplier = spec.family != "pba"
        stage_rows, origin_rows = _stage_rows_for_combo(
            year=year,
            combo=combo,
            selector_axes=spec.selector_axes,
            upstream_stages=upstream_stages,
            a_matrix=a_matrix,
            l_matrix=l_matrix,
            s_matrix=s_matrix,
            d_sel=d_sel,
            unit_map=unit_map,
            use_leontief_multiplier=use_leontief_multiplier,
            emit_stage_rows=emit_stage_rows,
        )
        if emit_stage_rows:
            stage_rows_all.append(stage_rows)
        origin_rows_all.append(origin_rows)
        if payload.fy_matrix is not None:
            fy_stage, fy_origin = _fy_rows_for_combo(
                year=year,
                combo=combo,
                spec=spec,
                fy_matrix=payload.fy_matrix,
                unit_map=unit_map,
            )
            if emit_stage_rows:
                stage_rows_all.append(fy_stage)
            origin_rows_all.append(fy_origin)
    stage_out = (
        pd.concat(stage_rows_all, ignore_index=True)
        if stage_rows_all
        else pd.DataFrame(
            columns=[
                *spec.selector_axes,
                "stage",
                "stage_r_p",
                "stage_s_p",
                "linked_from_stage",
                "linked_from_r_p",
                "linked_from_s_p",
                "impact",
                "impact_unit",
                "direct_at_stage",
                "embedded_from_deeper_stages",
                "stage_total",
            ]
        )
    )
    origin_out = (
        pd.concat(origin_rows_all, ignore_index=True)
        if origin_rows_all
        else pd.DataFrame(columns=origin_long_columns(spec.selector_axes))
    )
    return stage_out, origin_out

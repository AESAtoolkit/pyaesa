"""Allocation output frame assembly before deterministic persistence."""

import numpy as np
import pandas as pd

from ....runtime.output.contracts import OutputSpec, persisted_method_columns_for_output_spec
from pyaesa.asocc.orchestration.write.tables.wide_merge import group_output_rows, merge_wide_frames
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.runtime.scenario.time_routes import asocc_time_route_from_projection_subfolder


def prepare_allocation_frame(
    *,
    output_spec: OutputSpec,
    frames: list[pd.DataFrame],
    filters: dict[str, list[str] | None],
    aggreg_indices: bool,
    persisted_years: list[int],
) -> pd.DataFrame:
    """Prepare one allocation output frame before persistence."""
    method_columns = ("l1_l2_method", "l2_method", "l1_method")
    persisted_method_columns = persisted_method_columns_for_output_spec(output_spec)
    expected_method_values = {
        "l1_l2_method": output_spec.l1_l2_method,
        "l2_method": output_spec.l2_method,
        "l1_method": output_spec.l1_method,
    }

    def _normalize_frame_index(frame: pd.DataFrame) -> pd.DataFrame:
        if isinstance(frame.index, pd.RangeIndex):
            return frame
        if isinstance(frame.index, pd.MultiIndex):
            idx_names_raw = list(frame.index.names)
        else:
            idx_names_raw = [frame.index.name]
        if any(name is None for name in idx_names_raw):
            return frame
        idx_names = tuple(str(name) for name in idx_names_raw)
        expected_prefixed = (*method_columns, *identifier_columns)
        if idx_names != expected_prefixed:
            return frame
        normalized = frame.droplevel(list(method_columns))
        if len(identifier_columns) == 1:
            normalized.index.name = identifier_columns[0]
        else:
            normalized.index = normalized.index.set_names(list(identifier_columns))
        return normalized

    year_columns = tuple(str(int(y)) for y in persisted_years)
    identifier_columns = tuple(
        column for column in output_spec.identifier_columns if column not in method_columns
    )
    output_cols = {
        *method_columns,
        *identifier_columns,
    }
    scoped_filters: dict[str, list[str] | None] = {}
    for key, values in filters.items():
        if key not in {"r_p", "s_p", "r_c", "r_f"}:
            continue
        if key not in output_cols:
            continue
        scoped_filters[key] = values
    df = merge_wide_frames(
        frames=[_normalize_frame_index(frame) for frame in frames],
        identifier_columns=identifier_columns,
        year_columns=year_columns,
        where=f"output {output_spec.file_name}",
    )
    present_year_columns = [col for col in year_columns if col in df.columns]
    if present_year_columns:
        year_values = df.loc[:, present_year_columns].to_numpy(dtype=np.float64, copy=False)
        keep_rows = ~np.isnan(year_values).all(axis=1)
        if not bool(keep_rows.all()):
            df = df.loc[keep_rows].reset_index(drop=True)
    if aggreg_indices:
        df = group_output_rows(df, filters=scoped_filters, year_columns=year_columns)
    if present_year_columns or not df.empty:
        df[ASOCC_SSP_SCENARIO_COLUMN] = output_spec.route.ssp_scenario
        df[ASOCC_TIME_ROUTE_PUBLIC_COLUMN] = asocc_time_route_from_projection_subfolder(
            output_spec.route.projection_subfolder
        )
    method_block_columns = persisted_method_columns if present_year_columns or not df.empty else ()
    if method_block_columns:
        for column in reversed(method_block_columns):
            df.insert(0, column, expected_method_values[column])
    ordered_columns = [
        *method_block_columns,
        ASOCC_SSP_SCENARIO_COLUMN,
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
        *identifier_columns,
        *present_year_columns,
    ]
    return df.loc[:, [column for column in ordered_columns if column in df.columns]]

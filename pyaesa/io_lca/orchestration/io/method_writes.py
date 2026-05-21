"""Year level write ownership for IO-LCA method execution."""

from pathlib import Path

import pandas as pd

from pyaesa.asocc.orchestration.write.tables.wide_table_io import upsert_wide_table
from pyaesa.asocc.runtime.output.contracts import IdentifierSchema

from ...data.paths import (
    IOLCAPaths,
    main_results_path,
    origin_columns_defs_path,
    origin_ratio_results_path,
    origin_results_path,
    stage_columns_defs_path,
    stage_results_path,
)
from ...data.column_definitions import (
    write_origin_columns_defs,
    write_stage_columns_defs,
)
from ...data.writers import (
    long_to_year_wide,
    merge_with_existing,
    read_table,
    write_table,
)
from pyaesa.io_lca.orchestration.io.method_support import (
    main_key_columns,
    main_result_columns,
    origin_id_columns,
    stage_public_columns,
    stage_key_columns,
    to_origin_ratio_wide,
    validate_upstream_origin_matches_main,
)


def _normalize_blank_identifier_values(
    *,
    frame: pd.DataFrame,
    id_columns: list[str],
) -> pd.DataFrame:
    """Normalize blank identifier cells to NA for stable CSV round trips."""
    out = frame.copy()
    for col in id_columns:
        if col not in out.columns:
            continue
        series = out[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            normalized = series.astype(str).str.strip()
            out[col] = series
            blank_mask = normalized.eq("")
            if bool(blank_mask.any()):
                out.loc[blank_mask, col] = None
        else:
            out[col] = series
    return out


def _canonicalize_fy_origin_sector(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize F_Y origin sector key to avoid NaN/blank PK drift."""
    out = frame.copy()
    origin_r = out["origin_r_p"].astype(str).str.strip()
    origin_s = out["origin_s_p"]
    origin_s_text = origin_s.astype(str).str.strip()
    mask = origin_r.eq("F_Y") & (origin_s.isna() | origin_s_text.eq(""))
    if bool(mask.any()):
        out.loc[mask, "origin_s_p"] = "F_Y"
    return out


def _canonicalize_year_wide_columns(
    *,
    frame: pd.DataFrame,
    id_columns: list[str],
) -> pd.DataFrame:
    """Project one wide year table onto canonical id-plus-year column order."""
    year_columns = sorted(
        [column for column in frame.columns if column not in set(id_columns)],
        key=lambda column: int(str(column)),
    )
    return frame.loc[:, [*id_columns, *year_columns]].reset_index(drop=True)


def write_main_year(
    *,
    year_main_rows: pd.DataFrame,
    paths: IOLCAPaths,
    source: str,
    lcia_method: str,
    extension: str,
    output_format: str,
    effective_selector_axes: tuple[str, ...],
    written_main: list[Path],
) -> pd.DataFrame | None:
    """Write one year of main results."""
    if year_main_rows.empty:
        return None
    main_path = main_results_path(
        paths=paths,
        source=source,
        lcia_method=lcia_method,
        extension=extension,
    )
    merged_main = merge_with_existing(
        path=main_path,
        fresh=year_main_rows,
        key_columns=main_key_columns(effective_selector_axes),
    )
    merged_main = merged_main.loc[:, main_result_columns(effective_selector_axes)].reset_index(
        drop=True
    )
    write_table(path=main_path, frame=merged_main, output_format=output_format)
    if main_path not in written_main:
        written_main.append(main_path)
    return merged_main


def write_origin_year(
    *,
    year_origin_rows: pd.DataFrame,
    main_for_check: pd.DataFrame,
    lcia_method: str,
    paths: IOLCAPaths,
    source: str,
    extension: str,
    output_format: str,
    effective_selector_axes: tuple[str, ...],
    written_origin: list[Path],
) -> None:
    """Write one year of upstream origin outputs."""
    if year_origin_rows.empty:
        return
    validate_upstream_origin_matches_main(
        main_frame=main_for_check,
        origin_frame=year_origin_rows,
        selector_axes=effective_selector_axes,
        lcia_method=lcia_method,
    )
    origin_path = origin_results_path(
        paths=paths,
        source=source,
        lcia_method=lcia_method,
        extension=extension,
    )
    schema = IdentifierSchema(columns=tuple(origin_id_columns(effective_selector_axes)))
    year_origin_wide = long_to_year_wide(
        frame=year_origin_rows,
        id_columns=origin_id_columns(effective_selector_axes),
        value_column="lca_value",
        year_column="year",
    )
    year_origin_wide = _canonicalize_fy_origin_sector(frame=year_origin_wide)
    origin_ids = origin_id_columns(effective_selector_axes)
    year_origin_wide = _normalize_blank_identifier_values(
        frame=year_origin_wide,
        id_columns=origin_ids,
    )
    year_origin_wide = _canonicalize_year_wide_columns(
        frame=year_origin_wide,
        id_columns=origin_ids,
    )
    upsert_wide_table(
        path=origin_path,
        frame=year_origin_wide,
        schema=schema,
        refresh=False,
        output_format=output_format,
    )
    merged_origin = _canonicalize_year_wide_columns(
        frame=read_table(origin_path),
        id_columns=origin_ids,
    )
    write_table(path=origin_path, frame=merged_origin, output_format=output_format)
    origin_ratio_path = origin_ratio_results_path(
        paths=paths,
        source=source,
        lcia_method=lcia_method,
        extension=extension,
    )
    ratio_wide = to_origin_ratio_wide(
        frame=merged_origin,
        selector_axes=effective_selector_axes,
    )
    ratio_wide = _canonicalize_year_wide_columns(
        frame=ratio_wide,
        id_columns=origin_ids,
    )
    write_table(
        path=origin_ratio_path,
        frame=ratio_wide,
        output_format=output_format,
    )
    write_origin_columns_defs(
        path=origin_columns_defs_path(
            paths=paths,
            source=source,
        ),
        columns=[str(column) for column in merged_origin.columns],
    )
    if origin_path not in written_origin:
        written_origin.append(origin_path)
    if origin_ratio_path not in written_origin:
        written_origin.append(origin_ratio_path)


def write_stage_year(
    *,
    year: int,
    year_stage_rows: pd.DataFrame,
    paths: IOLCAPaths,
    source: str,
    lcia_method: str,
    extension: str,
    output_format: str,
    effective_selector_axes: tuple[str, ...],
    written_stage: list[Path],
) -> None:
    """Write one year of upstream stage outputs."""
    if year_stage_rows.empty:
        return
    out_path = stage_results_path(
        paths=paths,
        source=source,
        lcia_method=lcia_method,
        year=year,
        extension=extension,
    )
    merged_stage = merge_with_existing(
        path=out_path,
        fresh=year_stage_rows,
        key_columns=stage_key_columns(effective_selector_axes),
    )
    merged_stage = merged_stage.loc[:, stage_public_columns(effective_selector_axes)].reset_index(
        drop=True
    )
    write_table(path=out_path, frame=merged_stage, output_format=output_format)
    write_stage_columns_defs(
        path=stage_columns_defs_path(
            paths=paths,
            source=source,
        ),
        columns=[str(column) for column in merged_stage.columns],
    )
    if out_path not in written_stage:
        written_stage.append(out_path)

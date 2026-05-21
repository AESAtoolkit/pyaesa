"""Deterministic external LCA loading."""

from pathlib import Path
from typing import Any

import pandas as pd

from pyaesa.shared.runtime.scenario.columns import EXT_LCA_SSP_SCENARIO_COLUMN

from pyaesa.external_inputs.lca.io import (
    _MANDATORY_COLUMNS,
    _ordered_external_lca_columns,
    finalize_external_lca_loaded_rows,
    matching_external_lca_specs,
    parse_external_lca_filename,
    resolve_external_lca_year_assignments,
    validate_external_lca_contract,
    validate_external_lca_extra_columns,
    validate_external_lca_required_columns,
    validate_external_lca_selector_columns,
)
from pyaesa.external_inputs.shared.tabular import (
    arrow_table_to_pandas,
    arrow_wide_to_long,
    read_projected_table,
    tabular_columns,
    year_column_names,
)
from pyaesa.external_inputs.lca.paths import external_lca_deterministic_dir
from .scenario_metadata import with_lca_ssp_start_year


def load_external_lca_deterministic_rows(
    *,
    proj_base: Path,
    version_name: str,
    lcia_method: str,
    years: list[int],
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
    base_allocate_args: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame | None, tuple[Path, ...]]:
    """Load deterministic external LCA rows and the exact file paths used."""
    specs = matching_external_lca_specs(
        directory=external_lca_deterministic_dir(project_base=proj_base),
        version_name=version_name,
        lcia_method=lcia_method,
    )
    if not specs:
        return None, tuple()
    year_map = resolve_external_lca_year_assignments(
        specs=specs,
        version_name=version_name,
        lcia_method=lcia_method,
        years=years,
        ssp_scenario_options_by_year=ssp_scenario_options_by_year,
    )
    frames: list[pd.DataFrame] = []
    selected_paths: list[Path] = []
    for spec in specs:
        requested_years = year_map.get(spec.path, [])
        if not requested_years:
            continue
        frame = _load_deterministic_lca_file(
            path=spec.path,
            requested_years=requested_years,
            lcia_method=lcia_method,
            base_allocate_args=base_allocate_args,
        )
        frame[EXT_LCA_SSP_SCENARIO_COLUMN] = spec.scenario
        frames.append(frame)
        selected_paths.append(spec.path)
    if not frames:
        return None, tuple()
    out = with_lca_ssp_start_year(pd.concat(frames, ignore_index=True))
    return (
        finalize_external_lca_loaded_rows(frame=out).reset_index(drop=True),
        tuple(selected_paths),
    )


def load_external_lca_deterministic_rows_from_paths(
    *,
    paths: tuple[Path, ...],
    lcia_method: str,
    years: list[int],
    base_allocate_args: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load deterministic external LCA rows from already selected source files."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        spec = parse_external_lca_filename(path=path)
        frame = _load_deterministic_lca_file(
            path=spec.path,
            requested_years=years,
            lcia_method=lcia_method,
            base_allocate_args=base_allocate_args,
        )
        frame[EXT_LCA_SSP_SCENARIO_COLUMN] = spec.scenario
        frames.append(frame)
    out = with_lca_ssp_start_year(pd.concat(frames, ignore_index=True))
    return finalize_external_lca_loaded_rows(frame=out).reset_index(drop=True)


def _load_deterministic_lca_file(
    *,
    path: Path,
    requested_years: list[int],
    lcia_method: str,
    base_allocate_args: dict[str, Any] | None,
) -> pd.DataFrame:
    columns = tabular_columns(path)
    if {"year", "value"}.issubset(columns):
        raise ValueError(
            f"Deterministic external LCA file '{path}' must use wide year columns, not long "
            "'year'/'value' rows."
        )
    validate_external_lca_required_columns(
        columns=columns,
        path=path,
        required_columns=_MANDATORY_COLUMNS,
        file_label="External LCA file",
    )
    requested = [int(year) for year in requested_years]
    year_columns = [str(year) for year in requested]
    all_year_columns = set(year_column_names(columns))
    identity_columns = [column for column in columns if column not in all_year_columns]
    table = read_projected_table(path, columns=[*identity_columns, *year_columns])
    metadata = arrow_table_to_pandas(table.select(identity_columns))
    validate_external_lca_contract(frame=metadata, path=path, lcia_method=lcia_method)
    metadata = validate_external_lca_extra_columns(frame=metadata, path=path)
    validate_external_lca_selector_columns(
        frame=metadata,
        path=path,
        base_allocate_args=base_allocate_args,
    )
    long_table = arrow_wide_to_long(
        table=table.select([*metadata.columns, *year_columns]),
        identity_columns=list(metadata.columns),
        requested_years=requested,
    )
    long_frame = arrow_table_to_pandas(long_table)
    long_frame["impact"] = long_frame["impact"].astype(str)
    long_frame["impact_unit"] = long_frame["impact_unit"].astype(str)
    return long_frame.loc[:, _ordered_external_lca_columns(frame=long_frame)].reset_index(drop=True)

"""Deterministic external aSoCC loading."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pyaesa.asocc.runtime.paths.external import (
    get_asocc_external_method_level_dir,
)

from pyaesa.external_inputs.asocc.schema.contracts import ExternalMethodSelection
from pyaesa.external_inputs.asocc.schema.file_specs import (
    ExternalAsoCCFileSpec,
    candidate_files,
    external_asocc_expected_stems,
    resolve_external_asocc_year_assignments,
    validate_impact_contract,
    validate_lcia_inventory,
)
from pyaesa.external_inputs.asocc.schema.row_schema import normalize_external_asocc_long_rows
from pyaesa.external_inputs.shared.tabular import (
    arrow_table_to_pandas,
    arrow_wide_to_long,
    read_projected_table,
    tabular_columns,
    year_column_names,
)


@dataclass(frozen=True)
class ExternalDeterministicFileSelection:
    """One deterministic external aSoCC file slice used by a request."""

    path: Path
    lcia_method: str | None
    requested_years: tuple[int, ...]
    ssp_scenario: str | None


@dataclass(frozen=True)
class ExternalDeterministicRowsSource:
    """Resolved deterministic external aSoCC file source."""

    selection: ExternalMethodSelection
    file_selections: tuple[ExternalDeterministicFileSelection, ...]


@dataclass(frozen=True)
class ExternalDeterministicRowsResult:
    """Loaded deterministic external aSoCC rows and provenance."""

    rows: pd.DataFrame
    source: ExternalDeterministicRowsSource


def _melt_wide(
    *,
    path: Path,
    years: list[int],
    selection: ExternalMethodSelection,
    lcia_method: str | None,
    ssp_scenario: str | None,
    include_asocc_ssp_scenario_column: bool = True,
) -> pd.DataFrame:
    columns = tabular_columns(path)
    requested = [int(year) for year in years]
    year_columns = [str(year) for year in requested]
    all_year_columns = set(year_column_names(columns))
    identity_columns = [column for column in columns if column not in all_year_columns]
    table = read_projected_table(path, columns=[*identity_columns, *year_columns])
    metadata = arrow_table_to_pandas(table.select(identity_columns))
    validate_impact_contract(frame=metadata, path=path, lcia_method=lcia_method)
    long_table = arrow_wide_to_long(
        table=table,
        identity_columns=identity_columns,
        requested_years=requested,
    )
    return normalize_external_asocc_long_rows(
        frame=arrow_table_to_pandas(long_table),
        selection=selection,
        lcia_method=lcia_method,
        ssp_scenario=ssp_scenario,
        include_asocc_ssp_scenario_column=include_asocc_ssp_scenario_column,
    )


def _partition_specs(
    specs: tuple[ExternalAsoCCFileSpec, ...],
) -> dict[str | None, tuple[ExternalAsoCCFileSpec, ...]]:
    partitions: dict[str | None, list[ExternalAsoCCFileSpec]] = {}
    for spec in specs:
        partitions.setdefault(spec.lcia_method, []).append(spec)
    return {
        key: tuple(sorted(values, key=lambda item: (str(item.scenario), item.path.name)))
        for key, values in partitions.items()
    }


def _resolve_partition_years(
    *,
    specs: tuple[ExternalAsoCCFileSpec, ...],
    selection: ExternalMethodSelection,
    years: list[int],
    lcia_methods: list[str] | None,
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
) -> dict[Path, list[int]]:
    return resolve_external_asocc_year_assignments(
        specs=specs,
        selection=selection,
        years=years,
        lcia_methods=lcia_methods,
        ssp_scenario_options_by_year=ssp_scenario_options_by_year,
    )


def load_external_deterministic_rows(
    *,
    proj_base: Path,
    selection: ExternalMethodSelection,
    years: list[int],
    lcia_methods: list[str] | None,
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
) -> pd.DataFrame | None:
    """Load deterministic external aSoCC rows for one normalized selection."""
    result = load_external_deterministic_rows_with_source(
        proj_base=proj_base,
        selection=selection,
        years=years,
        lcia_methods=lcia_methods,
        ssp_scenario_options_by_year=ssp_scenario_options_by_year,
    )
    return None if result is None else result.rows


def load_external_deterministic_rows_with_source(
    *,
    proj_base: Path,
    selection: ExternalMethodSelection,
    years: list[int],
    lcia_methods: list[str] | None,
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
) -> ExternalDeterministicRowsResult | None:
    """Load deterministic external aSoCC rows and the exact files used."""
    specs = candidate_files(
        proj_base=proj_base,
        selection=selection,
        lcia_methods=lcia_methods,
    )
    if not specs:
        return None
    validate_lcia_inventory(specs=specs, selection=selection, lcia_methods=lcia_methods)
    frames: list[pd.DataFrame] = []
    file_selections: list[ExternalDeterministicFileSelection] = []
    for partition in _partition_specs(specs).values():
        year_map = _resolve_partition_years(
            specs=partition,
            selection=selection,
            years=years,
            lcia_methods=lcia_methods,
            ssp_scenario_options_by_year=ssp_scenario_options_by_year,
        )
        for spec in partition:
            requested_years = year_map.get(spec.path, [])
            if not requested_years:
                continue
            file_selections.append(
                ExternalDeterministicFileSelection(
                    path=spec.path,
                    lcia_method=spec.lcia_method,
                    requested_years=tuple(int(year) for year in requested_years),
                    ssp_scenario=spec.scenario,
                )
            )
            frames.append(
                _melt_wide(
                    path=spec.path,
                    years=requested_years,
                    selection=selection,
                    lcia_method=spec.lcia_method,
                    ssp_scenario=spec.scenario,
                    include_asocc_ssp_scenario_column=spec.scenario is not None,
                )
            )
    if not frames:
        return None
    return ExternalDeterministicRowsResult(
        rows=pd.concat(frames, ignore_index=True),
        source=ExternalDeterministicRowsSource(
            selection=selection,
            file_selections=tuple(file_selections),
        ),
    )


def expected_external_deterministic_stems(
    *,
    selection: ExternalMethodSelection,
    lcia_methods: list[str] | None,
    years: list[int],
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
) -> list[str]:
    """Return representative expected deterministic filename stems for one selection."""
    return external_asocc_expected_stems(
        fu_code=selection.fu_code,
        file_method_token=selection.file_method_token,
        l1_method=selection.l1_method,
        lcia_methods=lcia_methods,
        years=years,
        ssp_scenario_options_by_year=ssp_scenario_options_by_year,
    )


def describe_expected_external_deterministic_stems(
    *,
    proj_base: Path,
    selection: ExternalMethodSelection,
    stems: list[str],
) -> str:
    """Return a user-facing description of acceptable deterministic external filenames."""
    normalized = [f"{str(stem).strip()}.csv" for stem in stems if str(stem).strip()]
    if not normalized:
        return "no valid deterministic filenames could be derived"
    directory = get_asocc_external_method_level_dir(
        proj_base=proj_base,
        storage_mode="deterministic",
        level=str(selection.level),
    )
    return f"place one of these files in '{directory}': {normalized}"

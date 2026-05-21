"""Monte Carlo external aSoCC loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.shared.runtime.io.file_identity import file_identity_payload

from pyaesa.external_inputs.asocc.schema.contracts import ExternalMethodSelection
from pyaesa.external_inputs.asocc.schema.file_specs import (
    candidate_files,
    validate_lcia_inventory,
)
from pyaesa.external_inputs.shared.compact_matrix import is_compact_run_matrix_dir

EXTERNAL_MONTE_CARLO_MATRIX_CHUNK_ROWS = 500_000
EXTERNAL_MONTE_CARLO_CSV_BLOCK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ExternalMonteCarloFileSelection:
    """One external aSoCC Monte Carlo file slice needed by the request."""

    path: Path
    lcia_method: str | None
    requested_years: tuple[int, ...]
    ssp_scenario_options_by_year: dict[int, tuple[str | None, ...]] | None


@dataclass(frozen=True)
class ExternalMonteCarloRowsSource:
    """Resolved external aSoCC Monte Carlo file source."""

    selection: ExternalMethodSelection
    file_selections: tuple[ExternalMonteCarloFileSelection, ...]
    run_indices: tuple[int, ...]

    @property
    def available_runs(self) -> int:
        """Return the contiguous run count available from every selected file."""
        return len(self.run_indices)


@dataclass(frozen=True)
class ExternalMonteCarloRunMatrix:
    """External Monte Carlo values aligned by run_index and public row identity."""

    template: pd.DataFrame
    values: np.ndarray


@dataclass(frozen=True)
class MaterializedExternalMonteCarloRowsSource:
    """External Monte Carlo source with its compact run matrix loaded once."""

    metadata: ExternalMonteCarloRowsSource
    run_matrix: ExternalMonteCarloRunMatrix

    @property
    def selection(self) -> ExternalMethodSelection:
        """Return the selected external method identity."""
        return self.metadata.selection

    @property
    def file_selections(self) -> tuple[ExternalMonteCarloFileSelection, ...]:
        """Return external files contributing to this source."""
        return self.metadata.file_selections

    @property
    def run_indices(self) -> tuple[int, ...]:
        """Return the run_index inventory represented by matrix rows."""
        return self.metadata.run_indices

    @property
    def available_runs(self) -> int:
        """Return the number of empirical external run rows."""
        return self.metadata.available_runs


def resolve_external_monte_carlo_source(
    *,
    proj_base: Path,
    selection: ExternalMethodSelection,
    years: list[int],
    lcia_methods: list[str] | None,
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
) -> ExternalMonteCarloRowsSource | None:
    """Resolve external Monte Carlo file metadata and shared run inventory."""
    specs = candidate_files(
        proj_base=proj_base,
        selection=selection,
        lcia_methods=lcia_methods,
        storage_mode="monte_carlo",
    )
    if not specs:
        return None
    compact_specs = tuple(
        spec
        for spec in specs
        if is_compact_run_matrix_dir(spec.path, run_file_name="asocc_runs.csv")
    )
    if compact_specs:
        specs = compact_specs
    validate_lcia_inventory(specs=specs, selection=selection, lcia_methods=lcia_methods)
    file_selections: list[ExternalMonteCarloFileSelection] = []
    requested_year_set = {int(year) for year in years}
    if not requested_year_set:
        return None
    normalized_scenarios = _normalize_scenario_options_by_year(
        ssp_scenario_options_by_year=ssp_scenario_options_by_year
    )
    for spec in specs:
        file_selections.append(
            ExternalMonteCarloFileSelection(
                path=spec.path,
                lcia_method=spec.lcia_method,
                requested_years=tuple(sorted(requested_year_set)),
                ssp_scenario_options_by_year=normalized_scenarios,
            )
        )
    return ExternalMonteCarloRowsSource(
        selection=selection,
        file_selections=tuple(file_selections),
        run_indices=(),
    )


def materialize_external_monte_carlo_source(
    *,
    source: ExternalMonteCarloRowsSource,
) -> MaterializedExternalMonteCarloRowsSource:
    """Materialize one resolved external Monte Carlo source."""
    from pyaesa.external_inputs.asocc.monte_carlo.matrix import _materialize_source_matrix

    return _materialize_source_matrix(source=source)


def external_monte_carlo_source_for_years(
    *,
    source: MaterializedExternalMonteCarloRowsSource,
    years: tuple[int, ...],
) -> MaterializedExternalMonteCarloRowsSource | None:
    """Return a materialized source restricted to selected output years."""
    selected_years = {int(year) for year in years}
    file_selections = []
    for file_selection in source.file_selections:
        requested_years = tuple(
            int(year) for year in file_selection.requested_years if int(year) in selected_years
        )
        if requested_years:
            file_selections.append(
                ExternalMonteCarloFileSelection(
                    path=file_selection.path,
                    lcia_method=file_selection.lcia_method,
                    requested_years=requested_years,
                    ssp_scenario_options_by_year=file_selection.ssp_scenario_options_by_year,
                )
            )
    if not file_selections:
        return None
    year_values = cast(
        pd.Series,
        pd.to_numeric(
            pd.Series(source.run_matrix.template.loc[:, "year"], copy=False),
            errors="raise",
        ),
    ).astype(int)
    keep = year_values.isin(selected_years).to_numpy(dtype=bool)
    matrix = ExternalMonteCarloRunMatrix(
        template=source.run_matrix.template.loc[keep, :].reset_index(drop=True),
        values=source.run_matrix.values[:, keep],
    )
    return MaterializedExternalMonteCarloRowsSource(
        metadata=ExternalMonteCarloRowsSource(
            selection=source.selection,
            file_selections=tuple(file_selections),
            run_indices=source.run_indices,
        ),
        run_matrix=matrix,
    )


def external_monte_carlo_manifest_payload(
    *,
    source: MaterializedExternalMonteCarloRowsSource,
) -> dict[str, Any]:
    """Return the manifest identity payload for one materialized source."""
    return {
        "storage_mode": "monte_carlo",
        "selection": source.selection.asocc_method_label,
        "run_indices": list(source.run_indices),
        "files": [
            _file_selection_manifest_payload(file_selection)
            for file_selection in source.file_selections
        ],
    }


def _file_selection_manifest_payload(
    file_selection: ExternalMonteCarloFileSelection,
) -> dict[str, Any]:
    if file_selection.path.is_dir():
        return {
            "path": str(file_selection.path),
            "layout": "compact_run_matrix",
            "public_row_identity": file_identity_payload(
                path=file_selection.path / "public_row_identity.csv"
            ),
            "asocc_runs": file_identity_payload(path=file_selection.path / "asocc_runs.csv"),
            "lcia_method": file_selection.lcia_method,
            "requested_years": list(file_selection.requested_years),
        }
    return {
        **file_identity_payload(path=file_selection.path),
        "layout": "long_rows",
        "lcia_method": file_selection.lcia_method,
        "requested_years": list(file_selection.requested_years),
    }


def _normalize_scenario_options_by_year(
    *,
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
) -> dict[int, tuple[str | None, ...]] | None:
    if not ssp_scenario_options_by_year:
        return None
    return {
        int(year): tuple(None if value is None else str(value).strip() for value in values)
        for year, values in ssp_scenario_options_by_year.items()
    }

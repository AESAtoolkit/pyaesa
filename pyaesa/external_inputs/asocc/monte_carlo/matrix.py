"""Compact matrix materialization for external aSoCC Monte Carlo sources."""

from dataclasses import replace
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pyaesa.external_inputs.asocc.schema.file_specs import (
    read_external_asocc_table,
    validate_impact_contract,
)
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.external_inputs.asocc.monte_carlo.files import (
    EXTERNAL_MONTE_CARLO_MATRIX_CHUNK_ROWS,
    ExternalMonteCarloRunMatrix,
    ExternalMonteCarloFileSelection,
    ExternalMonteCarloRowsSource,
    MaterializedExternalMonteCarloRowsSource,
)
from pyaesa.external_inputs.asocc.monte_carlo.matrix_csv import (
    csv_matrix,
)
from pyaesa.external_inputs.shared.compact_matrix import (
    is_compact_run_matrix_dir,
    load_compact_run_matrix,
)
from pyaesa.external_inputs.shared.tabular import none_for_missing_series
from pyaesa.external_inputs.shared.matrix_identity import (
    InventoryTemplate,
    assign_values,
    contiguous_inventory,
    positions_for_frame,
    require_complete_matrix,
    shared_inventory_template,
    template_lookup,
)
from pyaesa.external_inputs.asocc.monte_carlo.matrix_inventory import ImpactInventory
from pyaesa.external_inputs.asocc.schema.row_schema import normalize_external_asocc_render_rows


def _materialize_source_matrix(
    *,
    source: ExternalMonteCarloRowsSource,
) -> MaterializedExternalMonteCarloRowsSource:
    """Load one external source into a run by public row matrix."""
    compact_selections = tuple(
        file
        for file in source.file_selections
        if is_compact_run_matrix_dir(file.path, run_file_name="asocc_runs.csv")
    )
    if compact_selections:
        source = replace(source, file_selections=compact_selections)
        return _materialize_compact_source(source=source)
    if all(file.path.suffix.lower() == ".csv" for file in source.file_selections):
        inventory_template, values = csv_matrix(source=source)
        source = replace(source, run_indices=inventory_template.run_indices)
        return _materialized(source=source, template=inventory_template.template, values=values)
    return _materialize_frame_source(source=source)


def _materialize_compact_source(
    *,
    source: ExternalMonteCarloRowsSource,
) -> MaterializedExternalMonteCarloRowsSource:
    inventories: dict[str, tuple[int, ...]] = {}
    templates: list[pd.DataFrame] = []
    matrices: list[np.ndarray] = []
    for selection in source.file_selections:
        compact = load_compact_run_matrix(
            directory=selection.path,
            run_file_name="asocc_runs.csv",
            context=f"External aSoCC compact Monte Carlo source '{selection.path.name}'",
        )
        compact_identity, compact_values = _compact_selection_identity_and_values(
            source=source,
            identity=compact.identity,
            values=compact.values,
            file_selection=selection,
        )
        template = compact_identity.drop(columns=["public_row_id"]).copy()
        template.insert(0, "run_index", 0)
        template["value"] = compact_values[0, :]
        inventories[selection.path.name] = tuple(int(value) for value in compact.run_indices)
        templates.append(template)
        matrices.append(compact_values)
    shared = shared_inventory_template(inventories=inventories, templates=templates)
    template, _lookup = template_lookup(template=shared.template)
    values = np.concatenate(matrices, axis=1)
    source = replace(source, run_indices=shared.run_indices)
    return _materialized(source=source, template=template, values=values)


def _compact_selection_identity_and_values(
    *,
    source: ExternalMonteCarloRowsSource,
    identity: pd.DataFrame,
    values: np.ndarray,
    file_selection: ExternalMonteCarloFileSelection,
) -> tuple[pd.DataFrame, np.ndarray]:
    identity = identity.copy()
    identity[ASOCC_SSP_SCENARIO_COLUMN] = none_for_missing_series(
        pd.Series(identity.loc[:, ASOCC_SSP_SCENARIO_COLUMN], copy=False)
    )
    years = pd.Series(
        pd.to_numeric(pd.Series(identity.loc[:, "year"], copy=False), errors="raise"),
        copy=False,
    ).astype(int)
    selected = years.isin([int(year) for year in file_selection.requested_years])
    filtered = _filter_frame_scenarios(
        frame=identity.loc[selected, :].copy(),
        options_by_year=file_selection.ssp_scenario_options_by_year,
    )
    if filtered.empty:
        raise ValueError(
            f"External aSoCC compact Monte Carlo source '{file_selection.path}' has no rows "
            f"for requested years {list(file_selection.requested_years)}."
        )
    positions = filtered["public_row_id"].to_numpy(dtype=np.int64)
    selected_values = values[:, positions]
    public_rows = filtered.drop(columns=["public_row_id"]).reset_index(drop=True)
    validate_impact_contract(
        frame=public_rows,
        path=file_selection.path / "public_row_identity.csv",
        lcia_method=file_selection.lcia_method,
    )
    run_zero = public_rows.copy()
    run_zero.insert(0, "run_index", 0)
    run_zero["value"] = selected_values[0, :]
    normalized = normalize_external_asocc_render_rows(
        frame=run_zero,
        selection=source.selection,
        lcia_method=file_selection.lcia_method,
        ssp_scenario=None,
        requested_years=list(file_selection.requested_years),
        include_asocc_ssp_scenario_column=True,
    )
    out = normalized.drop(columns=["run_index", "value"]).reset_index(drop=True)
    out.insert(0, "public_row_id", np.arange(len(out), dtype=np.int64))
    return out, selected_values


def _materialize_frame_source(
    *,
    source: ExternalMonteCarloRowsSource,
) -> MaterializedExternalMonteCarloRowsSource:
    inventory_template = _frame_inventory_template(source=source)
    source = replace(source, run_indices=inventory_template.run_indices)
    template, lookup = template_lookup(template=inventory_template.template)
    values = np.full((len(source.run_indices), len(template)), np.nan, dtype=np.float64)
    filled = np.zeros(values.shape, dtype=bool)
    for frame in _iter_frame_normalized_chunks(source=source):
        assign_values(
            values=values,
            filled=filled,
            row_positions=frame["run_index"].to_numpy(dtype=np.int64),
            column_positions=positions_for_frame(frame=frame, lookup=lookup),
            source_values=frame["value"].to_numpy(dtype=np.float64),
        )
    require_complete_matrix(filled=filled)
    return _materialized(source=source, template=template, values=values)


def _frame_inventory_template(*, source: ExternalMonteCarloRowsSource) -> InventoryTemplate:
    inventories: dict[str, tuple[int, ...]] = {}
    templates: list[pd.DataFrame] = []
    for selection in source.file_selections:
        run_values: set[int] = set()
        template_parts: list[pd.DataFrame] = []
        impact = ImpactInventory(path=selection.path, lcia_method=selection.lcia_method)
        for frame in _iter_selected_frame_chunks(file_selection=selection, run_indices=None):
            impact.observe_frame(frame)
            frame = _filter_frame_scenarios(
                frame=frame,
                options_by_year=selection.ssp_scenario_options_by_year,
            )
            if frame.empty:
                continue
            normalized = _normalize_frame_rows(source=source, file_selection=selection, frame=frame)
            run_index = normalized["run_index"].to_numpy(dtype=np.int64)
            run_values.update(int(value) for value in np.unique(run_index))
            run_zero = normalized.loc[run_index == 0]
            if not run_zero.empty:
                template_parts.append(run_zero)
        impact.validate()
        inventories[selection.path.name] = contiguous_inventory(
            values=np.asarray(sorted(run_values), dtype=np.int64),
            context=(
                f"External aSoCC Monte Carlo file '{selection.path.name}' after "
                "requested year and SSP scenario filtering"
            ),
        )
        templates.append(pd.concat(template_parts, ignore_index=True))
    return shared_inventory_template(inventories=inventories, templates=templates)


def _iter_frame_normalized_chunks(
    *,
    source: ExternalMonteCarloRowsSource,
) -> Iterator[pd.DataFrame]:
    selected_runs = set(source.run_indices)
    for selection in source.file_selections:
        for frame in _iter_selected_frame_chunks(
            file_selection=selection,
            run_indices=selected_runs,
        ):
            normalized = _normalize_frame_rows(source=source, file_selection=selection, frame=frame)
            yield _filter_frame_scenarios(
                frame=normalized,
                options_by_year=selection.ssp_scenario_options_by_year,
            )


def _iter_selected_frame_chunks(
    *,
    file_selection: ExternalMonteCarloFileSelection,
    run_indices: set[int] | None,
) -> Iterator[pd.DataFrame]:
    suffix = file_selection.path.suffix.lower()
    if suffix == ".parquet":
        parquet_file = pq.ParquetFile(file_selection.path)
        for batch in parquet_file.iter_batches(batch_size=EXTERNAL_MONTE_CARLO_MATRIX_CHUNK_ROWS):
            yield _select_frame_rows(
                frame=batch.to_pandas(),
                run_indices=run_indices,
                years=file_selection.requested_years,
            )
        return
    yield _select_frame_rows(
        frame=read_external_asocc_table(file_selection.path),
        run_indices=run_indices,
        years=file_selection.requested_years,
    )


def _normalize_frame_rows(
    *,
    source: ExternalMonteCarloRowsSource,
    file_selection: ExternalMonteCarloFileSelection,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return normalize_external_asocc_render_rows(
        frame=frame,
        selection=source.selection,
        lcia_method=file_selection.lcia_method,
        ssp_scenario=None,
        requested_years=list(file_selection.requested_years),
        include_asocc_ssp_scenario_column=True,
    )


def _select_frame_rows(
    *,
    frame: pd.DataFrame,
    run_indices: set[int] | None,
    years: tuple[int, ...],
) -> pd.DataFrame:
    if "year" not in frame.columns or "run_index" not in frame.columns:
        return frame.copy()
    selected = frame
    if run_indices is not None:
        selected = selected.loc[selected["run_index"].astype(int).isin(tuple(run_indices))]
    selected_years = tuple(int(year) for year in years)
    return selected.loc[selected["year"].astype(int).isin(selected_years)].copy()


def _filter_frame_scenarios(
    *,
    frame: pd.DataFrame,
    options_by_year: dict[int, tuple[str | None, ...]] | None,
) -> pd.DataFrame:
    if frame.empty or not options_by_year:
        return frame
    years = frame["year"].astype(int)
    scenarios = frame[ASOCC_SSP_SCENARIO_COLUMN].astype(object)
    keep = pd.Series(False, index=frame.index)
    for year in sorted(set(years.tolist())):
        allowed = options_by_year.get(int(year))
        year_mask = years.eq(int(year))
        if allowed is None:
            keep = keep | year_mask
        else:
            keep = keep | (year_mask & scenarios.isin(tuple(allowed)))
    return frame.loc[keep].reset_index(drop=True)


def _materialized(
    *,
    source: ExternalMonteCarloRowsSource,
    template: pd.DataFrame,
    values: np.ndarray,
) -> MaterializedExternalMonteCarloRowsSource:
    return MaterializedExternalMonteCarloRowsSource(
        metadata=source,
        run_matrix=ExternalMonteCarloRunMatrix(template=template, values=values),
    )

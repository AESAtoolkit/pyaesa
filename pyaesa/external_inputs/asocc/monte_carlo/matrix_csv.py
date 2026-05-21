"""Arrow CSV materialization for external aSoCC Monte Carlo files."""

from typing import Any, Iterator, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens

from pyaesa.external_inputs.asocc.monte_carlo.files import EXTERNAL_MONTE_CARLO_CSV_BLOCK_BYTES
from pyaesa.external_inputs.asocc.monte_carlo.files import (
    ExternalMonteCarloFileSelection,
    ExternalMonteCarloRowsSource,
)
from pyaesa.external_inputs.shared.matrix_identity import (
    IdentityLookup,
    InventoryTemplate,
    assign_values,
    combine,
    contiguous_inventory,
    duplicate_identity_error,
    positions_for_arrow,
    require_complete_matrix,
    shared_inventory_template,
    template_lookup,
)
from pyaesa.external_inputs.asocc.monte_carlo.matrix_inventory import (
    ImpactInventory,
    string_has_nonempty,
    unique_nonempty_strings,
)
from pyaesa.external_inputs.asocc.schema.row_schema import (
    external_asocc_render_row_columns,
    expected_external_selector_columns,
    validate_external_asocc_monte_carlo_column_names,
)

_PC = cast(Any, pc)
_ALL_SELECTOR_COLUMNS = ("r_p", "s_p", "r_c", "r_f")


def csv_matrix(*, source: ExternalMonteCarloRowsSource) -> tuple[InventoryTemplate, np.ndarray]:
    """Return run inventory, run zero template, and compact values for CSV files."""
    inventories: dict[str, tuple[int, ...]] = {}
    templates = []
    matrices = []
    for selection in source.file_selections:
        inventory, matrix = _csv_selection_matrix(source=source, selection=selection)
        inventories[selection.path.name] = inventory.run_indices
        templates.append(inventory.template)
        matrices.append(matrix)
    shared = shared_inventory_template(inventories=inventories, templates=templates)
    template, _lookup = template_lookup(template=shared.template)
    return InventoryTemplate(run_indices=shared.run_indices, template=template), np.concatenate(
        matrices,
        axis=1,
    )


def _csv_selection_matrix(
    *,
    source: ExternalMonteCarloRowsSource,
    selection: ExternalMonteCarloFileSelection,
) -> tuple[InventoryTemplate, np.ndarray]:
    run_chunks: list[np.ndarray] = []
    template_tables: list[pa.Table] = []
    impact = ImpactInventory(path=selection.path, lcia_method=selection.lcia_method)
    template_frame: pd.DataFrame | None = None
    lookup: IdentityLookup | None = None
    values: np.ndarray | None = None
    filled: np.ndarray | None = None
    last_run_index: int | None = None

    def build_template() -> tuple[pd.DataFrame, IdentityLookup, np.ndarray, np.ndarray]:
        nonlocal template_frame, lookup, values, filled
        if (
            template_frame is not None
            and lookup is not None
            and values is not None
            and filled is not None
        ):
            return template_frame, lookup, values, filled
        built_template = pa.concat_tables(template_tables).to_pandas()
        _validate_template_unique(template=built_template)
        built_matrix_template, built_lookup = template_lookup(template=built_template)
        template_frame = built_template
        lookup = built_lookup
        values = np.full((1024, len(built_matrix_template)), np.nan, dtype=np.float64)
        filled = np.zeros(values.shape, dtype=bool)
        for run_zero in template_tables:
            _assign_csv_table(
                table=run_zero,
                lookup=lookup,
                values=values,
                filled=filled,
            )
        return built_template, built_lookup, values, filled

    for table in iter_csv_selected_tables(file_selection=selection):
        validate_arrow_columns(table=table, source=source, path=selection.path)
        impact.validate_arrow_schema(table)
        table = filter_arrow_scenarios(
            table=table,
            options_by_year=selection.ssp_scenario_options_by_year,
        )
        if table.num_rows == 0:
            continue
        impact.observe_arrow(table)
        run_index = int64_numpy(pc.cast(table["run_index"], pa.int64()))
        _validate_non_decreasing_runs(
            run_index=run_index,
            last_run_index=last_run_index,
            path=selection.path,
        )
        last_run_index = int(run_index[-1])
        run_chunks.append(np.unique(run_index))
        run_zero_mask = run_index == 0
        if bool(run_zero_mask.any()):
            template_tables.append(
                normalize_arrow_rows(
                    source=source,
                    file_selection=selection,
                    table=table.filter(pa.array(run_zero_mask)),
                )
            )
        if lookup is None and bool((run_index > 0).any()) and template_tables:
            build_template()
        if lookup is not None:
            active = table.filter(pa.array(run_index > 0)) if bool(run_zero_mask.any()) else table
            _template, active_lookup, active_values, active_filled = build_template()
            active_values, active_filled = _ensure_render_capacity(
                values=active_values,
                filled=active_filled,
                stop_index=int(run_index[-1]),
            )
            values = active_values
            filled = active_filled
            _assign_csv_table(
                table=normalize_arrow_rows(
                    source=source,
                    file_selection=selection,
                    table=active,
                ),
                lookup=active_lookup,
                values=active_values,
                filled=active_filled,
            )
    impact.validate()
    inventories = contiguous_inventory(
        values=concatenate_int_chunks(run_chunks),
        context=(
            f"External aSoCC Monte Carlo file '{selection.path.name}' after "
            "requested year and SSP scenario filtering"
        ),
    )
    matrix_template, _matrix_lookup, matrix_values, matrix_filled = build_template()
    trimmed_values = matrix_values[: len(inventories), :]
    trimmed_filled = matrix_filled[: len(inventories), :]
    require_complete_matrix(filled=trimmed_filled)
    return InventoryTemplate(run_indices=inventories, template=matrix_template), trimmed_values


def _assign_csv_table(
    *,
    table: pa.Table,
    lookup: IdentityLookup,
    values: np.ndarray,
    filled: np.ndarray,
) -> None:
    assign_values(
        values=values,
        filled=filled,
        row_positions=int64_numpy(table["run_index"]),
        column_positions=positions_for_arrow(table=table, lookup=lookup),
        source_values=float64_numpy(table["value"]),
    )


def _validate_template_unique(*, template: pd.DataFrame) -> None:
    identity_columns = [
        column for column in template.columns if column not in {"run_index", "value"}
    ]
    if bool(template.duplicated(["run_index", *identity_columns], keep=False).any()):
        raise duplicate_identity_error()


def _ensure_render_capacity(
    *,
    values: np.ndarray | None,
    filled: np.ndarray | None,
    stop_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix_values = cast(np.ndarray, values)
    matrix_filled = cast(np.ndarray, filled)
    if stop_index < matrix_values.shape[0]:
        return matrix_values, matrix_filled
    new_rows = matrix_values.shape[0]
    while stop_index >= new_rows:
        new_rows *= 2
    next_values = np.full((new_rows, matrix_values.shape[1]), np.nan, dtype=np.float64)
    next_filled = np.zeros((new_rows, matrix_filled.shape[1]), dtype=bool)
    next_values[: matrix_values.shape[0], :] = matrix_values
    next_filled[: matrix_filled.shape[0], :] = matrix_filled
    return next_values, next_filled


def _validate_non_decreasing_runs(
    *,
    run_index: np.ndarray,
    last_run_index: int | None,
    path,
) -> None:
    prefix = np.fromiter(
        (value for value in (last_run_index,) if value is not None),
        dtype=np.int64,
    )
    ordered = np.concatenate((prefix, run_index))
    if bool(np.any(ordered[1:] < ordered[:-1])):
        raise _unsorted_run_index_error(path)


def _unsorted_run_index_error(path) -> ValueError:
    return ValueError(
        f"External aSoCC Monte Carlo CSV file '{path}' must be sorted by "
        "nondecreasing run_index. The streaming materializer reads one run "
        "block at a time and does not keep earlier run blocks in memory."
    )


def iter_csv_selected_tables(
    *,
    file_selection: ExternalMonteCarloFileSelection,
) -> Iterator[pa.Table]:
    """Yield requested year and run rows from one CSV as Arrow tables."""
    reader = pacsv.open_csv(
        file_selection.path,
        read_options=pacsv.ReadOptions(block_size=EXTERNAL_MONTE_CARLO_CSV_BLOCK_BYTES),
    )
    years = pa.array(file_selection.requested_years, type=pa.int64())
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            if "year" not in table.column_names or "run_index" not in table.column_names:
                yield table
            else:
                mask = _PC.is_in(pc.cast(table["year"], pa.int64()), value_set=years)
                yield table.filter(mask)
    finally:
        reader.close()


def normalize_arrow_rows(
    *,
    source: ExternalMonteCarloRowsSource,
    file_selection: ExternalMonteCarloFileSelection,
    table: pa.Table,
) -> pa.Table:
    """Return canonical external aSoCC run rows as an Arrow table."""
    count = table.num_rows
    impact = (
        pa.nulls(count, type=pa.string())
        if "impact" not in table.column_names
        else pc.cast(table["impact"], pa.string())
    )
    reference_year = (
        pa.nulls(count, type=pa.int64())
        if "reference_year" not in table.column_names
        else pc.cast(table["reference_year"], pa.int64())
    )
    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        "run_index": pc.cast(table["run_index"], pa.int64()),
        "year": pc.cast(table["year"], pa.int64()),
        "level": constant_string(source.selection.level, count),
        "l1_l2_method": constant_string(source.selection.l1_l2_method, count),
        "l2_method": constant_string(source.selection.l2_method, count),
        "l1_method": constant_string(source.selection.l1_method, count),
        "lcia_method": constant_string(file_selection.lcia_method, count),
        "impact": impact,
        ASOCC_SSP_SCENARIO_COLUMN: normalize_arrow_scenarios(table[ASOCC_SSP_SCENARIO_COLUMN]),
        "reference_year": reference_year,
        "value": pc.cast(table["value"], pa.float64()),
    }
    for selector in expected_external_selector_columns(fu_code=source.selection.fu_code):
        columns[selector] = pc.cast(table[selector], pa.string())
    return pa.table(
        {
            column: columns[column]
            for column in external_asocc_render_row_columns(
                selection=source.selection,
                include_asocc_ssp_scenario=True,
            )
        }
    )


def validate_arrow_columns(
    *,
    table: pa.Table,
    source: ExternalMonteCarloRowsSource,
    path,
) -> None:
    """Validate CSV columns at the external file boundary."""
    validate_external_asocc_monte_carlo_column_names(
        columns=table.column_names,
        selection=source.selection,
        path=path,
    )
    selectors = expected_external_selector_columns(fu_code=source.selection.fu_code)
    missing_selectors = [column for column in selectors if column not in table.column_names]
    if missing_selectors:
        raise ValueError(
            f"External aSoCC file '{path}' must provide the public selector columns expected "
            f"for fu_code='{source.selection.fu_code}'. Expected={list(selectors)}, "
            f"missing={missing_selectors}."
        )
    if table.num_rows == 0:
        return
    empty = [column for column in selectors if not string_all_nonempty(table[column])]
    if empty:
        raise ValueError(
            f"External aSoCC file '{path}' has empty public selector columns. They must "
            "be non empty so external rows "
            f"can align with pyaesa owned method rows. Empty={empty}."
        )
    unexpected = [
        column
        for column in _ALL_SELECTOR_COLUMNS
        if column not in selectors
        and column in table.column_names
        and string_has_nonempty(table[column])
    ]
    if unexpected:
        raise ValueError(
            f"External aSoCC file '{path}' must not provide selector columns outside the requested "
            f"functional unit identity. fu_code='{source.selection.fu_code}', "
            f"expected={list(selectors)}, unexpected={unexpected}."
        )


def filter_arrow_scenarios(
    *,
    table: pa.Table,
    options_by_year: dict[int, tuple[str | None, ...]] | None,
) -> pa.Table:
    """Filter rows to requested SSP options while keeping unrestricted years."""
    if (
        table.num_rows == 0
        or not options_by_year
        or ASOCC_SSP_SCENARIO_COLUMN not in table.column_names
    ):
        return table
    years = pc.cast(table["year"], pa.int64())
    option_years = pa.array(tuple(sorted(int(year) for year in options_by_year)), type=pa.int64())
    keep = _PC.invert(_PC.is_in(years, value_set=option_years))
    scenario_text = _PC.utf8_trim_whitespace(pc.cast(table[ASOCC_SSP_SCENARIO_COLUMN], pa.string()))
    if unique_nonempty_strings(scenario_text).size == 0:
        null_years = tuple(
            int(year) for year, allowed in options_by_year.items() if None in allowed
        )
        if null_years:
            keep = _PC.or_(keep, _PC.is_in(years, value_set=pa.array(null_years, type=pa.int64())))
        return table.filter(keep)
    scenarios = normalize_arrow_scenarios(scenario_text)
    for year in sorted(options_by_year):
        allowed = options_by_year[int(year)]
        year_mask = _PC.equal(years, pa.scalar(int(year), type=pa.int64()))
        scenario_mask = _PC.is_in(scenarios, value_set=pa.array(tuple(allowed), type=pa.string()))
        keep = _PC.or_(keep, _PC.and_(year_mask, scenario_mask))
    return table.filter(keep)


def normalize_arrow_scenarios(column: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return canonical nullable SSP labels for an Arrow column."""
    text = _PC.utf8_trim_whitespace(pc.cast(column, pa.string()))
    missing = _PC.or_(_PC.is_null(text), _PC.equal(text, ""))
    values = unique_nonempty_strings(text)
    if values.size == 0:
        return _PC.if_else(missing, pa.nulls(len(column), type=pa.string()), text)
    normalized = pa.array(
        tuple(
            normalize_ssp_tokens(
                [str(value)],
                context="External aSoCC Monte Carlo asocc_ssp_scenario column",
            )[0]
            for value in values.tolist()
        )
    )
    positions = _PC.index_in(text, value_set=pa.array(values.tolist()))
    mapped = pc.take(normalized, positions)
    return _PC.if_else(missing, pa.nulls(len(column), type=pa.string()), mapped)


def constant_string(value: object, count: int) -> pa.Array:
    if value is None:
        return pa.nulls(count, type=pa.string())
    return pa.repeat(str(value), count).cast(pa.string())


def int64_numpy(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    array = combine(pc.cast(values, pa.int64()))
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.int64)


def float64_numpy(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    array = combine(pc.cast(values, pa.float64()))
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)


def string_all_nonempty(values: pa.Array | pa.ChunkedArray) -> bool:
    text = _PC.utf8_trim_whitespace(pc.cast(values, pa.string()))
    present = _PC.and_(_PC.invert(_PC.is_null(text)), _PC.not_equal(text, ""))
    return bool(_PC.all(present).as_py())


def concatenate_int_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

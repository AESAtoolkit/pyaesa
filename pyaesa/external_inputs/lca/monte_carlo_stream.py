"""Vectorized external LCA Monte Carlo source scanning."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from pyaesa.external_inputs.shared.matrix_identity import (
    IdentityLookup,
    assign_values,
    contiguous_inventory,
    duplicate_identity_error,
    positions_for_arrow,
    require_complete_matrix,
    template_lookup,
)
from pyaesa.external_inputs.lca.io import (
    external_lca_no_requested_year_rows_error,
    validate_external_lca_contract,
    validate_external_lca_required_columns,
    validate_external_lca_reserved_columns,
)
from pyaesa.external_inputs.lca.scenario_metadata import with_lca_ssp_start_year
from pyaesa.external_inputs.shared.tabular import none_for_missing_series
from pyaesa.shared.figures.contracts import SELECTOR_COLUMNS
from pyaesa.shared.runtime.scenario.columns import EXT_LCA_SSP_SCENARIO_COLUMN
from pyaesa.shared.selectors.fu_axes import expected_fu_selector_columns
from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens

_PC = cast(Any, pc)
_BLOCK_BYTES = 64 * 1024 * 1024
_REQUIRED_COLUMNS = {
    "run_index",
    "year",
    EXT_LCA_SSP_SCENARIO_COLUMN,
    "impact",
    "impact_unit",
    "value",
}
_FILE_OWNED_COLUMNS = {"lcia_method", "ssp_scenario"}


@dataclass(frozen=True)
class ExternalLCALongMatrixSource:
    """Bounded external LCA long matrix source."""

    identity: pd.DataFrame
    run_indices: np.ndarray
    values_for_runs: Callable[[np.ndarray], np.ndarray]


def load_external_lca_long_matrix_source(
    *,
    path: Path,
    lcia_method: str,
    years: list[int],
    base_allocate_args: dict[str, Any],
) -> ExternalLCALongMatrixSource:
    """Return a bounded long external LCA Monte Carlo source."""
    run_chunks: list[np.ndarray] = []
    template_tables: list[pa.Table] = []
    observed_pairs: set[tuple[str, str]] = set()
    template_frame: pd.DataFrame | None = None
    lookup: IdentityLookup | None = None
    last_run_index: int | None = None
    current_run_index: int | None = None
    current_run_filled: np.ndarray | None = None

    def build_template() -> tuple[pd.DataFrame, IdentityLookup]:
        nonlocal template_frame, lookup
        if template_frame is not None and lookup is not None:
            return template_frame, lookup
        built = pa.concat_tables(template_tables, promote_options="default").to_pandas()
        _validate_template_unique(template=built)
        matrix_template, built_lookup = template_lookup(template=built)
        template_frame = built
        lookup = built_lookup
        return built, built_lookup

    def update_completeness(table: pa.Table, active_lookup: IdentityLookup) -> None:
        nonlocal current_run_index, current_run_filled
        row_count = len(cast(pd.DataFrame, template_frame))
        run_index = _int64_numpy(table["run_index"])
        positions = positions_for_arrow(table=table, lookup=active_lookup)
        for run in np.unique(run_index):
            run_id = int(run)
            if current_run_index is None:
                current_run_index = run_id
                current_run_filled = np.zeros(row_count, dtype=bool)
            elif run_id != current_run_index:
                require_complete_matrix(filled=cast(np.ndarray, current_run_filled)[None, :])
                current_run_index = run_id
                current_run_filled = np.zeros(row_count, dtype=bool)
            mask = run_index == run_id
            run_positions = positions[mask]
            filled = cast(np.ndarray, current_run_filled)
            if len(np.unique(run_positions)) != len(run_positions) or bool(
                filled[run_positions].any()
            ):
                raise duplicate_identity_error()
            filled[run_positions] = True

    for table in _iter_selected_tables(path=path, years=years):
        _validate_columns(table=table, path=path, base_allocate_args=base_allocate_args)
        observed_pairs.update(_impact_pairs(table))
        normalized = _normalize_table(table=table)
        run_index = _int64_numpy(normalized["run_index"])
        _validate_non_decreasing_runs(
            run_index=run_index,
            last_run_index=last_run_index,
            path=path,
        )
        last_run_index = int(run_index[-1])
        run_chunks.append(np.unique(run_index))
        run_zero_mask = run_index == 0
        if bool(run_zero_mask.any()):
            template_tables.append(normalized.filter(pa.array(run_zero_mask)))
        if lookup is None and bool((run_index > 0).any()) and template_tables:
            build_template()
        if lookup is not None:
            active = (
                normalized.filter(pa.array(run_index > 0))
                if bool(run_zero_mask.any())
                else normalized
            )
            _template, active_lookup = build_template()
            update_completeness(table=active, active_lookup=active_lookup)
    if not run_chunks:
        raise external_lca_no_requested_year_rows_error(path=path, years=years)
    _validate_impact_inventory(
        observed_pairs=observed_pairs,
        path=path,
        lcia_method=lcia_method,
    )
    run_indices = contiguous_inventory(
        values=_concatenate_int_chunks(run_chunks),
        context=f"External LCA Monte Carlo file '{path.name}' after requested year filtering",
    )
    matrix_template, matrix_lookup = build_template()
    if current_run_filled is not None:
        require_complete_matrix(filled=current_run_filled[None, :])
    identity = matrix_template.drop(columns=["value"]).reset_index(drop=True)
    identity[EXT_LCA_SSP_SCENARIO_COLUMN] = none_for_missing_series(
        pd.Series(identity.loc[:, EXT_LCA_SSP_SCENARIO_COLUMN], copy=False)
    )
    identity = with_lca_ssp_start_year(identity)
    identity.insert(0, "public_row_id", np.arange(len(identity), dtype=np.int64))
    run_index_array = np.asarray(run_indices, dtype=np.int64)
    return ExternalLCALongMatrixSource(
        identity=identity,
        run_indices=run_index_array,
        values_for_runs=lambda requested: _long_values_for_runs(
            path=path,
            years=years,
            lookup=matrix_lookup,
            run_indices=run_index_array,
            row_count=len(identity),
            requested_runs=np.asarray(requested, dtype=np.int64),
        ),
    )


def _long_values_for_runs(
    *,
    path: Path,
    years: list[int],
    lookup: IdentityLookup,
    run_indices: np.ndarray,
    row_count: int,
    requested_runs: np.ndarray,
) -> np.ndarray:
    if requested_runs.size == 0:
        return np.empty((0, int(row_count)), dtype=np.float64)
    unique_runs = np.unique(requested_runs.astype(np.int64, copy=False))
    _validate_requested_runs(run_indices=run_indices, requested_runs=unique_runs, path=path)
    values = np.full((len(unique_runs), int(row_count)), np.nan, dtype=np.float64)
    filled = np.zeros(values.shape, dtype=bool)
    for table in _iter_selected_tables(path=path, years=years):
        filtered = _filter_run_indices(table=table, run_indices=unique_runs)
        if not filtered.num_rows:
            continue
        normalized = _normalize_table(table=filtered)
        run = _int64_numpy(normalized["run_index"])
        assign_values(
            values=values,
            filled=filled,
            row_positions=np.searchsorted(unique_runs, run),
            column_positions=positions_for_arrow(table=normalized, lookup=lookup),
            source_values=_float64_numpy(normalized["value"]),
        )
    require_complete_matrix(filled=filled)
    return values[np.searchsorted(unique_runs, requested_runs)]


def _iter_selected_tables(*, path: Path, years: list[int]) -> Iterator[pa.Table]:
    selected_years = pa.array([int(year) for year in years], type=pa.int64())
    suffix = path.suffix.lower()
    if suffix == ".csv":
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(block_size=_BLOCK_BYTES),
        )
        try:
            for batch in reader:
                filtered = _filter_years(table=pa.Table.from_batches([batch]), years=selected_years)
                if filtered.num_rows:
                    yield filtered
        finally:
            reader.close()
        return
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches():
        filtered = _filter_years(table=pa.Table.from_batches([batch]), years=selected_years)
        if filtered.num_rows:
            yield filtered


def _filter_years(*, table: pa.Table, years: pa.Array) -> pa.Table:
    if "year" not in table.column_names:
        return table
    mask = _PC.is_in(pc.cast(table["year"], pa.int64()), value_set=years)
    return table.filter(mask)


def _filter_run_indices(*, table: pa.Table, run_indices: np.ndarray) -> pa.Table:
    selected_runs = pa.array(run_indices, type=pa.int64())
    mask = _PC.is_in(pc.cast(table["run_index"], pa.int64()), value_set=selected_runs)
    return table.filter(mask)


def _validate_requested_runs(
    *,
    run_indices: np.ndarray,
    requested_runs: np.ndarray,
    path: Path,
) -> None:
    if int(requested_runs[0]) >= 0 and int(requested_runs[-1]) < len(run_indices):
        return
    missing = requested_runs[(requested_runs < 0) | (requested_runs >= len(run_indices))]
    missing_values = missing.astype(int).tolist()
    raise ValueError(
        f"External LCA Monte Carlo file '{path}' is missing run_index values {missing_values}."
    )


def _validate_columns(*, table: pa.Table, path: Path, base_allocate_args: dict[str, Any]) -> None:
    validate_external_lca_required_columns(
        columns=table.column_names,
        path=path,
        required_columns=_REQUIRED_COLUMNS,
        file_label="External LCA Monte Carlo file",
    )
    validate_external_lca_reserved_columns(
        columns=table.column_names,
        path=path,
        forbidden_columns=_FILE_OWNED_COLUMNS,
        file_label="External LCA Monte Carlo file",
    )
    required_selectors = expected_fu_selector_columns(fu_code=str(base_allocate_args["fu_code"]))
    missing_selectors = [
        column for column in required_selectors if column not in table.column_names
    ]
    if missing_selectors:
        raise ValueError(
            f"External LCA file '{path}' must provide selector columns {list(required_selectors)}. "
            f"Missing={missing_selectors}."
        )
    empty_selectors = [
        column
        for column in required_selectors
        if not bool(_PC.all(_nonempty_text(table[column])).as_py())
    ]
    if empty_selectors:
        raise ValueError(
            f"External LCA file '{path}' contains empty selector values in required "
            f"columns {empty_selectors}."
        )
    unexpected = [
        column
        for column in SELECTOR_COLUMNS
        if column not in required_selectors
        and column in table.column_names
        and bool(_PC.any(_nonempty_text(table[column])).as_py())
    ]
    if unexpected:
        raise ValueError(
            f"External LCA file '{path}' contains selector columns outside the requested "
            f"fu_code='{base_allocate_args['fu_code']}'. Unexpected={unexpected}."
        )


def _normalize_table(*, table: pa.Table) -> pa.Table:
    count = table.num_rows
    base = {
        "run_index": pc.cast(table["run_index"], pa.int64()),
        "year": pc.cast(table["year"], pa.int64()),
        "impact": pc.cast(table["impact"], pa.string()),
        "impact_unit": pc.cast(table["impact_unit"], pa.string()),
        EXT_LCA_SSP_SCENARIO_COLUMN: _normalized_scenarios(table[EXT_LCA_SSP_SCENARIO_COLUMN]),
        "value": pc.cast(table["value"], pa.float64()),
    }
    selectors = {
        column: pc.cast(table[column], pa.string())
        for column in SELECTOR_COLUMNS
        if column in table.column_names
    }
    excluded = {*base, *selectors, *_FILE_OWNED_COLUMNS}
    extras = {column: table[column] for column in sorted(set(table.column_names) - excluded)}
    return pa.table({**base, **selectors, **extras}).slice(0, count)


def _normalized_scenarios(column: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    text = _PC.utf8_trim_whitespace(pc.cast(column, pa.string()))
    missing = _PC.or_(_PC.is_null(text), _PC.equal(text, ""))
    values = _unique_nonempty(text)
    if values.size == 0:
        return _PC.if_else(missing, pa.nulls(len(column), type=pa.string()), text)
    normalized = pa.array(
        tuple(
            normalize_ssp_tokens(
                [str(value)],
                context="External LCA Monte Carlo lca_ssp_scenario column",
            )[0]
            for value in values.tolist()
        )
    )
    mapped = pc.take(normalized, _PC.index_in(text, value_set=pa.array(values.tolist())))
    return _PC.if_else(missing, pa.nulls(len(column), type=pa.string()), mapped)


def _impact_pairs(table: pa.Table) -> set[tuple[str, str]]:
    impact = _PC.utf8_trim_whitespace(pc.cast(table["impact"], pa.string()))
    unit = _PC.utf8_trim_whitespace(pc.cast(table["impact_unit"], pa.string()))
    valid = _PC.and_(_PC.invert(_PC.is_null(impact)), _PC.invert(_PC.is_null(unit)))
    grouped = (
        pa.table(
            {
                "impact": impact.filter(valid),
                "impact_unit": unit.filter(valid),
            }
        )
        .group_by(["impact", "impact_unit"])
        .aggregate([])
    )
    return set(zip(grouped["impact"].to_pylist(), grouped["impact_unit"].to_pylist()))


def _validate_impact_inventory(
    *,
    observed_pairs: set[tuple[str, str]],
    path: Path,
    lcia_method: str,
) -> None:
    frame = pd.DataFrame.from_records(
        [{"impact": impact, "impact_unit": unit} for impact, unit in sorted(observed_pairs)]
    )
    validate_external_lca_contract(frame=frame, path=path, lcia_method=lcia_method)


def _validate_template_unique(*, template: pd.DataFrame) -> None:
    identity_columns = [
        column for column in template.columns if column not in {"run_index", "value"}
    ]
    if bool(template.duplicated(["run_index", *identity_columns], keep=False).any()):
        raise duplicate_identity_error()


def _validate_non_decreasing_runs(
    *,
    run_index: np.ndarray,
    last_run_index: int | None,
    path: Path,
) -> None:
    prefix = np.fromiter(
        (value for value in (last_run_index,) if value is not None),
        dtype=np.int64,
    )
    ordered = np.concatenate((prefix, run_index))
    if bool(np.any(ordered[1:] < ordered[:-1])):
        raise ValueError(
            f"External LCA Monte Carlo file '{path}' must be sorted by nondecreasing run_index."
        )


def _nonempty_text(column: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    text = _PC.utf8_trim_whitespace(pc.cast(column, pa.string()))
    return _PC.and_(_PC.invert(_PC.is_null(text)), _PC.not_equal(text, ""))


def _unique_nonempty(column: pa.Array | pa.ChunkedArray) -> np.ndarray:
    present = _nonempty_text(column)
    unique = _PC.unique(column.filter(present))
    return np.asarray(unique.to_numpy(zero_copy_only=False), dtype=object)


def _int64_numpy(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    array = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    return np.asarray(pc.cast(array, pa.int64()).to_numpy(zero_copy_only=False), dtype=np.int64)


def _float64_numpy(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    array = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    return np.asarray(pc.cast(array, pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)


def _concatenate_int_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

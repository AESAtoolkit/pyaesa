"""Numeric identity helpers for external Monte Carlo matrices."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

_PC = cast(Any, pc)


@dataclass(frozen=True)
class InventoryTemplate:
    """Shared run inventory and first run public row template."""

    run_indices: tuple[int, ...]
    template: pd.DataFrame


@dataclass(frozen=True)
class IdentityLookup:
    """Vectorized lookup from source row identity to template column position."""

    variable_columns: tuple[str, ...]
    constant_values: dict[str, object]
    value_sets: tuple[pa.Array, ...]
    multipliers: tuple[int, ...]
    sorted_codes: np.ndarray
    sorted_positions: np.ndarray


def template_lookup(*, template: pd.DataFrame) -> tuple[pd.DataFrame, IdentityLookup]:
    """Return canonical template columns and a vectorized identity lookup."""
    value_column = "value"
    identity_columns = [
        column for column in template.columns if column not in {"run_index", value_column}
    ]
    ordered = template.loc[:, [*identity_columns, value_column]].reset_index(drop=True)
    return ordered, build_lookup(template=ordered, identity_columns=identity_columns)


def build_lookup(*, template: pd.DataFrame, identity_columns: list[str]) -> IdentityLookup:
    """Build a compact lookup from row identity keys to matrix columns."""
    variable_columns: list[str] = []
    constant_values: dict[str, object] = {}
    for column in identity_columns:
        series = pd.Series(template.loc[:, column], copy=False)
        if int(series.nunique(dropna=False)) == 1:
            constant_values[column] = series.iloc[0]
        else:
            variable_columns.append(column)
    value_sets: list[pa.Array] = []
    multipliers: list[int] = []
    codes = np.zeros(len(template), dtype=np.int64)
    multiplier = 1
    for column in variable_columns:
        values = pandas_key_array(pd.Series(template.loc[:, column], copy=False))
        value_set = _PC.unique(values)
        codes += index_in(values=values, value_set=value_set) * multiplier
        value_sets.append(value_set)
        multipliers.append(multiplier)
        multiplier *= len(value_set)
    order = np.argsort(codes, kind="stable")
    return IdentityLookup(
        variable_columns=tuple(variable_columns),
        constant_values=constant_values,
        value_sets=tuple(value_sets),
        multipliers=tuple(multipliers),
        sorted_codes=codes[order],
        sorted_positions=order.astype(np.int64),
    )


def positions_for_arrow(*, table: pa.Table, lookup: IdentityLookup) -> np.ndarray:
    """Return template column positions for one normalized Arrow table."""
    validate_constant_arrow(table=table, lookup=lookup)
    if not lookup.variable_columns:
        return np.zeros(table.num_rows, dtype=np.int64)
    codes = np.zeros(table.num_rows, dtype=np.int64)
    for column, value_set, multiplier in zip(
        lookup.variable_columns,
        lookup.value_sets,
        lookup.multipliers,
        strict=True,
    ):
        column_codes = index_in(values=arrow_key_array(table[column]), value_set=value_set)
        if bool((column_codes < 0).any()):
            raise identity_mismatch_error()
        codes += column_codes * multiplier
    return positions_from_codes(codes=codes, lookup=lookup)


def positions_for_frame(*, frame: pd.DataFrame, lookup: IdentityLookup) -> np.ndarray:
    """Return template column positions for one normalized pandas frame."""
    validate_constant_frame(frame=frame, lookup=lookup)
    if not lookup.variable_columns:
        return np.zeros(len(frame), dtype=np.int64)
    codes = np.zeros(len(frame), dtype=np.int64)
    for column, value_set, multiplier in zip(
        lookup.variable_columns,
        lookup.value_sets,
        lookup.multipliers,
        strict=True,
    ):
        series = cast(pd.Series, frame.loc[:, column])
        column_codes = index_in(values=pandas_key_array(series), value_set=value_set)
        if bool((column_codes < 0).any()):
            raise identity_mismatch_error()
        codes += column_codes * multiplier
    return positions_from_codes(codes=codes, lookup=lookup)


def positions_from_codes(*, codes: np.ndarray, lookup: IdentityLookup) -> np.ndarray:
    """Map composite row identity codes to template positions."""
    locations = np.searchsorted(lookup.sorted_codes, codes)
    inside = locations < len(lookup.sorted_codes)
    matched = np.zeros(len(codes), dtype=bool)
    matched[inside] = lookup.sorted_codes[locations[inside]] == codes[inside]
    if not bool(matched.all()):
        raise identity_mismatch_error()
    return lookup.sorted_positions[locations]


def validate_constant_arrow(*, table: pa.Table, lookup: IdentityLookup) -> None:
    """Validate constant identity columns for one Arrow table."""
    for column, expected in lookup.constant_values.items():
        codes = index_in(
            values=arrow_key_array(table[column]), value_set=scalar_key_array(expected)
        )
        if bool(np.any(codes != 0)):
            raise identity_mismatch_error(column=column)


def validate_constant_frame(*, frame: pd.DataFrame, lookup: IdentityLookup) -> None:
    """Validate constant identity columns for one pandas frame."""
    for column, expected in lookup.constant_values.items():
        series = cast(pd.Series, frame.loc[:, column])
        codes = index_in(values=pandas_key_array(series), value_set=scalar_key_array(expected))
        if bool(np.any(codes != 0)):
            raise identity_mismatch_error(column=column)


def assign_values(
    *,
    values: np.ndarray,
    filled: np.ndarray,
    row_positions: np.ndarray,
    column_positions: np.ndarray,
    source_values: np.ndarray,
) -> None:
    """Write source values into the materialized run matrix."""
    if bool(filled[row_positions, column_positions].any()):
        raise duplicate_identity_error()
    values[row_positions, column_positions] = source_values
    filled[row_positions, column_positions] = True


def require_complete_matrix(*, filled: np.ndarray) -> None:
    """Fail if any requested run identity is missing."""
    if not bool(filled.all()):
        raise ValueError(
            "External Monte Carlo runs are incomplete for the requested public "
            "external row identity set. Every requested run_index must contain one "
            "value for every identity present in run_index 0. When reference_year is "
            "provided, each run_index must contain the complete requested "
            "reference_year candidate set for every studied year and selector identity."
        )


def contiguous_inventory(*, values: np.ndarray, context: str) -> tuple[int, ...]:
    """Return a contiguous run inventory starting at zero."""
    if values.size == 0:
        raise ValueError(f"{context} must provide at least one run_index value.")
    ordered = np.unique(values.astype(np.int64, copy=False))
    expected = np.arange(int(ordered[-1]) + 1, dtype=np.int64)
    if not np.array_equal(ordered, expected):
        missing = np.setdiff1d(expected, ordered, assume_unique=True).astype(int).tolist()
        raise ValueError(
            f"{context} run_index values must be contiguous from 0. Missing={missing}."
        )
    return tuple(int(value) for value in ordered.tolist())


def shared_inventory_template(
    *,
    inventories: dict[str, tuple[int, ...]],
    templates: list[pd.DataFrame],
) -> InventoryTemplate:
    """Return the shared inventory and deduplicated run zero template."""
    unique = set(inventories.values())
    if len(unique) != 1:
        raise ValueError(
            "External Monte Carlo files for one selection must provide the same "
            f"run_index inventory. Observed={inventories}."
        )
    template = pd.concat(templates, ignore_index=True).reset_index(drop=True)
    identity_columns = [
        column for column in template.columns if column not in {"run_index", "value"}
    ]
    if bool(template.duplicated(["run_index", *identity_columns], keep=False).any()):
        raise duplicate_identity_error()
    return InventoryTemplate(run_indices=next(iter(unique)), template=template)


def arrow_key_array(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return nullable Arrow values as comparable string keys."""
    return pc.cast(values, pa.string())


def pandas_key_array(values: pd.Series) -> pa.Array:
    """Return nullable pandas values as comparable string keys."""
    return pa.array(values.astype("string"), type=pa.string(), from_pandas=True)


def scalar_key_array(value: object) -> pa.Array:
    """Return one nullable scalar as a comparable Arrow key set."""
    normalized = None if bool(pd.isna(value)) else str(value)
    return pa.array([normalized], type=pa.string(), from_pandas=True)


def index_in(*, values: pa.Array | pa.ChunkedArray, value_set: pa.Array) -> np.ndarray:
    """Return Arrow index_in results as int64 NumPy codes."""
    codes = pc.fill_null(_PC.index_in(values, value_set=value_set), pa.scalar(-1, type=pa.int32()))
    return np.asarray(combine(codes).to_numpy(zero_copy_only=False), dtype=np.int64)


def combine(values: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Return one Arrow array for NumPy conversion."""
    return values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values


def duplicate_identity_error() -> ValueError:
    """Return the canonical duplicate external Monte Carlo identity error."""
    return ValueError(
        "External Monte Carlo rows contain more than one value for the same "
        "run_index and public row identity. Duplicate rows are invalid even when "
        "their numeric values are equal because each run cell must have exactly "
        "one source value."
    )


def identity_mismatch_error(*, column: str | None = None) -> ValueError:
    """Return the canonical external Monte Carlo identity mismatch error."""
    column_text = f" Column='{column}'." if column is not None else ""
    return ValueError(
        "External Monte Carlo runs do not expose the same public row "
        "identity set. Every requested run_index must contain exactly the same "
        "year, selector, method, LCIA, SSP scenario, and reference_year "
        f"identities as run_index 0.{column_text}"
    )

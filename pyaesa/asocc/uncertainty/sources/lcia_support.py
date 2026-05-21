"""Support row matching for combined aSoCC LCIA uncertainty routes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from pyaesa.asocc.runtime.scope.branch_resolution import asocc_l1_dir, asocc_l2_dir
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import (
    ASOCC_TIME_ROUTE_COLUMN,
    ASOCC_VALUE_COLUMN,
    LoadedAsoccFinalRows,
    read_deterministic_asocc_rows,
    table_paths_under_deterministic_root,
)
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN

_METHOD_IDENTITY_COLUMNS = {"l1_l2_method", "l1_method", "l2_method"}
_NON_SUPPORT_IDENTITY_COLUMNS = {
    "run_index",
    "year",
    ASOCC_VALUE_COLUMN,
    ASOCC_TIME_ROUTE_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
    "l2_reuse_year",
    *_METHOD_IDENTITY_COLUMNS,
}
_LCIA_WEIGHT_IDENTITY_COLUMNS = ("lcia_method", "impact", "reference_year")
_SUPPORT_TERM_CHUNK_SIZE = 2_000_000


@dataclass(frozen=True)
class PositionMatches:
    """One query row to many support rows expressed as integer positions."""

    query_positions: np.ndarray
    support_positions: np.ndarray


@dataclass(frozen=True)
class SupportCodeIndex:
    """Encoded support side lookup for one route term matcher."""

    categories: tuple[np.ndarray, ...]
    multipliers: tuple[int, ...]
    unique_codes: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    order: np.ndarray
    support_positions: np.ndarray
    direct_support_positions: np.ndarray | None


@dataclass
class LCIASupportRowCache:
    """Run scoped deterministic LCIA support row cache."""

    rows_by_key: dict[tuple[str, str, tuple[int, ...]], pd.DataFrame] = field(default_factory=dict)


def support_rows(
    *,
    loaded: LoadedAsoccFinalRows,
    bucket: str,
    stem: str,
    requested_years: list[int],
    support_cache: LCIASupportRowCache,
) -> pd.DataFrame:
    """Return deterministic support rows used by one combined LCIA route."""
    key = (str(bucket), str(stem), tuple(sorted(int(year) for year in requested_years)))
    cached = support_cache.rows_by_key.get(key)
    if cached is not None:
        return cached
    frames = [
        read_deterministic_asocc_rows(path=path, requested_years=requested_years)
        for path in _support_paths(loaded=loaded, bucket=bucket, stem=stem)
    ]
    rows = pd.concat(frames, ignore_index=True)
    support_cache.rows_by_key[key] = rows
    return rows


def combined_final_rows(*, rows: pd.DataFrame, l1_method: str, l2_method: str) -> pd.DataFrame:
    """Return public rows belonging to one combined deterministic route."""
    return rows.loc[
        rows["l1_method"].astype(str).eq(l1_method) & rows["l2_method"].astype(str).eq(l2_method)
    ]


def final_years(*, final_rows: pd.DataFrame) -> list[int]:
    """Return final public row years as deterministic integer support years."""
    years = pd.Series(
        pd.to_numeric(final_rows.loc[:, "year"], errors="raise"),
        index=final_rows.index,
    )
    return sorted({int(year) for year in years.tolist()})


def l2_support_years(*, final_rows: pd.DataFrame) -> list[int]:
    """Return L2 support years after deterministic L2 reuse year ownership."""
    years = pd.Series(
        pd.to_numeric(final_rows.loc[:, "year"], errors="raise"),
        index=final_rows.index,
    )
    if "l2_reuse_year" not in final_rows.columns:
        return sorted({int(year) for year in years.tolist()})
    reuse = pd.Series(
        pd.to_numeric(final_rows.loc[:, "l2_reuse_year"], errors="raise"),
        index=final_rows.index,
    )
    values = years.where(final_rows["l2_reuse_year"].isna(), reuse)
    return sorted({int(year) for year in values.tolist()})


def combined_route_coefficients(
    *,
    final_rows: pd.DataFrame,
    l1_rows: pd.DataFrame,
    l2_rows: pd.DataFrame,
    weight_axis: str,
    l1_axis: str,
    l1_sampled: bool,
) -> tuple[csr_matrix | None, csr_matrix | None]:
    """Return sparse coefficients that recompose combined public values."""
    l2_matches = _l2_support_matches(
        final_rows=final_rows,
        l2_rows=l2_rows,
    )
    final_positions, l1_positions, l2_positions = _matched_l1_terms(
        final_rows=final_rows,
        l1_rows=l1_rows,
        l2_rows=l2_rows,
        l2_matches=l2_matches,
        weight_axis=weight_axis,
        l1_axis=l1_axis,
    )
    if l1_sampled:
        coefficients = csr_matrix(
            (
                _numeric_column(l2_rows, ASOCC_VALUE_COLUMN)[l2_positions],
                (l1_positions, final_positions),
            ),
            shape=(len(l1_rows), len(final_rows)),
        )
        return coefficients, None
    coefficients = csr_matrix(
        (
            _numeric_column(l1_rows, ASOCC_VALUE_COLUMN)[l1_positions],
            (l2_positions, final_positions),
        ),
        shape=(len(l2_rows), len(final_rows)),
    )
    return None, coefficients


def _l2_support_matches(
    *,
    final_rows: pd.DataFrame,
    l2_rows: pd.DataFrame,
) -> PositionMatches:
    l2_axes = _l2_public_merge_axes(identity=final_rows, support=l2_rows)
    return _support_position_matches(
        query_values=(
            _l2_support_years_for_final_rows(final_rows=final_rows),
            *(_object_column(final_rows, column) for column in l2_axes),
        ),
        support_values=(
            _object_column(l2_rows, "year"),
            *(_object_column(l2_rows, column) for column in l2_axes),
        ),
        query_scenario=_scenario_column(final_rows),
        support_scenario=_scenario_column(l2_rows),
        query_positions=np.arange(len(final_rows), dtype=np.int64),
        support_positions=np.arange(len(l2_rows), dtype=np.int64),
    )


def _matched_l1_terms(
    *,
    final_rows: pd.DataFrame,
    l1_rows: pd.DataFrame,
    l2_rows: pd.DataFrame,
    l2_matches: PositionMatches,
    weight_axis: str,
    l1_axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    l1_lcia_columns = [
        column
        for column in _LCIA_WEIGHT_IDENTITY_COLUMNS
        if column in final_rows and column in l1_rows
    ]
    final_year = _object_column(final_rows, "year")
    final_scenario = _scenario_column(final_rows)
    final_lcia_values = {column: _object_column(final_rows, column) for column in l1_lcia_columns}
    l2_axis_value = _object_column(l2_rows, weight_axis)
    support_values = (
        _object_column(l1_rows, "year"),
        _object_column(l1_rows, l1_axis),
        *(_object_column(l1_rows, column) for column in l1_lcia_columns),
    )
    support_scenario = _scenario_column(l1_rows)
    support_positions = np.arange(len(l1_rows), dtype=np.int64)
    missing_scenario = _missing_mask(support_scenario)
    invariant_index = _support_code_index(
        values=tuple(values[missing_scenario] for values in support_values),
        support_positions=support_positions[missing_scenario],
    )
    scenario_index = _support_code_index(
        values=(
            *(values[~missing_scenario] for values in support_values),
            support_scenario[~missing_scenario],
        ),
        support_positions=support_positions[~missing_scenario],
    )
    invariant_final_codes, invariant_l2_codes = _l1_term_code_columns(
        index=invariant_index,
        final_year=final_year,
        l2_axis_value=l2_axis_value,
        final_values=tuple(final_lcia_values[column] for column in l1_lcia_columns),
    )
    scenario_final_codes, scenario_l2_codes = _l1_term_code_columns(
        index=scenario_index,
        final_year=final_year,
        l2_axis_value=l2_axis_value,
        final_values=(
            *(final_lcia_values[column] for column in l1_lcia_columns),
            final_scenario,
        ),
    )
    final_scenario_missing = _missing_mask(final_scenario)
    final_parts: list[np.ndarray] = []
    l1_parts: list[np.ndarray] = []
    l2_parts: list[np.ndarray] = []
    for start in range(0, len(l2_matches.query_positions), _SUPPORT_TERM_CHUNK_SIZE):
        stop = min(start + _SUPPORT_TERM_CHUNK_SIZE, len(l2_matches.query_positions))
        term_final = l2_matches.query_positions[start:stop]
        term_l2 = l2_matches.support_positions[start:stop]
        local_positions = np.arange(len(term_final), dtype=np.int64)
        invariant = _match_l1_term_chunk(
            index=invariant_index,
            final_codes=invariant_final_codes,
            l2_axis_codes=invariant_l2_codes,
            term_final=term_final,
            term_l2=term_l2,
            query_positions=local_positions,
        )
        scenario_offsets = np.flatnonzero(~final_scenario_missing[term_final])
        scenario = _match_l1_term_chunk(
            index=scenario_index,
            final_codes=scenario_final_codes,
            l2_axis_codes=scenario_l2_codes,
            term_final=term_final[scenario_offsets],
            term_l2=term_l2[scenario_offsets],
            query_positions=local_positions[scenario_offsets],
        )
        local_matches = PositionMatches(
            query_positions=np.concatenate([invariant.query_positions, scenario.query_positions]),
            support_positions=np.concatenate(
                [invariant.support_positions, scenario.support_positions]
            ),
        )
        final_parts.append(term_final[local_matches.query_positions])
        l1_parts.append(local_matches.support_positions)
        l2_parts.append(term_l2[local_matches.query_positions])
    return (
        np.concatenate(final_parts).astype(np.int64, copy=False),
        np.concatenate(l1_parts).astype(np.int64, copy=False),
        np.concatenate(l2_parts).astype(np.int64, copy=False),
    )


def _support_position_matches(
    *,
    query_values: tuple[np.ndarray, ...],
    support_values: tuple[np.ndarray, ...],
    query_scenario: np.ndarray,
    support_scenario: np.ndarray,
    query_positions: np.ndarray,
    support_positions: np.ndarray,
) -> PositionMatches:
    support_missing_scenario = _missing_mask(support_scenario)
    query_present_scenario = ~_missing_mask(query_scenario)
    support_present_scenario = ~support_missing_scenario
    invariant_index = _support_code_index(
        values=tuple(values[support_missing_scenario] for values in support_values),
        support_positions=support_positions[support_missing_scenario],
    )
    scenario_index = _support_code_index(
        values=(
            *(values[support_present_scenario] for values in support_values),
            support_scenario[support_present_scenario],
        ),
        support_positions=support_positions[support_present_scenario],
    )
    invariant = _matches_for_query_values(
        index=invariant_index,
        query_values=query_values,
        query_positions=query_positions,
    )
    scenario = _matches_for_query_values(
        index=scenario_index,
        query_values=(
            *(values[query_present_scenario] for values in query_values),
            query_scenario[query_present_scenario],
        ),
        query_positions=query_positions[query_present_scenario],
    )
    return PositionMatches(
        query_positions=np.concatenate([invariant.query_positions, scenario.query_positions]),
        support_positions=np.concatenate([invariant.support_positions, scenario.support_positions]),
    )


def _support_code_index(
    *,
    values: tuple[np.ndarray, ...],
    support_positions: np.ndarray,
) -> SupportCodeIndex:
    empty = np.empty(0, dtype=np.int64)
    if not len(support_positions):
        return SupportCodeIndex(
            categories=(),
            multipliers=(),
            unique_codes=empty,
            starts=empty,
            counts=empty,
            order=empty,
            support_positions=empty,
            direct_support_positions=None,
        )
    support_codes = np.zeros(len(support_positions), dtype=np.int64)
    categories: list[np.ndarray] = []
    multipliers: list[int] = []
    multiplier = 1
    for column in values:
        codes, uniques = pd.factorize(column, sort=False, use_na_sentinel=False)
        codes = codes.astype(np.int64, copy=False)
        categories.append(np.asarray(uniques, dtype=object))
        multipliers.append(multiplier)
        support_codes += codes * multiplier
        multiplier *= int(codes.max()) + 1
    order = np.argsort(support_codes, kind="stable")
    ordered_codes = support_codes[order]
    unique_codes, starts, counts = np.unique(
        ordered_codes,
        return_index=True,
        return_counts=True,
    )
    direct_support_positions = _direct_support_positions(
        unique_codes=unique_codes,
        counts=counts,
        order=order,
        starts=starts,
        support_positions=support_positions,
    )
    return SupportCodeIndex(
        categories=tuple(categories),
        multipliers=tuple(multipliers),
        unique_codes=unique_codes,
        starts=starts,
        counts=counts,
        order=order,
        support_positions=support_positions,
        direct_support_positions=direct_support_positions,
    )


def _direct_support_positions(
    *,
    unique_codes: np.ndarray,
    counts: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    support_positions: np.ndarray,
) -> np.ndarray | None:
    if not len(unique_codes) or int(np.max(counts, initial=0)) != 1:
        return None
    direct_size = int(unique_codes[-1]) + 1
    retained_index_bytes = (
        unique_codes.nbytes
        + starts.nbytes
        + counts.nbytes
        + order.nbytes
        + support_positions.nbytes
    )
    if direct_size * np.dtype(np.int64).itemsize > retained_index_bytes:
        return None
    direct = np.full(direct_size, -1, dtype=np.int64)
    direct[unique_codes] = support_positions[order[starts]]
    return direct


def _l1_term_code_columns(
    *,
    index: SupportCodeIndex,
    final_year: np.ndarray,
    l2_axis_value: np.ndarray,
    final_values: tuple[np.ndarray, ...],
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    if not index.categories:
        missing_final = tuple(
            np.full(len(values), -1, dtype=np.int64) for values in (final_year, *final_values)
        )
        return missing_final, np.full(len(l2_axis_value), -1, dtype=np.int64)
    final_codes = (
        _codes_for_categories(values=final_year, categories=index.categories[0]),
        *(
            _codes_for_categories(values=values, categories=index.categories[offset])
            for offset, values in enumerate(final_values, start=2)
        ),
    )
    return final_codes, _codes_for_categories(
        values=l2_axis_value,
        categories=index.categories[1],
    )


def _match_l1_term_chunk(
    *,
    index: SupportCodeIndex,
    final_codes: tuple[np.ndarray, ...],
    l2_axis_codes: np.ndarray,
    term_final: np.ndarray,
    term_l2: np.ndarray,
    query_positions: np.ndarray,
) -> PositionMatches:
    if not index.categories or not len(term_final):
        empty = np.empty(0, dtype=np.int64)
        return PositionMatches(query_positions=empty, support_positions=empty)
    codes = np.zeros(len(term_final), dtype=np.int64)
    valid = np.ones(len(term_final), dtype=bool)
    _add_l1_code(
        codes=codes,
        valid=valid,
        column_codes=final_codes[0][term_final],
        multiplier=index.multipliers[0],
    )
    _add_l1_code(
        codes=codes,
        valid=valid,
        column_codes=l2_axis_codes[term_l2],
        multiplier=index.multipliers[1],
    )
    for offset, column_codes in enumerate(final_codes[1:], start=2):
        _add_l1_code(
            codes=codes,
            valid=valid,
            column_codes=column_codes[term_final],
            multiplier=index.multipliers[offset],
        )
    return _matches_for_codes(
        codes=codes,
        valid=valid,
        query_positions=query_positions,
        index=index,
    )


def _matches_for_query_values(
    *,
    index: SupportCodeIndex,
    query_values: tuple[np.ndarray, ...],
    query_positions: np.ndarray,
) -> PositionMatches:
    if not index.categories or not len(query_positions):
        empty = np.empty(0, dtype=np.int64)
        return PositionMatches(query_positions=empty, support_positions=empty)
    codes = np.zeros(len(query_positions), dtype=np.int64)
    valid = np.ones(len(query_positions), dtype=bool)
    for column, categories, multiplier in zip(
        query_values,
        index.categories,
        index.multipliers,
        strict=True,
    ):
        _add_l1_code(
            codes=codes,
            valid=valid,
            column_codes=_codes_for_categories(values=column, categories=categories),
            multiplier=multiplier,
        )
    return _matches_for_codes(
        codes=codes,
        valid=valid,
        query_positions=query_positions,
        index=index,
    )


def _matches_for_codes(
    *,
    codes: np.ndarray,
    valid: np.ndarray,
    query_positions: np.ndarray,
    index: SupportCodeIndex,
) -> PositionMatches:
    offsets = np.flatnonzero(valid)
    if not len(offsets):
        empty = np.empty(0, dtype=np.int64)
        return PositionMatches(query_positions=empty, support_positions=empty)
    valid_codes = codes[offsets]
    direct_positions = index.direct_support_positions
    if direct_positions is not None:
        in_range = valid_codes < len(direct_positions)
        direct_matches = direct_positions[valid_codes[in_range]]
        matched = direct_matches >= 0
        matched_offsets = offsets[in_range][matched]
        return PositionMatches(
            query_positions=query_positions[matched_offsets],
            support_positions=direct_matches[matched],
        )
    locations = np.searchsorted(index.unique_codes, valid_codes)
    in_range = locations < len(index.unique_codes)
    matched = in_range.copy()
    matched[in_range] = index.unique_codes[locations[in_range]] == valid_codes[in_range]
    matched_locations = locations[matched]
    matched_offsets = offsets[np.flatnonzero(matched)]
    matched_counts = index.counts[matched_locations]
    if int(np.max(matched_counts, initial=0)) == 1:
        return PositionMatches(
            query_positions=query_positions[matched_offsets],
            support_positions=index.support_positions[index.order[index.starts[matched_locations]]],
        )
    starts = index.starts[matched_locations]
    total = int(matched_counts.sum())
    repeated_offsets = np.repeat(matched_offsets, matched_counts)
    repeated_starts = np.repeat(starts, matched_counts)
    repeated_bases = np.repeat(np.cumsum(matched_counts) - matched_counts, matched_counts)
    support_offsets = repeated_starts + np.arange(total, dtype=np.int64) - repeated_bases
    return PositionMatches(
        query_positions=query_positions[repeated_offsets],
        support_positions=index.support_positions[index.order[support_offsets]],
    )


def _codes_for_categories(*, values: np.ndarray, categories: np.ndarray) -> np.ndarray:
    return pd.Index(categories).get_indexer(values).astype(np.int64, copy=False)


def _add_l1_code(
    *,
    codes: np.ndarray,
    valid: np.ndarray,
    column_codes: np.ndarray,
    multiplier: int,
) -> None:
    column_valid = column_codes >= 0
    valid &= column_valid
    codes += np.where(column_valid, column_codes, 0) * multiplier


def _missing_mask(values: np.ndarray) -> np.ndarray:
    return pd.Series(values, copy=False).isna().to_numpy(dtype=bool)


def _object_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.Series(frame.loc[:, column], copy=False).to_numpy(dtype=object, copy=False)


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    numeric = cast(
        pd.Series,
        pd.to_numeric(pd.Series(frame.loc[:, column], copy=False), errors="raise"),
    )
    return numeric.to_numpy(dtype=np.float64)


def _scenario_column(frame: pd.DataFrame) -> np.ndarray:
    return _object_column(frame, ASOCC_SSP_SCENARIO_COLUMN)


def _l2_support_years_for_final_rows(*, final_rows: pd.DataFrame) -> np.ndarray:
    years = cast(
        pd.Series,
        pd.to_numeric(
            pd.Series(final_rows.loc[:, "year"], copy=False),
            errors="raise",
        ),
    ).astype("int64")
    if "l2_reuse_year" not in final_rows.columns:
        return years.to_numpy(copy=False)
    reuse = pd.Series(final_rows.loc[:, "l2_reuse_year"], copy=False)
    reuse_numeric = cast(pd.Series, pd.to_numeric(reuse, errors="raise"))
    support_years = years.where(reuse.isna(), reuse_numeric).astype("int64")
    return support_years.to_numpy(copy=False)


def _support_paths(*, loaded: LoadedAsoccFinalRows, bucket: str, stem: str) -> list[Path]:
    root = (
        asocc_l1_dir(
            scope=loaded.path_scope,
            lcia_sub=None,
            fu_code=str(loaded.base_asocc_args["fu_code"]),
        )
        if bucket == "level_1"
        else asocc_l2_dir(scope=loaded.path_scope, bucket=bucket, lcia_sub=None)
    )
    return sorted(
        {
            path
            for scope in loaded.persisted_scopes
            for path in table_paths_under_deterministic_root(
                raw_paths=scope.outputs,
                root=root,
            )
            if _matches_stem(path=path, stem=stem)
        }
    )


def _matches_stem(*, path: Path, stem: str) -> bool:
    path_stem = path.stem
    return path_stem == stem or path_stem.startswith(f"{stem}__ssp")


def _l2_public_merge_axes(identity: pd.DataFrame, support: pd.DataFrame) -> list[str]:
    support_columns = set(support.columns)
    return [
        column
        for column in identity.columns
        if column in support_columns
        and column not in _NON_SUPPORT_IDENTITY_COLUMNS
        and bool(pd.Series(identity.loc[:, column], copy=False).notna().any())
    ]

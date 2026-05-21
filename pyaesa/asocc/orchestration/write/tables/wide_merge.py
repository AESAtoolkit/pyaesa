"""Wide output row grouping and per year frame merging."""

from typing import cast

import numpy as np
import pandas as pd

from pyaesa.shared.selectors.aggregate_labels import aggregate_selector_label

from ...common_frame import coalesce_unique_non_null

_FILTER_COLUMN_MAP = {
    "r_p": "r_p",
    "s_p": "s_p",
    "r_c": "r_c",
    "r_f": "r_f",
}


def group_output_rows(
    df: pd.DataFrame,
    *,
    filters: dict[str, list[str] | None],
    year_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Aggregate outputs over selected multi value index filters."""
    out = df.copy()
    collapsed_cols: list[str] = []
    for key, col in _FILTER_COLUMN_MAP.items():
        values = filters.get(key)
        if not values or len(values) <= 1:
            continue
        out[col] = aggregate_selector_label(values)
        collapsed_cols.append(col)
    if not collapsed_cols:
        return out
    year_cols = [column for column in out.columns if str(column) in year_columns]
    id_cols = [column for column in out.columns if column not in year_cols]
    grouped = out.groupby(id_cols, dropna=False, as_index=False)[year_cols].sum(min_count=1)
    return cast(pd.DataFrame, grouped)


def _safe_cell_equal(left_value: object, right_value: object) -> bool:
    """Return robust scalar equality for overlap conflict checks."""
    try:
        is_equal = left_value == right_value
    except (TypeError, ValueError):
        return False
    if is_equal is pd.NA:
        return False
    if isinstance(is_equal, np.ndarray):
        if is_equal.size != 1:
            return False
        item = is_equal.item()
        if item is pd.NA:
            return False
        try:
            return bool(item)
        except (TypeError, ValueError):
            return False
    try:
        return bool(is_equal)
    except (TypeError, ValueError):
        return False


def merge_wide_frames(
    *,
    frames: list[pd.DataFrame],
    identifier_columns: tuple[str, ...],
    year_columns: tuple[str, ...],
    where: str,
) -> pd.DataFrame:
    """Merge indexed per year frames into one strict wide frame."""
    if not frames:
        return pd.DataFrame(columns=[*identifier_columns, *year_columns]).reset_index(drop=True)

    def _column_frame_from_indexed(frame_idx: pd.DataFrame) -> pd.DataFrame:
        index_frame = frame_idx.index.to_frame(index=False)
        index_frame.columns = list(identifier_columns)
        out = index_frame.reset_index(drop=True)
        values = frame_idx.to_numpy(copy=False)
        for col_pos, column in enumerate(frame_idx.columns):
            out[str(column)] = values[:, col_pos]
        return out

    def _column_frame_from_value_matrix(
        *,
        base_index: pd.Index,
        present_years: list[str],
        values: np.ndarray,
    ) -> pd.DataFrame:
        data: dict[str, object] = {}
        if isinstance(base_index, pd.MultiIndex):
            for level_position, column in enumerate(identifier_columns):
                data[column] = base_index.get_level_values(level_position).to_numpy(
                    dtype=object,
                    copy=False,
                )
        else:
            data[identifier_columns[0]] = base_index.to_numpy(dtype=object, copy=False)
        for column_position, column in enumerate(present_years):
            data[column] = values[:, column_position]
        return pd.DataFrame(data, columns=[*identifier_columns, *present_years])

    def _column_frame_from_common_index(
        *,
        base_index: pd.Index,
        present_years: list[str],
        series_by_year: dict[str, pd.Series],
    ) -> pd.DataFrame:
        data: dict[str, object] = {}
        if isinstance(base_index, pd.MultiIndex):
            for level_position, column in enumerate(identifier_columns):
                data[column] = base_index.get_level_values(level_position).to_numpy(
                    dtype=object,
                    copy=False,
                )
        else:
            data[identifier_columns[0]] = base_index.to_numpy(dtype=object, copy=False)
        for column in present_years:
            data[column] = series_by_year[column].to_numpy(dtype=np.float64, copy=False)
        return pd.DataFrame(data, columns=[*identifier_columns, *present_years])

    def _append_missing_index_values(
        *,
        base_index: pd.Index,
        next_index: pd.Index,
    ) -> pd.Index:
        missing_positions = base_index.get_indexer(next_index) < 0
        if not bool(missing_positions.any()):
            return base_index
        return base_index.append(next_index[missing_positions])

    def _column_frame_from_aligned_series(
        *,
        present_years: list[str],
        series_by_year: dict[str, pd.Series],
    ) -> pd.DataFrame:
        base_index = series_by_year[present_years[0]].index
        for column in present_years[1:]:
            base_index = _append_missing_index_values(
                base_index=base_index,
                next_index=series_by_year[column].index,
            )
        values = np.full((len(base_index), len(present_years)), np.nan, dtype=np.float64)
        for column_position, column in enumerate(present_years):
            series = series_by_year[column]
            positions = base_index.get_indexer(series.index)
            values[positions, column_position] = series.to_numpy(dtype=np.float64, copy=False)
        return _column_frame_from_value_matrix(
            base_index=base_index,
            present_years=present_years,
            values=values,
        )

    def _canonical_year_label(col: object) -> str | None:
        if isinstance(col, int):
            return str(col)
        if isinstance(col, float) and col.is_integer():
            return str(int(col))
        text = str(col)
        if text.isdigit():
            return text
        return None

    def _is_internal_hot_frame(frame: pd.DataFrame) -> bool:
        if isinstance(frame.index, pd.RangeIndex):
            return False
        if frame.shape[1] != 1:
            return False
        if isinstance(frame.index, pd.MultiIndex):
            idx_names_raw = list(frame.index.names)
        else:
            idx_names_raw = [frame.index.name]
        if any(name is None for name in idx_names_raw):
            return False
        idx_names = tuple(str(name) for name in idx_names_raw)
        if idx_names != identifier_columns:
            return False
        year_label = _canonical_year_label(frame.columns[0])
        if year_label is None or year_label not in year_columns:
            return False
        return True

    if frames and all(_is_internal_hot_frame(frame) for frame in frames):
        year_buckets: dict[str, list[pd.Series]] = {}
        for frame in frames:
            year_col = str(_canonical_year_label(frame.columns[0]))
            series = cast(pd.Series, frame.iloc[:, 0])
            if not pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="raise")
                series = pd.Series(numeric, index=series.index, copy=False)
            year_buckets.setdefault(year_col, []).append(series)

        year_series: dict[str, pd.Series] = {}
        for year_label, series_list in year_buckets.items():
            merged_series = (
                series_list[0]
                if len(series_list) == 1
                else cast(pd.Series, pd.concat(series_list, axis=0, sort=False))
            )
            if not merged_series.index.is_unique:
                level_arg = list(range(merged_series.index.nlevels))
                grouped = merged_series.groupby(level=level_arg, dropna=False, sort=False).agg(
                    lambda values: coalesce_unique_non_null(
                        values,
                        conflict_context=("the same output primary key during frame merge"),
                    )
                )
                merged_series = cast(pd.Series, grouped)
            year_series[year_label] = merged_series
        present_years = [column for column in year_columns if column in year_series]
        base_index = year_series[present_years[0]].index
        if all(year_series[column].index.equals(base_index) for column in present_years[1:]):
            return _column_frame_from_common_index(
                base_index=base_index,
                present_years=present_years,
                series_by_year=year_series,
            )
        return _column_frame_from_aligned_series(
            present_years=present_years,
            series_by_year=year_series,
        )

    def _prepare_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        if isinstance(frame.index, pd.RangeIndex):
            raise ValueError(
                f"{where}: deterministic aSoCC output frames must be indexed by "
                "identifier levels before wide table writing."
            )
        rename_map: dict[object, str] = {}
        for col in frame.columns:
            if isinstance(col, int):
                rename_map[col] = str(col)
            elif isinstance(col, float) and col.is_integer():
                rename_map[col] = str(int(col))
        frame_norm = frame.rename(columns=rename_map) if rename_map else frame
        if isinstance(frame.index, pd.MultiIndex):
            idx_names_raw = list(frame_norm.index.names)
        else:
            idx_names_raw = [frame_norm.index.name]
        if any(name is None for name in idx_names_raw):
            raise ValueError(
                f"{where}: indexed frames must have named identifier levels. "
                f"Got index names={idx_names_raw}."
            )
        idx_names = tuple(str(name) for name in idx_names_raw)
        if idx_names != identifier_columns:
            raise ValueError(
                f"{where}: frame index names do not match output identifiers. "
                f"expected={list(identifier_columns)} got={list(idx_names)}"
            )
        if isinstance(frame_norm.index, pd.MultiIndex):
            level_arg: int | list[int] = list(range(frame_norm.index.nlevels))
        else:
            level_arg = 0
        unknown_cols = [str(c) for c in frame_norm.columns if str(c) not in year_columns]
        if unknown_cols:
            raise ValueError(
                f"{where}: unexpected non-year columns in indexed frames: {unknown_cols}"
            )
        present_years = [c for c in year_columns if c in frame_norm.columns]
        work_idx = frame_norm.loc[:, present_years]
        if work_idx.index.duplicated(keep=False).any():
            work_idx = work_idx.groupby(level=level_arg, dropna=False, sort=False)[
                present_years
            ].agg(
                lambda values: coalesce_unique_non_null(
                    values,
                    conflict_context="the same output primary key during frame merge",
                )
            )
        if present_years:
            non_numeric = [
                col for col in present_years if not pd.api.types.is_numeric_dtype(work_idx[col])
            ]
            if non_numeric:
                work_idx = work_idx.copy()
                for col in non_numeric:
                    work_idx[col] = pd.to_numeric(work_idx[col], errors="raise")
        return work_idx, present_years

    prepared: list[pd.DataFrame] = []
    present_years_union: set[str] = set()
    for frame in frames:
        prepared_idx, present_years = _prepare_frame(frame)
        present_years_union.update(present_years)
        if prepared_idx.empty:
            continue
        prepared.append(prepared_idx)

    present_years_sorted = sorted(present_years_union, key=int)
    if not prepared:
        return pd.DataFrame(columns=[*identifier_columns, *present_years_sorted]).reset_index(
            drop=True
        )
    flat_cols = [str(col) for frame in prepared for col in frame.columns]
    if len(flat_cols) == len(set(flat_cols)):
        merged_idx = pd.concat(prepared, axis=1, sort=False)
        merged_idx = merged_idx.reindex(columns=present_years_sorted)
        return _column_frame_from_indexed(cast(pd.DataFrame, merged_idx))
    merged_idx = prepared[0]
    for next_idx_raw in prepared[1:]:
        next_cols = list(next_idx_raw.columns)
        overlap_cols = [col for col in next_cols if col in merged_idx.columns]
        add_cols = [col for col in next_cols if col not in merged_idx.columns]
        if add_cols:
            merged_idx = merged_idx.reindex(columns=[*merged_idx.columns, *add_cols])
        common_idx = merged_idx.index.intersection(next_idx_raw.index)
        if len(common_idx) > 0:
            if overlap_cols:
                left = merged_idx.loc[common_idx, overlap_cols]
                right = next_idx_raw.loc[common_idx, overlap_cols]
                both_present = left.notna() & right.notna()
                both_present_np = both_present.to_numpy(dtype=bool)
                conflict_positions: list[tuple[int, int]] = []
                if bool(both_present_np.any()):
                    left_np = left.to_numpy(dtype=object, copy=False)
                    right_np = right.to_numpy(dtype=object, copy=False)
                    row_idx, col_idx = np.where(both_present_np)
                    for r_pos, c_pos in zip(row_idx.tolist(), col_idx.tolist()):
                        if not _safe_cell_equal(left_np[r_pos, c_pos], right_np[r_pos, c_pos]):
                            conflict_positions.append((r_pos, c_pos))
                if conflict_positions:
                    sample = []
                    for r_pos, c_pos in conflict_positions[:5]:
                        idx_label = left.index[r_pos]
                        col_label = left.columns[c_pos]
                        sample.append(f"{idx_label}::{col_label}")
                    raise ValueError(
                        f"{where}: conflicting duplicate values for output primary key "
                        f"while merging frames. sample={sample}"
                    )
                fill_mask = left.isna() & right.notna()
                if bool(fill_mask.to_numpy().any()):
                    updated = left.where(~fill_mask, right)
                    for col in updated.columns:
                        updated[col] = pd.to_numeric(updated[col], errors="raise")
                    merged_idx.loc[common_idx, overlap_cols] = updated.to_numpy()
            if add_cols:
                add_frame = next_idx_raw.loc[common_idx, add_cols]
                add_values = add_frame.copy()
                for col in add_cols:
                    add_values[col] = pd.to_numeric(add_values[col], errors="raise")
                merged_idx.loc[common_idx, add_cols] = add_values.to_numpy()
        new_idx = next_idx_raw.index.difference(merged_idx.index)
        if len(new_idx) > 0:
            to_add = next_idx_raw.loc[new_idx].reindex(columns=merged_idx.columns)
            merged_idx = pd.concat([merged_idx, to_add], axis=0)
    merged_idx = merged_idx.reindex(columns=present_years_sorted)
    return _column_frame_from_indexed(cast(pd.DataFrame, merged_idx))

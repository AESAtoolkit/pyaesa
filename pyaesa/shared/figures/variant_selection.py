"""Shared variant selection helpers for figure readability."""

from dataclasses import dataclass

import pandas as pd

from pyaesa.shared.figures.contracts import DETERMINISTIC_PROSPECTIVE_COLUMNS
from pyaesa.shared.figures.reference_year_variant_coverage import (
    eligible_full_window_reference_tokens,
    filter_full_window_reference_years,
)
from pyaesa.shared.tabular.scalars import display_scalar

_NON_VARIANT_COLUMNS = {"year", "value", *DETERMINISTIC_PROSPECTIVE_COLUMNS}


@dataclass(frozen=True)
class VariantCompression:
    """One compressed variant column retained in the main figure."""

    column: str
    kept_values: tuple[object, object]
    filtered: bool = False
    base_key: tuple[str, ...] = ()


def compression_base_columns(
    frame: pd.DataFrame,
    *,
    variant_columns: tuple[str, ...],
    ignored_columns: set[str],
) -> list[str]:
    """Return the non-variant columns that actually distinguish families."""
    candidates = [
        column for column in frame.columns if column not in {*ignored_columns, *variant_columns}
    ]
    return [
        column
        for column in candidates
        if pd.Series(frame.loc[:, column], copy=False).dropna().drop_duplicates().size > 1
    ]


def base_group_key_from_row(row: pd.Series, *, base_columns: list[str]) -> tuple[str, ...]:
    """Return one stable base family key from the selected base columns."""
    return tuple(_stable_key(row.get(column)) for column in base_columns)


def split_variant_frames(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    variant_columns: tuple[str, ...] = ("reference_year", "l2_reuse_year"),
    value_column: str = "value",
    ignored_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, tuple[VariantCompression, ...]]:
    """Return the main figure rows plus any compressed variant metadata."""
    if frame.empty:
        return frame.copy(), tuple()
    work = frame.copy()
    if "year" in work.columns:
        years = pd.Series(
            pd.to_numeric(pd.Series(work["year"], copy=False), errors="raise"),
            copy=False,
        ).astype(int)
        work = work.loc[years.isin([int(year) for year in requested_years])].copy()
    if work.empty or value_column not in work.columns:
        return frame.copy(), tuple()
    numeric_value = pd.to_numeric(pd.Series(work[value_column], copy=False), errors="raise")
    work[value_column] = pd.Series(numeric_value, copy=False)
    # ``series_label`` is a display artifact and can embed variant selectors
    # such as ref year or l2_reuse_year. It must not define the compression family.
    group_columns = compression_base_columns(
        frame,
        variant_columns=variant_columns,
        ignored_columns={
            *_NON_VARIANT_COLUMNS,
            value_column,
            "series_label",
            *(ignored_columns or set()),
        },
    )
    work = work.copy()
    frame = frame.copy()
    work["__compression_group_key"] = _group_keys(work, group_columns)
    frame["__compression_group_key"] = _group_keys(frame, group_columns)
    filtered_groups: list[pd.DataFrame] = []
    compressions: list[VariantCompression] = []
    for group_key, group_work in work.groupby("__compression_group_key", sort=True, dropna=False):
        group_frame = frame.loc[frame["__compression_group_key"] == group_key].copy()
        kept_group, group_compressions = _compress_group(
            frame=group_frame,
            work=group_work.copy(),
            variant_columns=variant_columns,
            value_column=value_column,
            group_columns=group_columns,
            requested_years=[int(year) for year in requested_years],
        )
        filtered_groups.append(kept_group)
        compressions.extend(group_compressions)
    main_frame = pd.concat(filtered_groups, ignore_index=True).drop(
        columns="__compression_group_key"
    )
    return main_frame.reset_index(drop=True), tuple(compressions)


def variant_footer_note(
    compressions: tuple[VariantCompression, ...],
    *,
    average_over_years: bool,
    display_aliases: dict[str, str] | None = None,
) -> str | None:
    """Return one generic footer note for retained min and max variants."""
    if not compressions:
        return None
    active_columns = {compression.column for compression in compressions}
    variant_scope = _variant_scope_text(active_columns, display_aliases=display_aliases)
    if variant_scope is None:
        return None
    first_line = (
        f"For a given allocation method across {variant_scope}: "
        "dotted = average max over years, plain = average min over years."
        if average_over_years
        else f"For a given allocation method across {variant_scope}: dotted = max, plain = min."
    )
    if "l2_reuse_year" not in active_columns:
        return first_line
    reuse_name = _variant_display_name("l2_reuse_year", display_aliases=display_aliases)
    return f"{first_line}\n{reuse_name} affects only the L2 prospective allocation weighting."


def _variant_scope_text(
    active_columns: set[str],
    *,
    display_aliases: dict[str, str] | None = None,
) -> str | None:
    """Return one user-facing phrase for the active variant dimensions."""
    has_reference_year = "reference_year" in active_columns
    has_l2_reuse_year = "l2_reuse_year" in active_columns
    reference_name = _variant_display_name("reference_year", display_aliases=display_aliases)
    reuse_name = _variant_display_name("l2_reuse_year", display_aliases=display_aliases)
    if has_reference_year and has_l2_reuse_year:
        return f"{reference_name} and {reuse_name}"
    if has_reference_year:
        return reference_name
    if has_l2_reuse_year:
        return reuse_name
    return None


def _variant_display_name(column: str, *, display_aliases: dict[str, str] | None) -> str:
    """Return one visible variant axis name."""
    default_names = {"reference_year": "ref year", "l2_reuse_year": "l2_reuse_year"}
    if display_aliases is not None and column in display_aliases:
        return str(display_aliases[column])
    return default_names.get(column, column)


def _compress_group(
    *,
    frame: pd.DataFrame,
    work: pd.DataFrame,
    variant_columns: tuple[str, ...],
    value_column: str,
    group_columns: list[str],
    requested_years: list[int],
) -> tuple[pd.DataFrame, tuple[VariantCompression, ...]]:
    """Compress one family group by its active variant columns."""
    active_variant_columns = tuple(
        column
        for column in variant_columns
        if column in work.columns and bool(pd.Series(work[column], copy=False).notna().any())
    )
    if not active_variant_columns:
        return frame, tuple()
    if "reference_year" in active_variant_columns:
        eligible_tokens = eligible_full_window_reference_tokens(
            frame=work,
            required_years=requested_years,
            group_columns=[],
        )
        frame = filter_full_window_reference_years(frame, eligible_tokens=eligible_tokens)
        work = filter_full_window_reference_years(work, eligible_tokens=eligible_tokens)
        active_variant_columns = tuple(
            column
            for column in active_variant_columns
            if column in work.columns and bool(pd.Series(work[column], copy=False).notna().any())
        )
    varying_variant_columns = tuple(
        column
        for column in active_variant_columns
        if pd.Series(work.loc[work[column].notna(), column], copy=False).drop_duplicates().size > 1
    )
    if not varying_variant_columns:
        return frame, tuple()
    base_key = _base_key(frame, group_columns)
    if len(varying_variant_columns) == 1:
        return _compress_single_variant_group(
            frame=frame,
            work=work,
            variant_column=varying_variant_columns[0],
            value_column=value_column,
            base_key=base_key,
        )
    return _compress_multi_variant_group(
        frame=frame,
        work=work,
        variant_columns=varying_variant_columns,
        value_column=value_column,
        base_key=base_key,
    )


def _compress_single_variant_group(
    *,
    frame: pd.DataFrame,
    work: pd.DataFrame,
    variant_column: str,
    value_column: str,
    base_key: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[VariantCompression, ...]]:
    """Compress one group with one active variant column."""
    non_null = work.loc[work[variant_column].notna(), [variant_column, value_column]].copy()
    unique_values = pd.Series(non_null[variant_column], copy=False).drop_duplicates().tolist()
    averages = (
        non_null.groupby(variant_column, dropna=False, sort=True)[value_column]
        .mean()
        .sort_values(kind="mergesort")
    )
    kept_values = (averages.index[0], averages.index[-1])
    filtered = len(unique_values) > 2
    main_frame = frame
    if filtered:
        main_frame = frame.loc[
            frame[variant_column].isna() | frame[variant_column].isin(list(kept_values))
        ].copy()
    return main_frame, (
        VariantCompression(
            column=variant_column,
            kept_values=kept_values,
            filtered=filtered,
            base_key=base_key,
        ),
    )


def _compress_multi_variant_group(
    *,
    frame: pd.DataFrame,
    work: pd.DataFrame,
    variant_columns: tuple[str, ...],
    value_column: str,
    base_key: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[VariantCompression, ...]]:
    """Compress one group from true combination level min and max."""
    combo_frame = work.loc[:, [*variant_columns, value_column]].copy()
    complete_combo_frame = combo_frame.loc[
        combo_frame.loc[:, list(variant_columns)].notna().all(axis=1)
    ].copy()
    combo_averages = (
        complete_combo_frame.groupby(list(variant_columns), dropna=False, sort=True)[value_column]
        .mean()
        .sort_values(kind="mergesort")
    )
    min_combo = combo_averages.index[0]
    max_combo = combo_averages.index[-1]
    keep_combinations = {tuple(min_combo), tuple(max_combo)}
    complete_frame_mask = frame.loc[:, list(variant_columns)].notna().all(axis=1)
    kept_complete_mask = (
        frame.loc[complete_frame_mask, list(variant_columns)]
        .apply(tuple, axis=1)
        .isin(keep_combinations)
    )
    filtered = len(combo_averages) > 2
    filtered_frame = frame
    if filtered:
        filtered_frame = frame.loc[(~complete_frame_mask) | kept_complete_mask].copy()
    compressions = tuple(
        VariantCompression(
            column=column,
            kept_values=(min_combo[index], max_combo[index]),
            filtered=filtered,
            base_key=base_key,
        )
        for index, column in enumerate(variant_columns)
    )
    return filtered_frame, compressions


def _base_key(frame: pd.DataFrame, group_columns: list[str]) -> tuple[str, ...]:
    """Return one stable base key shared with line style grouping."""
    first_row = pd.Series(frame.iloc[0], copy=False)
    return base_group_key_from_row(first_row, base_columns=group_columns)


def _group_keys(frame: pd.DataFrame, group_columns: list[str]) -> pd.Series:
    """Return stable family keys for grouping compression decisions."""
    if not group_columns:
        return pd.Series(["__all__"] * len(frame), index=frame.index, dtype="object")
    return frame.loc[:, group_columns].apply(
        lambda row: "|".join(_stable_key(row[column]) for column in group_columns),
        axis=1,
    )


def _stable_key(value: object) -> str:
    """Return one deterministic grouping key part."""
    text = display_scalar(value)
    if text is None:
        return "missing"
    stable = str(text).strip()
    return stable or "missing"

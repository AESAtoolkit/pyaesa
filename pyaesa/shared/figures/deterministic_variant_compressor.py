"""Shared deterministic figure variant compression."""

from dataclasses import dataclass
from typing import cast

import pandas as pd
from pandas.api.types import is_numeric_dtype

from pyaesa.shared.figures.reference_year_variant_coverage import (
    eligible_full_window_reference_tokens,
    filter_full_window_reference_years,
)
from pyaesa.shared.tabular.scalars import display_scalar

VALUE_COLUMN = "value"
YEAR_COLUMN = "year"
VARIANT_COLUMNS = ("reference_year", "l2_reuse_year")
ROLE_COLUMN = "__variant_role"
MIN_ROLE = "min"
MAX_ROLE = "max"
IDENTITY_COLUMNS = (
    "__method",
    "__figure_ssp_scope",
    "l1_l2_method",
    "l1_method",
    "l2_method",
    "r_c",
    "s_p",
    "r_p",
    "r_f",
    "lcia_method",
    "impact",
)


@dataclass(frozen=True)
class _VariantColumnView:
    tokens: pd.Series
    missing: pd.Series


def compress_variants(
    frame: pd.DataFrame,
    *,
    identity_columns: tuple[str, ...] = IDENTITY_COLUMNS,
    score_column: str = VALUE_COLUMN,
) -> pd.DataFrame:
    """Keep scope level min and max reference year or L2 reuse year variants."""
    variant_columns = [column for column in VARIANT_COLUMNS if column in frame.columns]
    if not variant_columns:
        return frame.copy()
    return _compress_group(
        frame,
        variant_columns=variant_columns,
        identity_columns=identity_columns,
        score_column=score_column,
    )


def _compress_group(
    frame: pd.DataFrame,
    *,
    variant_columns: list[str],
    identity_columns: tuple[str, ...],
    score_column: str,
) -> pd.DataFrame:
    active_columns = [
        column
        for column in variant_columns
        if column in frame.columns and _has_display_value(cast(pd.Series, frame[column]))
    ]
    if not active_columns:
        return frame.copy()
    frame = _with_full_window_reference_years(
        frame=frame,
        active_columns=active_columns,
        identity_columns=identity_columns,
    )
    active_columns = _active_variant_columns(frame, active_columns=active_columns)
    complete = frame.loc[_complete_variant_mask(frame, active_columns)].copy()
    scores = _variant_scores(
        complete,
        active_columns=active_columns,
        identity_columns=identity_columns,
        score_column=score_column,
    )
    if len(scores) == 1:
        return _selected_variant_rows(
            frame=frame,
            active_columns=active_columns,
            selected=[(None, scores.index[0])],
            identity_columns=identity_columns,
        )
    ordered_scores = scores.sort_values(kind="mergesort")
    selected: list[tuple[str | None, object]] = [
        (MIN_ROLE, cast(object, ordered_scores.index[0])),
        (MAX_ROLE, cast(object, ordered_scores.index[-1])),
    ]
    return _selected_variant_rows(
        frame=frame,
        active_columns=active_columns,
        selected=selected,
        identity_columns=identity_columns,
    )


def _selected_variant_rows(
    *,
    frame: pd.DataFrame,
    active_columns: list[str],
    selected: list[tuple[str | None, object]],
    identity_columns: tuple[str, ...],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    group_columns = [column for column in identity_columns if column in frame.columns]
    grouped = (
        frame.groupby(group_columns, dropna=False, sort=False) if group_columns else [(None, frame)]
    )
    for _key, group in grouped:
        group_active = _active_variant_columns(group, active_columns=active_columns)
        if not group_active:
            parts.append(group.copy())
            continue
        projected = _projected_selected_variants(
            active_columns=active_columns,
            group_active=group_active,
            selected=selected,
        )
        views = _variant_column_views(group, active_columns=group_active)
        if len(projected) == 1:
            _role, variant_key = projected[0]
            parts.append(
                _variant_frame_with_prefix(
                    frame=group,
                    active_columns=group_active,
                    variant_key=variant_key,
                    views=views,
                )
            )
            continue
        for role, variant_key in projected:
            variant_frame = _variant_frame_with_prefix(
                frame=group,
                active_columns=group_active,
                variant_key=variant_key,
                views=views,
            )
            variant_frame[ROLE_COLUMN] = role
            parts.append(variant_frame)
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def _active_variant_columns(frame: pd.DataFrame, *, active_columns: list[str]) -> list[str]:
    return [
        column
        for column in active_columns
        if column in frame.columns and _has_display_value(cast(pd.Series, frame[column]))
    ]


def _with_full_window_reference_years(
    *,
    frame: pd.DataFrame,
    active_columns: list[str],
    identity_columns: tuple[str, ...],
) -> pd.DataFrame:
    if "reference_year" not in active_columns:
        return frame
    years = pd.Series(pd.to_numeric(frame[YEAR_COLUMN], errors="raise"), copy=False).astype(int)
    group_columns = [column for column in identity_columns if column in frame.columns]
    eligible_tokens = eligible_full_window_reference_tokens(
        frame=frame,
        required_years=sorted({int(year) for year in years.tolist()}),
        group_columns=group_columns,
    )
    return filter_full_window_reference_years(frame, eligible_tokens=eligible_tokens)


def _projected_selected_variants(
    *,
    active_columns: list[str],
    group_active: list[str],
    selected: list[tuple[str | None, object]],
) -> list[tuple[str | None, object]]:
    active_positions = {column: index for index, column in enumerate(active_columns)}
    projected: list[tuple[str | None, object]] = []
    seen: set[tuple[str, ...]] = set()
    for role, variant_key in selected:
        key_tuple = _as_tuple(variant_key)
        values = tuple(key_tuple[active_positions[column]] for column in group_active)
        group_key = values[0] if len(values) == 1 else values
        token = _variant_key_token(group_key)
        if token in seen:
            continue
        seen.add(token)
        projected.append((role, group_key))
    return projected


def _variant_scores(
    frame: pd.DataFrame,
    *,
    active_columns: list[str],
    identity_columns: tuple[str, ...],
    score_column: str,
) -> pd.Series:
    raw_scores = cast(
        pd.Series,
        frame.groupby(active_columns, dropna=False)[score_column].mean(),
    )
    if len(raw_scores) <= 1:
        return raw_scores
    comparison_columns = [
        column
        for column in dict.fromkeys([*identity_columns, YEAR_COLUMN])
        if column in frame.columns and column not in active_columns
    ]
    values = cast(
        pd.DataFrame,
        cast(
            pd.Series,
            frame.groupby([*comparison_columns, *active_columns], dropna=False)[
                score_column
            ].mean(),
        ).reset_index(),
    )
    numeric = pd.Series(pd.to_numeric(values[score_column], errors="raise"), copy=False)
    grouped = values.groupby(comparison_columns, dropna=False, sort=False)[score_column]
    low = pd.Series(grouped.transform("min"), copy=False).astype(float)
    high = pd.Series(grouped.transform("max"), copy=False).astype(float)
    spread = high - low
    weight = spread / (high.abs() + low.abs())
    valid = spread.gt(0.0)
    if not bool(valid.any()):
        return raw_scores
    weighted = values.loc[valid, [*active_columns]].copy()
    weighted["__score_sum"] = ((numeric - low) / spread * weight).loc[valid].to_numpy()
    weighted["__weight_sum"] = weight.loc[valid].to_numpy()
    totals = weighted.groupby(active_columns, dropna=False, sort=True)[
        ["__score_sum", "__weight_sum"]
    ].sum()
    scores = cast(pd.Series, totals["__score_sum"] / totals["__weight_sum"])
    scores = scores.reindex(raw_scores.index).dropna()
    return scores


def _variant_key_token(value: object) -> tuple[str, ...]:
    tokens: list[str] = []
    for item in _as_tuple(value):
        tokens.append(str(display_scalar(item)))
    return tuple(tokens)


def _variant_frame_with_prefix(
    *,
    frame: pd.DataFrame,
    active_columns: list[str],
    variant_key: object,
    views: dict[str, _VariantColumnView],
) -> pd.DataFrame:
    key_tuple = _as_tuple(variant_key)
    variant_frame = _variant_slice(
        frame,
        active_columns,
        variant_key,
        views=views,
    )
    first_year = int(pd.Series(pd.to_numeric(variant_frame[YEAR_COLUMN]), copy=False).min())
    prefix = _compatible_prefix_rows(
        frame=frame,
        active_columns=active_columns,
        variant_key=key_tuple,
        before_year=first_year,
        views=views,
    )
    if prefix.empty:
        return variant_frame
    for column, value in zip(active_columns, key_tuple, strict=True):
        prefix[column] = value
    return pd.concat([prefix, variant_frame], ignore_index=True)


def _variant_slice(
    frame: pd.DataFrame,
    variant_columns: list[str],
    variant_key: object,
    *,
    views: dict[str, _VariantColumnView],
) -> pd.DataFrame:
    mask = pd.Series([True] * len(frame), index=frame.index)
    for column, value in zip(variant_columns, _as_tuple(variant_key), strict=True):
        mask = mask & views[column].tokens.eq(display_scalar(value))
    return frame.loc[mask].copy()


def _compatible_prefix_rows(
    *,
    frame: pd.DataFrame,
    active_columns: list[str],
    variant_key: tuple[object, ...],
    before_year: int,
    views: dict[str, _VariantColumnView],
) -> pd.DataFrame:
    years = pd.Series(pd.to_numeric(frame[YEAR_COLUMN], errors="raise"), copy=False).astype(int)
    mask = years < int(before_year)
    missing_any_variant = pd.Series([False] * len(frame), index=frame.index)
    for column, value in zip(active_columns, variant_key, strict=True):
        column_missing = views[column].missing
        missing_any_variant = missing_any_variant | column_missing
        same_value = views[column].tokens.eq(display_scalar(value))
        mask = mask & (column_missing | same_value)
    return frame.loc[mask & missing_any_variant].copy()


def _variant_column_views(
    frame: pd.DataFrame,
    *,
    active_columns: list[str],
) -> dict[str, _VariantColumnView]:
    out: dict[str, _VariantColumnView] = {}
    for column in active_columns:
        series = cast(pd.Series, frame[column])
        missing = _display_missing_mask(series)
        out[column] = _VariantColumnView(
            tokens=_display_token_series(series, missing=missing),
            missing=missing,
        )
    return out


def _as_tuple(value: object) -> tuple[object, ...]:
    return value if isinstance(value, tuple) else (value,)


def _complete_variant_mask(frame: pd.DataFrame, active_columns: list[str]) -> pd.Series:
    complete = pd.Series([True] * len(frame), index=frame.index)
    for column in active_columns:
        complete = complete & ~_display_missing_mask(cast(pd.Series, frame[column]))
    return complete


def _has_display_value(series: pd.Series) -> bool:
    return bool((~_display_missing_mask(series)).any())


def _display_missing_mask(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series.dtype):
        return cast(pd.Series, series.isna())
    text = series.astype("string").str.strip()
    return cast(
        pd.Series,
        series.isna() | text.eq("") | text.str.lower().isin({"nan", "none", "nat"}),
    )


def _display_token_series(series: pd.Series, *, missing: pd.Series | None = None) -> pd.Series:
    if not is_numeric_dtype(series.dtype):
        return cast(pd.Series, series.map(display_scalar))
    text = series.astype("string").str.replace(r"^([+-]?\d+)\.0+$", r"\1", regex=True)
    out = text.astype("object")
    out.loc[_display_missing_mask(series) if missing is None else missing] = None
    return cast(pd.Series, out)

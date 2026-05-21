"""Deterministic transition panel and series payload helpers."""

from dataclasses import dataclass

import pandas as pd

from pyaesa.shared.figures.contracts import (
    DETERMINISTIC_PROSPECTIVE_COLUMNS,
    deterministic_prospective_series,
)
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_PROSPECTIVE_TIME_ROUTE_VALUES,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from .deterministic_legends_methods import legend_group_from_row
from .multi_year_transitions import (
    expand_historical_rows_for_prospective_series,
    markers_from_frame,
)
from .series_labels import require_series_label
from .variant_selection import (
    VariantCompression,
    base_group_key_from_row,
    compression_base_columns,
)
from pyaesa.shared.tabular.scalars import is_display_missing

_MARKER_COLUMNS = {
    "__transition_marker_year",
    "__transition_marker_label",
    "__transition_marker_color",
}
_DETERMINISTIC_TRANSITION_METADATA_COLUMNS = {
    "asocc_ssp_start_year",
    "lca_ssp_start_year",
}
_DETERMINISTIC_TRANSITION_SCENARIO_COLUMN = "__deterministic_prospective_scenario"


@dataclass(frozen=True)
class VariantLineSpec:
    """Resolved style contract for one deterministic series."""

    base_color_key: tuple[str, ...]
    line_style: str
    show_in_legend: bool
    prospective_only: bool = False


def _excluded_transition_grouping_columns() -> set[str]:
    """Return non-identity columns excluded from deterministic transition grouping."""
    return {
        *DETERMINISTIC_PROSPECTIVE_COLUMNS,
        *_DETERMINISTIC_TRANSITION_METADATA_COLUMNS,
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
        _DETERMINISTIC_TRANSITION_SCENARIO_COLUMN,
    }


def prepare_transition_frame(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    marker_label: str,
    marker_color: str,
    transition_grouping_skip_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Normalize one long deterministic frame for historical and SSP rendering."""
    work = frame.copy()
    numeric_years = pd.to_numeric(pd.Series(work.loc[:, "year"], copy=False), errors="raise")
    work["year"] = pd.Series(numeric_years, copy=False).astype(int)
    requested = sorted({int(year) for year in requested_years})
    work = work.loc[pd.Series(work.loc[:, "year"], copy=False).isin(requested)].copy()
    work["value"] = pd.to_numeric(work["value"], errors="raise")
    work[_DETERMINISTIC_TRANSITION_SCENARIO_COLUMN] = _transition_scope_series(work)
    grouping_columns = [
        column
        for column in work.columns
        if column
        not in {
            "year",
            "value",
            *_excluded_transition_grouping_columns(),
            *_MARKER_COLUMNS,
            *(transition_grouping_skip_columns or set()),
        }
    ]
    expanded = expand_historical_rows_for_prospective_series(
        work,
        grouping_columns=grouping_columns,
        propagate_columns=sorted(transition_grouping_skip_columns or set()),
        prospective_column=_DETERMINISTIC_TRANSITION_SCENARIO_COLUMN,
        marker_label=marker_label,
        marker_color=marker_color,
    )
    transition_scope = pd.Series(
        expanded.loc[:, _DETERMINISTIC_TRANSITION_SCENARIO_COLUMN],
        copy=False,
    )
    for column in DETERMINISTIC_PROSPECTIVE_COLUMNS:
        if column not in expanded.columns:
            continue
        expanded[column] = pd.Series(transition_scope, copy=False).astype("object")
    return expanded.drop(
        columns=[
            column
            for column in (_DETERMINISTIC_TRANSITION_SCENARIO_COLUMN,)
            if column in expanded.columns
        ]
    )


def _transition_scope_series(frame: pd.DataFrame) -> pd.Series:
    """Return row-level transition scope from SSP scenario or deterministic route."""
    scenario = deterministic_prospective_series(frame)
    if ASOCC_TIME_ROUTE_PUBLIC_COLUMN not in frame.columns:
        return scenario
    route = pd.Series(frame.loc[:, ASOCC_TIME_ROUTE_PUBLIC_COLUMN], copy=False)
    values = [
        scenario_value if scenario_value is not None else _prospective_route(route_value)
        for scenario_value, route_value in zip(scenario.tolist(), route.tolist(), strict=True)
    ]
    return pd.Series(values, index=frame.index, dtype="object")


def _prospective_route(value: object) -> str | None:
    text = str(value).strip()
    return text if text in ASOCC_PROSPECTIVE_TIME_ROUTE_VALUES else None


def panel_groups(
    frame: pd.DataFrame,
    *,
    panel_column: str | None,
) -> list[tuple[str, pd.DataFrame]]:
    """Return deterministic panel slices."""
    if panel_column is None:
        return [("value", frame.copy())]
    return [
        (str(panel_value), subset.copy())
        for panel_value, subset in frame.groupby(panel_column, dropna=False, sort=True)
    ]


def series_payloads(
    frame: pd.DataFrame,
    *,
    requested_years: list[int],
    panel_column: str | None,
    value_scale: float = 1.0,
    skip_columns: set[str] | None = None,
) -> dict[tuple[str, ...], tuple[str, list[int], list[float], str]]:
    """Return labeled deterministic series payloads for one prepared panel frame."""
    ignored = {
        "year",
        "value",
        *_excluded_transition_grouping_columns(),
        "series_label",
        *_MARKER_COLUMNS,
        *(skip_columns or set()),
    }
    if panel_column is not None:
        ignored.add(panel_column)
    series_columns = [column for column in frame.columns if column not in ignored]
    groups = (
        [(None, frame)]
        if not series_columns
        else list(frame.groupby(series_columns, dropna=False, sort=True))
    )
    payloads: dict[tuple[str, ...], tuple[str, list[int], list[float], str]] = {}
    for index, (_group_key, group) in enumerate(groups, start=1):
        ordered = group.sort_values("year", kind="stable")
        years = [int(year) for year in ordered["year"].tolist()]
        first_row = pd.Series(ordered.iloc[0], copy=False)
        label = require_series_label(
            first_row,
            context="Multi-year deterministic figure rendering",
        )
        legend_group = legend_group_from_row(first_row)
        values = [float(value_scale) * float(value) for value in ordered["value"].tolist()]
        key = (
            tuple(_stable_key(first_row.get(column)) for column in series_columns)
            if series_columns
            else (f"__series_{index}",)
        )
        payloads[key] = (label, years, values, legend_group)
    return payloads


def series_transition_years(
    frame: pd.DataFrame,
    *,
    panel_column: str | None,
    skip_columns: set[str] | None = None,
) -> dict[tuple[str, ...], int | None]:
    """Return transition year per deterministic series key when present."""
    ignored = {
        "year",
        "value",
        *_excluded_transition_grouping_columns(),
        *_MARKER_COLUMNS,
        *(skip_columns or set()),
    }
    if panel_column is not None:
        ignored.add(panel_column)
    series_columns = [column for column in frame.columns if column not in ignored]
    groups = (
        [(None, frame)]
        if not series_columns
        else list(frame.groupby(series_columns, dropna=False, sort=True))
    )
    transition_years: dict[tuple[str, ...], int | None] = {}
    for index, (_group_key, group) in enumerate(groups, start=1):
        first_row = pd.Series(group.iloc[0], copy=False)
        key = (
            tuple(_stable_key(first_row.get(column)) for column in series_columns)
            if series_columns
            else (f"__series_{index}",)
        )
        marker_year_series = pd.Series(group.loc[:, "__transition_marker_year"], copy=False)
        numeric_marker_years = pd.Series(
            pd.to_numeric(marker_year_series, errors="coerce"),
            copy=False,
        ).dropna()
        transition_years[key] = (
            int(numeric_marker_years.min()) if not numeric_marker_years.empty else None
        )
    return transition_years


def panel_markers(frame: pd.DataFrame) -> list:
    """Return panel-level transition markers from one prepared frame."""
    return markers_from_frame(frame)


def series_line_specs(
    frame: pd.DataFrame,
    *,
    panel_column: str | None,
    compressions: tuple[VariantCompression, ...],
    skip_columns: set[str] | None = None,
) -> dict[tuple[str, ...], VariantLineSpec]:
    """Return variant styling payloads keyed by deterministic series key."""
    ignored = {
        "year",
        "value",
        *_excluded_transition_grouping_columns(),
        *_MARKER_COLUMNS,
        *(skip_columns or set()),
    }
    if panel_column is not None:
        ignored.add(panel_column)
    series_columns = [col for col in frame.columns if col not in ignored]
    variant_columns = tuple(
        column
        for column in ("reference_year", "l2_reuse_year")
        if column in frame.columns and bool(pd.Series(frame[column], copy=False).notna().any())
    )
    base_columns = compression_base_columns(
        frame,
        variant_columns=variant_columns,
        ignored_columns=ignored,
    )
    compressions_by_base: dict[tuple[str, ...], dict[str, object]] = {}
    for compression in compressions:
        compressions_by_base.setdefault(compression.base_key, {})[compression.column] = (
            compression.kept_values[1]
        )
    variant_col_names = {compression.column for compression in compressions}
    variant_indices = [i for i, col in enumerate(series_columns) if col in variant_col_names]
    groups_iter = (
        [(None, frame)]
        if not series_columns
        else list(frame.groupby(series_columns, dropna=False, sort=True))
    )
    all_keys: list[tuple[str, ...]] = []
    for index, (_group_key, group) in enumerate(groups_iter, start=1):
        first_row = pd.Series(group.iloc[0], copy=False)
        key = (
            tuple(_stable_key(first_row.get(col)) for col in series_columns)
            if series_columns
            else (f"__series_{index}",)
        )
        all_keys.append(key)
    base_to_keys: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for key in all_keys:
        key_row = pd.Series(
            {column: key[index] for index, column in enumerate(series_columns)},
            copy=False,
        )
        base_key = base_group_key_from_row(key_row, base_columns=base_columns)
        base_to_keys.setdefault(base_key, []).append(key)
    result: dict[tuple[str, ...], VariantLineSpec] = {}
    for base_key, keys in base_to_keys.items():
        variant_max_values = compressions_by_base.get(base_key, {})
        has_reference_variant = "reference_year" in variant_max_values
        has_reuse_variant = "l2_reuse_year" in variant_max_values
        if len(keys) <= 1 or not variant_indices or not variant_max_values:
            for key in keys:
                result[key] = VariantLineSpec(
                    base_color_key=base_key,
                    line_style="solid",
                    show_in_legend=True,
                )
            continue
        for key in keys:
            is_max = all(
                key[i] == _stable_key(variant_max_values.get(series_columns[i]))
                for i in variant_indices
                if series_columns[i] in variant_max_values
            )
            result[key] = VariantLineSpec(
                base_color_key=base_key,
                line_style="dotted" if is_max else "solid",
                show_in_legend=True,
                prospective_only=bool(is_max and has_reuse_variant and not has_reference_variant),
            )
    return result


def _stable_key(value: object) -> str:
    if is_display_missing(value):
        return "missing"
    text = str(value).strip()
    return text or "missing"

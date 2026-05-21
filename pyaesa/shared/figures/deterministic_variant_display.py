"""Shared display policy for deterministic retained variant figures."""

import re

import pandas as pd

from pyaesa.shared.figures.deterministic_variant_compressor import (
    IDENTITY_COLUMNS,
    MAX_ROLE,
    MIN_ROLE,
    ROLE_COLUMN,
    VARIANT_COLUMNS,
    YEAR_COLUMN,
)
from pyaesa.shared.figures.scope_values import visible_scope_values
from pyaesa.shared.tabular.scalars import is_display_missing

_INTEGER_FLOAT_TEXT = re.compile(r"^[+-]?\d+\.0+$")


def variant_styles(frame: pd.DataFrame) -> list[str]:
    """Return dotted styles for max retained variants and solid styles otherwise."""
    if ROLE_COLUMN not in frame.columns:
        return ["solid"] * len(frame)
    return [
        "dotted" if str(role).strip() == MAX_ROLE else "solid"
        for role in frame[ROLE_COLUMN].tolist()
    ]


def base_variant_groups(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Return one group per visible deterministic base identity."""
    columns = [column for column in IDENTITY_COLUMNS if column in frame.columns]
    return [group for _key, group in frame.groupby(columns, dropna=False, sort=False)]


def has_complete_variant_roles(group: pd.DataFrame) -> bool:
    """Return whether a grouped series has lower and upper retained variants."""
    if ROLE_COLUMN not in group.columns:
        return False
    roles = {
        str(value).strip()
        for value in group[ROLE_COLUMN].tolist()
        if not is_display_missing(value) and str(value).strip()
    }
    return roles == {MIN_ROLE, MAX_ROLE}


def variant_role_row(group: pd.DataFrame, *, role: str) -> pd.Series:
    """Return the first row for one retained variant role."""
    scoped = group.loc[group[ROLE_COLUMN].astype(str).eq(str(role))]
    return pd.Series(scoped.iloc[0], copy=False)


def variant_note(
    frame: pd.DataFrame,
    *,
    single_year: bool = False,
    geometry_override: str | None = None,
) -> str | None:
    """Return the visible retained variant note for one deterministic figure scope."""
    active_columns = [
        column
        for column in VARIANT_COLUMNS
        if column in frame.columns
        and bool(frame[column].map(lambda value: not is_display_missing(value)).any())
    ]
    if not active_columns:
        return None
    if ROLE_COLUMN not in frame.columns:
        selected_variant = _single_variant_text(frame, active_columns)
        if selected_variant is None:
            return None
        first_line = f"{selected_variant}."
    elif _scope_has_variant_roles(frame):
        basis = _variant_compression_basis(frame, single_year=single_year)
        basis_suffix = f" {basis}" if basis else ""
        min_key = _scope_role_variant_key(frame, role=MIN_ROLE, variant_columns=active_columns)
        max_key = _scope_role_variant_key(frame, role=MAX_ROLE, variant_columns=active_columns)
        geometry = str(geometry_override).strip() if geometry_override is not None else ""
        if not geometry:
            geometry = (
                "Solid bar = retained lower combination; "
                "dotted whisker and cap = retained upper combination."
                if single_year
                else "Plain = retained lower combination; dotted = retained upper combination."
            )
        first_line = "\n".join(
            [
                "Variant compression: lower and upper retained combinations minimize and "
                "maximize the average relative position between minimum and maximum values"
                f"{basis_suffix}.",
                geometry,
                f"Lower retained combination: {variant_combo_text(active_columns, min_key)}.",
                f"Upper retained combination: {variant_combo_text(active_columns, max_key)}.",
            ]
        )
    else:
        return None
    axis_note = _variant_axis_application_note(frame, active_columns=active_columns)
    if axis_note is not None:
        first_line = f"{first_line}\n{axis_note}"
    if "l2_reuse_year" not in active_columns:
        return first_line
    return (
        f"{first_line}\n"
        f"{variant_display_name('l2_reuse_year')} affects only the L2 in L1 prospective "
        "allocation weighting."
    )


def variant_combo_text(columns: list[str], values: tuple[object, ...]) -> str:
    """Return visible variant year metadata using integer year formatting."""
    return ", ".join(
        f"{variant_display_name(column)}={format_year_scalar(value)}"
        for column, value in zip(columns, values, strict=True)
        if not is_display_missing(value)
    )


def variant_display_name(column: str) -> str:
    """Return compact visible variant column names."""
    return {"reference_year": "ref_year", "l2_reuse_year": "l2_reuse_year"}[column]


def format_year_scalar(value: object) -> str:
    """Return year like scalar values without a trailing decimal."""
    text = _visible_value(value)
    if text is None:
        return ""
    if _INTEGER_FLOAT_TEXT.fullmatch(text):
        return text.split(".", maxsplit=1)[0]
    return text


def _visible_value(value: object) -> str | None:
    if is_display_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _scope_has_variant_roles(frame: pd.DataFrame) -> bool:
    roles = {
        str(value).strip()
        for value in frame[ROLE_COLUMN].tolist()
        if not is_display_missing(value) and str(value).strip()
    }
    return roles == {MIN_ROLE, MAX_ROLE}


def _scope_role_variant_key(
    frame: pd.DataFrame,
    *,
    role: str,
    variant_columns: list[str],
) -> tuple[object, ...]:
    scoped = frame.loc[frame[ROLE_COLUMN].astype(str).eq(str(role))]
    complete = scoped.loc[~scoped[variant_columns].map(is_display_missing).any(axis=1)]
    if not complete.empty:
        scoped = complete
    first_row = pd.Series(scoped.iloc[0], copy=False)
    return tuple(first_row[column] for column in variant_columns)


def _single_variant_text(frame: pd.DataFrame, active_columns: list[str]) -> str | None:
    values = list(
        frame.loc[:, active_columns]
        .dropna(how="all")
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if len(values) != 1:
        return None
    return variant_combo_text(active_columns, tuple(values[0]))


def _variant_compression_basis(frame: pd.DataFrame, *, single_year: bool) -> str:
    method_count = len(visible_scope_values(frame, "__method"))
    impact_count = len(visible_scope_values(frame, "impact"))
    year_count = len(visible_scope_values(frame, YEAR_COLUMN))
    parts: list[str] = []
    if method_count > 1:
        parts.append("allocation methods")
    if impact_count > 1:
        parts.append("impact categories")
    if not single_year and year_count > 1:
        parts.append("years")
    return f"across {_join_text(parts)}" if parts else ""


def _variant_axis_application_note(
    frame: pd.DataFrame,
    *,
    active_columns: list[str],
) -> str | None:
    if not active_columns or "__method" not in frame.columns:
        return None
    if len(visible_scope_values(frame, "__method")) <= 1:
        return None
    buckets: dict[tuple[str, ...], list[str]] = {}
    for group in base_variant_groups(frame):
        group_columns = tuple(
            column
            for column in active_columns
            if column in group.columns
            and bool(group[column].map(lambda value: not is_display_missing(value)).any())
        )
        method = str(pd.Series(group.iloc[0], copy=False)["__method"]).strip()
        if method:
            buckets.setdefault(group_columns, []).append(method)
    entries: list[str] = []
    for columns in _ordered_variant_axis_buckets(buckets):
        if not columns:
            continue
        label = _variant_axis_bucket_label(columns)
        methods = list(dict.fromkeys(buckets[columns]))
        entries.append(f"{label}: {'; '.join(methods)}")
    if not entries:
        return None
    return "\n".join(entries)


def _ordered_variant_axis_buckets(
    buckets: dict[tuple[str, ...], list[str]],
) -> list[tuple[str, ...]]:
    order = {
        ("reference_year",): 0,
        ("l2_reuse_year",): 1,
        ("reference_year", "l2_reuse_year"): 2,
    }
    return sorted(buckets, key=lambda columns: (order.get(columns, 99), columns))


def _variant_axis_bucket_label(columns: tuple[str, ...]) -> str:
    if len(columns) == 1:
        return f"{variant_display_name(columns[0])} only"
    return " and ".join(variant_display_name(column) for column in columns)


def _join_text(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"

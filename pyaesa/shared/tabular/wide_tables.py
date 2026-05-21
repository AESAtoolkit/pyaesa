"""Shared wide-table normalization helpers."""

from typing import Any, Sequence

import numpy as np
import pandas as pd

from pyaesa.shared.tabular.scalars import canonical_scalar_text

METHOD_IDENTITY_COLUMNS = frozenset(
    {
        "l1_l2_method",
        "l2_method",
        "l1_method",
        "lcia_method",
    }
)
"""Canonical persisted method identity columns used by wide aSoCC tables."""

PERSISTED_METHOD_BLOCK_COLUMNS = ("l1_l2_method", "l1_method", "l2_method")
"""Canonical persisted method-block order for simplified public result tables."""

FIGURE_COMPARISON_METHOD_IDENTITY_COLUMNS = frozenset(
    {
        "l1_l2_method",
    }
)
"""Canonical method identity columns used for figure comparison gating."""


def method_identity_columns(frame: pd.DataFrame) -> list[str]:
    """Return canonical method identity columns present in one frame."""
    return [column for column in frame.columns if str(column) in METHOD_IDENTITY_COLUMNS]


def persisted_method_block_columns(frame: pd.DataFrame) -> list[str]:
    """Return the canonical persisted method block represented in one frame."""
    has_l1_l2 = _has_nonmissing_method_text(frame, column="l1_l2_method")
    has_l1 = _has_nonmissing_method_text(frame, column="l1_method")
    has_l2 = _has_nonmissing_method_text(frame, column="l2_method")
    if not has_l2 and not has_l1_l2:
        return ["l1_method"] if has_l1 else []
    columns = ["l1_l2_method"]
    if has_l1:
        columns.append("l1_method")
    columns.append("l2_method")
    return columns


def _has_nonmissing_method_text(frame: pd.DataFrame, *, column: str) -> bool:
    """Return whether one method column contains any meaningful value."""
    if column not in frame.columns or frame.empty:
        return False
    series = pd.Series(frame.loc[:, column], copy=False)
    non_null = series.loc[series.notna()]
    if non_null.empty:
        return False
    text = non_null.astype(str).str.strip().str.lower()
    return bool((~text.isin({"", "none", "nan", "null", "<na>"})).any())


def resolved_allocation_method_identities(frame: pd.DataFrame) -> list[str]:
    """Return normalized allocation method identities represented in one frame.

    The persisted canonical method column is `l1_l2_method` for L2 outputs.
    L1 outputs may omit that redundant column and expose only `l1_method`.
    One-step L2 outputs may omit `l1_method` and expose only `l2_method`.
    This helper resolves one stable allocation identity from the available
    persisted method columns and fails fast when redundant columns conflict.
    """
    if frame.empty:
        return []
    explicit = _method_text_series(frame, column="l1_l2_method")
    l2_values = _method_text_series(frame, column="l2_method")
    l1_values = _method_text_series(frame, column="l1_method")
    resolved: list[str] = []
    for explicit_value, l2_value, l1_value in zip(
        explicit.tolist(),
        l2_values.tolist(),
        l1_values.tolist(),
        strict=True,
    ):
        derived_value: str | None
        if l2_value is not None:
            derived_value = l2_value if l1_value is None else f"{l1_value}_{l2_value}"
        else:
            derived_value = l1_value
        if (
            explicit_value is not None
            and derived_value is not None
            and explicit_value != derived_value
        ):
            raise ValueError(
                "Persisted deterministic method columns are inconsistent. "
                f"Observed l1_l2_method={explicit_value!r} and derived method={derived_value!r}."
            )
        resolved_value = explicit_value if explicit_value is not None else derived_value
        if resolved_value is not None:
            resolved.append(resolved_value)
    return resolved


def resolve_single_allocation_method_identity(frame: pd.DataFrame, *, where: str) -> str:
    """Return one canonical allocation method identity or fail explicitly."""
    values = set(resolved_allocation_method_identities(frame))
    if not values:
        raise ValueError(f"{where} has no canonical allocation method identity.")
    if len(values) != 1:
        raise ValueError(
            f"{where} must expose exactly one canonical allocation method identity. "
            f"Observed values: {sorted(values)}."
        )
    return next(iter(values))


def distinct_method_identity_count(frame: pd.DataFrame) -> int:
    """Return the number of distinct method identities represented in one frame."""
    columns = method_identity_columns(frame)
    return _distinct_identity_count(frame=frame, columns=columns)


def figure_comparison_method_identity_columns(frame: pd.DataFrame) -> list[str]:
    """Return canonical method identity columns used for figure comparisons."""
    return [
        column
        for column in frame.columns
        if str(column) in FIGURE_COMPARISON_METHOD_IDENTITY_COLUMNS
    ]


def distinct_figure_comparison_method_identity_count(frame: pd.DataFrame) -> int:
    """Return the number of distinct method identities relevant to figure comparisons."""
    columns = figure_comparison_method_identity_columns(frame)
    return _distinct_identity_count(frame=frame, columns=columns)


def has_multiple_figure_comparison_method_identities(frame: pd.DataFrame) -> bool:
    """Return whether one frame contains more than one figure-comparison method identity."""
    return distinct_figure_comparison_method_identity_count(frame) > 1


def _distinct_identity_count(
    frame: pd.DataFrame,
    *,
    columns: list[str],
) -> int:
    """Return the number of distinct normalized identities in one frame."""
    if not columns or frame.empty:
        return 0
    normalized = frame.loc[:, columns].copy()
    for column in columns:
        normalized[column] = pd.Series(normalized[column], copy=False).map(canonical_scalar_text)
    return int(len(normalized.drop_duplicates(keep="first")))


def _method_text_series(frame: pd.DataFrame, *, column: str) -> pd.Series:
    """Return one normalized optional method text series."""
    if column not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")
    raw = pd.Series(frame.loc[:, column], copy=False)
    text = raw.astype("string").str.strip()
    missing = raw.isna() | text.eq("") | text.str.lower().isin({"nan", "none", "nat"})
    out = text.astype("object")
    out.loc[missing] = None
    return pd.Series(out, index=frame.index, dtype="object")


def has_multiple_method_identities(frame: pd.DataFrame) -> bool:
    """Return whether one frame contains more than one distinct method identity."""
    return distinct_method_identity_count(frame) > 1


def detect_year_columns(frame: pd.DataFrame) -> list[str]:
    """Return year-like columns from a wide deterministic output table."""
    years: list[str] = []
    for column in frame.columns:
        try:
            year = int(column)
        except (TypeError, ValueError):
            continue
        if 1900 < year < 2200:
            years.append(str(column))
    return sorted(years, key=int)


def id_columns(
    frame: pd.DataFrame,
    *,
    year_columns: list[str],
    ignored_columns: set[str] | None = None,
) -> list[str]:
    """Return non-year identifier columns excluding explicitly ignored fields."""
    ignored = {str(column) for column in year_columns}
    ignored.update(str(column) for column in (ignored_columns or set()))
    return [column for column in frame.columns if str(column) not in ignored]


def requested_year_columns(
    frame: pd.DataFrame,
    *,
    requested_years: Sequence[int],
) -> list[str]:
    """Return requested year columns present in one wide deterministic table."""
    requested = {int(year) for year in requested_years}
    return [column for column in detect_year_columns(frame) if int(column) in requested]


def first_non_null_scenario_year(
    frame: pd.DataFrame,
    *,
    scenario_column: str,
    year_column: str = "year",
) -> int | None:
    """Return the first year whose scenario column is non-null."""
    if scenario_column not in frame.columns:
        return None
    scenario_series = pd.Series(frame.loc[:, scenario_column], copy=False)
    mask = scenario_series.notna()
    if not bool(mask.any()):
        return None
    years = pd.to_numeric(pd.Series(frame.loc[mask, year_column], copy=False), errors="raise")
    return int(pd.Series(years, copy=False).astype(int).min())


def validate_complete_wide_year_values(
    frame: pd.DataFrame,
    *,
    year_columns: Sequence[str],
    where: str,
) -> None:
    """Fail when wide year columns contain missing values.

    Args:
        frame: Wide input table containing year-like columns.
        year_columns: Year columns that must be fully populated.
        where: User-facing context string for the error message.

    Raises:
        ValueError: If any requested year column contains missing values.
    """
    columns = [str(column) for column in year_columns]
    if not columns:
        return
    missing_mask = frame.loc[:, columns].isna()
    if not bool(missing_mask.to_numpy().any()):
        return
    sample_positions = np.argwhere(missing_mask.to_numpy())[:5]
    sample_labels = [
        f"row={int(row_index) + 1}, year={columns[int(column_index)]}"
        for row_index, column_index in sample_positions.tolist()
    ]
    raise ValueError(
        f"{where} must not contain missing values in declared year columns. "
        f"Sample missing cells: {sample_labels}."
    )


def planned_melt_columns(
    frame: pd.DataFrame,
    *,
    requested_years: Sequence[int],
    ignored_columns: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return detected year columns, requested melt columns, and metadata columns."""
    all_year_columns = detect_year_columns(frame)
    requested_columns = requested_year_columns(frame, requested_years=requested_years)
    metadata_columns = id_columns(
        frame,
        year_columns=all_year_columns,
        ignored_columns=ignored_columns,
    )
    return all_year_columns, requested_columns, metadata_columns


def melt_requested_year_value_rows(
    frame: pd.DataFrame,
    *,
    requested_years: Sequence[int],
    ignored_columns: set[str] | None = None,
    year_name: str = "year",
    value_name: str = "value",
) -> pd.DataFrame:
    """Return canonical long rows for requested wide year columns.

    Args:
        frame: Wide input table containing year-like columns.
        requested_years: Requested years that should be retained in the long
            output when present in ``frame``.
        ignored_columns: Optional non-year columns that should be excluded from
            the identifier columns used for the melt.
        year_name: Output column name for the melted year token.
        value_name: Output column name for the melted numeric value.

    Returns:
        Long-form DataFrame containing metadata columns plus ``year_name`` and
        ``value_name``. Returns an empty frame with the same output columns
        when none of the requested years are present.
    """
    _all_year_columns, requested_columns, metadata_columns = planned_melt_columns(
        frame,
        requested_years=requested_years,
        ignored_columns=ignored_columns,
    )
    if not requested_columns:
        return pd.DataFrame(columns=[*metadata_columns, year_name, value_name])
    values = frame.loc[:, requested_columns].to_numpy()
    flat_values = values.T.reshape(-1)
    keep = pd.notna(flat_values)
    row_positions = np.tile(np.arange(len(frame), dtype=np.int64), len(requested_columns))[keep]
    year_positions = np.repeat(np.arange(len(requested_columns), dtype=np.int64), len(frame))[keep]
    out = pd.DataFrame(
        {column: frame.loc[:, column].to_numpy()[row_positions] for column in metadata_columns}
    )
    out[year_name] = np.asarray(requested_columns, dtype=object)[year_positions]
    out[value_name] = flat_values[keep]
    return out


def row_label(*, row: pd.Series, columns: list[str], default_prefix: str) -> str:
    """Return a compact label for one wide table row."""
    parts: list[str] = []
    for column in columns:
        value = row.get(column)
        if _is_missing_scalar(value):
            continue
        parts.append(f"{column}={value}")
    if parts:
        return ", ".join(parts)
    index = int(getattr(row, "name", 0)) + 1
    return f"{default_prefix} {index}"


def row_identity_key(
    *,
    row: pd.Series,
    columns: list[str],
    ordinal: int,
) -> tuple[str, ...]:
    """Return a stable key for one row, including duplicate ordinal."""
    if not columns:
        return ("__row__", str(int(ordinal)))
    base = tuple(canonical_scalar_text(row.get(column)) for column in columns)
    return (*base, f"__dup_{int(ordinal)}")


def _is_missing_scalar(value: Any) -> bool:
    """Return whether ``value`` is a missing scalar."""
    missing = pd.isna(value)
    if not isinstance(missing, (bool, np.bool_)):
        raise TypeError("Wide-table row labeling expects scalar values, not array-like objects.")
    return bool(missing)

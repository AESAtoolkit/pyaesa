"""I/O helpers for allocation method tests."""

from pathlib import Path
import re
from typing import Any, cast

import numpy as np
import pandas as pd

_SUPPORTED_EXTENSIONS: tuple[str, ...] = ("pickle", "csv", "parquet")
_READERS = {
    "csv": pd.read_csv,
    "pickle": pd.read_pickle,
    "parquet": pd.read_parquet,
}
_METHOD_SOURCE_COLUMNS: tuple[str, ...] = ("l1_l2_method", "method", "l1_method", "l2_method")
VALIDATION_METHOD_COLUMN = "_validation_method"


def _normalize_format(value: str | None) -> str:
    """Normalize optional format string to one supported extension name."""
    fmt = str(value or "").strip().lower()
    if fmt in _SUPPORTED_EXTENSIONS:
        return fmt
    return "pickle"


def read_output(path: Path, *, output_format: str | None = None) -> pd.DataFrame:
    """Read one deterministic aSoCC output file using file suffix first."""
    suffix = path.suffix.lower().lstrip(".")
    fmt = suffix if suffix in _SUPPORTED_EXTENSIONS else _normalize_format(output_format)
    loaded = _READERS.get(fmt, pd.read_pickle)(path)
    if isinstance(loaded, pd.Series):
        return loaded.to_frame()
    return loaded


def output_extension(output_format: str) -> str:
    """Return canonical extension for a configured output format."""
    return _normalize_format(output_format)


def list_output_files(directory: Path, *, preferred_format: str | None = None) -> list[Path]:
    """List output files in deterministic order across supported formats."""
    preferred = output_extension(preferred_format or "pickle")
    files: list[Path] = []
    for ext in (preferred, *[e for e in _SUPPORTED_EXTENSIONS if e != preferred]):
        files.extend(sorted(directory.glob(f"*.{ext}")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def has_output_files(directory: Path, *, preferred_format: str | None = None) -> bool:
    """Return whether directory contains any supported output file."""
    return bool(list_output_files(directory, preferred_format=preferred_format))


def aggregate_share_by_group_keys(
    df: pd.DataFrame,
    *,
    year_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    """Aggregate share column by groups; return one row frame if no groups."""
    if not group_cols:
        year_frame = df.filter(items=[year_col])
        if year_frame.empty:
            raise KeyError(f"Missing year column '{year_col}' in validation output frame.")
        series = year_frame.iloc[:, 0]
        numeric = pd.Series(pd.to_numeric(series, errors="coerce"), copy=False)
        sum_share = scalar_float(numeric.sum(min_count=1))
        return pd.DataFrame([{"sum_share": sum_share}])
    grouped_raw = df.groupby(group_cols, dropna=False)[year_col].sum(min_count=1)
    grouped_frame = pd.DataFrame(grouped_raw).reset_index()
    value_cols = [col for col in grouped_frame.columns if col not in group_cols]
    if value_cols:
        grouped_frame = grouped_frame.rename(columns={value_cols[0]: "sum_share"})
    return grouped_frame


def with_validation_method_column(
    frame: pd.DataFrame,
    *,
    column_name: str = VALIDATION_METHOD_COLUMN,
) -> pd.DataFrame:
    """Return frame with one canonical internal method label column.

    The validation suite consumes current deterministic aSoCC outputs, where
    method identity is published through ``l1_l2_method`` plus optional split
    columns such as ``l1_method`` and ``l2_method``. This helper centralizes
    that schema into one internal label used across all validators.
    """
    if column_name in frame.columns:
        return frame

    out = frame.copy()
    method_columns = [col for col in _METHOD_SOURCE_COLUMNS if col in out.columns]
    if not method_columns:
        return out

    method_labels = pd.Series(pd.NA, index=out.index, dtype="string")
    for column in method_columns:
        candidate = out[column].astype("string").str.strip()
        candidate = candidate.mask(candidate == "")
        method_labels = method_labels.fillna(candidate)

    out[column_name] = method_labels
    return out


def method_label_from_row(
    row: pd.Series,
    *,
    column_name: str = VALIDATION_METHOD_COLUMN,
) -> str:
    """Return canonical method label from one grouped report row."""
    if column_name in row.index:
        label = clean_text(row.get(column_name))
        if label:
            return label
    for source_column in _METHOD_SOURCE_COLUMNS:
        if source_column not in row.index:
            continue
        label = clean_text(row.get(source_column))
        if label:
            return label
    return ""


def is_lcia_output(path: Path) -> bool:
    """Return whether output file stem includes an LCIA suffix."""
    return "_lcia" in path.stem


def parse_lcia_method(path: Path) -> str | None:
    """Extract LCIA method suffix from output filename if present."""
    match = re.search(r"([a-zA-Z0-9]+_lcia)", path.stem)
    return match.group(1) if match else None


def is_missing_scalar(value: object) -> bool:
    """Return True for scalar missing sentinels used in report rows."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def clean_text(value: object) -> str:
    """Return stable text representation for optional scalar fields."""
    return "" if is_missing_scalar(value) else str(value)


def parse_optional_int(value: object) -> int | None:
    """Parse optional scalar into int; return None for missing/blank values."""
    if is_missing_scalar(value):
        return None

    parsed_value: object | None
    if isinstance(value, str):
        text = value.strip()
        parsed_value = text or None
    elif isinstance(value, (int, np.integer)):
        parsed_value = int(value)
    elif isinstance(value, (float, np.floating)):
        parsed_value = None if not np.isfinite(value) else int(value)
    else:
        parsed_value = str(value)

    if parsed_value is None:
        return None
    return int(parsed_value)


def parse_optional_int_or_empty(value: object) -> int | str:
    """Parse optional scalar into int; return empty string when missing."""
    parsed = parse_optional_int(value)
    return "" if parsed is None else parsed


def scalar_float(value: object) -> float:
    """Convert row scalar like values (including Series/ndarray) to float."""
    sample = value
    if isinstance(value, pd.Series):
        sample = np.nan if value.empty else value.iloc[0]
    elif isinstance(value, np.ndarray):
        sample = np.nan if value.size == 0 else value.reshape(-1)[0]

    if is_missing_scalar(sample):
        return np.nan
    if isinstance(sample, (int, float, np.integer, np.floating, str)):
        return float(sample)
    return float(str(cast(Any, sample)))

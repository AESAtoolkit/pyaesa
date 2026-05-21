"""Detect and normalize bundled static carrying capacity CSV schemas.

The package currently accepts the standard and planetary boundary bundled CSV
layouts. This module validates those scientific input schemas and converts them
into one normalized row contract used by later aCC, ASR, and external LCA
validation steps.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

StaticCCSchemaKind = Literal["standard", "planetary boundary"]

STANDARD_REQUIRED_COLUMNS = (
    "impact_full_name",
    "impact",
    "impact_unit",
    "min_cc",
    "max_cc",
)
PLANETARY_BOUNDARY_REQUIRED_COLUMNS = (
    "Planetary boundary",
    "Control variable",
    "impact",
    "impact_unit",
    "min_cc",
    "max_cc",
)


@dataclass(frozen=True)
class NormalizedStaticCCRow:
    """One normalized bundled static carrying capacity row."""

    impact: str
    impact_unit: str
    min_cc: float
    max_cc: float
    impact_full_name_normalized: str
    planetary_boundary: str | None
    control_variable: str | None


def _csv_text(value) -> str:
    """Return one CSV cell as stripped text, or an empty string when missing."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _required_float(*, row: pd.Series, column: str, path: Path) -> float:
    """Return one required numeric column as float."""
    value = row.get(column)
    if value is None or pd.isna(value):
        impact = _csv_text(row.get("impact")) or "<missing impact>"
        raise ValueError(
            f"Bundled static carrying capacity CSV '{path}' is missing required numeric "
            f"column '{column}' for impact '{impact}'."
        )
    try:
        return float(value)
    except (TypeError, ValueError):
        impact = _csv_text(row.get("impact")) or "<missing impact>"
        raise ValueError(
            f"Bundled static carrying capacity CSV '{path}' has non-numeric value "
            f"{value!r} in column '{column}' for impact '{impact}'."
        ) from None


def detect_static_cc_schema(*, frame: pd.DataFrame, path: Path) -> StaticCCSchemaKind:
    """Detect which accepted bundled static carrying capacity schema is used."""
    columns = {str(column) for column in frame.columns}
    if "impact_full_name" in columns:
        missing = sorted(set(str(column) for column in STANDARD_REQUIRED_COLUMNS) - columns)
        if missing:
            raise ValueError(
                f"Bundled static carrying capacity CSV '{path}' is missing standard-schema "
                f"columns {missing}."
            )
        return "standard"
    if {"Planetary boundary", "Control variable"}.issubset(columns):
        missing = sorted(
            set(str(column) for column in PLANETARY_BOUNDARY_REQUIRED_COLUMNS) - columns
        )
        if missing:
            raise ValueError(
                f"Bundled static carrying capacity CSV '{path}' is missing planetary boundary "
                f"schema columns {missing}."
            )
        return "planetary boundary"
    raise ValueError(
        "Bundled static carrying capacity CSV does not match any accepted schema. "
        f"CSV: '{path}'. Found columns: {sorted(columns)}. "
        f"Standard required columns: {list(STANDARD_REQUIRED_COLUMNS)}. "
        f"Planetary boundary required columns: {list(PLANETARY_BOUNDARY_REQUIRED_COLUMNS)}."
    )


def _build_standard_label(*, row: pd.Series) -> str:
    """Build the normalized display label for one standard schema row."""
    full_name = _csv_text(row.get("impact_full_name"))
    impact = _csv_text(row.get("impact"))
    return full_name or impact


def _build_planetary_boundary_label(*, row: pd.Series) -> str:
    """Build the normalized display label for one planetary boundary row."""
    boundary = _csv_text(row.get("Planetary boundary"))
    control = _csv_text(row.get("Control variable"))
    if boundary and control:
        return f"{boundary}: {control}"
    return boundary


def standardize_static_cc_rows(
    *,
    frame: pd.DataFrame,
    path: Path,
) -> tuple[StaticCCSchemaKind, tuple[NormalizedStaticCCRow, ...]]:
    """Normalize one bundled static carrying capacity CSV to one common row contract."""
    schema_kind = detect_static_cc_schema(frame=frame, path=path)
    rows: list[NormalizedStaticCCRow] = []
    seen_impacts: set[str] = set()
    for _, row in frame.iterrows():
        impact = _csv_text(row.get("impact"))
        if not impact:
            continue
        if impact in seen_impacts:
            raise ValueError(
                f"Bundled static carrying capacity CSV '{path}' contains duplicate impact "
                f"code '{impact}'."
            )
        impact_unit = _csv_text(row.get("impact_unit"))
        if not impact_unit:
            raise ValueError(
                f"Bundled static carrying capacity CSV '{path}' is missing 'impact_unit' "
                f"for impact '{impact}'."
            )
        min_cc = _required_float(row=row, column="min_cc", path=path)
        max_cc = _required_float(row=row, column="max_cc", path=path)
        if schema_kind == "standard":
            label = _build_standard_label(row=row)
        else:
            label = _build_planetary_boundary_label(row=row)
        rows.append(
            NormalizedStaticCCRow(
                impact=impact,
                impact_unit=impact_unit,
                min_cc=min_cc,
                max_cc=max_cc,
                impact_full_name_normalized=label or impact,
                planetary_boundary=(
                    _csv_text(row.get("Planetary boundary")) or None
                    if schema_kind == "planetary boundary"
                    else None
                ),
                control_variable=(
                    _csv_text(row.get("Control variable")) or None
                    if schema_kind == "planetary boundary"
                    else None
                ),
            )
        )
        seen_impacts.add(impact)
    if not rows:
        raise ValueError(
            f"Bundled static carrying capacity CSV '{path}' has no usable impact rows."
        )
    return schema_kind, tuple(rows)

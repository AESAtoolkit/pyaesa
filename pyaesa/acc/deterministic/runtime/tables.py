"""Deterministic aCC table transforms and writing."""

import pandas as pd

from pyaesa.shared.acc_asr_common.deterministic.downstream.tabular_io import write_output_table
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.tabular.wide_tables import (
    detect_year_columns,
    persisted_method_block_columns,
)

_DYNAMIC_CC_COLUMNS = (
    "cc_model",
    "cc_scenario",
    "cc_category",
    AR6_CC_SSP_SCENARIO_COLUMN,
    "cc_flow",
    "cc_variable",
)
_SELECTOR_COLUMNS = ("r_p", "s_p", "r_c", "r_f")


def materialize_acc_scope(
    df: pd.DataFrame,
    *,
    l1_l2_method: str,
    impact: str,
    impact_unit: str,
    asocc_ssp_start_year: int | None = None,
) -> pd.DataFrame:
    """Return one aCC frame with explicit method and LCIA scope materialized."""
    out = df.copy()
    if "share_stem" in out.columns:
        out = out.drop(columns=["share_stem"])
    if "lcia_method" in out.columns:
        out = out.drop(columns=["lcia_method"])
    out = _materialize_method_block(out, l1_l2_method=l1_l2_method)
    out = _ensure_constant_text_column(
        out,
        column="impact",
        expected_value=impact,
        context="aCC impact scope",
    )
    out = _ensure_constant_text_column(
        out,
        column="impact_unit",
        expected_value=impact_unit,
        context="aCC impact unit scope",
    )
    if asocc_ssp_start_year is not None:
        out["asocc_ssp_start_year"] = int(asocc_ssp_start_year)
    return out


def resolve_acc_l1_l2_method(*, frame: pd.DataFrame, source_label: str) -> str:
    """Return one canonical allocation method identity for deterministic aCC output writing."""
    del source_label
    has_l1_l2 = _column_has_values(frame, "l1_l2_method")
    has_l1 = _column_has_values(frame, "l1_method")
    has_l2 = _column_has_values(frame, "l2_method")
    if has_l1_l2:
        return _first_non_empty_text(frame, "l1_l2_method")
    if has_l2:
        value = _first_non_empty_text(frame, "l2_method")
        if has_l1:
            return f"{_first_non_empty_text(frame, 'l1_method')}_{value}"
        return value
    return _first_non_empty_text(frame, "l1_method")


def write_acc_output(
    df: pd.DataFrame,
    output_path,
    output_format: str,
) -> None:
    """Write one deterministic aCC output table."""
    ordered = ordered_acc_output_columns(df)
    write_output_table(
        df=df.loc[:, ordered].copy(),
        output_path=output_path,
        output_format=output_format,
    )


def ordered_acc_output_columns(frame: pd.DataFrame) -> list[str]:
    """Return canonical deterministic aCC output column order."""
    year_columns = detect_year_columns(frame)
    ordered: list[str] = []
    ordered.extend(column for column in _DYNAMIC_CC_COLUMNS if column in frame.columns)
    ordered.extend(persisted_method_block_columns(frame))
    if "cc_bound" in frame.columns:
        ordered.append("cc_bound")
    ordered.extend(column for column in ("impact", "impact_unit") if column in frame.columns)
    ordered.extend(column for column in _SELECTOR_COLUMNS if column in frame.columns)
    if ASOCC_SSP_SCENARIO_COLUMN in frame.columns:
        ordered.append(ASOCC_SSP_SCENARIO_COLUMN)
    if ASOCC_TIME_ROUTE_PUBLIC_COLUMN in frame.columns:
        ordered.append(ASOCC_TIME_ROUTE_PUBLIC_COLUMN)
    if "reference_year" in frame.columns:
        ordered.append("reference_year")
    if "asocc_ssp_start_year" in frame.columns:
        ordered.append("asocc_ssp_start_year")
    ordered.extend(
        column for column in frame.columns if column not in ordered and column not in year_columns
    )
    return [*ordered, *year_columns]


def _materialize_method_block(
    frame: pd.DataFrame,
    *,
    l1_l2_method: str,
) -> pd.DataFrame:
    """Return one deterministic aCC frame with the simplified method block."""
    expected_method = str(l1_l2_method).strip()
    out = frame.copy()
    has_l2 = _column_has_values(out, "l2_method")
    has_l1 = _column_has_values(out, "l1_method")
    if has_l2 and has_l1:
        out["l1_l2_method"] = expected_method
        return out
    if has_l2:
        out["l1_l2_method"] = expected_method
        return out.drop(columns=["l1_method"], errors="ignore")
    out["l1_method"] = expected_method
    return out.drop(columns=["l1_l2_method", "l2_method"], errors="ignore")


def _column_has_values(frame: pd.DataFrame, column: str) -> bool:
    """Return whether one optional persisted text column carries any non-empty value."""
    return bool(_non_empty_text_values(frame, column))


def _first_non_empty_text(frame: pd.DataFrame, column: str) -> str:
    """Return the first non-empty text value from a canonical method column."""
    values = _non_empty_text_values(frame, column)
    return values[0]


def _ensure_constant_text_column(
    frame: pd.DataFrame,
    *,
    column: str,
    expected_value: str,
    context: str,
) -> pd.DataFrame:
    """Fill missing values in one constant text column and fail on conflicts."""
    out = frame.copy()
    expected_text = str(expected_value).strip()
    if column not in out.columns:
        out[column] = expected_text
        return out
    out[column] = pd.Series(out.loc[:, column], copy=False).astype("object")
    present_values = _non_empty_text_values(out, column)
    incompatible = sorted(value for value in present_values if value != expected_text)
    if incompatible:
        raise ValueError(
            f"{context} found rows that conflict with the target scope '{expected_text}': "
            f"{incompatible}."
        )
    series = pd.Series(out.loc[:, column], copy=False)
    missing_mask = series.isna() | series.astype("string").fillna("").str.strip().eq("")
    out.loc[missing_mask, column] = expected_text
    return out


def _non_empty_text_values(frame: pd.DataFrame, column: str) -> list[str]:
    """Return sorted non-empty normalized text values from one optional column."""
    if column not in frame.columns:
        return []
    values = {
        str(value).strip()
        for value in pd.Series(frame.loc[:, column], copy=False).tolist()
        if value is not None and not pd.isna(value) and str(value).strip()
    }
    return sorted(values)

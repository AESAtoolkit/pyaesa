"""Helpers to shape validation report DataFrames."""

from typing import Sequence

import pandas as pd


def normalize_report_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize validator frame (currently pass through with light cleanup)."""
    return df.drop(columns=[c for c in ("expected",) if c in df.columns]).copy()


def per_fu_columns() -> list[str]:
    """Return unified ordered output columns for one FU report."""
    return combined_columns()


def select_existing_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return DataFrame reindexed to requested columns (missing -> NaN)."""
    return df.reindex(columns=list(columns)).copy()


def combined_columns() -> list[str]:
    """Return ordered output columns for the combined report."""
    return [
        "source",
        "fu_code",
        "year",
        "l1_reg_aggreg",
        "bucket",
        "l2_country_axis",
        "l2_country_code",
        "file",
        "method",
        "impact",
        "reference_year",
        "group_key",
        "sum_share_observed",
        "fy_add_share_observed",
        "all_incl_fy_share_observed",
        "ratio_expected",
        "abs_error",
        "atol_used",
        "passed",
        "rule",
        "validation_note",
    ]

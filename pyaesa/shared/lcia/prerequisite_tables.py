"""Utilities for bundled LCIA prerequisite CSV tables."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped column labels and no unnamed placeholder columns."""
    cleaned = frame.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    keep_columns = [
        col for col in cleaned.columns if col and not col.lower().startswith("unnamed:")
    ]
    return cleaned.loc[:, keep_columns].copy()


def clean_characterization_matrix_frame(
    *,
    frame: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    """Validate and normalize one characterization matrix.

    Every row defines one factor for an exact extension, stressor and impact.
    Missing factors are not numerical zeros. Inventory coverage belongs in the
    accompanying method documentation rather than incomplete numerical rows.
    """
    normalized = _clean_columns(frame)
    text_columns = [
        "extension",
        "stressor",
        "stressor_unit",
        "impact",
        "impact_parent",
        "impact_unit",
    ]
    missing = sorted(set([*text_columns, "factor"]) - set(normalized.columns))
    if missing:
        raise ValueError(f"Characterization matrix is missing columns {missing}. CSV: {path}")
    if normalized.empty or normalized.columns.duplicated().any():
        raise ValueError(
            f"Characterization matrix must contain rows and unique columns. CSV: {path}"
        )
    for column in text_columns:
        values = normalized[column].astype("string").str.strip()
        if values.isna().any() or values.str.lower().isin(["", "nan", "none", "null"]).any():
            raise ValueError(
                f"Characterization matrix contains missing or empty '{column}' values. CSV: {path}"
            )
        normalized[column] = values.astype(str)
    factors = pd.Series(pd.to_numeric(normalized["factor"], errors="raise"), dtype=float)
    if not np.isfinite(factors.to_numpy()).all():
        raise ValueError(f"Characterization factors must all be finite numbers. CSV: {path}")
    normalized["factor"] = factors
    if normalized.duplicated(subset=["extension", "stressor", "impact"]).any():
        raise ValueError(f"Characterization matrix contains duplicate factor keys. CSV: {path}")
    for keys, value in (
        (["extension", "stressor"], "stressor_unit"),
        (["impact"], "impact_parent"),
        (["impact"], "impact_unit"),
        (["impact_parent"], "impact_unit"),
    ):
        unique_counts = cast(pd.Series, normalized.groupby(keys, sort=False)[value].nunique())
        if bool(unique_counts.gt(1).any()):
            raise ValueError(
                f"Characterization matrix has conflicting '{value}' values for {keys}. CSV: {path}"
            )
    return normalized


def clean_responsibility_period_frame(
    *,
    frame: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    """Return one responsibility period frame with cleaned public columns."""
    normalized = _clean_columns(frame)
    if "impact" not in normalized.columns:
        raise ValueError(f"RPS file missing impact column: {path}")
    duplicated = normalized["impact"].duplicated(keep=False)
    if bool(duplicated.any()):
        duplicate_impacts = normalized.loc[duplicated, "impact"].astype(str).drop_duplicates()
        sample = duplicate_impacts.tolist()[:10]
        raise ValueError(
            "RPS file contains duplicate impact rows for responsibility periods. "
            f"Impacts (sample): {sample}. CSV: {path}"
        )
    return normalized

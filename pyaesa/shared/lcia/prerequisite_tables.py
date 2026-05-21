"""Utilities for bundled LCIA prerequisite CSV tables."""

from pathlib import Path

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
    """Return one characterization matrix frame with cleaned public columns."""
    normalized = _clean_columns(frame)
    if "extension" not in normalized.columns:
        raise ValueError(f"Characterization matrix must contain an 'extension' column. CSV: {path}")
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

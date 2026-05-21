"""Utilities for MRIO region/sector grouping via user mapping CSVs."""

from pathlib import Path
from typing import Iterable, List, Sequence, cast

import pandas as pd

REQUIRED_COLUMNS = ("original_classification", "grouped_mrio")


def read_group_map(csv_path: str | Path) -> pd.DataFrame:
    """Read a grouping map CSV and validate required columns."""
    path = Path(csv_path)
    encodings = ("utf-8", "utf-8-sig", "utf-16", "latin-1", "cp1252")
    last_exc = None
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except (UnicodeError, pd.errors.ParserError, OSError) as exc:
            last_exc = exc
            continue

    if df is None:
        raise ValueError(f"Failed to read CSV {path}: {last_exc}") from last_exc

    if not set(REQUIRED_COLUMNS).issubset(df.columns):
        raise ValueError(
            f"{path} must contain columns: {', '.join(REQUIRED_COLUMNS)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.copy()
    original_missing = cast(pd.Series, df["original_classification"].isna())
    if bool(original_missing.any()):
        examples = df.loc[original_missing, "original_classification"].head(10).tolist()
        raise ValueError(f"{path} contains empty original labels: {examples}")
    grouped_missing = cast(pd.Series, df["grouped_mrio"].isna())
    if bool(grouped_missing.any()):
        examples = df.loc[grouped_missing, "original_classification"].head(10).tolist()
        raise ValueError(f"{path} contains empty grouped labels for originals: {examples}")

    df["original_classification"] = df["original_classification"].astype(str).str.strip()
    df["grouped_mrio"] = df["grouped_mrio"].astype(str).str.strip()

    empty_original = cast(pd.Series, df["original_classification"] == "")
    if bool(empty_original.any()):
        examples = df.loc[empty_original, "original_classification"].head(10).tolist()
        raise ValueError(f"{path} contains empty original labels: {examples}")

    empty_grouped = cast(pd.Series, df["grouped_mrio"] == "")
    if bool(empty_grouped.any()):
        examples = df.loc[empty_grouped, "original_classification"].head(10).tolist()
        raise ValueError(f"{path} contains empty grouped labels for originals: {examples}")
    return df


def build_agg_vector(
    original_order: Sequence[str],
    map_df: pd.DataFrame,
    *,
    label_kind: str,
    csv_path: str | Path,
) -> List[str]:
    """Build an aggregation vector aligned to ``original_order``."""
    path = Path(csv_path)
    originals = map_df["original_classification"].tolist()
    if pd.Series(originals).duplicated().any():
        dup = (
            map_df.loc[map_df["original_classification"].duplicated(), "original_classification"]
            .unique()
            .tolist()
        )
        raise ValueError(f"Duplicate {label_kind} labels in {path}: {dup[:20]}")

    mapping = dict(zip(map_df["original_classification"], map_df["grouped_mrio"]))
    missing = [label for label in original_order if label not in mapping]
    if missing:
        preview = missing[:20]
        raise ValueError(
            f"Missing {len(missing)} {label_kind} labels in {path}. Examples: {preview}"
        )

    return [mapping[label] for label in original_order]


def unique_in_order(values: Iterable[str]) -> list[str]:
    """Return unique values preserving first seen order."""
    seen = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out

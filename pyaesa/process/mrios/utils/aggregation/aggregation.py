"""Utilities for MRIO region and sector aggregation and disaggregation via user mapping CSVs."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence, cast

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("original_classification", "aggregated_mrio")
WEIGHT_COLUMN = "weight"
_WEIGHT_SUM_RTOL = 1e-10


@dataclass(frozen=True)
class AggregationSpec:
    """Resolved aggregation rows aligned to the source label order."""

    original_order: tuple[str, ...]
    aggregated_labels: tuple[str, ...]
    weighted: bool
    rows: tuple[tuple[int, int, float], ...]


def read_agg_map(csv_path: str | Path) -> pd.DataFrame:
    """Read an MRIO aggregation and disaggregation map CSV and validate required columns."""
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
    aggregated_missing = cast(pd.Series, df["aggregated_mrio"].isna())
    if bool(aggregated_missing.any()):
        examples = df.loc[aggregated_missing, "original_classification"].head(10).tolist()
        raise ValueError(f"{path} contains empty aggregated labels for originals: {examples}")

    df["original_classification"] = df["original_classification"].astype(str).str.strip()
    df["aggregated_mrio"] = df["aggregated_mrio"].astype(str).str.strip()

    empty_original = cast(pd.Series, df["original_classification"] == "")
    if bool(empty_original.any()):
        examples = df.loc[empty_original, "original_classification"].head(10).tolist()
        raise ValueError(f"{path} contains empty original labels: {examples}")

    empty_aggregated = cast(pd.Series, df["aggregated_mrio"] == "")
    if bool(empty_aggregated.any()):
        examples = df.loc[empty_aggregated, "original_classification"].head(10).tolist()
        raise ValueError(f"{path} contains empty aggregated labels for originals: {examples}")

    if WEIGHT_COLUMN in df.columns:
        weight_missing = cast(pd.Series, df[WEIGHT_COLUMN].isna())
        if bool(weight_missing.any()):
            examples = df.loc[weight_missing, "original_classification"].head(10).tolist()
            raise ValueError(f"{path} contains empty mapping weights for originals: {examples}")
        weights = cast(pd.Series, pd.to_numeric(df[WEIGHT_COLUMN], errors="coerce"))
        invalid = cast(pd.Series, weights.isna() | ~np.isfinite(weights) | (weights < 0.0))
        if bool(invalid.any()):
            examples = df.loc[invalid, "original_classification"].head(10).tolist()
            raise ValueError(f"{path} contains invalid mapping weights for originals: {examples}")
        df[WEIGHT_COLUMN] = weights.astype(float)
        duplicate_pairs = cast(
            pd.Series,
            df.duplicated(subset=["original_classification", "aggregated_mrio"]),
        )
        if bool(duplicate_pairs.any()):
            examples = (
                df.loc[duplicate_pairs, ["original_classification", "aggregated_mrio"]]
                .head(10)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"{path} contains duplicate weighted MRIO aggregation and "
                f"disaggregation rows: {examples}"
            )
        sums = cast(pd.Series, df.groupby("original_classification")[WEIGHT_COLUMN].sum())
        bad_sum = cast(pd.Series, ~np.isclose(sums, 1.0, rtol=_WEIGHT_SUM_RTOL, atol=0.0))
        if bool(bad_sum.any()):
            examples = sums.loc[bad_sum].head(10).to_dict()
            raise ValueError(
                f"{path} mapping weights must sum to 1 per original label. Invalid sums: {examples}"
            )
    return df


def build_agg_vector(
    original_order: Sequence[str],
    map_df: pd.DataFrame,
    *,
    label_kind: str,
    csv_path: str | Path,
) -> list[str]:
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

    mapping = dict(zip(map_df["original_classification"], map_df["aggregated_mrio"]))
    missing = [label for label in original_order if label not in mapping]
    if missing:
        preview = missing[:20]
        raise ValueError(
            f"Missing {len(missing)} {label_kind} labels in {path}. Examples: {preview}"
        )

    return [mapping[label] for label in original_order]


def build_aggregation_spec(
    original_order: Sequence[str],
    map_df: pd.DataFrame,
    *,
    label_kind: str,
    csv_path: str | Path,
) -> AggregationSpec:
    """Build an MRIO aggregation and disaggregation spec aligned to source labels."""
    path = Path(csv_path)
    normalized_order = tuple(str(label) for label in original_order)
    weighted = WEIGHT_COLUMN in map_df.columns
    if not weighted:
        strict_labels = build_agg_vector(
            normalized_order,
            map_df,
            label_kind=label_kind,
            csv_path=path,
        )
        aggregated_labels = tuple(unique_in_order(strict_labels))
        aggregated_index = {label: idx for idx, label in enumerate(aggregated_labels)}
        rows = tuple(
            (original_idx, aggregated_index[aggregated], 1.0)
            for original_idx, aggregated in enumerate(strict_labels)
        )
        return AggregationSpec(
            original_order=normalized_order,
            aggregated_labels=aggregated_labels,
            weighted=False,
            rows=rows,
        )

    originals = set(map_df["original_classification"].tolist())
    missing = [label for label in normalized_order if label not in originals]
    if missing:
        preview = missing[:20]
        raise ValueError(
            f"Missing {len(missing)} {label_kind} labels in {path}. Examples: {preview}"
        )

    aggregated_labels = tuple(unique_in_order(map_df["aggregated_mrio"].astype(str).tolist()))
    aggregated_index = {label: idx for idx, label in enumerate(aggregated_labels)}
    original_index = {label: idx for idx, label in enumerate(normalized_order)}
    rows_list: list[tuple[int, int, float]] = []
    for original, aggregated, weight in map_df[
        ["original_classification", "aggregated_mrio", WEIGHT_COLUMN]
    ].itertuples(index=False, name=None):
        original_label = str(original)
        if original_label not in original_index:
            continue
        rows_list.append(
            (original_index[original_label], aggregated_index[str(aggregated)], float(weight))
        )
    return AggregationSpec(
        original_order=normalized_order,
        aggregated_labels=aggregated_labels,
        weighted=True,
        rows=tuple(rows_list),
    )


def agg_map_fingerprint(map_df: pd.DataFrame) -> str:
    """Return a stable fingerprint for a validated MRIO aggregation and disaggregation map."""
    columns = ["original_classification", "aggregated_mrio"]
    if WEIGHT_COLUMN in map_df.columns:
        columns.append(WEIGHT_COLUMN)
    rows = [
        {
            "original_classification": str(record["original_classification"]),
            "aggregated_mrio": str(record["aggregated_mrio"]),
            **({"weight": float(record[WEIGHT_COLUMN])} if WEIGHT_COLUMN in map_df.columns else {}),
        }
        for record in map_df.loc[:, columns].to_dict(orient="records")
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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

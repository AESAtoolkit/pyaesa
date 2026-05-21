"""Shared ownership for AR6 sampling convergence diagnostics."""

import zlib
from typing import Any

import numpy as np
import pandas as pd


def distribution_stats_from_counts(
    values: np.ndarray,
    counts: np.ndarray,
) -> dict[str, float] | None:
    """Return exact sampled statistics from source values plus repetition counts."""
    clean_values = np.asarray(values, dtype=float)
    clean_counts = np.asarray(counts, dtype=np.int64)
    keep_mask = np.isfinite(clean_values) & (clean_counts > 0)
    if not bool(np.any(keep_mask)):
        return None
    clean_values = clean_values[keep_mask]
    clean_counts = clean_counts[keep_mask]
    order = np.argsort(clean_values, kind="stable")
    sorted_values = clean_values[order]
    sorted_counts = clean_counts[order]
    cumulative_counts = np.cumsum(sorted_counts, dtype=np.int64)
    total_count = int(cumulative_counts[-1])
    return {
        "mean": float(np.dot(sorted_values, sorted_counts) / total_count),
        "median": _linear_quantile_from_counts(sorted_values, cumulative_counts, total_count, 0.50),
        "p25": _linear_quantile_from_counts(sorted_values, cumulative_counts, total_count, 0.25),
        "p75": _linear_quantile_from_counts(sorted_values, cumulative_counts, total_count, 0.75),
        "p5": _linear_quantile_from_counts(sorted_values, cumulative_counts, total_count, 0.05),
        "p95": _linear_quantile_from_counts(sorted_values, cumulative_counts, total_count, 0.95),
    }


def _linear_quantile_from_counts(
    sorted_values: np.ndarray,
    cumulative_counts: np.ndarray,
    total_count: int,
    quantile: float,
) -> float:
    """Return the linear sample quantile without expanding duplicate runs."""
    if total_count <= 1:
        return float(sorted_values[0])
    position = (total_count - 1) * float(quantile)
    lower_rank = int(np.floor(position))
    upper_rank = int(np.ceil(position))
    lower_value = float(sorted_values[np.searchsorted(cumulative_counts, lower_rank, side="right")])
    upper_value = float(sorted_values[np.searchsorted(cumulative_counts, upper_rank, side="right")])
    if lower_rank == upper_rank:
        return lower_value
    return lower_value + (position - lower_rank) * (upper_value - lower_value)


def snapshots_are_stable(
    previous_snapshot: dict[tuple[str, str, str | int, str], float],
    current_snapshot: dict[tuple[str, str, str | int, str], float],
    *,
    relative_tolerance: float,
) -> bool:
    """Return whether all monitored summaries have stabilized."""
    if previous_snapshot.keys() != current_snapshot.keys():
        return False
    for key, current_value in current_snapshot.items():
        previous_value = previous_snapshot[key]
        if not np.isfinite(previous_value) or not np.isfinite(current_value):
            return False
        if current_value == 0.0 and previous_value == 0.0:
            continue
        scale = max(abs(previous_value), abs(current_value))
        if abs(current_value - previous_value) / scale > relative_tolerance:
            return False
    return True


def flatten_sampled_index_from_counts(
    buckets: list[dict[str, Any]],
    sampled_counts: dict[tuple[str, int], np.ndarray],
) -> list[tuple]:
    """Return sampled MultiIndex labels for plotting."""
    sampled_index: list[tuple] = []
    for bucket in buckets:
        counts = np.asarray(sampled_counts[bucket["key"]], dtype=np.int64)
        if not bool(np.any(counts)):
            continue
        for label, count in zip(bucket["labels"], counts, strict=True):
            if int(count) > 0:
                sampled_index.extend([label] * int(count))
    return sampled_index


def sampling_seed(var_sel: str, sampling_method: str) -> int:
    """Return a deterministic RNG seed for one variable and one run method."""
    return zlib.crc32(f"{var_sel}|{sampling_method}".encode("utf-8")) & 0xFFFFFFFF


def _grouped_row_sort_key(
    key: tuple[str, str, str | int],
) -> tuple[str, str, tuple[int, int, str]]:
    """Return a stable sort key for convergence log rows."""
    distribution_kind, category, ssp_family = key
    ssp_text = str(ssp_family)
    if isinstance(ssp_family, (int, np.integer)):
        ssp_key = (0, int(ssp_family), "")
    elif isinstance(ssp_family, (float, np.floating)) and float(ssp_family).is_integer():
        ssp_key = (0, int(ssp_family), "")
    elif ssp_text.isdigit():
        ssp_key = (0, int(ssp_text), "")
    else:
        ssp_key = (1, 0, ssp_text)
    return (str(distribution_kind), str(category), ssp_key)


def snapshot_to_log_rows(
    snapshot: dict[tuple[str, str, str | int, str], float],
    *,
    variable: str,
    sampling_method: str,
    rng_seed_value: int,
    final_runs_per_bucket: int,
    run_batch_size: int,
    maximum_runs_per_bucket: int,
    relative_tolerance: float,
    stable_checks_required: int,
) -> list[dict[str, str | int | float]]:
    """Return one structured log row per distribution/category/SSP summary."""
    grouped_rows: dict[tuple[str, str, str | int], dict[str, str | int | float]] = {}
    for (distribution_kind, category, ssp_family, metric), value in snapshot.items():
        key = (distribution_kind, category, ssp_family)
        row = grouped_rows.setdefault(
            key,
            {
                "variable": variable,
                "method": sampling_method,
                "distribution_kind": distribution_kind,
                "category": category,
                "ssp_family": ssp_family,
                "rng_seed": int(rng_seed_value),
                "final_runs_per_bucket": int(final_runs_per_bucket),
                "run_batch_size": int(run_batch_size),
                "maximum_runs_per_bucket": int(maximum_runs_per_bucket),
                "relative_tolerance": float(relative_tolerance),
                "stable_checks_required": int(stable_checks_required),
            },
        )
        row[metric] = float(value)
    return [grouped_rows[key] for key in sorted(grouped_rows, key=_grouped_row_sort_key)]


def study_rows_to_frame(
    rows: list[dict[str, str | int | float]],
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Return a study summary dataframe indexed like the ratio figures."""
    if not rows:
        my_mi = pd.MultiIndex(
            levels=[[], [], []],
            codes=[[], [], []],
            names=["variable", "Category", "Ssp_family"],
        )
        return pd.DataFrame(index=my_mi, columns=list(metrics))
    out_df = pd.DataFrame(rows)
    out_df = out_df.set_index(["variable", "Category", "Ssp_family"]).sort_index()
    return out_df.loc[:, list(metrics)].apply(pd.to_numeric, errors="raise")

"""Shared summary kernels for uncertainty run values."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from pyaesa.shared.runtime.memory import memory_bounded_rows

SUMMARY_STATISTICS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "p5",
    "p25",
    "median",
    "p75",
    "p95",
    "max",
)


def summary_scan_max_numeric_cells(
    *,
    output_format: str,
    memory_budget_bytes: int | None = None,
) -> int:
    """Return the cell budget for one public summary scan."""
    del output_format
    return memory_bounded_rows(
        bytes_per_row=_summary_scan_bytes_per_cell(),
        memory_budget_bytes=memory_budget_bytes,
    )


def _summary_scan_bytes_per_cell() -> int:
    float_bytes = np.dtype(np.float64).itemsize * len(
        ("input_values", "clean_values", "statistic_workspace", "quantile_workspace")
    )
    mask_bytes = np.dtype(np.bool_).itemsize * len(("valid_mask",))
    return float_bytes + mask_bytes


def column_block_width(
    *,
    run_count: int,
    row_count: int,
    max_numeric_cells_per_block: int | None = None,
) -> int:
    """Return a summary scan column width bounded by the shared cell budget."""
    max_cells = (
        summary_scan_max_numeric_cells(output_format="parquet")
        if max_numeric_cells_per_block is None
        else int(max_numeric_cells_per_block)
    )
    return max(1, min(int(row_count), int(max_cells) // max(1, run_count)))


def public_row_groups_are_identity_ordered(
    *,
    public_row_groups: tuple[tuple[str, ...], ...],
) -> bool:
    """Return whether groups map one to one onto public row ids."""
    return all(
        len(group) == 1 and int(group[0]) == index for index, group in enumerate(public_row_groups)
    )


def group_block_stop(
    *,
    groups: tuple[tuple[str, ...], ...],
    start: int,
    run_count: int,
    max_numeric_cells_per_block: int | None = None,
) -> int:
    """Return the exclusive group block stop for a grouped scan."""
    max_cells = (
        summary_scan_max_numeric_cells(output_format="parquet")
        if max_numeric_cells_per_block is None
        else int(max_numeric_cells_per_block)
    )
    total_columns = 0
    stop = start
    while stop < len(groups):
        next_total = total_columns + len(groups[stop])
        if stop > start and next_total * int(run_count) > int(max_cells):
            break
        total_columns = next_total
        stop += 1
    return stop


def assign_summary_columns(*, summary: pd.DataFrame, values: np.ndarray) -> None:
    """Assign summary statistic columns for a run by row value block."""
    for statistic, data in summary_arrays(values=values).items():
        summary[statistic] = data


def observed_group_summary_arrays(
    *,
    group_count: int,
    value_groups: Iterable[tuple[int, np.ndarray]],
    frequency_threshold: float | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return statistics from sparse observed values grouped by summary row."""
    stats = {
        statistic: np.full(int(group_count), np.nan, dtype=np.float64)
        for statistic in SUMMARY_STATISTICS
    }
    frequency = (
        np.full(int(group_count), np.nan, dtype=np.float64)
        if frequency_threshold is not None
        else np.empty(0, dtype=np.float64)
    )
    for group_index, values in value_groups:
        group_values = np.asarray(values, dtype=np.float64)
        nan_mask = np.isnan(group_values)
        observed = group_values[~nan_mask] if bool(nan_mask.any()) else group_values
        stats["mean"][group_index] = np.mean(observed)
        stats["std"][group_index] = np.std(observed, ddof=1) if observed.size > 1 else 0.0
        stats["min"][group_index] = np.min(observed)
        stats["max"][group_index] = np.max(observed)
        quantiles = _linear_quantiles_1d(values=observed)
        stats["p5"][group_index] = quantiles[0]
        stats["p25"][group_index] = quantiles[1]
        stats["median"][group_index] = quantiles[2]
        stats["p75"][group_index] = quantiles[3]
        stats["p95"][group_index] = quantiles[4]
        if frequency_threshold is not None:
            frequency[group_index] = np.count_nonzero(observed <= float(frequency_threshold)) / len(
                observed
            )
    return stats, frequency


def collapse_grouped_run_values(
    *,
    values: np.ndarray,
    columns: list[str],
    public_row_groups: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    """Collapse public row values to per run summary group values."""
    positions = {column: index for index, column in enumerate(columns)}
    collapsed = np.empty((values.shape[0], len(public_row_groups)), dtype=np.float64)
    group_sizes = sorted({len(group) for group in public_row_groups})
    for group_size in group_sizes:
        group_indexes = [
            index for index, group in enumerate(public_row_groups) if len(group) == group_size
        ]
        column_positions = np.array(
            [[positions[column] for column in public_row_groups[index]] for index in group_indexes],
            dtype=np.int64,
        )
        selected = values[:, column_positions]
        counts = np.sum(~np.isnan(selected), axis=2)
        sums = np.nansum(selected, axis=2)
        collapsed[:, group_indexes] = np.divide(
            sums,
            counts,
            out=np.full((values.shape[0], len(group_indexes)), np.nan, dtype=np.float64),
            where=counts > 0,
        )
    return collapsed


def summary_arrays(*, values: np.ndarray) -> dict[str, np.ndarray]:
    """Return summary statistic arrays for each value column."""
    nan_mask = np.isnan(values)
    if not bool(nan_mask.any()):
        return _finite_summary_arrays(values=values)
    empty = nan_mask.all(axis=0)
    valid = ~empty
    stats = {
        statistic: np.full(values.shape[1], np.nan, dtype=np.float64)
        for statistic in SUMMARY_STATISTICS
    }
    if np.any(valid):
        valid_values = values[:, valid]
        stats["mean"][valid] = np.nanmean(valid_values, axis=0)
        stats["std"][valid] = _nanstd_columns(values=valid_values)
        stats["min"][valid] = np.nanmin(valid_values, axis=0)
        stats["max"][valid] = np.nanmax(valid_values, axis=0)
        quantiles = _linear_quantiles_from_sorted_columns(
            values=valid_values,
            counts=np.sum(~np.isnan(valid_values), axis=0),
        )
        stats["p5"][valid] = quantiles[0]
        stats["p25"][valid] = quantiles[1]
        stats["median"][valid] = quantiles[2]
        stats["p75"][valid] = quantiles[3]
        stats["p95"][valid] = quantiles[4]
    return stats


def _finite_summary_arrays(*, values: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.mean(values, axis=0)
    std = (
        np.std(values, axis=0, ddof=1)
        if values.shape[0] > 1
        else np.zeros(values.shape[1], dtype=np.float64)
    )
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    quantiles = _linear_quantiles_from_sorted_columns(
        values=values,
        counts=np.full(values.shape[1], values.shape[0], dtype=np.int64),
    )
    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "p5": quantiles[0],
        "p25": quantiles[1],
        "median": quantiles[2],
        "p75": quantiles[3],
        "p95": quantiles[4],
        "max": maximum,
    }


def _nanstd_columns(*, values: np.ndarray) -> np.ndarray:
    counts = np.sum(~np.isnan(values), axis=0)
    means = np.nanmean(values, axis=0)
    centered = np.where(np.isnan(values), 0.0, values - means)
    variance = np.sum(centered * centered, axis=0) / np.maximum(counts - 1, 1)
    return np.where(counts > 1, np.sqrt(variance), 0.0)


def _linear_quantiles_from_sorted_columns(
    *,
    values: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Return default NumPy linear quantiles using in place column sorting."""
    values.sort(axis=0)
    positions = np.array([0.05, 0.25, 0.5, 0.75, 0.95]).reshape(-1, 1) * (counts - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    weight = positions - lower
    columns = np.arange(values.shape[1])
    return values[lower, columns] * (1.0 - weight) + values[upper, columns] * weight


def _linear_quantiles_1d(*, values: np.ndarray) -> np.ndarray:
    """Return default NumPy linear quantiles for one observed vector."""
    values.sort()
    positions = np.array([0.05, 0.25, 0.5, 0.75, 0.95]) * (len(values) - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    weight = positions - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight

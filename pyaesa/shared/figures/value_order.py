"""Shared ordering helpers for visible figure value categories."""

from collections.abc import Iterable, Mapping
import math
from typing import Any

import numpy as np
import pandas as pd


def finite_average(values: Iterable[float]) -> float | None:
    """Return the arithmetic mean of finite numeric values."""
    finite: list[float] = []
    for value in values:
        numeric = float(value)
        if math.isfinite(numeric):
            finite.append(numeric)
    if not finite:
        return None
    return sum(finite) / len(finite)


def order_labels_by_average_score(scores: Mapping[str, Iterable[float]]) -> list[str]:
    """Order labels by decreasing average visible value."""
    averages: dict[str, float] = {}
    for label, values in scores.items():
        average = finite_average(values)
        if average is not None:
            averages[str(label)] = average
    return sorted(averages, key=lambda label: (-averages[label], label))


def order_labels_by_average_within_group_rank(
    values: Iterable[tuple[str, str, float | None]],
) -> list[str]:
    """Order labels by average rank inside comparable value groups."""
    grouped_scores: dict[str, dict[str, list[float]]] = {}
    for group, label, value in values:
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        grouped_scores.setdefault(str(group), {}).setdefault(str(label), []).append(numeric)
    label_ranks: dict[str, list[float]] = {}
    for label_scores in grouped_scores.values():
        averages = {
            label: average
            for label, scores in label_scores.items()
            if (average := finite_average(scores)) is not None
        }
        for rank, label in enumerate(
            sorted(averages, key=lambda item: (-averages[item], item)),
            start=1,
        ):
            label_ranks.setdefault(label, []).append(float(rank))
    rank_averages = {
        label: average
        for label, ranks in label_ranks.items()
        if (average := finite_average(ranks)) is not None
    }
    return sorted(rank_averages, key=lambda label: (rank_averages[label], label))


def row_average_score(
    row: pd.Series,
    *,
    value_array_column: str | None = None,
    scalar_columns: tuple[str, ...] = ("mean", "median", "value"),
) -> float | None:
    """Return the representative plotted value for one figure row."""
    if value_array_column is not None and value_array_column in row.index:
        numeric = np.asarray(row[value_array_column], dtype=np.float64)
        return finite_average(numeric.tolist())
    data: dict[str, Any] = row.to_dict()
    for column in scalar_columns:
        if column not in data:
            continue
        value = data[column]
        if bool(pd.isna(value)):
            continue
        return float(value)
    return None


def frame_average_score(
    frame: pd.DataFrame,
    *,
    value_array_column: str | None = None,
    scalar_columns: tuple[str, ...] = ("mean", "median", "value"),
) -> float:
    """Return the average representative plotted value for a figure frame."""
    scores = [
        score
        for _index, row in frame.iterrows()
        if (
            score := row_average_score(
                pd.Series(row, copy=False),
                value_array_column=value_array_column,
                scalar_columns=scalar_columns,
            )
        )
        is not None
    ]
    average = finite_average(scores)
    return average if average is not None else float("-inf")

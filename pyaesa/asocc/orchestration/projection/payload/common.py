"""Shared payload normalization for projection builders."""

from typing import cast

import pandas as pd

from .basis import require_frame, require_series


def series_payload(value: pd.Series | pd.DataFrame, label: str) -> pd.Series:
    """Read one named payload object as a numeric series."""
    return require_series(cast(pd.Series, value), label=label)


def frame_payload(value: pd.Series | pd.DataFrame, label: str) -> pd.DataFrame:
    """Read one named payload object as a numeric frame."""
    return require_frame(cast(pd.DataFrame, value), label=label)


def stack_series_payload(
    frame: pd.DataFrame,
    label: str,
    *,
    names: list[str],
) -> pd.Series:
    """Convert one wide frame to a long numeric series with explicit index names."""
    stacked = series_payload(frame.stack(), label)
    if stacked.index.nlevels != len(names):
        raise ValueError(
            f"Stacked payload '{label}' has {stacked.index.nlevels} levels, expected {len(names)}."
        )
    stacked.index = stacked.index.set_names(names)
    return stacked


def reorder_series_levels_payload(
    series: pd.Series,
    *,
    order: list[str],
    label: str,
) -> pd.Series:
    """Reorder MultiIndex levels by names using integer positions."""
    if not isinstance(series.index, pd.MultiIndex):
        raise ValueError(f"Cannot reorder levels for payload '{label}': index is not MultiIndex.")
    index_names = [str(name) for name in series.index.names]
    missing = [name for name in order if name not in index_names]
    if missing:
        raise ValueError(f"Cannot reorder levels for payload '{label}': missing levels {missing}.")
    positions = [index_names.index(name) for name in order]
    reordered = series.reorder_levels(positions)
    reordered.index = reordered.index.set_names(order)
    return reordered

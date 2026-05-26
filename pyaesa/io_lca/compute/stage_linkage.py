"""Linkage ownership for upstream stage step by step attribution."""

import numpy as np
import pandas as pd

from pyaesa.shared.runtime.memory import memory_bounded_rows


def _pair_tokens(value: object) -> tuple[str, str]:
    """Return deterministic ``(region, sector)`` tokens from axis labels."""
    if isinstance(value, tuple):
        first = str(value[0]) if len(value) > 0 else ""
        second = str(value[1]) if len(value) > 1 else ""
        return first, second
    return str(value), ""


def dominant_parent_link_map(
    *,
    a_matrix: pd.DataFrame,
    q_prev: pd.Series,
    eps: float,
) -> dict[tuple[str, str], tuple[str, str]]:
    """Return mapping ``(stage_r_p, stage_s_p) -> (linked_from_r_p, linked_from_s_p)``."""
    q_aligned = pd.Series(
        pd.to_numeric(
            q_prev.reindex(a_matrix.columns, fill_value=0.0),
            errors="raise",
        ),
        copy=False,
    ).fillna(0.0)
    a_values = a_matrix.to_numpy(dtype=float, copy=False)
    q_values = q_aligned.to_numpy(dtype=float, copy=False)
    child_labels = list(a_matrix.index)
    parent_labels = list(a_matrix.columns)
    out: dict[tuple[str, str], tuple[str, str]] = {}
    chunk_size = _dominant_parent_link_chunk_size(parent_count=len(parent_labels))
    for start in range(0, len(child_labels), chunk_size):
        stop = min(start + chunk_size, len(child_labels))
        weighted = a_values[start:stop, :] * q_values[None, :]
        if weighted.size == 0:
            continue
        best_parent_idx = np.argmax(weighted, axis=1)
        best_vals = weighted[np.arange(weighted.shape[0]), best_parent_idx]
        for offset, child_label in enumerate(child_labels[start:stop]):
            child_pair = _pair_tokens(child_label)
            if abs(float(best_vals[offset])) <= float(eps):
                out[child_pair] = ("", "")
            else:
                out[child_pair] = _pair_tokens(parent_labels[int(best_parent_idx[offset])])
    return out


def _dominant_parent_link_chunk_size(*, parent_count: int) -> int:
    weighted_row_bytes = np.dtype(np.float64).itemsize * int(parent_count)
    result_row_bytes = np.dtype(np.int64).itemsize + np.dtype(np.float64).itemsize
    return memory_bounded_rows(bytes_per_row=max(1, weighted_row_bytes + result_row_bytes))

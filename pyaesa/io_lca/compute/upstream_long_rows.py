"""NumPy backed long row builders for IO-LCA upstream outputs."""

import numpy as np
import pandas as pd


def pair_labels(index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic string arrays for two level product labels."""
    if isinstance(index, pd.MultiIndex):
        first = index.get_level_values(0).astype(str).to_numpy()
        if index.nlevels > 1:
            second = index.get_level_values(1).astype(str).to_numpy()
        else:
            second = np.full(len(index), "", dtype=object)
        return first, second
    first_vals: list[str] = []
    second_vals: list[str] = []
    for raw in index.tolist():
        if isinstance(raw, tuple):
            first_vals.append(str(raw[0]) if len(raw) > 0 else "")
            second_vals.append(str(raw[1]) if len(raw) > 1 else "")
        else:
            first_vals.append(str(raw))
            second_vals.append("")
    return np.asarray(first_vals, dtype=object), np.asarray(second_vals, dtype=object)


def stage_rows_from_values(
    *,
    impact_labels: np.ndarray,
    stage_r_labels: np.ndarray,
    stage_s_labels: np.ndarray,
    direct_values: np.ndarray,
    embedded_values: np.ndarray,
    total_values: np.ndarray,
    eps: float,
) -> pd.DataFrame:
    """Build stage long rows from dense impact by product arrays."""
    mask = (np.abs(direct_values) + np.abs(embedded_values) + np.abs(total_values)) > float(eps)
    row_idx, col_idx = np.nonzero(mask)
    if row_idx.size == 0:
        return pd.DataFrame(
            columns=[
                "impact",
                "stage_r_p",
                "stage_s_p",
                "direct_at_stage",
                "embedded_from_deeper_stages",
                "stage_total",
            ]
        )
    return pd.DataFrame(
        {
            "impact": impact_labels[row_idx],
            "stage_r_p": stage_r_labels[col_idx],
            "stage_s_p": stage_s_labels[col_idx],
            "direct_at_stage": direct_values[row_idx, col_idx],
            "embedded_from_deeper_stages": embedded_values[row_idx, col_idx],
            "stage_total": total_values[row_idx, col_idx],
        }
    )


def value_rows_from_values(
    *,
    impact_labels: np.ndarray,
    r_labels: np.ndarray,
    s_labels: np.ndarray,
    values: np.ndarray,
    eps: float,
    value_column: str,
    r_column: str,
    s_column: str,
) -> pd.DataFrame:
    """Build long rows for one value matrix using numeric mask filtering."""
    mask = np.abs(values) > float(eps)
    row_idx, col_idx = np.nonzero(mask)
    if row_idx.size == 0:
        return pd.DataFrame(columns=["impact", r_column, s_column, value_column])
    return pd.DataFrame(
        {
            "impact": impact_labels[row_idx],
            r_column: r_labels[col_idx],
            s_column: s_labels[col_idx],
            value_column: values[row_idx, col_idx],
        }
    )

"""Prepared payloads and count based snapshots for AR6 figure sampling."""

from typing import Any

import numpy as np
import pandas as pd

from .plot_helpers import MT_TO_GT, remaining_budget_end_year, var_df, year_slice
from .sampling_convergence_utils import distribution_stats_from_counts


def _numeric_series(values: pd.Series) -> pd.Series:
    """Return one numeric pandas Series for sampling payload calculations."""
    return pd.Series(pd.to_numeric(values, errors="raise"), copy=False)


def _numeric_array(values: pd.Series) -> np.ndarray:
    """Return one float numpy array for sampling payload calculations."""
    return _numeric_series(values).to_numpy(dtype=float)


def build_variable_payload(
    *,
    harmonized_data: pd.DataFrame,
    tmp_proba_df: pd.DataFrame,
    var_sel: str,
    categories: list[str],
    study_period: list[int],
    remaining_budget_end_year_value: int | None,
    sampling_method: str,
) -> dict[str, Any]:
    """Return precomputed per variable sampling payloads."""
    selected_var_df = var_df(harmonized_data, var_sel).sort_index()
    study_years = year_slice(selected_var_df, int(study_period[0]), int(study_period[1]))
    budget_end_year = (
        remaining_budget_end_year(selected_var_df)
        if remaining_budget_end_year_value is None
        else int(remaining_budget_end_year_value)
    )
    remaining_years = (
        year_slice(selected_var_df, int(study_period[1]) + 1, int(budget_end_year))
        if budget_end_year is not None
        else []
    )
    labels = np.asarray(selected_var_df.index.to_list(), dtype=object)
    category_s = selected_var_df["Category"]
    ssp_s = _numeric_series(pd.Series(selected_var_df["Ssp_family"], copy=False))
    study_values = (
        _numeric_array(
            pd.Series(pd.DataFrame(selected_var_df.loc[:, study_years]).sum(axis=1), copy=False)
        )
        * MT_TO_GT
    )
    remaining_values = np.full(len(selected_var_df), np.nan, dtype=float)
    if remaining_years and budget_end_year in selected_var_df.columns:
        remaining_end = _numeric_array(pd.Series(selected_var_df[budget_end_year], copy=False))
        remaining_ok = np.isfinite(remaining_end)
        if bool(np.any(remaining_ok)):
            remaining_values[remaining_ok] = (
                _numeric_array(
                    pd.Series(
                        pd.DataFrame(selected_var_df.loc[remaining_ok, remaining_years]).sum(
                            axis=1
                        ),
                        copy=False,
                    )
                )
                * MT_TO_GT
            )
    proba_col = "proba_SRS" if sampling_method == "SRS" else "proba_LHS"
    buckets = []
    category_bucket_keys: dict[str, list[tuple[str, int]]] = {}
    all_ssps_l = list(sorted(set(ssp_s.dropna())))
    for curr_cat in categories:
        for curr_ssp in all_ssps_l:
            bucket_mask = (category_s == curr_cat).to_numpy() & (ssp_s.to_numpy() == curr_ssp)
            if not bool(np.any(bucket_mask)):
                continue
            bucket_index = selected_var_df.index[bucket_mask]
            bucket_probabilities = _numeric_array(
                pd.Series(tmp_proba_df.loc[bucket_index, proba_col], copy=False)
            )
            bucket_probabilities = np.nan_to_num(bucket_probabilities, nan=0.0)
            if float(bucket_probabilities.sum()) <= 0.0:
                continue
            positions = np.flatnonzero(bucket_mask)
            bucket_key = (str(curr_cat), int(curr_ssp))
            buckets.append(
                {
                    "key": bucket_key,
                    "category": str(curr_cat),
                    "ssp": int(curr_ssp),
                    "positions": positions,
                    "labels": labels[positions],
                    "study_values": study_values[positions],
                    "remaining_values": remaining_values[positions],
                    "probabilities": bucket_probabilities / bucket_probabilities.sum(),
                }
            )
            category_bucket_keys.setdefault(str(curr_cat), []).append(bucket_key)
    return {
        "buckets": buckets,
        "categories": list(category_bucket_keys),
        "category_bucket_keys": category_bucket_keys,
    }


def expected_snapshot_key_count(buckets: list[dict[str, Any]]) -> int:
    """Return the expected number of converged snapshot entries."""
    metric_count = 6
    study_group_count = 0
    remaining_group_count = 0
    categories = sorted({str(bucket["category"]) for bucket in buckets})
    for curr_cat in categories:
        study_group_count += 1
        if any(
            np.isfinite(np.asarray(bucket["remaining_values"], dtype=float)).any()
            for bucket in buckets
            if str(bucket["category"]) == curr_cat
        ):
            remaining_group_count += 1
    for bucket in buckets:
        study_group_count += 1
        if np.isfinite(np.asarray(bucket["remaining_values"], dtype=float)).any():
            remaining_group_count += 1
    return (study_group_count + remaining_group_count) * metric_count


def build_snapshot_from_counts(
    *,
    sampled_counts: dict[tuple[str, int], np.ndarray],
    payload: dict[str, Any],
    var_sel: str,
) -> tuple[dict[tuple[str, str, str | int, str], float], list[dict[str, str | int | float]]]:
    """Return the current convergence snapshot and study summary rows."""
    snapshot: dict[tuple[str, str, str | int, str], float] = {}
    study_rows: list[dict[str, str | int | float]] = []
    buckets_by_key = {bucket["key"]: bucket for bucket in payload["buckets"]}
    for curr_cat in payload["categories"]:
        study_values = []
        study_counts = []
        remaining_values = []
        remaining_counts = []
        for bucket_key in payload["category_bucket_keys"][curr_cat]:
            bucket = buckets_by_key[bucket_key]
            study_values.append(np.asarray(bucket["study_values"], dtype=float))
            study_counts.append(np.asarray(sampled_counts[bucket_key], dtype=np.int64))
            remaining_values.append(np.asarray(bucket["remaining_values"], dtype=float))
            remaining_counts.append(np.asarray(sampled_counts[bucket_key], dtype=np.int64))
        _append_summary_rows_from_counts(
            snapshot=snapshot,
            study_rows=study_rows,
            var_sel=var_sel,
            category=str(curr_cat),
            ssp="all",
            study_values=np.concatenate(study_values),
            study_counts=np.concatenate(study_counts),
            remaining_values=np.concatenate(remaining_values),
            remaining_counts=np.concatenate(remaining_counts),
        )
    for bucket in payload["buckets"]:
        bucket_counts = np.asarray(sampled_counts[bucket["key"]], dtype=np.int64)
        _append_summary_rows_from_counts(
            snapshot=snapshot,
            study_rows=study_rows,
            var_sel=var_sel,
            category=str(bucket["category"]),
            ssp=int(bucket["ssp"]),
            study_values=np.asarray(bucket["study_values"], dtype=float),
            study_counts=bucket_counts,
            remaining_values=np.asarray(bucket["remaining_values"], dtype=float),
            remaining_counts=bucket_counts,
        )
    return snapshot, study_rows


def _append_summary_rows_from_counts(
    *,
    snapshot: dict[tuple[str, str, str | int, str], float],
    study_rows: list[dict[str, str | int | float]],
    var_sel: str,
    category: str,
    ssp: str | int,
    study_values: np.ndarray,
    study_counts: np.ndarray,
    remaining_values: np.ndarray,
    remaining_counts: np.ndarray,
) -> None:
    """Append study and remaining summaries for one plotted distribution."""
    study_stats = distribution_stats_from_counts(study_values, study_counts)
    if study_stats is not None:
        for metric, value in study_stats.items():
            snapshot[("study", category, ssp, metric)] = value
        study_rows.append(
            {
                "variable": var_sel,
                "Category": category,
                "Ssp_family": ssp,
                **study_stats,
            }
        )
    remaining_stats = distribution_stats_from_counts(remaining_values, remaining_counts)
    if remaining_stats is not None:
        for metric, value in remaining_stats.items():
            snapshot[("remaining", category, ssp, metric)] = value

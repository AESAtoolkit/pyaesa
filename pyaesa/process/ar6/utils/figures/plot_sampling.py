"""Sampling diagnostics used by the AR6 LHS/SRS figures."""

from typing import Any, Callable

import numpy as np
import pandas as pd

from .figure_sampling_config import (
    CONVERGENCE_LOG_COLUMNS,
    RUN_BATCH_SIZE,
    STABLE_CHECKS_REQUIRED,
)
from .plot_helpers import var_df
from .sampling_convergence_utils import (
    flatten_sampled_index_from_counts,
    sampling_seed,
    snapshot_to_log_rows,
    snapshots_are_stable,
    study_rows_to_frame,
)
from .sampling_payloads import (
    build_snapshot_from_counts,
    build_variable_payload,
    expected_snapshot_key_count,
)

CONVERGENCE_METRICS = ("mean", "median", "p25", "p75", "p5", "p95")


def _ssp_family_sort_rank(value: object) -> float:
    """Return a stable sort rank for numeric SSP families and the aggregate 'all' bucket."""
    if value is None or value is pd.NA:
        return float("inf")
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return float("inf")
    if isinstance(value, str) and value.strip().lower() == "all":
        return float("inf")
    return float(str(value).strip())


def build_sampling_probability_df(
    harmonized_data: pd.DataFrame,
    all_variables_l: list[str],
    categories: list[str],
) -> pd.DataFrame:
    """Return the per row SRS and LHS sampling probabilities."""
    proba_srs = pd.Series(np.nan, index=harmonized_data.index, dtype=float)
    proba_lhs = pd.Series(np.nan, index=harmonized_data.index, dtype=float)
    cat_col = pd.Series(index=harmonized_data.index, dtype=object)
    ssp_col = pd.Series(index=harmonized_data.index, dtype=object)

    all_ssps_l = sorted(set(harmonized_data["Ssp_family"]))
    for var_sel in all_variables_l:
        tmp_filter_var_df = var_df(harmonized_data, var_sel)
        var_cat = tmp_filter_var_df["Category"]
        var_ssp = tmp_filter_var_df["Ssp_family"]
        for curr_cat in categories:
            for curr_ssp in all_ssps_l:
                bucket_mask = (var_cat == curr_cat) & (var_ssp == curr_ssp)
                bucket_idx = tmp_filter_var_df.index[bucket_mask]
                n_bucket = len(bucket_idx)
                if n_bucket == 0:
                    continue
                model_level = bucket_idx.droplevel(["scenario", "variable"])
                n_models = len(model_level.unique())
                model_counts = model_level.value_counts()
                per_model_n = model_counts.reindex(model_level).to_numpy(dtype=float)

                proba_srs.loc[bucket_idx] = 1.0 / n_bucket
                proba_lhs.loc[bucket_idx] = 1.0 / (n_models * per_model_n)
                cat_col.loc[bucket_idx] = curr_cat
                ssp_col.loc[bucket_idx] = curr_ssp

    tmp_proba_df = pd.DataFrame(
        {
            "Category": cat_col,
            "Ssp_family": ssp_col,
            "proba_SRS": proba_srs,
            "proba_LHS": proba_lhs,
        },
        index=harmonized_data.index,
    )
    return tmp_proba_df


def build_sampling_runs_until_convergence(
    harmonized_data: pd.DataFrame,
    tmp_proba_df: pd.DataFrame,
    all_variables_l: list[str],
    categories: list[str],
    study_period: list[int],
    remaining_budget_end_year_value: int | None,
    relative_tolerance: float,
    max_runs_per_bucket: int,
    status_callback: Callable[[str], None] | None = None,
    run_batch_size: int = RUN_BATCH_SIZE,
    stable_checks_required: int = STABLE_CHECKS_REQUIRED,
) -> tuple[dict[str, list[tuple]], dict[str, list[tuple]], pd.DataFrame, pd.DataFrame]:
    """Return sampled indices, study budget ratios, and convergence log rows."""
    sampled_index = {
        "SRS": {var: [] for var in all_variables_l},
        "LHS": {var: [] for var in all_variables_l},
    }
    study_rows = {"SRS": [], "LHS": []}
    convergence_rows: list[dict[str, str | int | float]] = []
    for sampling_method in ["SRS", "LHS"]:
        for var_sel in all_variables_l:
            var_result = _sample_variable_until_converged(
                harmonized_data=harmonized_data,
                tmp_proba_df=tmp_proba_df,
                var_sel=var_sel,
                categories=categories,
                study_period=study_period,
                remaining_budget_end_year_value=remaining_budget_end_year_value,
                sampling_method=sampling_method,
                relative_tolerance=relative_tolerance,
                max_runs_per_bucket=max_runs_per_bucket,
                status_callback=status_callback,
                run_batch_size=run_batch_size,
                stable_checks_required=stable_checks_required,
            )
            sampled_index[sampling_method][var_sel] = var_result["sampled_index"]
            study_rows[sampling_method].extend(var_result["study_rows"])
            convergence_rows.extend(var_result["convergence_rows"])
    srs_stats = study_rows_to_frame(study_rows["SRS"], CONVERGENCE_METRICS)
    lhs_stats = study_rows_to_frame(study_rows["LHS"], CONVERGENCE_METRICS)
    ratio_lhs_vs_srs = lhs_stats / srs_stats
    convergence_log_df = pd.DataFrame(convergence_rows, columns=list(CONVERGENCE_LOG_COLUMNS))
    if not convergence_log_df.empty:
        ssp_sort_rank = pd.Series(
            [_ssp_family_sort_rank(value) for value in convergence_log_df["ssp_family"].tolist()],
            index=convergence_log_df.index,
            dtype=float,
        )
        convergence_log_df = convergence_log_df.assign(
            _ssp_family_sort_rank=ssp_sort_rank,
            _ssp_family_sort_text=convergence_log_df["ssp_family"].astype(str),
        )
        convergence_log_df = convergence_log_df.sort_values(
            by=[
                "variable",
                "method",
                "distribution_kind",
                "category",
                "_ssp_family_sort_rank",
                "_ssp_family_sort_text",
            ],
            kind="stable",
        ).drop(columns=["_ssp_family_sort_rank", "_ssp_family_sort_text"])
    return sampled_index["SRS"], sampled_index["LHS"], ratio_lhs_vs_srs, convergence_log_df


def _sample_variable_until_converged(
    *,
    harmonized_data: pd.DataFrame,
    tmp_proba_df: pd.DataFrame,
    var_sel: str,
    categories: list[str],
    study_period: list[int],
    remaining_budget_end_year_value: int | None,
    sampling_method: str,
    relative_tolerance: float,
    max_runs_per_bucket: int,
    status_callback: Callable[[str], None] | None = None,
    run_batch_size: int = RUN_BATCH_SIZE,
    stable_checks_required: int = STABLE_CHECKS_REQUIRED,
) -> dict[str, Any]:
    """Sample one variable with one method until study and remaining budgets stabilize."""
    payload = build_variable_payload(
        harmonized_data=harmonized_data,
        tmp_proba_df=tmp_proba_df,
        var_sel=var_sel,
        categories=categories,
        study_period=study_period,
        remaining_budget_end_year_value=remaining_budget_end_year_value,
        sampling_method=sampling_method,
    )
    if not payload["buckets"]:
        return {"sampled_index": [], "study_rows": [], "convergence_rows": []}
    rng_seed_value = sampling_seed(var_sel, sampling_method)
    rng = np.random.default_rng(rng_seed_value)
    sampled_counts = {
        bucket["key"]: np.zeros(len(bucket["positions"]), dtype=np.int64)
        for bucket in payload["buckets"]
    }
    previous_snapshot = None
    stable_checks = 0
    runs_per_bucket = 0
    required_snapshot_size = expected_snapshot_key_count(payload["buckets"])
    _show_sampling_status(
        status_callback=status_callback,
        sampling_method=sampling_method,
        variable=var_sel,
        runs_per_bucket=runs_per_bucket,
        max_runs_per_bucket=max_runs_per_bucket,
        stable_checks=stable_checks,
        stable_checks_required=stable_checks_required,
    )
    while runs_per_bucket < max_runs_per_bucket:
        next_batch_size = min(run_batch_size, max_runs_per_bucket - runs_per_bucket)
        for bucket in payload["buckets"]:
            sampled_counts[bucket["key"]] += rng.multinomial(
                next_batch_size,
                bucket["probabilities"],
            ).astype(np.int64, copy=False)
        runs_per_bucket += next_batch_size
        _show_sampling_status(
            status_callback=status_callback,
            sampling_method=sampling_method,
            variable=var_sel,
            runs_per_bucket=runs_per_bucket,
            max_runs_per_bucket=max_runs_per_bucket,
            stable_checks=stable_checks,
            stable_checks_required=stable_checks_required,
        )
        current_snapshot, study_rows = build_snapshot_from_counts(
            sampled_counts=sampled_counts,
            payload=payload,
            var_sel=var_sel,
        )
        if len(current_snapshot) < required_snapshot_size:
            previous_snapshot = current_snapshot
            stable_checks = 0
            continue
        if previous_snapshot is not None and snapshots_are_stable(
            previous_snapshot,
            current_snapshot,
            relative_tolerance=relative_tolerance,
        ):
            stable_checks += 1
            _show_sampling_status(
                status_callback=status_callback,
                sampling_method=sampling_method,
                variable=var_sel,
                runs_per_bucket=runs_per_bucket,
                max_runs_per_bucket=max_runs_per_bucket,
                stable_checks=stable_checks,
                stable_checks_required=stable_checks_required,
            )
            if stable_checks >= stable_checks_required:
                return {
                    "sampled_index": flatten_sampled_index_from_counts(
                        payload["buckets"],
                        sampled_counts,
                    ),
                    "study_rows": study_rows,
                    "convergence_rows": snapshot_to_log_rows(
                        current_snapshot,
                        variable=var_sel,
                        sampling_method=sampling_method,
                        rng_seed_value=rng_seed_value,
                        final_runs_per_bucket=runs_per_bucket,
                        run_batch_size=run_batch_size,
                        maximum_runs_per_bucket=max_runs_per_bucket,
                        relative_tolerance=relative_tolerance,
                        stable_checks_required=stable_checks_required,
                    ),
                }
        else:
            stable_checks = 0
        previous_snapshot = current_snapshot
    raise RuntimeError(
        "AR6 sampling figures could not reach convergence for "
        f"{sampling_method} and variable {var_sel}. Increase 'figure_convergence_max_runs' and/or "
        "'figure_convergence_tol'."
    )


def _show_sampling_status(
    *,
    status_callback: Callable[[str], None] | None,
    sampling_method: str,
    variable: str,
    runs_per_bucket: int,
    max_runs_per_bucket: int,
    stable_checks: int,
    stable_checks_required: int,
) -> None:
    """Emit one transient progress line for the current sampling loop."""
    if status_callback is None:
        return
    status_callback(
        f"Generating {sampling_method} runs until convergence for {variable}: "
        f"{runs_per_bucket} runs/bucket, "
        f"stable {stable_checks}/{stable_checks_required}, "
        f"cap {max_runs_per_bucket}"
    )

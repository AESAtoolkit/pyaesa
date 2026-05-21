"""Model table normalization and merge guards for regression diagnostics."""

from typing import cast

import pandas as pd

from ...projection.regression.projection_clipping_log import CLIP_KEY_COLUMNS
from pyaesa.asocc.orchestration.write.regression_stats.columns import (
    REG_KEY,
    REGRESSION_MODELS_COLUMNS,
)


def _column_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one frame column as a pandas Series."""
    return pd.Series(frame.loc[:, column], copy=False)


def _is_blank_series(series: pd.Series) -> pd.Series:
    """Return mask for null/blank values in one frame column."""
    return series.isna() | series.astype(str).str.strip().eq("")


def _sample_key_values(frame: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    """Return compact regression-key diagnostics without row dictionaries."""
    sample_frame = frame.loc[mask, REG_KEY].head(5)
    return {
        "columns": list(REG_KEY),
        "values": [tuple(values) for values in sample_frame.itertuples(index=False, name=None)],
    }


def _raise_on_duplicate_keys(*, frame: pd.DataFrame, label: str) -> None:
    """Raise when one source frame contains duplicate regression keys."""
    if frame.empty:
        return
    missing_keys = [column for column in REG_KEY if column not in frame.columns]
    if missing_keys:
        raise ValueError(f"{label} is missing required regression key columns: {missing_keys}.")
    dup_mask = frame.duplicated(subset=REG_KEY, keep=False)
    if not bool(dup_mask.any()):
        return
    sample = _sample_key_values(frame, dup_mask)
    raise ValueError(f"{label} contains duplicate regression keys: sample={sample}")


def _raise_on_missing_uncertainty(*, frame: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise when merged stats rows are missing uncertainty scalars."""
    missing = pd.Series(False, index=frame.index)
    for column in required_columns:
        if column not in frame.columns:
            missing |= True
            continue
        missing |= _is_blank_series(_column_series(frame, column))
    if not bool(missing.any()):
        return
    sample = _sample_key_values(frame, missing)
    raise ValueError(
        f"Missing uncertainty scalars for regression stats rows: sample_keys={sample}."
    )


def _normalize_regression_models_frame(
    frame: pd.DataFrame, *, clip_counts: dict[tuple[str, ...], int]
) -> pd.DataFrame:
    """Normalize merged stats+uncertainty frame to canonical schema."""
    out = frame.copy()
    for column in REGRESSION_MODELS_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    out = out.astype(object)

    model_type = out["model_type"].astype(str)
    is_ols_level = model_type.eq("ols_level")
    is_log_ratio = model_type.eq("log_ratio_time")

    out.loc[
        is_ols_level & _is_blank_series(_column_series(out, "x_transform")),
        "x_transform",
    ] = "level"
    out.loc[
        is_ols_level & _is_blank_series(_column_series(out, "y_transform")),
        "y_transform",
    ] = "level"
    out.loc[is_ols_level, "x_center_value"] = ""

    out.loc[
        is_log_ratio & _is_blank_series(_column_series(out, "x_transform")),
        "x_transform",
    ] = "centered"
    out.loc[
        is_log_ratio & _is_blank_series(_column_series(out, "y_transform")),
        "y_transform",
    ] = "log_ratio"
    share_missing_center = is_log_ratio & _is_blank_series(_column_series(out, "x_center_value"))
    if bool(share_missing_center.any()):
        sample = _sample_key_values(out, share_missing_center)
        raise ValueError(
            "log_ratio_time rows require x_center_value from fit-time metadata: "
            f"sample_keys={sample}."
        )

    share_blank_baseline = is_log_ratio & _is_blank_series(_column_series(out, "baseline_object"))
    out.loc[share_blank_baseline, "baseline_object"] = out.loc[
        share_blank_baseline, "denominator_object"
    ]
    share_blank_category = is_log_ratio & _is_blank_series(_column_series(out, "category_object"))
    out.loc[share_blank_category, "category_object"] = out.loc[
        share_blank_category, "numerator_object"
    ]

    out.loc[is_ols_level, "deterministic_clip_lower"] = 0.0
    out.loc[is_ols_level, "deterministic_clip_applied_count_hint"] = "no clipping"
    if clip_counts:
        level_keys = out.loc[is_ols_level, CLIP_KEY_COLUMNS].astype(str)
        level_counts = [int(clip_counts.get(tuple(values), 0)) for values in level_keys.to_numpy()]
        for idx, count in zip(level_keys.index, level_counts):
            if int(count) <= 0:
                continue
            out.at[idx, "deterministic_clip_applied_count_hint"] = (
                f"{int(count)} clipped value(s) (see projection_clipping_log.csv)"
            )
    out.loc[~is_ols_level, "deterministic_clip_lower"] = ""
    out.loc[~is_ols_level, "deterministic_clip_applied_count_hint"] = ""

    out = out.loc[:, REGRESSION_MODELS_COLUMNS]
    return cast(
        pd.DataFrame,
        out.sort_values(REG_KEY, kind="mergesort").reset_index(drop=True),
    )


def _merge_regression_models_frame(
    *,
    stats_frame: pd.DataFrame,
    uncertainty_frame: pd.DataFrame,
    clip_counts: dict[tuple[str, ...], int],
    required_uncertainty_columns: list[str],
) -> pd.DataFrame:
    """Merge deterministic stats and uncertainty rows on strict regression key."""
    _raise_on_duplicate_keys(frame=stats_frame, label="regression_stats_rows")
    _raise_on_duplicate_keys(
        frame=uncertainty_frame,
        label="regression_uncertainty_rows",
    )
    merged = stats_frame.merge(
        uncertainty_frame,
        on=REG_KEY,
        how="left",
        suffixes=("", "_unc"),
    )
    _raise_on_missing_uncertainty(
        frame=merged,
        required_columns=required_uncertainty_columns,
    )
    return _normalize_regression_models_frame(merged, clip_counts=clip_counts)

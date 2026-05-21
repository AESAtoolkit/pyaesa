"""Projection clipping diagnostics for nonnegative level constraints."""

from pathlib import Path

import pandas as pd
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent

from ....runtime.paths.deterministic import (
    allocate_regression_logs_dir,
    projection_clipping_log_path,
)
from ..config.types import BASE_REGRESSION_KEY

CLIP_KEY_COLUMNS = list(BASE_REGRESSION_KEY)


def _numeric_series(series: pd.Series) -> pd.Series:
    """Return one numeric Series with stable pandas metadata."""
    numeric = pd.to_numeric(pd.Series(series, copy=False), errors="raise")
    return pd.Series(numeric, index=series.index, name=series.name, copy=False)


def clip_counts_by_key(
    *,
    proj_base: Path,
    fit_start_year: int,
    fit_end_year: int,
    source: str,
    group_version: str | None,
) -> dict[tuple[str, ...], int]:
    """Return clipping event counts keyed by deterministic regression identity."""
    path = (
        allocate_regression_logs_dir(
            proj_base=proj_base,
            source=source,
            group_version=group_version,
        )
        / "projection_clipping_log.csv"
    )
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}
    if frame.empty:
        return {}
    if any(column not in frame.columns for column in CLIP_KEY_COLUMNS):
        return {}
    grouped = (
        frame.loc[:, CLIP_KEY_COLUMNS]
        .astype(str)
        .groupby(CLIP_KEY_COLUMNS, sort=False, dropna=False)
        .size()
    )
    counts = grouped.rename("clip_count").reset_index()
    return {
        tuple(str(value) for value in key_values): int(clip_count)
        for *key_values, clip_count in counts.loc[:, [*CLIP_KEY_COLUMNS, "clip_count"]].itertuples(
            index=False, name=None
        )
    }


def write_projection_clipping_log(
    *,
    before: pd.Series,
    source: str,
    projection_branch: str,
    fu_code: str,
    l2_method: str,
    target_object: str,
    year: int,
    unit: str,
    fit_start_year: int,
    fit_end_year: int,
    state,
) -> None:
    """Append rows for negative projected level values that were clipped to 0."""
    negatives = _numeric_series(before)
    negatives = pd.Series(negatives.loc[negatives < 0.0], copy=False)
    if negatives.empty:
        return
    path = projection_clipping_log_path(
        state=state,
    )
    path = ensure_file_parent(path)
    out = pd.DataFrame(
        {
            "projection_branch": str(projection_branch),
            "source": str(source),
            "fu_code": str(fu_code),
            "l2_method": str(l2_method),
            "target_object": str(target_object),
            "year": int(year),
            "unit": str(unit),
            "domain_key": [str(key) for key in negatives.index],
            "original_value": negatives.to_numpy(dtype=float),
            "clipped_value": 0.0,
        }
    )
    mode = "a" if path.exists() else "w"
    out.to_csv(path, mode=mode, index=False, header=mode == "w")

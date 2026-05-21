"""Helpers for targeted baseline vs refactor output equivalence checks."""

from pathlib import Path
from typing import Any, cast

import pandas as pd


def _as_frame(value: object) -> pd.DataFrame:
    """Return DataFrame payload for table readers that may yield Series."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame()
    return pd.DataFrame(value)


def _read_table(path: Path) -> pd.DataFrame:
    """Read one output table by suffix."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _as_frame(pd.read_csv(path))
    if suffix == ".pickle":
        return _as_frame(pd.read_pickle(path))
    if suffix == ".parquet":
        return _as_frame(pd.read_parquet(path))
    raise ValueError(f"Unsupported output suffix: {path.suffix}")


def _sort_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic row/column ordering for value comparison."""
    year_cols = sorted([c for c in df.columns if str(c).isdigit()], key=lambda c: int(str(c)))
    id_cols = [c for c in df.columns if c not in year_cols]
    out = df[id_cols + year_cols].copy()
    if not id_cols:
        return cast(pd.DataFrame, out.reset_index(drop=True))
    out_any: Any = out
    return out_any.sort_values(id_cols, kind="mergesort").reset_index(drop=True)


def compare_output_files(
    *,
    baseline_path: Path,
    candidate_path: Path,
    atol: float = 1e-12,
    rtol: float = 0.0,
) -> None:
    """Assert strict numeric equivalence ignoring row order."""
    base = _sort_for_compare(_read_table(baseline_path))
    cand = _sort_for_compare(_read_table(candidate_path))
    if list(base.columns) != list(cand.columns):
        raise AssertionError(
            f"Column mismatch for {baseline_path.name}: "
            f"{list(base.columns)} != {list(cand.columns)}"
        )
    id_cols = [c for c in base.columns if not str(c).isdigit()]
    year_cols = [c for c in base.columns if str(c).isdigit()]
    pd.testing.assert_frame_equal(base[id_cols], cand[id_cols], check_like=False)
    pd.testing.assert_frame_equal(
        base[year_cols],
        cand[year_cols],
        check_like=False,
        atol=atol,
        rtol=rtol,
    )

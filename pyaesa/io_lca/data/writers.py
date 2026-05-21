"""Small persistence ownership for IO-LCA tabular outputs."""

from pathlib import Path
import shutil
from typing import cast

import pandas as pd
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent, write_via_atomic_temp


def write_table(
    *,
    path: Path,
    frame: pd.DataFrame,
    output_format: str,
) -> None:
    """Write DataFrame in deterministic column/index free form.

    Args:
        path: Destination file path.
        frame: Tabular payload.
        output_format: One of ``csv``, ``pickle``, ``parquet``.
    """
    path = ensure_file_parent(path)
    if output_format == "csv":
        write_via_atomic_temp(path, writer=lambda tmp_path: frame.to_csv(tmp_path, index=False))
        return
    if output_format == "pickle":
        write_via_atomic_temp(path, writer=lambda tmp_path: frame.to_pickle(tmp_path))
        return
    write_via_atomic_temp(path, writer=lambda tmp_path: frame.to_parquet(tmp_path, index=False))


def read_table(path: Path) -> pd.DataFrame:
    """Read persisted DataFrame from supported extensions."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".pickle", ".pkl"}:
        return cast(pd.DataFrame, pd.read_pickle(path))
    return pd.read_parquet(path)


def merge_with_existing(
    *,
    path: Path,
    fresh: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    """Merge fresh rows with existing file and deduplicate by key columns."""
    if not path.exists():
        return fresh
    existing = read_table(path)
    combined = pd.concat([existing, fresh], ignore_index=True)
    if key_columns:
        combined = combined.drop_duplicates(subset=key_columns, keep="last")
    return combined


def long_to_year_wide(
    *,
    frame: pd.DataFrame,
    id_columns: list[str],
    value_column: str,
    year_column: str = "year",
) -> pd.DataFrame:
    """Pivot long rows to one row per id with year columns.

    Args:
        frame: Long form rows.
        id_columns: Identifier columns preserved as row keys.
        value_column: Numeric value column to pivot.
        year_column: Year column to pivot into wide columns.
    """
    if frame.empty:
        return pd.DataFrame(columns=[*id_columns])
    work = frame.copy()
    work[year_column] = pd.Series(work[year_column], copy=False).astype(int).astype(str)
    work[value_column] = pd.to_numeric(work[value_column], errors="raise")
    if id_columns:
        grouped = cast(
            pd.Series,
            work.groupby([*id_columns, year_column], dropna=False)[value_column].sum(min_count=1),
        )
        wide = grouped.unstack(year_column).reset_index().rename_axis(columns=None)
    else:
        totals = cast(
            pd.Series,
            work.groupby(year_column, dropna=False)[value_column].sum(min_count=1),
        )
        wide = totals.to_frame().T.reset_index(drop=True).rename_axis(columns=None)
    year_cols = sorted(
        [col for col in wide.columns if col not in set(id_columns)],
        key=lambda col: int(str(col)),
    )
    ordered = [*id_columns, *year_cols]
    out = wide.loc[:, ordered]
    if id_columns:
        out = out.sort_values(id_columns, kind="stable").reset_index(drop=True)
    return out


def clear_scope_outputs(*, scope_root: Path) -> None:
    """Delete one output subtree when it exists."""
    if scope_root.exists():
        shutil.rmtree(scope_root)

"""Shared explicit series-label helpers for deterministic figure rendering."""

import pandas as pd

from pyaesa.shared.figures.method_identity import display_pair
from pyaesa.shared.tabular.scalars import is_display_missing


def resolve_series_label(
    row: pd.Series,
    *,
    label_columns: tuple[str, ...],
    skip_columns: set[str] | None = None,
    display_aliases: dict[str, str] | None = None,
    context: str,
) -> str:
    """Return one visible label from explicit columns only."""
    ignored = set(skip_columns or set())
    if (
        "l1_l2_method" in row.index
        and "l1_l2_method" not in ignored
        and not is_display_missing(row["l1_l2_method"])
    ):
        ignored.update({"l2_method", "l1_method"})
    if not label_columns:
        raise ValueError(f"{context} requires non-empty explicit label columns.")
    effective_columns = [column for column in label_columns if column not in ignored]
    parts: list[str] = []
    for column in effective_columns:
        if column not in row.index:
            continue
        part = display_pair(column, row[column], display_aliases=display_aliases)
        if part is not None:
            parts.append(part)
    if parts:
        return ", ".join(parts)
    missing_columns = [column for column in effective_columns if column not in row.index]
    empty_columns = [
        column
        for column in effective_columns
        if column in row.index
        and display_pair(column, row[column], display_aliases=display_aliases) is None
    ]
    available_columns = [column for column in row.index if column not in ignored]
    raise ValueError(
        f"{context} could not build a visible label from explicit label columns. "
        f"Configured label columns={list(label_columns)}. "
        f"Effective label columns={effective_columns}. "
        f"Missing label columns={missing_columns}. "
        f"Present but empty label columns={empty_columns}. "
        f"Available non-skipped columns={available_columns}."
    )


def require_series_label(
    row: pd.Series,
    *,
    context: str,
) -> str:
    """Return one non-empty precomputed ``series_label`` from one row."""
    if "series_label" not in row.index:
        raise ValueError(
            f"{context} received a deterministic figure row without the required "
            "'series_label' column. The figure frame must include populated series "
            "labels before rendering."
        )
    label = str(row["series_label"]).strip()
    if not label:
        raise ValueError(f"{context} found an empty 'series_label' value.")
    return label


def with_series_label_column(
    frame: pd.DataFrame,
    *,
    label_columns: tuple[str, ...],
    skip_columns: set[str] | None = None,
    display_aliases: dict[str, str] | None = None,
    context: str,
) -> pd.DataFrame:
    """Return a frame copy with one explicit ``series_label`` column."""
    if frame.empty:
        out = frame.copy()
        out["series_label"] = pd.Series(dtype="object")
        return out
    out = frame.copy()
    out["series_label"] = [
        resolve_series_label(
            pd.Series(row, copy=False),
            label_columns=label_columns,
            skip_columns=skip_columns,
            display_aliases=display_aliases,
            context=context,
        )
        for _, row in out.iterrows()
    ]
    return out

"""Canonical row-owned ``l2_reuse_year`` helpers."""

from pathlib import Path

import pandas as pd


def canonicalize_l2_reuse_year_column(
    frame: pd.DataFrame,
    *,
    path: Path | None = None,
) -> pd.DataFrame:
    """Return one frame with canonical row-owned ``l2_reuse_year`` values.

    Args:
        frame: Input frame that may contain ``l2_reuse_year``.
        path: Optional persisted path used only for precise error messages.

    Returns:
        Copy of ``frame`` with a validated nullable integer ``l2_reuse_year``
        column when the identity is present.

    Raises:
        ValueError: If ``l2_reuse_year`` contains non-numeric values.
    """
    out = frame.copy()
    out.attrs.update(frame.attrs)
    l2_reuse_series = _coerce_optional_l2_reuse_year_series(out, path=path)
    if l2_reuse_series is not None:
        out["l2_reuse_year"] = l2_reuse_series
    return out


def frame_l2_reuse_years(frame: pd.DataFrame) -> tuple[int, ...]:
    """Return sorted row-owned L2 reuse years present in one frame."""
    if "l2_reuse_year" not in frame.columns:
        return tuple()
    series = pd.Series(frame.loc[:, "l2_reuse_year"], copy=False).dropna()
    if series.empty:
        return tuple()
    numeric = pd.Series(pd.to_numeric(series, errors="raise"), copy=False).astype(int)
    return tuple(sorted({int(value) for value in numeric.tolist()}))


def _coerce_optional_l2_reuse_year_series(
    frame: pd.DataFrame,
    *,
    path: Path | None,
) -> pd.Series | None:
    """Return one nullable integer ``l2_reuse_year`` series when the column exists."""
    column = "l2_reuse_year"
    if column not in frame.columns:
        return None
    try:
        numeric_series = pd.Series(
            pd.to_numeric(pd.Series(frame.loc[:, column], copy=False), errors="raise"),
            copy=False,
        )
    except (TypeError, ValueError) as exc:
        location = f" in '{path.name}'" if path is not None else ""
        raise ValueError(f"Column 'l2_reuse_year'{location} contains non-numeric values.") from exc
    return numeric_series.astype("Int64")

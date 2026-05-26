"""Shared LCIA-method and impact scope helpers for figure planning."""

from pathlib import Path
from collections.abc import Iterator
from typing import cast

import numpy as np
import pandas as pd

from pyaesa.shared.figures.lcia_metadata import resolve_frame_impact_title
from pyaesa.shared.tabular.scalars import is_display_missing, sanitize_token


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return is_display_missing(value)


def _reject_lcia_method_without_impact(
    frame: pd.DataFrame,
    *,
    impact_column: str,
    lcia_method_column: str = "lcia_method",
) -> None:
    """Fail when an LCIA-scoped row has no impact identity."""
    if impact_column not in frame.columns or lcia_method_column not in frame.columns:
        return
    impact_series = pd.Series(frame.loc[:, impact_column], copy=False)
    method_series = pd.Series(frame.loc[:, lcia_method_column], copy=False)
    invalid_mask = impact_series.map(_is_missing) & method_series.map(
        lambda value: not _is_missing(value)
    )
    if invalid_mask.any():
        methods = sorted(
            {
                str(value).strip()
                for value in method_series.loc[invalid_mask].tolist()
                if not _is_missing(value)
            }
        )
        raise ValueError(
            "Figure rendering found LCIA rows with missing impact identity. "
            f"Observed lcia_method values: {methods}."
        )


def lcia_method_slices(
    frame: pd.DataFrame,
    *,
    column: str = "lcia_method",
    fill_generic_method: bool = True,
) -> Iterator[tuple[str, str, pd.DataFrame, str | None]]:
    """Yield figure slices scoped to one LCIA method."""
    if frame.empty or column not in frame.columns:
        yield "all", "", frame.copy(), None
        return
    method_series = pd.Series(frame.loc[:, column], copy=False)
    methods = sorted(
        {str(value).strip() for value in method_series.tolist() if not _is_missing(value)}
    )
    if not methods:
        yield "all", "", frame.copy(), None
        return
    generic_mask = method_series.map(_is_missing)
    for lcia_method_label in methods:
        method_mask = method_series.astype(str).eq(str(lcia_method_label))
        scoped = frame.loc[method_mask | generic_mask].copy()
        if fill_generic_method:
            scoped[column] = str(lcia_method_label)
        token = sanitize_token(lcia_method_label)
        yield (
            token,
            str(lcia_method_label),
            scoped.reset_index(drop=True),
            str(lcia_method_label),
        )


def resolve_unique_lcia_method(
    frame: pd.DataFrame,
    *,
    column: str = "lcia_method",
) -> str | None:
    """Return the unique LCIA method carried by one figure frame when available."""
    if frame.empty or column not in frame.columns:
        return None
    method_series = pd.Series(frame.loc[:, column], copy=False)
    values = sorted(
        {str(value).strip() for value in method_series.tolist() if not _is_missing(value)}
    )
    if len(values) != 1:
        return None
    return values[0]


def impact_slices(
    frame: pd.DataFrame,
    *,
    impact_column: str | None,
    repeat_generic: bool,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield impact specific slices, optionally repeating generic rows."""
    if frame.empty or impact_column is None or impact_column not in frame.columns:
        yield "value", frame.copy()
        return
    _reject_lcia_method_without_impact(frame, impact_column=impact_column)
    impact_series = pd.Series(frame.loc[:, impact_column], copy=False)
    impacts = sorted(
        {str(value).strip() for value in impact_series.tolist() if not _is_missing(value)}
    )
    if not impacts:
        yield "value", frame.copy()
        return
    generic_mask = impact_series.map(_is_missing)
    yielded = False
    for impact in impacts:
        impact_mask = impact_series.astype(str).eq(str(impact))
        if repeat_generic:
            scoped = frame.loc[impact_mask | generic_mask].copy()
        else:
            scoped = frame.loc[impact_mask].copy()
        if scoped.empty:
            continue
        scoped.loc[
            scoped[impact_column].notna() & scoped[impact_column].astype(str).eq(str(impact)),
            impact_column,
        ] = str(impact)
        yielded = True
        yield str(impact), scoped.reset_index(drop=True)
    if not yielded:
        yield "value", frame.copy()


def combined_impact_slices(
    frame: pd.DataFrame,
    *,
    impact_column: str = "impact",
    lcia_method_column: str = "lcia_method",
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Yield combined comparison slices scoped to one studied impact."""
    if frame.empty or impact_column not in frame.columns:
        yield "all", "", frame.copy()
        return
    _reject_lcia_method_without_impact(
        frame,
        impact_column=impact_column,
        lcia_method_column=lcia_method_column,
    )
    impact_series = pd.Series(frame.loc[:, impact_column], copy=False)
    nonmissing_impacts = sorted(
        {str(value).strip() for value in impact_series.tolist() if not _is_missing(value)}
    )
    if not nonmissing_impacts:
        yield "all", "", frame.copy()
        return
    if lcia_method_column not in frame.columns:
        raise ValueError(
            "Combined deterministic figure rendering requires 'lcia_method' when impact rows "
            "are present."
        )
    method_series = pd.Series(frame.loc[:, lcia_method_column], copy=False)
    mismatched = impact_series.map(lambda value: not _is_missing(value)) & method_series.map(
        _is_missing
    )
    if mismatched.any():
        raise ValueError(
            "Combined deterministic figure rendering found impact rows with missing "
            f"'{lcia_method_column}'."
        )
    generic_mask = impact_series.map(_is_missing)
    yielded = False
    for impact in nonmissing_impacts:
        impact_mask = impact_series.astype(str).eq(str(impact))
        scoped = frame.loc[impact_mask | generic_mask].copy()
        if scoped.empty:
            continue
        scoped.loc[:, impact_column] = str(impact)
        title = cast(str, resolve_frame_impact_title(frame.loc[impact_mask].copy()))
        yielded = True
        yield sanitize_token(impact), title, scoped.reset_index(drop=True)
    if not yielded:
        yield "all", "", frame.copy()


def combined_lcia_impact_slices(
    frame: pd.DataFrame,
    *,
    impact_column: str = "impact",
    lcia_method_column: str = "lcia_method",
) -> Iterator[tuple[str, str, pd.DataFrame, str | None]]:
    """Yield combined comparison slices without expanding generic rows globally."""
    if frame.empty or impact_column not in frame.columns:
        yield "all", "", frame.copy(), None
        return
    _reject_lcia_method_without_impact(
        frame,
        impact_column=impact_column,
        lcia_method_column=lcia_method_column,
    )
    if lcia_method_column not in frame.columns:
        raise ValueError(
            "Combined deterministic figure rendering requires 'lcia_method' when impact rows "
            "are present."
        )
    impact_series = pd.Series(frame.loc[:, impact_column], copy=False)
    method_series = pd.Series(frame.loc[:, lcia_method_column], copy=False)
    impact_present = impact_series.map(lambda value: not _is_missing(value))
    method_missing = method_series.map(_is_missing)
    if bool((impact_present & method_missing).any()):
        raise ValueError(
            "Combined deterministic figure rendering found impact rows with missing "
            f"'{lcia_method_column}'."
        )
    methods = sorted(
        {str(value).strip() for value in method_series.tolist() if not _is_missing(value)}
    )
    if not methods:
        yield "all", "", frame.copy(), None
        return
    generic = frame.loc[method_missing & ~impact_present].copy()
    yielded = False
    for method in methods:
        method_rows = frame.loc[method_series.astype(str).eq(str(method))].copy()
        method_impacts = sorted(
            {
                str(value).strip()
                for value in method_rows[impact_column].tolist()
                if not _is_missing(value)
            }
        )
        for impact in method_impacts:
            impact_rows = method_rows.loc[method_rows[impact_column].astype(str).eq(str(impact))]
            if impact_rows.empty:
                continue
            scoped = (
                pd.concat([impact_rows.copy(), generic.copy()], ignore_index=True)
                if not generic.empty
                else impact_rows.copy()
            )
            scoped.loc[:, impact_column] = str(impact)
            title = cast(str, resolve_frame_impact_title(impact_rows.copy()))
            yielded = True
            yield sanitize_token(impact), title, scoped.reset_index(drop=True), method
    if not yielded:
        yield "all", "", frame.copy(), None


def suffix_path(base: Path, token: str) -> Path:
    """Return a path with one stable token appended to the base stem."""
    return base.parent / f"{base.name}__{sanitize_token(token)}"

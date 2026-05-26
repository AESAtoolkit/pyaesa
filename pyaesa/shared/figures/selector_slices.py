"""Shared selector slice helpers for package figure rendering."""

from collections.abc import Iterator

import numpy as np
import pandas as pd

from pyaesa.shared.figures.contracts import SELECTOR_COLUMNS, figure_selector_columns
from pyaesa.shared.figures.title_contract import SelectorScopeRequest, resolve_selector_scope
from pyaesa.shared.selectors.path_tokens import (
    deduplicated_selector_value_tokens,
    selector_axis_token,
    selector_value_text,
)


def selector_slices(
    frame: pd.DataFrame,
    *,
    selector_columns: tuple[str, ...] = SELECTOR_COLUMNS,
    selector_scope_request: SelectorScopeRequest | None = None,
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Yield figure slices split by FU selector combinations."""
    scoped_selector_columns = figure_selector_columns(
        frame,
        selector_columns=selector_columns,
    )
    if frame.empty:
        yield (
            "all",
            resolve_selector_scope(
                frame=frame,
                selector_columns=scoped_selector_columns,
                selector_scope_request=selector_scope_request,
            )
            or "",
            frame.copy(),
        )
        return
    present = [column for column in scoped_selector_columns if column in frame.columns]
    if not present:
        yield (
            "all",
            resolve_selector_scope(
                frame=frame,
                selector_columns=scoped_selector_columns,
                selector_scope_request=selector_scope_request,
            )
            or "",
            frame.copy(),
        )
        return
    reference_frame = frame
    value_token_maps = {
        column: deduplicated_selector_value_tokens(reference_frame[column].tolist())
        for column in present
    }
    for key, subset in frame.groupby(present, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        token_parts: list[str] = []
        for column, value in zip(present, key_tuple, strict=True):
            value_token = value_token_maps[column][selector_value_text(value)]
            token_parts.append(f"{selector_axis_token(column)}_{value_token}")
        yield (
            "__".join(token_parts) if token_parts else "all",
            resolve_selector_scope(
                frame=subset,
                reference_frame=reference_frame,
                selector_columns=scoped_selector_columns,
                selector_scope_request=selector_scope_request,
            )
            or "",
            subset.copy().reset_index(drop=True),
        )


def matching_selector_mask(
    frame: pd.DataFrame,
    *,
    reference_row: pd.Series,
    selector_columns: tuple[str, ...] = SELECTOR_COLUMNS,
) -> pd.Series:
    """Return one missing-aware selector equality mask for one reference row."""
    scoped_selector_columns = figure_selector_columns(
        frame,
        selector_columns=selector_columns,
    )
    present = [
        column
        for column in scoped_selector_columns
        if column in frame.columns and column in reference_row.index
    ]
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for column in present:
        frame_series = pd.Series(frame[column], copy=False)
        reference_value = reference_row[column]
        missing = pd.isna(reference_value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            mask = mask & frame_series.isna()
            continue
        mask = mask & frame_series.eq(reference_value)
    return mask

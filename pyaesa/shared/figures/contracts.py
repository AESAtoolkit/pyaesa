"""Shared selector and prospective-scope figure contracts."""

from collections.abc import Sequence

import pandas as pd

from pyaesa.shared.selectors.fu_axes import expected_fu_selector_columns
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
)
from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens
from pyaesa.shared.tabular.scalars import display_scalar

FIGURE_OUTPUT_FORMAT_SET = frozenset({"pdf", "png", "svg"})
SELECTOR_COLUMNS = ("r_p", "s_p", "r_c", "r_f")
DETERMINISTIC_PROSPECTIVE_COLUMNS = (
    ASOCC_SSP_SCENARIO_COLUMN,
    AR6_CC_SSP_SCENARIO_COLUMN,
)


def normalize_figure_output_format(
    output_format: str,
    *,
    argument_name: str = "figure_output_format",
) -> str:
    """Validate and normalize one figure output format token."""
    fmt = str(output_format).strip().lower()
    if fmt not in FIGURE_OUTPUT_FORMAT_SET:
        raise ValueError(
            f"{argument_name} must be one of {sorted(FIGURE_OUTPUT_FORMAT_SET)}, "
            f"got '{output_format}'."
        )
    return fmt


def validate_figure_dpi(dpi: int) -> int:
    """Validate figure DPI is a positive integer."""
    raw_dpi = dpi
    dpi = int(dpi)
    if dpi < 1:
        raise ValueError(f"figure_dpi must be a positive integer. Received {raw_dpi!r}.")
    return dpi


def resolved_selector_columns(
    frame: pd.DataFrame | None = None,
    *,
    selector_columns: Sequence[str] = SELECTOR_COLUMNS,
    require_non_null: bool = True,
) -> tuple[str, ...]:
    """Return canonical selector columns present in one frame."""
    if frame is None or frame.empty:
        return tuple()
    resolved: list[str] = []
    for column in selector_columns:
        if column not in frame.columns:
            continue
        if require_non_null:
            series = pd.Series(frame.loc[:, column], copy=False)
            if not bool(series.notna().any()):
                continue
        resolved.append(str(column))
    return tuple(resolved)


def figure_selector_columns(
    frame: pd.DataFrame | None,
    *,
    selector_columns: Sequence[str] = SELECTOR_COLUMNS,
) -> tuple[str, ...]:
    """Return selector columns owned by the frame functional unit."""
    requested = tuple(str(column) for column in selector_columns)
    if frame is None or "fu_code" not in frame.columns:
        return requested
    fu_values = _visible_values(frame=frame, column="fu_code")
    expected = tuple(
        column
        for column in expected_fu_selector_columns(fu_code=fu_values[0])
        if column in requested
    )
    return expected


def deterministic_prospective_series(frame: pd.DataFrame) -> pd.Series:
    """Return one conflict-checked deterministic prospective scenario series."""
    return _coalesced_visible_series(
        frame=frame,
        columns=DETERMINISTIC_PROSPECTIVE_COLUMNS,
        value_normalizer=_normalize_visible_scalar,
        conflict_context=(
            "Figure scope mixes conflicting prospective scenario values across canonical "
            f"scenario columns {list(DETERMINISTIC_PROSPECTIVE_COLUMNS)}"
        ),
    )


def deterministic_prospective_values(frame: pd.DataFrame) -> list[str]:
    """Return visible deterministic prospective scenario values."""
    series = deterministic_prospective_series(frame)
    return sorted({value for value in series.tolist() if value is not None})


def _coalesced_visible_series(
    *,
    frame: pd.DataFrame,
    columns: Sequence[str],
    value_normalizer,
    conflict_context: str,
) -> pd.Series:
    """Return one canonical visible-value series after conflict validation."""
    if frame.empty:
        return pd.Series([], dtype="object")
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")
    normalized = {column: value_normalizer(frame=frame, column=column) for column in present}
    values: list[str | None] = []
    for row_values in zip(*(series.tolist() for series in normalized.values()), strict=True):
        visible = {text for text in row_values if text is not None}
        if len(visible) > 1:
            raise ValueError(f"{conflict_context}: {sorted(visible)}.")
        values.append(next(iter(visible)) if visible else None)
    return pd.Series(values, index=frame.index, dtype="object")


def _visible_values(*, frame: pd.DataFrame, column: str) -> list[str]:
    return sorted(
        {
            text
            for value in frame[column].dropna().tolist()
            for text in [display_scalar(value)]
            if text is not None
        }
    )


def _normalize_visible_scalar(*, frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one visible deterministic scope series."""
    series = pd.Series(frame.loc[:, column], copy=False)
    if column == AR6_CC_SSP_SCENARIO_COLUMN:
        values: list[str | None] = []
        for value in series.tolist():
            text = display_scalar(value)
            if text is None:
                values.append(None)
                continue
            stripped = str(text).strip()
            values.append(None if not stripped else normalize_ssp_tokens([stripped])[0])
        return pd.Series(values, index=frame.index, dtype="object")
    values: list[str | None] = []
    for value in series.tolist():
        values.append(display_scalar(value))
    return pd.Series(values, index=frame.index, dtype="object")

"""Shared figure title and selector display contract."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from pyaesa.shared.figures.contracts import (
    SELECTOR_COLUMNS,
    deterministic_prospective_series,
    deterministic_prospective_values,
    figure_selector_columns,
)
from pyaesa.shared.selectors.fu_axes import expected_fu_selector_columns
from pyaesa.shared.selectors.path_tokens import selector_scope_request_axes_token
from pyaesa.shared.tabular.scalars import display_scalar, sanitize_token


@dataclass(frozen=True)
class SelectorScopeRequest:
    """Explicit selector scope carried from normalized request inputs."""

    axes: tuple[tuple[str, tuple[str, ...] | None], ...]


def _visible_values(*, frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    values = sorted(
        {
            text
            for value in frame[column].dropna().tolist()
            for text in [display_scalar(value)]
            if text is not None
        }
    )
    return values


def single_visible_value(*, frame: pd.DataFrame, column: str) -> str | None:
    """Return one unique visible column value or ``None``."""
    values = _visible_values(frame=frame, column=column)
    if len(values) == 1:
        return values[0]
    return None


def resolve_prospective_scope(frame: pd.DataFrame) -> str | None:
    """Return one prospective scenario title block when the figure scope is scenario dependent."""
    values = prospective_scenario_values(frame)
    if len(values) > 1:
        raise ValueError(
            "Figure scope mixes multiple prospective scenarios. Render one figure per SSP."
        )
    if values:
        return f"Prospective: {values[0]}"
    return None


def prospective_scenario_values(frame: pd.DataFrame) -> list[str]:
    """Return visible prospective scenario values from one figure scope."""
    return deterministic_prospective_values(frame)


def prospective_scope_slices(
    frame: pd.DataFrame,
) -> list[tuple[str, str | None, pd.DataFrame]]:
    """Return figure slices scoped to one prospective scenario."""
    series = deterministic_prospective_series(frame)
    values = sorted({value for value in series.tolist() if value is not None})
    if values:
        generic_mask = series.isna()
        slices: list[tuple[str, str | None, pd.DataFrame]] = []
        for value in values:
            scoped = frame.loc[series.eq(str(value)) | generic_mask].copy()
            token = f"prospective_{sanitize_token(value)}"
            slices.append((token, f"Prospective: {value}", scoped.reset_index(drop=True)))
        return slices
    return [("all", None, frame.copy())]


def format_selector_axis(
    *,
    frame: pd.DataFrame,
    column: str,
    reference_frame: pd.DataFrame | None = None,
) -> str | None:
    """Return one selector axis label for a scoped figure slice."""
    values = _visible_values(frame=frame, column=column)
    if not values:
        return None
    reference = frame if reference_frame is None else reference_frame
    reference_values = _visible_values(frame=reference, column=column)
    if len(reference_values) > 1 and values == reference_values:
        return f"all {column}"
    if len(values) == 1:
        return f"{column}={values[0]}"
    return f"{column}={' + '.join(values)}"


def format_selector_scope(
    *,
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame | None = None,
    selector_columns: Sequence[str] = SELECTOR_COLUMNS,
) -> str | None:
    """Return the selector block for one figure title."""
    scoped_columns = figure_selector_columns(
        reference_frame if reference_frame is not None else frame,
        selector_columns=selector_columns,
    )
    parts = [
        part
        for column in scoped_columns
        for part in [
            format_selector_axis(
                frame=frame,
                column=str(column),
                reference_frame=reference_frame,
            )
        ]
        if part is not None
    ]
    return " | ".join(parts) if parts else None


def resolve_selector_scope(
    *,
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame | None = None,
    selector_columns: Sequence[str] = SELECTOR_COLUMNS,
    selector_scope_request: SelectorScopeRequest | None = None,
) -> str | None:
    """Return one selector scope from figure data or explicit request metadata."""
    scope = format_selector_scope(
        frame=frame,
        reference_frame=reference_frame,
        selector_columns=selector_columns,
    )
    if scope is not None:
        return scope
    return format_selector_scope_request(selector_scope_request=selector_scope_request)


def join_title_blocks(*blocks: str | None) -> str:
    """Return one canonical title from visible blocks."""
    return " | ".join(
        str(block).strip() for block in blocks if block is not None and str(block).strip()
    )


def clean_panel_title(*, panel_title: str | None) -> str | None:
    """Return one normalized panel title text or ``None``."""
    if panel_title is None:
        return None
    text = str(panel_title).strip()
    if not text or text == "value":
        return None
    return text


def build_figure_title(
    *,
    family: str,
    selector_scope: str | None = None,
    lcia_method: str | None = None,
    user_facing_override_label: str | None = None,
    prospective_scope: str | None = None,
    year: int | None = None,
) -> str:
    """Return one canonical figure title."""
    return join_title_blocks(
        str(family).strip(),
        selector_scope,
        None if lcia_method is None else str(lcia_method).strip(),
        (None if user_facing_override_label is None else str(user_facing_override_label).strip()),
        prospective_scope,
        None if year is None else str(int(year)),
    )


def build_resolved_figure_title(
    *,
    title_parts: Mapping[str, str | None],
    year: int | None = None,
    panel_title: str | None = None,
    panel_count: int = 1,
) -> str:
    """Return one canonical figure title with centralized panel omission rules."""
    normalized_panel = clean_panel_title(panel_title=panel_title)
    base_title = build_figure_title(
        family=title_parts["family"] or "",
        selector_scope=title_parts["selector_scope"],
        lcia_method=title_parts["lcia_method"],
        user_facing_override_label=title_parts["user_facing_override_label"],
        prospective_scope=title_parts.get("prospective_scope"),
        year=year,
    )
    return (
        base_title
        if panel_count != 1 or normalized_panel is None
        else join_title_blocks(base_title, normalized_panel)
    )


def resolve_panel_title(
    *,
    panel_title: str | None,
    panel_count: int,
) -> str | None:
    """Return one subplot title after centralized omission rules."""
    normalized_panel = clean_panel_title(panel_title=panel_title)
    if panel_count <= 1:
        return None
    return normalized_panel


def selector_scope_request_from_filters(
    *,
    filters: Mapping[str, Sequence[str] | None],
    selector_columns: Sequence[str] = SELECTOR_COLUMNS,
) -> SelectorScopeRequest | None:
    """Return explicit selector scope metadata from normalized request filters."""
    axes: list[tuple[str, tuple[str, ...] | None]] = []
    for column in selector_columns:
        column_name = str(column)
        if column_name not in filters:
            continue
        values = filters[column_name]
        if values is None:
            axes.append((column_name, None))
            continue
        visible = tuple(
            sorted(
                dict.fromkeys(
                    text for value in values for text in [display_scalar(value)] if text is not None
                )
            )
        )
        if not visible:
            continue
        axes.append((column_name, visible))
    if not axes:
        return None
    return SelectorScopeRequest(axes=tuple(axes))


def selector_scope_request_from_selector_values(
    *,
    selector_values: Mapping[str, object],
    selector_columns: Sequence[str] = SELECTOR_COLUMNS,
) -> SelectorScopeRequest | None:
    """Return explicit selector scope metadata from selector arguments."""
    normalized: dict[str, tuple[str, ...] | None] = {}
    scoped_columns = _selector_columns_for_values(
        selector_values=selector_values,
        selector_columns=selector_columns,
    )
    for column in scoped_columns:
        column_name = str(column)
        if column_name not in selector_values:
            continue
        values = selector_values[column_name]
        if values is None:
            normalized[column_name] = None
            continue
        if isinstance(values, str):
            text = display_scalar(values)
            if text is None:
                continue
            normalized[column_name] = (text,)
            continue
        if not isinstance(values, Sequence):
            text = display_scalar(values)
            if text is None:
                continue
            normalized[column_name] = (text,)
            continue
        visible = tuple(
            sorted(
                dict.fromkeys(
                    text for value in values for text in [display_scalar(value)] if text is not None
                )
            )
        )
        if not visible:
            continue
        normalized[column_name] = visible
    if not normalized:
        return None
    return SelectorScopeRequest(
        axes=tuple(
            (column, normalized[column]) for column in scoped_columns if column in normalized
        )
    )


def format_selector_scope_request(
    *,
    selector_scope_request: SelectorScopeRequest | None,
) -> str | None:
    """Return one selector scope block from preserved request metadata."""
    if selector_scope_request is None:
        return None
    parts: list[str] = []
    for column, values in selector_scope_request.axes:
        if values is None:
            parts.append(f"all {column}")
            continue
        if len(values) == 1:
            parts.append(f"{column}={values[0]}")
            continue
        parts.append(f"{column}={' + '.join(values)}")
    return " | ".join(parts) if parts else None


def selector_scope_request_token(
    *,
    selector_scope_request: SelectorScopeRequest | None,
    empty_token: str = "all_selectors",
) -> str:
    """Return one filesystem safe token for a preserved selector scope request."""
    if selector_scope_request is None:
        return empty_token
    return selector_scope_request_axes_token(
        selector_scope_request.axes,
        empty_token=empty_token,
    )


def selector_scope_request_from_base_allocate_args(
    *,
    base_allocate_args: Mapping[str, object],
    selector_columns: Sequence[str] = SELECTOR_COLUMNS,
) -> SelectorScopeRequest | None:
    """Return canonical selector scope metadata from allocation base arguments."""
    return selector_scope_request_from_selector_values(
        selector_values=base_allocate_args,
        selector_columns=selector_columns,
    )


def _selector_columns_for_values(
    *,
    selector_values: Mapping[str, object],
    selector_columns: Sequence[str],
) -> tuple[str, ...]:
    requested = tuple(str(column) for column in selector_columns)
    fu_code = display_scalar(selector_values.get("fu_code"))
    if fu_code is None:
        return requested
    return tuple(
        column for column in expected_fu_selector_columns(fu_code=fu_code) if column in requested
    )


def uncertainty_family_label(*, value_column: str, transition_policy: str) -> str:
    """Return the uncertainty family label for one shared postprocess run."""
    if value_column == "acc_value":
        return "aCC"
    if value_column == "asr_value":
        return "ASR"
    if transition_policy == "asocc":
        return "aSoCC"
    return "IO-LCA"

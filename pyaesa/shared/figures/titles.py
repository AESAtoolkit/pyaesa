"""Shared figure title rendering helpers."""

from typing import Any

from pyaesa.shared.figures.scientific_text import format_scientific_figure_text

FIGURE_TITLE_Y = 0.985
FIGURE_TITLE_SIZE = 15
FIGURE_DETAIL_SIZE = 13
FIGURE_TITLE_CLEARANCE_IN = 0.22
FIGURE_TITLE_DETAIL_CLEARANCE_IN = 0.32
FIGURE_TITLE_DETAIL_OFFSET_IN = 0.22
_TITLE_DETAIL_OFFSET_IN = FIGURE_TITLE_DETAIL_OFFSET_IN
_TITLE_CLEARANCE_IN = FIGURE_TITLE_CLEARANCE_IN
_TITLE_DETAIL_CLEARANCE_IN = FIGURE_TITLE_DETAIL_CLEARANCE_IN


def render_figure_title(
    fig: Any,
    title: str,
    *,
    y: float = FIGURE_TITLE_Y,
    detail_offset_in: float = FIGURE_TITLE_DETAIL_OFFSET_IN,
) -> None:
    """Render one figure title with an optional lighter second line."""
    main_title, _separator, detail_line = str(title).partition("\n")
    fig.suptitle(
        format_scientific_figure_text(main_title),
        fontsize=FIGURE_TITLE_SIZE,
        fontweight="bold",
        y=y,
    )
    if not detail_line.strip():
        return
    height = max(float(fig.get_size_inches()[1]), 1.0)
    offset = max(0.0, float(detail_offset_in))
    fig.text(
        0.5,
        y - offset / height,
        format_scientific_figure_text(detail_line),
        ha="center",
        va="top",
        fontsize=FIGURE_DETAIL_SIZE,
        fontstyle="italic",
    )


def render_dynamic_ar6_title(
    fig: Any,
    title: str,
    *,
    y: float = FIGURE_TITLE_Y,
    detail_offset_in: float = FIGURE_TITLE_DETAIL_OFFSET_IN,
) -> None:
    """Render one dynamic AR6 title with the shared title spacing contract."""
    render_figure_title(fig, title, y=y, detail_offset_in=detail_offset_in)


def title_layout_top(
    fig: Any,
    title: str,
    *,
    default_top: float,
    panel_title_pad: int | float = 0,
    y: float = FIGURE_TITLE_Y,
    detail_offset_in: float = FIGURE_TITLE_DETAIL_OFFSET_IN,
    detail_clearance_in: float | None = None,
) -> float:
    """Return an axes top position that keeps panel titles below the figure title."""
    height = max(float(fig.get_size_inches()[1]), 1.0)
    main_bottom = y - float(FIGURE_TITLE_SIZE) / 72.0 / height
    _main, _separator, detail = str(title).partition("\n")
    offset = max(0.0, float(detail_offset_in))
    content_bottom = (
        y - (offset + float(FIGURE_DETAIL_SIZE) / 72.0) / height if detail.strip() else main_bottom
    )
    clearance_in = (
        _TITLE_DETAIL_CLEARANCE_IN
        if detail_clearance_in is None
        else max(0.0, float(detail_clearance_in))
    )
    clearance = (clearance_in if detail.strip() else _TITLE_CLEARANCE_IN) / height
    panel_pad = max(float(panel_title_pad), 0.0) / 72.0 / height
    return min(float(default_top), content_bottom - clearance - panel_pad)

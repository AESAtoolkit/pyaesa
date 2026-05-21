"""ASR frequency of no-transgression figure labels."""

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
from matplotlib.lines import Line2D

from pyaesa.asr.figures.threshold_contract import build_asr_threshold_contract

FNT_FRACTION_COLUMN = "__fnt_fraction"
CUMULATIVE_FNT_FRACTION_COLUMN = "__cumulative_fnt_fraction"
_FNT_BOX_GRAPH_GAP_PT = 2.0
_FNT_BOX_CONNECTOR_GAP_PT = 0.6


def format_fnt_percent(value: object) -> str:
    """Return the shared percent label for ASR frequency of no-transgression."""
    percent = 100.0 * float(cast(float | int | str, value))
    if np.isclose(percent, 0.0) or np.isclose(percent, 100.0):
        return f"{percent:.0f}%"
    return f"{percent:.1f}%"


def format_fnt_math_label(value: object) -> str:
    """Return the polar math label for ASR frequency of no-transgression."""
    text = format_fnt_percent(value)
    escaped = text.replace("%", r"\%")
    return rf"$f^{{\mathrm{{NT}}}}={escaped}$"


def fnt_legend_entry(*, cc_source: str) -> tuple[Any, str]:
    """Return a figure legend entry explaining ASR frequency of no-transgression."""
    label = build_asr_threshold_contract(
        cc_source=cc_source,
        has_max_threshold=False,
    ).fnt_label
    return Line2D([], [], color="none", label=label), label


def fnt_box_legend_entry(*, cc_source: str) -> tuple[Any, str]:
    """Return a legend entry explaining frequency box labels above ASR axes."""
    _handle, label = fnt_legend_entry(cc_source=cc_source)
    box_label = f"Percent box labels show {label}."
    return Line2D([], [], color="none", label=box_label), box_label


def render_fnt_box(
    axis: Any,
    *,
    x: float,
    value: object,
    y: float = 1.012,
) -> None:
    """Render one frequency of no-transgression box above a subplot."""
    render_fnt_boxes(axis, entries=[(float(x), value)], y=y)


def render_fnt_boxes(
    axis: Any,
    *,
    entries: Iterable[tuple[float, object]],
    y: float = 1.012,
) -> None:
    """Render frequency of no-transgression boxes above a subplot."""
    render_fnt_box_groups([(axis, entries)], y=y)


def render_fnt_box_groups(
    groups: Iterable[tuple[Any, Iterable[tuple[float, object]]]],
    *,
    y: float = 1.012,
) -> None:
    """Render frequency of no-transgression boxes above several subplots."""
    materialized = [(axis, list(entries)) for axis, entries in groups]
    active_groups = [(axis, entries) for axis, entries in materialized if entries]
    if not active_groups:
        return
    fig = active_groups[0][0].figure
    if any(axis.figure is not fig for axis, _entries in active_groups):
        raise ValueError("Frequency boxes for one batch must share the same figure.")
    payloads = []
    for axis, entries in active_groups:
        box_entries = [
            (float(x), _add_fnt_box_text(axis=axis, x=float(x), value=value, y=y))
            for x, value in entries
        ]
        payloads.append((axis, box_entries))
    renderer = _position_fnt_box_groups(fig=fig, groups=payloads)
    for axis, box_entries in payloads:
        for x, text in box_entries:
            axis.plot(
                [float(x), float(x)],
                [1.0, _fnt_box_connector_top(axis=axis, text=text, renderer=renderer)],
                transform=axis.get_xaxis_transform(),
                color="#3d3d3d",
                linewidth=0.7,
                zorder=9,
                clip_on=False,
            )


def _add_fnt_box_text(
    *,
    axis: Any,
    x: float,
    value: object,
    y: float,
) -> Any:
    return axis.text(
        float(x),
        float(y),
        format_fnt_percent(value),
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#2f2f2f",
        bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "#3d3d3d", "alpha": 1.0},
        zorder=10,
        clip_on=False,
    )


def _position_fnt_box_groups(*, fig: Any, groups: list[tuple[Any, list[tuple[float, Any]]]]) -> Any:
    fig.canvas.draw()
    renderer = cast(Any, fig.canvas).get_renderer()
    adjusted = False
    for axis, box_entries in groups:
        axis_bbox = axis.get_window_extent(renderer=renderer)
        target_bottom = float(axis_bbox.y1) + _points_to_pixels(fig, _FNT_BOX_GRAPH_GAP_PT)
        for _box_x, text in box_entries:
            text_bbox = text.get_window_extent(renderer=renderer)
            delta_y = target_bottom - float(text_bbox.y0)
            if np.isclose(delta_y, 0.0, rtol=0.0, atol=0.1):
                continue
            x, y = text.get_position()
            text.set_position((float(x), float(y) + delta_y / max(float(axis_bbox.height), 1.0)))
            adjusted = True
    if adjusted:
        fig.canvas.draw()
        renderer = cast(Any, fig.canvas).get_renderer()
    return renderer


def _fnt_box_connector_top(*, axis: Any, text: Any, renderer: Any) -> float:
    fig = axis.figure
    bbox = text.get_window_extent(renderer=renderer)
    bottom_y = float(bbox.y0) - _points_to_pixels(fig, _FNT_BOX_CONNECTOR_GAP_PT)
    return float(axis.transAxes.inverted().transform((0.0, bottom_y))[1])


def _points_to_pixels(fig: Any, points: float) -> float:
    return float(fig.dpi) * float(points) / 72.0

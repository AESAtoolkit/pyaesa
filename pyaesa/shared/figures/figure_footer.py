"""Shared below figure legend and note helpers."""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, cast

from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pyaesa.shared.figures.scientific_text import format_scientific_figure_text

_LEGEND_ROW_HEIGHT_IN = 0.18
_LEGEND_TITLED_OVERHEAD_IN = 0.26
_LEGEND_PLAIN_OVERHEAD_IN = 0.18
_XAXIS_FOOTER_MIN_GAP_IN = 0.09
_XAXIS_LEGEND_PADDING_IN = 0.11
_NOTE_LINE_HEIGHT_IN = 0.12
_NOTE_FONT_SIZE = 8
_NOTE_WRAP_WIDTH_FACTOR = 1.08
_BOTTOM_MARGIN_IN = 0.15
_BOTTOM_MAX = 0.95
_MIN_PLOT_HEIGHT_IN = 6.0
_FOOTER_BOTTOM_ATTR = "_pyaesa_reserved_footer_bottom"
_FOOTER_MIN_PLOT_HEIGHT_ATTR = "_pyaesa_footer_min_plot_height_in"


@dataclass(frozen=True)
class LegendLayout:
    """Shared below figure legend layout metrics."""

    anchor_y: float
    bottom: float
    top_y: float


def center_legend_text(legend: Any) -> None:
    """Center all visible text inside one Matplotlib legend."""
    legend.get_title().set_ha("center")
    legend.get_title().set_multialignment("center")
    for text in legend.get_texts():
        text.set_ha("center")
        text.set_multialignment("center")


def set_footer_min_plot_height(fig: Figure, *, height_in: float) -> None:
    """Set the minimum plotting area height for subsequent footer reservations."""
    setattr(fig, _FOOTER_MIN_PLOT_HEIGHT_ATTR, max(0.1, float(height_in)))


def legend_note_lines(fig: Figure, note: str | None) -> list[str]:
    """Return legend note lines wrapped to the current plotting area width."""
    lines = _raw_note_lines(note)
    if not lines:
        return []
    fig.canvas.draw()
    renderer = cast(Any, fig.canvas).get_renderer()
    max_width_in = _plotting_area_width_inches(fig) * _NOTE_WRAP_WIDTH_FACTOR
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(
            _wrap_note_line(
                fig,
                renderer=renderer,
                line=line,
                max_width_in=max_width_in,
                fontsize=_NOTE_FONT_SIZE,
            )
        )
    return wrapped


def _raw_note_lines(note: str | None) -> list[str]:
    if note is None:
        return []
    stripped = str(note).strip()
    if not stripped:
        return []
    return [
        format_scientific_figure_text(line.strip())
        for line in stripped.splitlines()
        if line.strip()
    ]


def legend_label_line_count(label: str) -> int:
    """Return the visible line count for one legend label."""
    stripped = str(label).strip()
    if not stripped:
        return 0
    return len([line for line in stripped.splitlines() if line.strip()])


def legend_display_rows(labels: list[str], *, ncol: int) -> int:
    """Return the visible text row count needed by a multi column legend."""
    if not labels:
        return 0
    columns = max(1, int(ncol))
    rows = 0
    for start in range(0, len(labels), columns):
        row_labels = labels[start : start + columns]
        rows += max(legend_label_line_count(label) for label in row_labels)
    return rows


def render_below_figure_legend(
    fig: Figure,
    *,
    legend_note: str | None = None,
    max_columns: int = 2,
    extra_entries: Sequence[tuple[Artist, str]] | None = None,
    extra_height_in: float = 0.0,
) -> bool:
    """Render one figure level legend below the plotting area."""
    entries = _with_extra_entries(
        _collect_figure_legend_entries(fig),
        extra_entries=extra_entries,
    )
    note_entries = legend_note_entries(legend_note)
    if not entries and not note_entries:
        return False
    ncol = _legend_columns(entry_count=len(entries), max_columns=max_columns) if entries else 1
    labels = [label for _handle, label in entries]
    rows = legend_display_rows(labels, ncol=ncol) + legend_display_rows(
        [label for _handle, label in note_entries],
        ncol=1,
    )
    layout = reserve_footer_space(
        fig,
        rows=rows,
        note_lines=0,
        extra_height_in=extra_height_in,
    )
    legends = []
    if note_entries:
        note_legend = fig.legend(
            handles=[handle for handle, _label in note_entries],
            labels=[label for _handle, label in note_entries],
            loc="lower center",
            bbox_to_anchor=(0.5, layout.anchor_y),
            ncol=1,
            frameon=False,
            fontsize="small",
            handlelength=0,
            handletextpad=0,
        )
        center_legend_text(note_legend)
        legends.append(note_legend)
    if entries:
        legend = fig.legend(
            handles=[handle for handle, _label in entries],
            labels=[label for _handle, label in entries],
            loc="lower center",
            bbox_to_anchor=(0.5, layout.anchor_y),
            ncol=ncol,
            frameon=False,
            fontsize="small",
        )
        legends.append(legend)
    align_lower_legend_stack_top_to_layout(fig, legends, layout=layout)
    return True


def legend_note_entries(note: str | None) -> list[tuple[Artist, str]]:
    """Return one text only legend entry for a deterministic explanatory note."""
    stripped = str(note or "").strip()
    if not stripped:
        return []
    handle = Line2D([], [], linestyle="none", marker=None, color="none")
    return [(handle, format_scientific_figure_text(stripped))]


def render_two_panel_legends_below(
    fig: Figure,
    *,
    left_axis: Any,
    right_axis: Any,
    left_handles: Sequence[Any],
    right_handles: Sequence[Any],
    left_ncol: int,
    right_ncol: int,
    left_title: str | None = None,
    right_handler_map: Mapping[type, object] | None = None,
    right_legend_kwargs: Mapping[str, object] | None = None,
    extra_height_in: float = 0.0,
    title_rows: int = 0,
) -> None:
    """Render separate below figure legends aligned to two panels."""
    left_labels = [
        format_scientific_figure_text(str(handle.get_label()).strip()) for handle in left_handles
    ]
    right_labels = [
        format_scientific_figure_text(str(handle.get_label()).strip()) for handle in right_handles
    ]
    layout = reserve_footer_space(
        fig,
        rows=max(
            legend_display_rows(left_labels, ncol=left_ncol),
            legend_display_rows(right_labels, ncol=right_ncol),
        ),
        note_lines=0,
        title_rows=title_rows,
        extra_height_in=extra_height_in,
    )
    fig.canvas.draw()
    legend_top_y = layout.top_y
    if left_handles:
        fig.legend(
            handles=list(left_handles),
            labels=left_labels,
            title=None if left_title is None else format_scientific_figure_text(left_title),
            loc="upper center",
            bbox_to_anchor=(_axis_center_x(left_axis), legend_top_y),
            ncol=int(left_ncol),
            frameon=True,
            fontsize="small",
        )
    if right_handles:
        fig.legend(
            handles=list(right_handles),
            labels=right_labels,
            loc="upper center",
            bbox_to_anchor=(_axis_center_x(right_axis), legend_top_y),
            ncol=int(right_ncol),
            frameon=True,
            fontsize="small",
            handler_map=right_handler_map,
            **dict(right_legend_kwargs or {}),
        )


def grouped_entries_from_figure(fig: Figure) -> OrderedDict[str, list[tuple[Artist, str]]]:
    """Return grouped deterministic legend entries gathered from all axes."""
    visible: list[tuple[Artist, str, str]] = []
    for axis in fig.axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            stripped_label = format_scientific_figure_text(str(label).strip())
            if not stripped_label:
                continue
            group_title = str(getattr(handle, "_pyaesa_group_title", "")).strip()
            visible.append((handle, stripped_label, group_title))
    grouped: OrderedDict[str, list[tuple[Artist, str]]] = OrderedDict()
    for handle, label, group_title in visible:
        grouped.setdefault(group_title, [])
        grouped[group_title].append((handle, label))
    return OrderedDict((title, items) for title, items in grouped.items() if items)


def _axis_center_x(axis: Any) -> float:
    position = axis.get_position()
    return 0.5 * (float(position.x0) + float(position.x1))


def reserve_footer_space(
    fig: Figure,
    *,
    rows: int,
    note_lines: int,
    title_rows: int = 0,
    extra_height_in: float = 0.0,
) -> LegendLayout:
    """Reserve enough bottom space for below figure legends and notes."""
    width, height = fig.get_size_inches()
    extra_height = max(0.0, float(extra_height_in))
    legend_height_in = _legend_height_inches(rows=rows, title_rows=title_rows)
    xaxis_gap = (
        _xaxis_footer_gap_inches(fig) if rows > 0 or note_lines > 0 or extra_height > 0.0 else 0.0
    )
    footer_in = (
        _BOTTOM_MARGIN_IN
        + legend_height_in
        + _NOTE_LINE_HEIGHT_IN * max(0, note_lines)
        + xaxis_gap
        + extra_height
    )
    minimum_anchor_in = _BOTTOM_MARGIN_IN + _NOTE_LINE_HEIGHT_IN * max(0, note_lines)
    top = float(fig.subplotpars.top)
    top_margin_in = max(0.0, (1.0 - top) * height)
    min_plot_height = float(getattr(fig, _FOOTER_MIN_PLOT_HEIGHT_ATTR, _MIN_PLOT_HEIGHT_IN))
    min_height = (footer_in + min_plot_height) / max(top, 0.01)
    resized_height = max(height, min_height)
    if resized_height > height:
        fig.set_size_inches(width, resized_height, forward=True)
        height = resized_height
        top = min(0.99, max(0.01, 1.0 - top_margin_in / height))
        fig.subplots_adjust(top=top)
    requested_bottom = min(_BOTTOM_MAX, footer_in / height)
    previous_bottom = getattr(fig, _FOOTER_BOTTOM_ATTR, None)
    bottom = (
        requested_bottom
        if previous_bottom is None
        else max(float(previous_bottom), requested_bottom)
    )
    fig.subplots_adjust(bottom=bottom)
    setattr(fig, _FOOTER_BOTTOM_ATTR, bottom)
    anchor_offset_in = minimum_anchor_in
    if rows > 0:
        fig.canvas.draw()
        content_bottom_in = _xaxis_content_bottom_inches(fig)
        legend_top_in = content_bottom_in - _XAXIS_LEGEND_PADDING_IN - extra_height
        anchor_offset_in = max(minimum_anchor_in, legend_top_in - legend_height_in)
    anchor_y = anchor_offset_in / height
    top_y = (anchor_offset_in + legend_height_in) / height
    return LegendLayout(anchor_y=anchor_y, bottom=bottom, top_y=top_y)


def footer_content_top_limit(fig: Figure, *, padding_in: float = _XAXIS_LEGEND_PADDING_IN) -> float:
    """Return the highest footer content coordinate that clears visible x axis labels."""
    fig.canvas.draw()
    height = float(fig.get_size_inches()[1])
    return (_xaxis_content_bottom_inches(fig) - max(0.0, float(padding_in))) / height


def align_lower_legend_top_to_layout(
    fig: Figure,
    legend: Any,
    *,
    layout: LegendLayout,
    anchor_x: float = 0.5,
) -> None:
    """Align one lower centered figure legend to the reserved footer top."""
    renderer = _drawn_renderer(fig)
    legend_height = _legend_bbox_height(fig, legend, renderer=renderer)
    legend.set_bbox_to_anchor(
        (float(anchor_x), max(0.0, float(layout.top_y) - legend_height)),
        transform=fig.transFigure,
    )


def align_lower_legend_stack_top_to_layout(
    fig: Figure,
    legends: Sequence[Any],
    *,
    layout: LegendLayout,
    anchor_x: float = 0.5,
) -> None:
    """Align a bottom-to-top stack of lower centered legends to the footer top."""
    renderer = _drawn_renderer(fig)
    current_top = float(layout.top_y)
    for legend in reversed(list(legends)):
        legend_height = _legend_bbox_height(fig, legend, renderer=renderer)
        anchor_y = max(0.0, current_top - legend_height)
        legend.set_bbox_to_anchor((float(anchor_x), anchor_y), transform=fig.transFigure)
        current_top = anchor_y


def _legend_bbox_height(fig: Figure, legend: Any, *, renderer: Any) -> float:
    bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    return float(bbox.height)


def _drawn_renderer(fig: Figure) -> Any:
    canvas = cast(Any, fig.canvas)
    renderer = getattr(canvas, "renderer", None)
    if renderer is not None:
        return renderer
    fig.canvas.draw()
    return canvas.get_renderer()


def _legend_height_inches(*, rows: int, title_rows: int) -> float:
    if rows <= 0:
        return 0.0
    legend_overhead = (
        _LEGEND_TITLED_OVERHEAD_IN * max(0, title_rows)
        if title_rows > 0
        else _LEGEND_PLAIN_OVERHEAD_IN
    )
    return _LEGEND_ROW_HEIGHT_IN * max(0, rows) + legend_overhead


def _xaxis_content_bottom_inches(fig: Figure) -> float:
    """Return the lowest visible x-axis text bottom or axis baseline in inches."""
    renderer = cast(Any, fig.canvas).get_renderer()
    height = float(fig.get_size_inches()[1])
    bottoms: list[float] = []
    axis_bottoms: list[float] = []
    for axis in fig.axes:
        axis_bottoms.append(float(axis.get_position().y0) * height)
        labels = [
            label
            for label in [*axis.get_xticklabels(), axis.xaxis.label]
            if label.get_visible() and str(label.get_text()).strip()
        ]
        for label in labels:
            bbox = label.get_window_extent(renderer=renderer).transformed(
                fig.dpi_scale_trans.inverted()
            )
            bottoms.append(float(bbox.y0))
    if bottoms:
        return min(bottoms)
    return min(axis_bottoms)


def _xaxis_footer_gap_inches(fig: Figure) -> float:
    """Return footer room needed for visible x axis labels and padding."""
    fig.canvas.draw()
    renderer = cast(Any, fig.canvas).get_renderer()
    height = float(fig.get_size_inches()[1])
    required = 0.0
    for axis in fig.axes:
        axis_bottom_in = float(axis.get_position().y0) * height
        labels = [
            label
            for label in [*axis.get_xticklabels(), axis.xaxis.label]
            if label.get_visible() and str(label.get_text()).strip()
        ]
        for label in labels:
            bbox = label.get_window_extent(renderer=renderer).transformed(
                fig.dpi_scale_trans.inverted()
            )
            required = max(required, axis_bottom_in - float(bbox.y0))
    if required <= 0.0:
        return _XAXIS_FOOTER_MIN_GAP_IN
    return max(_XAXIS_FOOTER_MIN_GAP_IN, required + _XAXIS_LEGEND_PADDING_IN)


def _collect_figure_legend_entries(fig: Figure) -> list[tuple[Artist, str]]:
    entries: list[tuple[Artist, str]] = []
    seen_labels: set[str] = set()
    for axis in fig.axes:
        axis_entries: list[tuple[Artist, str]] = []
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            stripped = format_scientific_figure_text(str(label).strip())
            if not stripped:
                continue
            axis_entries.append((handle, stripped))
        for handle, label in axis_entries:
            if label in seen_labels:
                continue
            entries.append((handle, label))
            seen_labels.add(label)
    return entries


def _with_extra_entries(
    entries: list[tuple[Artist, str]],
    *,
    extra_entries: Sequence[tuple[Artist, str]] | None,
) -> list[tuple[Artist, str]]:
    if not extra_entries:
        return entries
    out = list(entries)
    seen = {label for _handle, label in out}
    for handle, label in extra_entries:
        stripped = format_scientific_figure_text(str(label).strip())
        if not stripped or stripped in seen:
            continue
        out.append((handle, stripped))
        seen.add(stripped)
    return out


def _legend_columns(*, entry_count: int, max_columns: int) -> int:
    capped = max(1, min(int(max_columns), int(entry_count)))
    for ncol in range(capped, 0, -1):
        if _legend_rows(entry_count=entry_count, ncol=ncol) <= 3:
            return ncol
    return capped


def _legend_rows(*, entry_count: int, ncol: int) -> int:
    return int(math.ceil(int(entry_count) / max(1, int(ncol))))


def _plotting_area_width_inches(fig: Figure) -> float:
    width = float(fig.get_size_inches()[0])
    axes = [axis for axis in fig.axes if axis.get_visible()]
    positions = [axis.get_position() for axis in axes]
    left = min(float(position.x0) for position in positions)
    right = max(float(position.x1) for position in positions)
    return max(0.1, (right - left) * width)


def _wrap_note_line(
    fig: Figure,
    *,
    renderer: Any,
    line: str,
    max_width_in: float,
    fontsize: int,
) -> list[str]:
    words = str(line).split()
    wrapped: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if (
            _note_text_width_inches(
                fig,
                renderer=renderer,
                text=candidate,
                fontsize=fontsize,
            )
            <= max_width_in
        ):
            current = candidate
            continue
        wrapped.append(current)
        current = word
    wrapped.append(current)
    return wrapped


def _note_text_width_inches(
    fig: Figure,
    *,
    renderer: Any,
    text: str,
    fontsize: int,
) -> float:
    artist = fig.text(0.0, 0.0, str(text), fontsize=fontsize, alpha=0.0)
    try:
        bbox = artist.get_window_extent(renderer=renderer).transformed(
            fig.dpi_scale_trans.inverted()
        )
        return float(bbox.width)
    finally:
        artist.remove()

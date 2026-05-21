"""ASR dynamic aCC versus LCA row color keys."""

from typing import Any

from matplotlib import patheffects as path_effects
from matplotlib.patches import Rectangle

from pyaesa.shared.figures.colors import DEFAULT_SINGLE_SERIES_COLOR, single_or_distinct_colors
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text

ACC_COMPONENT_COLOR = "#54A24B"
LCA_COMPONENT_COLOR = "#111111"

_KEY_RECT_WIDTH = 0.012
_KEY_RECT_HEIGHT = 0.010
_KEY_GAP = 0.004
_KEY_ENTRY_GAP = 0.006
_KEY_AXIS_PADDING_IN = 0.135
_LCA_LINEWIDTH = 1.9
_LCA_MULTI_METHOD_LINEWIDTH = 2.8
_LCA_MULTI_METHOD_HALO_WIDTH = 5.0


def acc_component_label(*, emissions_mode: str | None) -> str:
    """Return the visible aCC component label for one dynamic ASR figure."""
    mode = str(emissions_mode or "net").strip()
    if mode in {"gross", "gross_alt"}:
        return f"aCC ({mode})"
    return "aCC"


def ar6_cc_positive_flow_label(*, emissions_mode: str | None) -> str:
    """Return the visible Global AR6 CC positive flow label."""
    mode = str(emissions_mode or "net").strip()
    if mode == "gross_alt":
        return "gross_alt emissions"
    if mode == "gross":
        return "gross emissions"
    return "net emissions"


def ar6_cc_flow_key_entries(
    *,
    emissions_mode: str | None,
    positive_color: str,
    negative_color: str,
    visible_negative_flow: bool,
    negative_style: str,
) -> list[tuple[str, str, str]]:
    """Return visible Global AR6 CC flow key entries."""
    entries = [
        (
            ar6_cc_positive_flow_label(emissions_mode=emissions_mode),
            positive_color,
            "-",
        )
    ]
    if visible_negative_flow:
        entries.append(("Negative sequestration", negative_color, negative_style))
    return entries


def acc_lca_pathway_title(*, emissions_mode: str | None) -> str:
    """Return the dynamic ASR aCC versus LCA pathway panel title."""
    return f"{acc_component_label(emissions_mode=emissions_mode)} vs. LCA pathways"


def acc_lca_cumulative_title(*, emissions_mode: str | None) -> str:
    """Return the dynamic ASR aCC versus LCA cumulative panel title."""
    return f"Cumulative {acc_component_label(emissions_mode=emissions_mode)} vs. LCA"


def frequency_color_map(labels: list[str], *, include_method_in_label: bool) -> dict[str, str]:
    """Return colors for frequency of no-transgression series."""
    if include_method_in_label:
        return single_or_distinct_colors(labels)
    return {label: DEFAULT_SINGLE_SERIES_COLOR for label in labels}


def lca_component_linewidth(*, include_method_in_label: bool) -> float:
    """Return the LCA line width for one ASR component row."""
    return _LCA_MULTI_METHOD_LINEWIDTH if include_method_in_label else _LCA_LINEWIDTH


def lca_component_path_effects(*, include_method_in_label: bool) -> list[Any]:
    """Return the LCA line halo for method dense ASR component rows."""
    if include_method_in_label:
        return [
            path_effects.Stroke(linewidth=_LCA_MULTI_METHOD_HALO_WIDTH, foreground="white"),
            path_effects.Normal(),
        ]
    return []


def render_acc_lca_row_key(
    *,
    fig: Any,
    left_axis: Any,
    right_axis: Any,
    include_method_in_label: bool,
    acc_color: str = ACC_COMPONENT_COLOR,
    emissions_mode: str | None = None,
) -> None:
    """Render the aCC versus LCA color key directly below the component row."""
    acc_label = acc_component_label(emissions_mode=emissions_mode)
    fig.canvas.draw()
    left = left_axis.get_position()
    center_x = 0.5 * (float(left.x0) + float(left.x1))
    y = _key_y(fig=fig, component_axis=left_axis)
    if include_method_in_label:
        _render_entries(
            fig=fig,
            center_x=center_x,
            y=y,
            entries=[
                ("LCA", LCA_COMPONENT_COLOR, True),
                (f"Other colors: {acc_label}", "", False),
            ],
        )
        return
    _render_entries(
        fig=fig,
        center_x=center_x,
        y=y,
        entries=[
            (acc_label, acc_color, True),
            ("LCA", LCA_COMPONENT_COLOR, True),
        ],
    )


def render_ar6_cc_row_key(
    *,
    fig: Any,
    left_axis: Any,
    right_axis: Any,
    entries: list[tuple[str, str, str]],
) -> None:
    """Render the Global AR6 CC row key directly below its row."""
    fig.canvas.draw()
    left = left_axis.get_position()
    center_x = 0.5 * (float(left.x0) + float(left.x1))
    y = _key_y(fig=fig, component_axis=left_axis)
    _render_entries(
        fig=fig,
        center_x=center_x,
        y=y,
        entries=[(label, color, True) for label, color, _style in entries],
    )


def _render_entries(
    *,
    fig: Any,
    center_x: float,
    y: float,
    entries: list[tuple[str, str, bool]],
) -> None:
    total_width = sum(
        _entry_width(label, has_rectangle=has_rectangle) for label, _color, has_rectangle in entries
    )
    total_width += _KEY_ENTRY_GAP * max(0, len(entries) - 1)
    x = float(center_x) - 0.5 * total_width
    for label, color, has_rectangle in entries:
        formatted_label = format_scientific_figure_text(label)
        if has_rectangle:
            fig.add_artist(
                Rectangle(
                    (x, y - 0.5 * _KEY_RECT_HEIGHT),
                    _KEY_RECT_WIDTH,
                    _KEY_RECT_HEIGHT,
                    transform=fig.transFigure,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=20,
                )
            )
            x += _KEY_RECT_WIDTH + _KEY_GAP
        fig.text(
            x,
            y,
            formatted_label,
            ha="left",
            va="center",
            fontsize=8,
            transform=fig.transFigure,
        )
        x += _text_width(formatted_label) + _KEY_ENTRY_GAP


def _key_y(*, fig: Any, component_axis: Any) -> float:
    component = component_axis.get_position()
    height = max(float(fig.get_size_inches()[1]), 1.0)
    return float(component.y0) - _KEY_AXIS_PADDING_IN / height


def _entry_width(label: str, *, has_rectangle: bool) -> float:
    width = _text_width(label)
    if has_rectangle:
        width += _KEY_RECT_WIDTH + _KEY_GAP
    return width


def _text_width(label: str) -> float:
    return max(0.016, 0.0042 * len(str(label)))

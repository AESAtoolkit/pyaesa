"""ASR risk guides for deterministic and uncertainty figures."""

from typing import Any

from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from pyaesa.asr.figures.polar_artists import risk_scale_rgba
from pyaesa.asr.figures.risk_style import (
    MAX_RISK_COLOR,
    RISK_RAMP_END,
    SAFE_COLOR,
    SAFE_FRAC,
    SCALE_LIGHTEN,
)
from pyaesa.asr.figures.threshold_contract import (
    build_asr_threshold_contract,
    has_max_asr_threshold,
)
from pyaesa.shared.figures.figure_footer import (
    _BOTTOM_MARGIN_IN,
    footer_content_top_limit,
    reserve_footer_space,
)
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text

ASR_RISK_LEGEND_GROUP_TITLE = "Uncertainty"
ASR_RISK_BACKGROUND_VIOLIN_ALPHA = 0.48
ASR_NON_POLAR_RISK_BACKGROUND_ALPHA_SCALE = 0.5
_SAFE_ZONE_COLOR = SAFE_COLOR
_MIDDLE_ZONE_COLOR = "#f5e1b8"
_HIGH_RISK_ZONE_COLOR = MAX_RISK_COLOR
_MIN_THRESHOLD_COLOR = SAFE_COLOR
_MAX_THRESHOLD_COLOR = MAX_RISK_COLOR
_MIN_THRESHOLD_GUIDE_COLOR = "#00a81f"
_MAX_THRESHOLD_GUIDE_COLOR = "#c51616"
_THRESHOLD_GUIDE_LINE_WIDTH = 1.6
_SAFE_ZONE_ALPHA = 0.1375
_MIDDLE_ZONE_ALPHA = 0.19
_HIGH_RISK_ZONE_ALPHA = 0.085
_RISK_SCALE_FOOTER_HEIGHT_IN = 0.64
_RISK_SCALE_FOOTER_GAP_IN = 0.04
_RISK_SCALE_FOOTER_TOP_GAP_IN = 0.08
_RISK_SCALE_FOOTER_WIDTH = 0.24
_RISK_SCALE_XTICK_GAP_IN = 0.11


def asr_risk_scale_footer_extra_height() -> float:
    """Return footer height reserved for the ASR risk color scale."""
    return _RISK_SCALE_FOOTER_HEIGHT_IN + _RISK_SCALE_FOOTER_GAP_IN + _RISK_SCALE_FOOTER_TOP_GAP_IN


def render_asr_threshold_guides(
    axis,
    *,
    has_max_threshold: bool,
    max_threshold: float | None = None,
    grouped_title: str | None = None,
    background_alpha_scale: float = ASR_NON_POLAR_RISK_BACKGROUND_ALPHA_SCALE,
) -> None:
    """Render ASR threshold lines and attach the shared legend contract.

    Args:
        axis: Matplotlib axis receiving the visible guide lines and legend handles.
        has_max_threshold: Whether this figure scope exposes the max-threshold semantics.
        max_threshold: Optional numeric max threshold. When omitted, only the
            shared legend wording is attached for the max threshold.
        grouped_title: Optional deterministic grouped-legend title.
    """
    del grouped_title
    render_asr_risk_background(
        axis,
        max_threshold=max_threshold if has_max_threshold else None,
        alpha_scale=background_alpha_scale,
    )
    axis.axhline(
        1.0,
        color=_MIN_THRESHOLD_GUIDE_COLOR,
        linestyle=":",
        linewidth=_THRESHOLD_GUIDE_LINE_WIDTH,
        alpha=1.0,
        zorder=30,
    )
    if has_max_threshold and max_threshold is not None:
        axis.axhline(
            float(max_threshold),
            color=_MAX_THRESHOLD_GUIDE_COLOR,
            linestyle=":",
            linewidth=_THRESHOLD_GUIDE_LINE_WIDTH,
            alpha=1.0,
            zorder=30,
        )


def render_asr_risk_scale_footer(
    fig,
    *,
    frame: pd.DataFrame,
    background_text: str | None = None,
) -> None:
    """Render the ASR risk color scale in the reserved below figure footer."""
    reserve_footer_space(
        fig,
        rows=0,
        note_lines=0,
        extra_height_in=asr_risk_scale_footer_extra_height(),
    )
    contract = build_asr_threshold_contract(
        cc_source=_cc_source(frame),
        has_max_threshold=has_max_asr_threshold(frame=frame),
    )
    render_asr_risk_scale_contents(
        fig,
        x=(1.0 - _RISK_SCALE_FOOTER_WIDTH) / 2.0,
        y=_footer_y(
            fig,
            scale_height=_RISK_SCALE_FOOTER_HEIGHT_IN / float(fig.get_size_inches()[1]),
        ),
        width=_RISK_SCALE_FOOTER_WIDTH,
        height=_RISK_SCALE_FOOTER_HEIGHT_IN / float(fig.get_size_inches()[1]),
        min_label=contract.min_line_label,
        max_label=contract.max_line_label,
        lower_zone_label=contract.lower_zone_label,
        middle_zone_label=contract.middle_zone_label,
        upper_zone_label=contract.upper_zone_label,
        background_text=background_text,
    )


def render_asr_risk_scale_contents(
    fig: Any,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    min_label: str,
    max_label: str | None,
    lower_zone_label: str,
    middle_zone_label: str | None,
    upper_zone_label: str,
    background_text: str | None,
) -> None:
    """Render the ASR risk color scale contents at figure coordinates."""
    axis = fig.add_axes([x, y, width, height], zorder=3)
    axis.set_facecolor("none")
    axis.axis("off")
    axis.text(
        0.5,
        0.90,
        "Risk level color scale",
        ha="center",
        fontsize=9,
        fontweight="bold",
        transform=axis.transAxes,
    )
    bar_axis = fig.add_axes([x, y + 0.50 * height, width, 0.14 * height], zorder=4)
    bar_axis.set_xlim(0.0, 1.0)
    bar_axis.set_ylim(0.0, 1.0)
    bar_axis.axis("off")
    u_values = np.linspace(0.0, 1.0, 1200)
    if max_label is None:
        rgba = np.asarray(
            [
                risk_scale_rgba(
                    SAFE_FRAC if value <= SAFE_FRAC else 1.0,
                    alpha=1.0,
                    lighten=SCALE_LIGHTEN,
                )
                for value in u_values
            ]
        )
    else:
        rgba = np.asarray(
            [risk_scale_rgba(float(value), alpha=1.0, lighten=SCALE_LIGHTEN) for value in u_values]
        )
    bar_axis.imshow(rgba[np.newaxis, :, :], extent=(0, 1, 0, 1), origin="lower", aspect="auto")
    bar_axis.add_patch(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=1.0))
    _render_scale_marker(bar_axis, xpos=SAFE_FRAC, color=_MIN_THRESHOLD_COLOR)
    if max_label is not None:
        _render_scale_marker(bar_axis, xpos=RISK_RAMP_END, color=_MAX_THRESHOLD_COLOR)
    axis.text(
        SAFE_FRAC,
        0.67,
        format_scientific_figure_text(min_label),
        ha="center",
        va="bottom",
        color=_MIN_THRESHOLD_COLOR,
        fontsize=8,
        transform=axis.transAxes,
    )
    if max_label is not None:
        axis.text(
            RISK_RAMP_END,
            0.67,
            format_scientific_figure_text(max_label),
            ha="center",
            va="bottom",
            color=_MAX_THRESHOLD_COLOR,
            fontsize=8,
            transform=axis.transAxes,
        )
    axis.text(
        SAFE_FRAC / 2.0,
        0.44,
        format_scientific_figure_text(_zone_label(lower_zone_label)),
        ha="center",
        va="top",
        fontsize=8,
        transform=axis.transAxes,
    )
    if max_label is None:
        axis.text(
            (SAFE_FRAC + 1.0) / 2.0,
            0.44,
            format_scientific_figure_text(_zone_label(upper_zone_label)),
            ha="center",
            va="top",
            fontsize=8,
            transform=axis.transAxes,
        )
    else:
        axis.text(
            (SAFE_FRAC + RISK_RAMP_END) / 2.0,
            0.44,
            format_scientific_figure_text(middle_zone_label or ""),
            ha="center",
            va="top",
            fontsize=8,
            transform=axis.transAxes,
        )
        axis.text(
            (RISK_RAMP_END + 1.0) / 2.0,
            0.44,
            format_scientific_figure_text(_zone_label(upper_zone_label)),
            ha="center",
            va="top",
            fontsize=8,
            transform=axis.transAxes,
        )
    if background_text is not None and str(background_text).strip():
        axis.text(
            0.0,
            0.01,
            format_scientific_figure_text(str(background_text).strip()),
            ha="left",
            va="bottom",
            fontsize=8,
            transform=axis.transAxes,
        )


def render_asr_risk_background(
    axis,
    *,
    max_threshold: float | None,
    alpha_scale: float = 1.0,
) -> None:
    """Render the static ASR risk background behind plotted values."""
    ymin, ymax = axis.get_ylim()
    alpha = max(0.0, min(1.0, float(alpha_scale)))
    axis.axhspan(ymin, 1.0, color=_SAFE_ZONE_COLOR, alpha=_SAFE_ZONE_ALPHA * alpha, zorder=0)
    if max_threshold is not None and max_threshold > 1.0:
        axis.axhspan(
            1.0,
            float(max_threshold),
            color=_MIDDLE_ZONE_COLOR,
            alpha=_MIDDLE_ZONE_ALPHA * alpha,
            zorder=0,
        )
        axis.axhspan(
            float(max_threshold),
            ymax,
            color=_HIGH_RISK_ZONE_COLOR,
            alpha=_HIGH_RISK_ZONE_ALPHA * alpha,
            zorder=0,
        )
    else:
        axis.axhspan(
            1.0,
            ymax,
            color=_HIGH_RISK_ZONE_COLOR,
            alpha=_HIGH_RISK_ZONE_ALPHA * alpha,
            zorder=0,
        )


def _footer_y(fig: Any, *, scale_height: float) -> float:
    fig.canvas.draw()
    height_in = float(fig.get_size_inches()[1])
    legend_top = _legend_top(fig)
    if legend_top is not None:
        candidate = legend_top + _RISK_SCALE_FOOTER_GAP_IN / height_in
        max_y = footer_content_top_limit(fig, padding_in=_RISK_SCALE_XTICK_GAP_IN) - scale_height
        return max(0.01, min(candidate, max_y))
    note_floor = max(0.01, (_BOTTOM_MARGIN_IN + _RISK_SCALE_FOOTER_GAP_IN) / height_in)
    tick_bottom = _lowest_visible_xaxis_label_bottom(fig)
    if tick_bottom is None:
        return note_floor
    tick_y = float(tick_bottom) - float(scale_height) - _RISK_SCALE_XTICK_GAP_IN / height_in
    return max(note_floor, tick_y)


def _legend_top(fig: Any) -> float | None:
    if not fig.legends:
        return None
    renderer = fig.canvas.get_renderer()
    tops = [
        legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).y1
        for legend in fig.legends
        if legend.get_visible()
    ]
    return max(tops) if tops else None


def _lowest_visible_xaxis_label_bottom(fig: Any) -> float | None:
    renderer = fig.canvas.get_renderer()
    bottoms = []
    for axis in fig.axes:
        labels = [
            label
            for label in [*axis.get_xticklabels(), axis.xaxis.label]
            if label.get_visible() and str(label.get_text()).strip()
        ]
        for label in labels:
            bottoms.append(
                float(
                    label.get_window_extent(renderer=renderer)
                    .transformed(fig.transFigure.inverted())
                    .y0
                )
            )
    return min(bottoms) if bottoms else None


def _render_scale_marker(axis: Any, *, xpos: float, color: str) -> None:
    axis.plot(
        [xpos, xpos],
        [0.0, 1.0],
        color=color,
        linewidth=0.85,
        linestyle=(0.0, (0.6, 2.2)),
        dash_capstyle="round",
    )


def _zone_label(label: str) -> str:
    return str(label).replace(" ", "\n")


def _cc_source(frame: pd.DataFrame) -> str:
    values = frame["lcia_method"].dropna().astype(str).str.strip()
    return str(values.loc[values.ne("")].iloc[0])

"""ASR polar labels, ticks, and legend layout."""

from typing import Any

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.patches import FancyBboxPatch

from pyaesa.asr.figures.axis import (
    ASR_LOG_SCALE,
    ASRScaleMode,
    normal_asr_tick_text,
    normal_asr_ticks,
)
from pyaesa.asr.figures.risk_guides import render_asr_risk_scale_contents
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.titles import FIGURE_TITLE_CLEARANCE_IN

_POLAR_LABEL_LEGEND_GAP = 0.018
_POLAR_LEGEND_MIN_Y = 0.018
_POLAR_IMPACT_LABEL_CLEARANCE_PT = 2.0
_POLAR_IMPACT_LABEL_POSITION_PASSES = 8
_POLAR_TITLE_SIZE = 12
_POLAR_LEGEND_HEIGHT = 0.13
_POLAR_DETERMINISTIC_LEGEND_WIDTH = 0.368
_POLAR_UNCERTAINTY_LEGEND_WIDTH = 0.92
_POLAR_DETERMINISTIC_NOTE_LEGEND_SCALE = 1.20


def render_impact_labels(
    fig: Any,
    axis: Any,
    *,
    theta_bounds: np.ndarray,
    labels: list[str],
    frequency_labels: list[str | None],
    r_max: float,
) -> float:
    """Render impact labels and frequency boxes outside the polar axis."""
    label_radius = _impact_label_radius(r_max=r_max)
    specs = []
    label_artists = []
    label_angles = []
    for index, label in enumerate(labels):
        theta_mid = 0.5 * (theta_bounds[index] + theta_bounds[index + 1])
        artist = axis.text(
            theta_mid,
            label_radius,
            label,
            ha="center",
            va="center",
            fontsize=10,
            clip_on=False,
            zorder=20,
        )
        artist.set_path_effects(
            [path_effects.withStroke(linewidth=1.2, foreground="white"), path_effects.Normal()]
        )
        label_artists.append(artist)
        label_angles.append(float(theta_mid))
        frequency = frequency_labels[index]
        if frequency is not None:
            specs.append((index, frequency, bool(np.sin(theta_mid) < 0.0)))
    label_boxes = _resolve_impact_label_positions(
        fig=fig,
        axis=axis,
        artists=label_artists,
        theta_values=label_angles,
        radius=label_radius,
    )
    inv_fig = fig.transFigure.inverted()
    label_bottom = min(float(inv_fig.transform((0.0, box[1]))[1]) for box in label_boxes)
    for index, label, on_left in specs:
        x0, y0, x1, y1 = label_boxes[index]
        y_display = 0.5 * (y0 + y1)
        x_display = x0 - 8.0 if on_left else x1 + 8.0
        x_fig, y_fig = inv_fig.transform((x_display, y_display))
        fig.text(
            x_fig,
            y_fig,
            label,
            ha="right" if on_left else "left",
            va="center",
            fontsize=8.0,
            bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#3d3d3d", "alpha": 1.0},
            zorder=30,
        )
    return label_bottom


def _impact_label_radius(*, r_max: float) -> float:
    return float(r_max)


def _resolve_impact_label_positions(
    *,
    fig: Any,
    axis: Any,
    artists: list[Any],
    theta_values: list[float],
    radius: float,
) -> list[tuple[float, float, float, float]]:
    current_radii = [float(radius) for _artist in artists]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    content_center, content_radius = _polar_content_circle(axis=axis, renderer=renderer)
    clearance = _points_to_pixels(fig, _POLAR_IMPACT_LABEL_CLEARANCE_PT)
    label_boxes = [
        _label_box_spec(axis=axis, renderer=renderer, artist=artist, theta=theta, radius=radius)
        for artist, theta in zip(artists, theta_values, strict=True)
    ]
    for _pass_index in range(_POLAR_IMPACT_LABEL_POSITION_PASSES):
        required_deltas = _required_label_radius_deltas(
            axis=axis,
            content_center=content_center,
            content_radius=content_radius,
            clearance=clearance,
            label_boxes=label_boxes,
            theta_values=theta_values,
            radii=current_radii,
        )
        if max(required_deltas, default=0.0) <= 0.0:
            return _label_display_boxes(
                axis=axis,
                label_boxes=label_boxes,
                theta_values=theta_values,
                radii=current_radii,
            )
        for index, (artist, theta) in enumerate(zip(artists, theta_values, strict=True)):
            current_radii[index] += required_deltas[index]
            artist.set_position((float(theta), current_radii[index]))
    return _label_display_boxes(
        axis=axis,
        label_boxes=label_boxes,
        theta_values=theta_values,
        radii=current_radii,
    )


def _label_box_spec(
    *,
    axis: Any,
    renderer: Any,
    artist: Any,
    theta: float,
    radius: float,
) -> tuple[float, float, np.ndarray]:
    bbox = artist.get_window_extent(renderer=renderer)
    data_center = np.asarray(axis.transData.transform((float(theta), float(radius))), dtype=float)
    bbox_center = np.asarray(
        [0.5 * (float(bbox.x0) + float(bbox.x1)), 0.5 * (float(bbox.y0) + float(bbox.y1))],
        dtype=float,
    )
    return (
        0.5 * float(bbox.width),
        0.5 * float(bbox.height),
        bbox_center - data_center,
    )


def _label_display_boxes(
    *,
    axis: Any,
    label_boxes: list[tuple[float, float, np.ndarray]],
    theta_values: list[float],
    radii: list[float],
) -> list[tuple[float, float, float, float]]:
    return [
        _label_display_box(
            center=(
                np.asarray(axis.transData.transform((float(theta), float(radius))), dtype=float)
                + offset
            ),
            half_width=half_width,
            half_height=half_height,
        )
        for (half_width, half_height, offset), theta, radius in zip(
            label_boxes,
            theta_values,
            radii,
            strict=True,
        )
    ]


def _label_display_box(
    *,
    center: np.ndarray,
    half_width: float,
    half_height: float,
) -> tuple[float, float, float, float]:
    return (
        float(center[0]) - float(half_width),
        float(center[1]) - float(half_height),
        float(center[0]) + float(half_width),
        float(center[1]) + float(half_height),
    )


def _required_label_radius_deltas(
    *,
    axis: Any,
    content_center: np.ndarray,
    content_radius: float,
    clearance: float,
    label_boxes: list[tuple[float, float, np.ndarray]],
    theta_values: list[float],
    radii: list[float],
) -> list[float]:
    deltas = []
    for label_box, theta, radius in zip(label_boxes, theta_values, radii, strict=True):
        half_width, half_height, offset = label_box
        label_center = (
            np.asarray(axis.transData.transform((float(theta), float(radius))), dtype=float)
            + offset
        )
        required_pixels = (
            content_radius
            + clearance
            - _box_distance_from_point(
                center=label_center,
                half_width=half_width,
                half_height=half_height,
                point=content_center,
            )
        )
        if required_pixels > 0.0:
            deltas.append(
                _radial_data_delta_for_pixels(
                    axis=axis,
                    theta=theta,
                    radius=radius,
                    pixels=required_pixels,
                )
            )
            continue
        deltas.append(0.0)
    return deltas


def _box_distance_from_point(
    *,
    center: np.ndarray,
    half_width: float,
    half_height: float,
    point: np.ndarray,
) -> float:
    dx = max(abs(float(point[0]) - float(center[0])) - float(half_width), 0.0)
    dy = max(abs(float(point[1]) - float(center[1])) - float(half_height), 0.0)
    return float(np.hypot(dx, dy))


def _polar_content_circle(*, axis: Any, renderer: Any) -> tuple[np.ndarray, float]:
    bbox = axis.get_window_extent(renderer=renderer)
    center = np.asarray(
        [float(bbox.x0) + 0.5 * float(bbox.width), float(bbox.y0) + 0.5 * float(bbox.height)],
        dtype=float,
    )
    radius = 0.5 * min(float(bbox.width), float(bbox.height))
    return center, radius


def _radial_data_delta_for_pixels(
    *,
    axis: Any,
    theta: float,
    radius: float,
    pixels: float,
) -> float:
    probe = max(abs(float(radius)) * 0.02, 0.01)
    start = np.asarray(axis.transData.transform((float(theta), float(radius))), dtype=float)
    end = np.asarray(axis.transData.transform((float(theta), float(radius) + probe)), dtype=float)
    pixels_per_data = float(np.hypot(*(end - start))) / probe
    return float(pixels) / pixels_per_data


def _points_to_pixels(fig: Any, points: float) -> float:
    return float(fig.dpi) * float(points) / 72.0


def render_polar_tick_marks(
    axis: Any,
    *,
    theta_bounds: np.ndarray,
    r_min: float,
    r_max: float,
    scale_mode: ASRScaleMode,
) -> None:
    """Render tick marks on every polar sector axis."""
    if scale_mode == ASR_LOG_SCALE:
        _render_log_tick_marks(axis, theta_bounds=theta_bounds, r_min=r_min, r_max=r_max)
        return
    _render_normal_tick_marks(axis, theta_bounds=theta_bounds, r_min=r_min, r_max=r_max)


def render_polar_title(fig: Any, axis: Any, *, title: str) -> None:
    """Render the polar title with a measured gap above visible polar content."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    height = max(float(fig.get_size_inches()[1]), 1.0)
    content_top = _polar_content_top(fig=fig, axis=axis, renderer=renderer)
    title_bottom = content_top + FIGURE_TITLE_CLEARANCE_IN / height
    fig.text(
        0.5,
        float(title_bottom),
        format_scientific_figure_text(title),
        ha="center",
        va="bottom",
        fontsize=_POLAR_TITLE_SIZE,
        fontweight="bold",
        color="black",
        transform=fig.transFigure,
    )


def _polar_content_top(*, fig: Any, axis: Any, renderer: Any) -> float:
    content_tops: list[float] = [float(axis.get_position().y1)]
    for artist in [*axis.texts, *fig.texts]:
        if not artist.get_visible() or not str(artist.get_text()).strip():
            continue
        bbox = artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        content_tops.append(float(bbox.y1))
    return max(content_tops)


def _render_log_tick_marks(
    axis: Any,
    *,
    theta_bounds: np.ndarray,
    r_min: float,
    r_max: float,
) -> None:
    major_exponents = range(int(np.floor(r_min)), int(np.ceil(r_max)) + 1)
    major_ticks = np.asarray(
        [float(exponent) for exponent in major_exponents if r_min <= exponent <= r_max],
        dtype=float,
    )
    minor_ticks = []
    for exponent in range(int(np.floor(r_min)), int(np.ceil(r_max))):
        base = 10.0 ** float(exponent)
        for sub_tick in range(2, 10):
            minor_ticks.append(np.log10(float(sub_tick) * base))
    minor_ticks_array = np.asarray(minor_ticks, dtype=float)
    minor_ticks_array = minor_ticks_array[
        (minor_ticks_array >= r_min) & (minor_ticks_array <= r_max)
    ]
    for theta in theta_bounds[:-1]:
        for tick in minor_ticks_array:
            _tick(axis, theta=theta, r=float(tick), half_width=0.010, lw=0.55, color="#666666")
        for tick in major_ticks:
            _tick(axis, theta=theta, r=float(tick), half_width=0.018, lw=1.15, color="#111111")
    for exponent in range(int(np.floor(r_min)), int(np.ceil(r_max)) + 1):
        log_value = float(exponent)
        if r_min <= log_value <= r_max:
            text = axis.text(
                0.0,
                log_value + 0.02,
                rf"$10^{{{exponent}}}$",
                ha="center",
                va="bottom",
                fontsize=8.5,
                zorder=45,
            )
            text.set_path_effects([path_effects.withStroke(linewidth=0.8, foreground="white")])


def _render_normal_tick_marks(
    axis: Any,
    *,
    theta_bounds: np.ndarray,
    r_min: float,
    r_max: float,
) -> None:
    ticks = normal_asr_ticks(lower=float(r_min), upper=float(r_max))
    label_offset = max(0.03, 0.008 * (float(r_max) - float(r_min)))
    for theta in theta_bounds[:-1]:
        for tick in ticks:
            _tick(axis, theta=theta, r=float(tick), half_width=0.018, lw=1.0, color="#111111")
    for tick in ticks:
        if np.isclose(float(tick), 0.0, rtol=0.0, atol=1e-12):
            continue
        text = axis.text(
            0.0,
            float(tick) + label_offset,
            normal_asr_tick_text(float(tick)),
            ha="center",
            va="bottom",
            fontsize=8.5,
            zorder=45,
        )
        text.set_path_effects([path_effects.withStroke(linewidth=0.8, foreground="white")])


def render_bottom_legend(
    fig: Any,
    *,
    label_bottom_y: float,
    style: str,
    min_label: str,
    max_label: str | None,
    lower_zone_label: str,
    middle_zone_label: str | None,
    upper_zone_label: str,
    fnt_label: str,
    deterministic_note: str | None = None,
) -> None:
    """Render the polar legend below the figure."""
    has_deterministic_note = (
        style == "deterministic"
        and deterministic_note is not None
        and bool(str(deterministic_note).strip())
    )
    height = _POLAR_LEGEND_HEIGHT
    if has_deterministic_note:
        height *= _POLAR_DETERMINISTIC_NOTE_LEGEND_SCALE
    y0 = max(
        _POLAR_LEGEND_MIN_Y,
        float(label_bottom_y) - _POLAR_LABEL_LEGEND_GAP - height,
    )
    box_w = (
        _POLAR_UNCERTAINTY_LEGEND_WIDTH
        if style != "deterministic"
        else _POLAR_DETERMINISTIC_LEGEND_WIDTH
    )
    if has_deterministic_note:
        box_w *= _POLAR_DETERMINISTIC_NOTE_LEGEND_SCALE
    box_x = 0.5 - box_w / 2.0
    fig.add_artist(
        FancyBboxPatch(
            (box_x, y0),
            box_w,
            height,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            transform=fig.transFigure,
            facecolor=(1.0, 1.0, 1.0, 1.0),
            edgecolor="#b6b6b6",
            linewidth=1.0,
            zorder=1,
        )
    )
    left_x = box_x + 0.015
    if style == "deterministic":
        right_x = left_x
        right_w = box_w - 0.030
    else:
        left_w = 0.42
        right_x = left_x + left_w + 0.006
        right_w = box_x + box_w - right_x - 0.015
        _render_glyph_legend(
            fig,
            x=left_x,
            y=y0,
            width=left_w,
            height=height,
            style=style,
            fnt_label=fnt_label,
        )
    render_asr_risk_scale_contents(
        fig,
        x=right_x,
        y=y0,
        width=right_w,
        height=height,
        min_label=min_label,
        max_label=max_label,
        lower_zone_label=lower_zone_label,
        middle_zone_label=middle_zone_label,
        upper_zone_label=upper_zone_label,
        background_text=(
            _deterministic_background_text(deterministic_note)
            if style == "deterministic"
            else "Background risk colors are shown until median ASR."
        ),
    )


def _deterministic_background_text(note: str | None) -> str:
    text = "Background risk colors are shown until the ASR value."
    if note is None or not str(note).strip():
        return text
    return f"{text}\n{str(note).strip()}"


def _tick(axis: Any, *, theta: float, r: float, half_width: float, lw: float, color: str) -> None:
    axis.plot([theta - half_width, theta + half_width], [r, r], color=color, lw=lw, zorder=40)


def _render_glyph_legend(
    fig: Any,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    style: str,
    fnt_label: str,
) -> None:
    axis = fig.add_axes([x, y, width, height], zorder=3)
    axis.set_facecolor("none")
    axis.axis("off")
    title = {
        "violin": "ASR violins",
        "whisker": "ASR whiskers",
    }[style]
    axis.text(0.0, 0.90, title, fontsize=9, fontweight="bold", transform=axis.transAxes)
    x0 = 0.13
    if style == "violin":
        y_values = np.linspace(0.22, 0.68, 120)
        widths = 0.012 + 0.034 * np.exp(-(((y_values - 0.45) / 0.13) ** 2))
        axis.fill_betweenx(
            y_values,
            x0 - widths,
            x0 + widths,
            facecolor="#f2c97d",
            edgecolor="black",
            linewidth=1.0,
            transform=axis.transAxes,
        )
        axis.text(
            0.0,
            0.78,
            _violin_width_label(style),
            va="center",
            fontsize=8,
            transform=axis.transAxes,
        )
    if style == "whisker":
        axis.plot([x0, x0], [0.22, 0.79], color="black", lw=1.6, transform=axis.transAxes)
        axis.plot(
            [x0 - 0.03, x0 + 0.03], [0.22, 0.22], color="black", lw=1.8, transform=axis.transAxes
        )
        axis.plot(
            [x0 - 0.03, x0 + 0.03], [0.79, 0.79], color="black", lw=1.8, transform=axis.transAxes
        )
        axis.plot([x0, x0], [0.33, 0.68], color="black", lw=6.0, transform=axis.transAxes)
        axis.text(0.20, 0.79, "95th percentile", va="center", fontsize=8, transform=axis.transAxes)
        axis.text(0.20, 0.22, "5th percentile", va="center", fontsize=8, transform=axis.transAxes)
    axis.scatter([x0], [0.57], s=43.2, color="white", edgecolors="none", transform=axis.transAxes)
    axis.scatter(
        [x0],
        [0.57],
        s=31.2,
        facecolors="none",
        edgecolors="#8b5a00",
        linewidths=1.4,
        transform=axis.transAxes,
    )
    axis.plot(
        [x0 - 0.025, x0 + 0.025], [0.44, 0.44], color="#8b5a00", lw=2.6, transform=axis.transAxes
    )
    axis.text(
        0.20,
        0.57,
        "Mean (dot, color by risk zone)",
        va="center",
        fontsize=8,
        transform=axis.transAxes,
    )
    axis.text(
        0.20,
        0.44,
        "Median (thick line, color by risk zone)",
        va="center",
        fontsize=8,
        transform=axis.transAxes,
    )
    axis.text(
        0.0,
        0.01,
        format_scientific_figure_text(fnt_label),
        fontsize=8,
        ha="left",
        va="bottom",
        transform=axis.transAxes,
    )


def _violin_width_label(style: str) -> str:
    del style
    return (
        "Width is proportional to the distribution of Monte Carlo ASR runs. "
        "\nHeight extends from minimum to maximum."
    )

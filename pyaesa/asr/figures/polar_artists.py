"""ASR polar rendering primitives."""

from typing import Any, Mapping

from matplotlib import colors
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
import numpy as np

from pyaesa.asr.figures.axis import ASR_LOG_SCALE, ASRScaleMode
from pyaesa.asr.figures.risk_style import (
    MAX_RISK_COLOR,
    RISK_RAMP_END,
    SAFE_COLOR,
    SAFE_FRAC,
    SCALE_LIGHTEN,
    make_bg_risk_cmap,
)

SOS_LINE_LIGHTEN = 0.0
VIOLIN_LIGHTEN = 0.20
WHISKER_LIGHTEN = 0.20
EDGE_COLOR = "#2f2f2f"
MIN_SCALE_COLOR = SAFE_COLOR
MAX_SCALE_COLOR = MAX_RISK_COLOR
BACKGROUND_BAND_COUNT = 1000
BACKGROUND_BAND_OVERLAP = 0.04
VIOLIN_GEOM_RES = 1000
VIOLIN_STRIP_COUNT = 1000
VIOLIN_ARC_STEPS = 48
VIOLIN_BORDER_GAP_PX = 5.0
WHISKER_THIN_HALF_PX = 2.4
WHISKER_BOX_HALF_PX = 7.2
WHISKER_CAP_HALF_PX = 6.8
WHISKER_MED_HALF_PX = 11.5
WHISKER_MED_BAND_PX = 3.0
MEAN_DOT_SIZE = 23.4
MEAN_DOT_RING_SIZE = 17.55
VIOLIN_MEDIAN_HALF_PX = 5.72

_RISK_CMAP = make_bg_risk_cmap()


def risk_rgba(value: float, max_ratio: float, *, alpha: float, lighten: float) -> tuple:
    """Return one risk color for an ASR value."""
    u = _risk_scale_u(value=value, max_ratio=max_ratio)
    return risk_scale_rgba(u, alpha=alpha, lighten=lighten)


def risk_scale_rgba(position: float, *, alpha: float, lighten: float) -> tuple:
    """Return one color at a normalized ASR risk scale position."""
    u = float(np.clip(position, 0.0, 1.0))
    if u <= SAFE_FRAC:
        base = colors.to_rgba(SAFE_COLOR, alpha=alpha)
    elif u >= RISK_RAMP_END:
        base = colors.to_rgba(MAX_RISK_COLOR, alpha=alpha)
    else:
        t = (u - SAFE_FRAC) / max(RISK_RAMP_END - SAFE_FRAC, 1e-12)
        red, green, blue, _alpha = _RISK_CMAP(float(np.clip(t, 0.0, 1.0)))
        base = (red, green, blue, alpha)
    return _lighten(base, lighten)


def render_threshold_arcs(
    axis: Any,
    *,
    theta_bounds: np.ndarray,
    max_radii: np.ndarray,
    has_max_threshold: bool,
    scale_mode: ASRScaleMode,
) -> None:
    """Render custom ASR sector axes and threshold arcs."""
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    min_color = risk_scale_rgba(SAFE_FRAC, alpha=1.0, lighten=SOS_LINE_LIGHTEN)
    max_color = risk_scale_rgba(RISK_RAMP_END, alpha=1.0, lighten=SOS_LINE_LIGHTEN)
    min_radius = _radius_from_value(1.0, scale_mode=scale_mode)
    axis.plot(theta, np.full_like(theta, min_radius), ":", color=min_color, lw=2.5, zorder=3)
    y_min, y_max = axis.get_ylim()
    for theta_bound in theta_bounds:
        axis.plot([theta_bound, theta_bound], [y_min, y_max], color="#777777", lw=1.0, zorder=40)
    if not has_max_threshold:
        return
    for index in range(len(theta_bounds) - 1):
        segment = np.linspace(theta_bounds[index], theta_bounds[index + 1], 150)
        axis.plot(
            segment,
            np.full_like(segment, float(max_radii[index])),
            ":",
            color=max_color,
            lw=2.5,
            zorder=3,
        )


def render_risk_background(
    axis: Any,
    *,
    theta0: float,
    theta1: float,
    r_min: float,
    r_end: float,
    max_ratio: float,
    scale_mode: ASRScaleMode,
) -> None:
    """Render sector background risk color up to the selected ASR value."""
    min_radius = _radius_from_value(1.0, scale_mode=scale_mode)
    max_radius = _radius_from_value(max(max_ratio, 1.0000001), scale_mode=scale_mode)
    radial_edges = _split_edges(
        np.linspace(r_min, r_end, BACKGROUND_BAND_COUNT),
        points=[min_radius, max_radius],
        lower=r_min,
        upper=r_end,
    )
    theta_arc = np.linspace(theta0, theta1, 72)
    raw0 = radial_edges[:-1]
    raw1 = radial_edges[1:]
    delta = raw1 - raw0
    r0 = np.maximum(float(r_min), raw0 - (BACKGROUND_BAND_OVERLAP * delta))
    r1 = np.minimum(float(r_end), raw1 + (BACKGROUND_BAND_OVERLAP * delta))
    r0, r1 = _clipped_risk_bands(r0, r1, raw1, min_radius, max_radius)
    middle = 0.5 * (r0 + r1)
    face_colors = _risk_rgba_values(
        _values_from_radii(middle, scale_mode=scale_mode),
        max_ratio,
        alpha=1.0,
        lighten=SCALE_LIGHTEN,
    )
    polygons = np.empty((r0.size, theta_arc.size * 2, 2), dtype=float)
    polygons[:, : theta_arc.size, 0] = theta_arc
    polygons[:, : theta_arc.size, 1] = r1[:, np.newaxis]
    polygons[:, theta_arc.size :, 0] = theta_arc[::-1]
    polygons[:, theta_arc.size :, 1] = r0[:, np.newaxis]
    polygon_list = [polygon for polygon in polygons]
    axis.add_collection(
        PolyCollection(
            polygon_list,
            closed=True,
            facecolors=face_colors,
            edgecolors="none",
            antialiaseds=False,
            rasterized=True,
            zorder=1,
            transform=axis.transData,
        )
    )


def render_uncertainty_glyph(
    axis: Any,
    *,
    theta_mid: float,
    sector_width: float,
    radial_payload: np.ndarray,
    summary: Mapping[str, float],
    max_ratio: float,
    density_scale: float,
    style: str,
    scale_mode: ASRScaleMode,
) -> None:
    """Render one uncertainty violin or whisker polar glyph."""
    if style == "violin":
        _render_violin(
            axis,
            theta_mid,
            sector_width,
            radial_payload,
            max_ratio,
            density_scale,
            scale_mode,
        )
        median = _summary_radius(summary, "median", scale_mode=scale_mode)
        _render_violin_median_marker(
            axis,
            theta=theta_mid,
            r=median,
            max_ratio=max_ratio,
            scale_mode=scale_mode,
        )
    else:
        _render_whisker(axis, theta_mid, sector_width, summary, max_ratio, scale_mode)
    mean = _summary_radius(summary, "mean", scale_mode=scale_mode)
    _render_mean_marker(axis, theta=theta_mid, r=mean, max_ratio=max_ratio, scale_mode=scale_mode)


def _render_violin(
    axis: Any,
    theta_mid: float,
    sector_width: float,
    radial_payload: np.ndarray,
    max_ratio: float,
    density_scale: float,
    scale_mode: ASRScaleMode,
) -> None:
    profile = _violin_profile(radial_payload)
    if profile is None:
        return
    radial_points, density = profile
    half_width = _violin_half_width(
        axis,
        theta_mid,
        radial_points,
        density / density_scale,
        sector_width,
    )
    r_line = np.linspace(float(radial_points[0]), float(radial_points[-1]), VIOLIN_GEOM_RES)
    width_line = np.interp(
        r_line,
        radial_points,
        half_width,
        left=float(half_width[0]),
        right=float(half_width[-1]),
    )
    body = _closed_profile(theta_mid, r_line, width_line)
    polygons = [body]
    face_colors = [risk_rgba(1.0, max_ratio, alpha=1.0, lighten=VIOLIN_LIGHTEN)]
    min_radius = _radius_from_value(1.0, scale_mode=scale_mode)
    max_radius = _radius_from_value(max(max_ratio, 1.0000001), scale_mode=scale_mode)
    strip_lower = max(float(radial_points[0]), min_radius)
    strip_upper = float(radial_points[-1])
    strip_edges = (
        _split_edges(
            np.linspace(strip_lower, strip_upper, VIOLIN_STRIP_COUNT),
            points=[max_radius],
            lower=strip_lower,
            upper=strip_upper,
        )
        if strip_upper > strip_lower
        else np.asarray([], dtype=float)
    )
    strip_middles = 0.5 * (strip_edges[:-1] + strip_edges[1:])
    strip_colors = _risk_rgba_values(
        _values_from_radii(strip_middles, scale_mode=scale_mode),
        max_ratio,
        alpha=1.0,
        lighten=VIOLIN_LIGHTEN,
    )
    for lower, upper in zip(strip_edges[:-1], strip_edges[1:], strict=True):
        strip_r = np.linspace(float(lower), float(upper), 10)
        strip_w = np.interp(strip_r, r_line, width_line)
        strip = _closed_profile(theta_mid, strip_r, strip_w)
        polygons.append(strip)
    face_colors.extend([tuple(color) for color in strip_colors])
    axis.add_collection(
        PolyCollection(
            polygons,
            closed=True,
            facecolors=face_colors,
            edgecolors="none",
            antialiaseds=False,
            rasterized=True,
            zorder=5,
            transform=axis.transData,
        )
    )
    axis.add_patch(
        Polygon(
            body,
            closed=True,
            facecolor="none",
            edgecolor=EDGE_COLOR,
            linewidth=0.95,
            joinstyle="round",
            zorder=7,
        )
    )


def _render_whisker(
    axis: Any,
    theta_mid: float,
    sector_width: float,
    summary: Mapping[str, float],
    max_ratio: float,
    scale_mode: ASRScaleMode,
) -> None:
    quantiles = [
        _summary_radius(summary, column, scale_mode=scale_mode)
        for column in ("p5", "p25", "median", "p75", "p95")
    ]
    p5, p25, median, p75, p95 = [float(value) for value in quantiles]
    lo = float(min(p5, p95))
    hi = float(max(p5, p95))
    if hi <= lo:
        return
    radial = _split_edges(
        np.linspace(lo, hi, 560),
        points=[
            _radius_from_value(1.0, scale_mode=scale_mode),
            _radius_from_value(max(max_ratio, 1.0000001), scale_mode=scale_mode),
            p25,
            median,
            p75,
        ],
        lower=lo,
        upper=hi,
    )
    band = _median_band(axis, theta_mid, float(median), lo, hi)
    border = 0.5 * sector_width * 0.999
    widths = _whisker_half_widths(
        axis,
        theta_mid,
        radial,
        p5,
        p25,
        median,
        p75,
        p95,
        band,
        border,
    )
    lower = radial[:-1]
    upper = radial[1:]
    lower_width = widths[:-1]
    upper_width = widths[1:]
    polygons = np.empty((lower.size, 4, 2), dtype=float)
    polygons[:, 0, 0] = theta_mid - lower_width
    polygons[:, 0, 1] = lower
    polygons[:, 1, 0] = theta_mid + lower_width
    polygons[:, 1, 1] = lower
    polygons[:, 2, 0] = theta_mid + upper_width
    polygons[:, 2, 1] = upper
    polygons[:, 3, 0] = theta_mid - upper_width
    polygons[:, 3, 1] = upper
    polygon_list = [polygon for polygon in polygons]
    middle = 0.5 * (lower + upper)
    face_colors = _risk_rgba_values(
        _values_from_radii(middle, scale_mode=scale_mode),
        max_ratio,
        alpha=1.0,
        lighten=WHISKER_LIGHTEN,
    )
    axis.add_collection(
        PolyCollection(
            polygon_list,
            closed=True,
            facecolors=face_colors,
            edgecolors="none",
            antialiaseds=False,
            rasterized=True,
            zorder=6,
            transform=axis.transData,
        )
    )
    _render_whisker_median_bar(
        axis,
        theta=theta_mid,
        r=float(median),
        max_ratio=max_ratio,
        scale_mode=scale_mode,
    )
    outline = np.column_stack(
        [
            np.concatenate([theta_mid - widths, (theta_mid + widths)[::-1]]),
            np.concatenate([radial, radial[::-1]]),
        ]
    )
    axis.add_patch(
        Polygon(
            outline,
            closed=True,
            facecolor="none",
            edgecolor=EDGE_COLOR,
            linewidth=0.95,
            joinstyle="round",
            zorder=7.5,
        )
    )


def _whisker_half_widths(
    axis: Any,
    theta_mid: float,
    radial: np.ndarray,
    p5: float,
    p25: float,
    median: float,
    p75: float,
    p95: float,
    band: float,
    border: float,
) -> np.ndarray:
    r_values = np.asarray(radial, dtype=float)
    half_pixels = np.full(r_values.shape, WHISKER_THIN_HALF_PX, dtype=float)
    box_mask = (p25 <= r_values) & (r_values <= p75)
    half_pixels[box_mask] = WHISKER_BOX_HALF_PX
    cap_mask = (np.abs(r_values - p5) <= band) | (np.abs(r_values - p95) <= band)
    half_pixels[cap_mask] = np.maximum(half_pixels[cap_mask], WHISKER_CAP_HALF_PX)
    median_mask = np.abs(r_values - median) <= band
    half_pixels[median_mask] = np.maximum(half_pixels[median_mask], WHISKER_MED_HALF_PX)
    return np.minimum(
        _theta_half_spans_for_px(axis, theta_mid=theta_mid, r_values=r_values, half_px=half_pixels),
        border,
    )


def violin_density_peak(log_payload: np.ndarray) -> float:
    """Return the maximum empirical log density for one polar violin payload."""
    profile = _violin_profile(log_payload)
    return 0.0 if profile is None else float(np.max(profile[1]))


def _violin_profile(log_payload: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if log_payload.size < 2:
        return None
    lo = float(np.nanmin(log_payload))
    hi = float(np.nanmax(log_payload))
    if hi <= lo:
        return None
    edges = np.asarray(np.histogram_bin_edges(log_payload, bins="fd"), dtype=float)
    if edges.size < 8:
        edges = np.linspace(lo, hi, 9)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density, _edges = np.histogram(log_payload, bins=edges, density=True)
    numeric = np.maximum(np.asarray(density, dtype=float), 0.0)
    return centers, numeric


def _closed_profile(theta_mid: float, r_line: np.ndarray, half_width: np.ndarray) -> np.ndarray:
    left = _wrap_theta(theta_mid - half_width, theta_mid)
    right = _wrap_theta(theta_mid + half_width, theta_mid)
    top_count = _arc_count(left[-1], right[-1])
    bottom_count = _arc_count(left[0], right[0])
    top = _wrap_theta(np.linspace(left[-1], right[-1], top_count), theta_mid)
    bottom = _wrap_theta(np.linspace(right[0], left[0], bottom_count), theta_mid)
    theta = np.concatenate([left, top[1:], right[::-1], bottom[1:]])
    radius = np.concatenate(
        [
            r_line,
            np.full(top_count - 1, r_line[-1]),
            r_line[::-1],
            np.full(bottom_count - 1, r_line[0]),
        ]
    )
    return np.column_stack([theta, radius])


def _violin_half_width(
    axis: Any,
    theta_mid: float,
    centers: np.ndarray,
    normalized_width: np.ndarray,
    sector_width: float,
) -> np.ndarray:
    border = 0.5 * sector_width
    gaps = _theta_half_spans_for_px(
        axis,
        theta_mid=theta_mid,
        r_values=np.asarray(centers, dtype=float),
        half_px=np.full(np.asarray(centers).shape, VIOLIN_BORDER_GAP_PX, dtype=float),
    )
    available = np.maximum(border - gaps, 0.0)
    return np.asarray(normalized_width, dtype=float) * available


def _render_violin_median_marker(
    axis: Any,
    *,
    theta: float,
    r: float,
    max_ratio: float,
    scale_mode: ASRScaleMode,
) -> None:
    color = risk_rgba(
        _value_from_radius(r, scale_mode=scale_mode),
        max_ratio,
        alpha=1.0,
        lighten=0.08,
    )
    half_width = _theta_half_span_for_px(
        axis,
        theta_mid=theta,
        r_val=r,
        half_px=VIOLIN_MEDIAN_HALF_PX,
    )
    half_height = _radial_half_span_for_px(axis, theta=theta, r=r, half_px=1.2)
    outer = np.asarray(
        [
            [theta - half_width, r - half_height],
            [theta + half_width, r - half_height],
            [theta + half_width, r + half_height],
            [theta - half_width, r + half_height],
        ],
        dtype=float,
    )
    inner = np.asarray(
        [
            [theta - half_width * 0.72, r - half_height * 0.62],
            [theta + half_width * 0.72, r - half_height * 0.62],
            [theta + half_width * 0.72, r + half_height * 0.62],
            [theta - half_width * 0.72, r + half_height * 0.62],
        ],
        dtype=float,
    )
    axis.add_patch(
        Polygon(outer, closed=True, facecolor=color, edgecolor=color, linewidth=0.8, zorder=13)
    )
    axis.add_patch(
        Polygon(inner, closed=True, facecolor="white", edgecolor="none", linewidth=0.0, zorder=14)
    )


def _render_whisker_median_bar(
    axis: Any,
    *,
    theta: float,
    r: float,
    max_ratio: float,
    scale_mode: ASRScaleMode,
) -> None:
    color = risk_rgba(
        _value_from_radius(r, scale_mode=scale_mode),
        max_ratio,
        alpha=1.0,
        lighten=WHISKER_LIGHTEN,
    )
    half_width = _theta_half_span_for_px(
        axis,
        theta_mid=theta,
        r_val=r,
        half_px=WHISKER_MED_HALF_PX,
    )
    axis.plot(
        [theta - half_width, theta + half_width],
        [r, r],
        color=color,
        lw=2.6,
        solid_capstyle="butt",
        zorder=7.2,
    )


def _render_mean_marker(
    axis: Any,
    *,
    theta: float,
    r: float,
    max_ratio: float,
    scale_mode: ASRScaleMode,
) -> None:
    color = risk_rgba(
        _value_from_radius(r, scale_mode=scale_mode),
        max_ratio,
        alpha=1.0,
        lighten=0.08,
    )
    axis.scatter(
        [theta], [r], marker="o", s=MEAN_DOT_SIZE, color="white", edgecolors="none", zorder=15
    )
    axis.scatter(
        [theta],
        [r],
        marker="o",
        s=MEAN_DOT_RING_SIZE,
        facecolors="none",
        edgecolors=color,
        linewidths=1.2,
        zorder=16,
    )


def _theta_half_span_for_px(axis: Any, *, theta_mid: float, r_val: float, half_px: float) -> float:
    probe = 1e-4
    center = axis.transData.transform((theta_mid, r_val))
    shifted = axis.transData.transform((theta_mid + probe, r_val))
    pixels_per_radian = float(np.hypot(*(shifted - center)) / probe)
    return float(half_px) / pixels_per_radian


def _theta_half_spans_for_px(
    axis: Any,
    *,
    theta_mid: float,
    r_values: np.ndarray,
    half_px: np.ndarray,
) -> np.ndarray:
    probe = 1e-4
    r_array = np.asarray(r_values, dtype=float)
    theta = np.full(r_array.shape, float(theta_mid), dtype=float)
    center = axis.transData.transform(np.column_stack([theta, r_array]))
    shifted = axis.transData.transform(np.column_stack([theta + probe, r_array]))
    pixels_per_radian = np.hypot(*(shifted - center).T) / probe
    return np.asarray(half_px, dtype=float) / pixels_per_radian


def _radial_half_span_for_px(axis: Any, *, theta: float, r: float, half_px: float) -> float:
    probe = 1e-4
    center = axis.transData.transform((theta, r))
    shifted = axis.transData.transform((theta, r + probe))
    pixels_per_unit = float(np.hypot(*(shifted - center)) / probe)
    return float(half_px) / pixels_per_unit


def _median_band(axis: Any, theta: float, r: float, lower: float, upper: float) -> float:
    return float(
        np.clip(
            _radial_half_span_for_px(axis, theta=theta, r=r, half_px=WHISKER_MED_BAND_PX),
            1e-6,
            max(upper - lower, 1e-6),
        )
    )


def _split_edges(
    edges: np.ndarray, *, points: list[float], lower: float, upper: float
) -> np.ndarray:
    in_range = [point for point in points if lower < float(point) < upper]
    return (
        np.sort(np.unique(np.concatenate([edges, np.asarray(in_range, dtype=float)])))
        if in_range
        else edges
    )


def _clipped_risk_bands(
    r0: np.ndarray,
    r1: np.ndarray,
    raw1: np.ndarray,
    min_radius: float,
    max_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(r0, dtype=float).copy()
    upper = np.asarray(r1, dtype=float).copy()
    raw_upper = np.asarray(raw1, dtype=float)
    below_min = raw_upper <= min_radius
    upper = np.where(below_min, np.minimum(upper, min_radius), upper)
    lower = np.where(~below_min, np.maximum(lower, min_radius), lower)
    below_max = raw_upper <= max_radius
    upper = np.where(below_max, np.minimum(upper, max_radius), upper)
    lower = np.where(~below_max, np.maximum(lower, max_radius), lower)
    return lower, upper


def _risk_scale_u(*, value: float, max_ratio: float) -> float:
    if value <= 1.0:
        return SAFE_FRAC
    if max_ratio <= 1.0 or value >= max_ratio:
        return 1.0
    position = np.log10(value) / max(np.log10(max_ratio), 1e-12)
    return SAFE_FRAC + float(np.clip(position, 0.0, 1.0)) * (RISK_RAMP_END - SAFE_FRAC)


def _risk_scale_values(values: np.ndarray, max_ratio: float) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    positions = np.ones(numeric.shape, dtype=float)
    safe = numeric <= 1.0
    positions[safe] = SAFE_FRAC
    if max_ratio > 1.0:
        between = (~safe) & (numeric < max_ratio)
        scaled = np.log10(numeric[between]) / max(np.log10(max_ratio), 1e-12)
        positions[between] = SAFE_FRAC + np.clip(scaled, 0.0, 1.0) * (RISK_RAMP_END - SAFE_FRAC)
    return positions


def _risk_rgba_values(
    values: np.ndarray,
    max_ratio: float,
    *,
    alpha: float,
    lighten: float,
) -> np.ndarray:
    positions = _risk_scale_values(values, max_ratio)
    out = np.empty((positions.size, 4), dtype=float)
    safe = positions <= SAFE_FRAC
    maximum = positions >= RISK_RAMP_END
    middle = ~(safe | maximum)
    out[safe] = colors.to_rgba(SAFE_COLOR, alpha=alpha)
    out[maximum] = colors.to_rgba(MAX_RISK_COLOR, alpha=alpha)
    if np.any(middle):
        scaled = (positions[middle] - SAFE_FRAC) / max(RISK_RAMP_END - SAFE_FRAC, 1e-12)
        out[middle] = _RISK_CMAP(np.clip(scaled, 0.0, 1.0))
        out[middle, 3] = alpha
    amount = float(np.clip(lighten, 0.0, 1.0))
    out[:, :3] = out[:, :3] + (1.0 - out[:, :3]) * amount
    return out


def _arc_count(theta_left: float, theta_right: float) -> int:
    span = abs(float(theta_right) - float(theta_left))
    return max(3, int(np.ceil(max(span / (2.0 * np.pi), 1e-6) * VIOLIN_ARC_STEPS)))


def _wrap_theta(theta_values: np.ndarray, theta_ref: float) -> np.ndarray:
    values = np.asarray(theta_values, dtype=float)
    return ((values - theta_ref + np.pi) % (2.0 * np.pi)) - np.pi + theta_ref


def _lighten(rgba: tuple, fraction: float) -> tuple:
    red, green, blue, alpha = rgba
    amount = float(np.clip(fraction, 0.0, 1.0))
    return (
        red + (1.0 - red) * amount,
        green + (1.0 - green) * amount,
        blue + (1.0 - blue) * amount,
        alpha,
    )


def _summary_radius(
    summary: Mapping[str, float],
    column: str,
    *,
    scale_mode: ASRScaleMode,
) -> float:
    numeric = float(summary[column])
    return _radius_from_value(numeric, scale_mode=scale_mode)


def _radius_from_value(value: float, *, scale_mode: ASRScaleMode) -> float:
    numeric = float(value)
    return float(np.log10(numeric)) if scale_mode == ASR_LOG_SCALE else numeric


def _values_from_radii(radii: np.ndarray, *, scale_mode: ASRScaleMode) -> np.ndarray:
    numeric = np.asarray(radii, dtype=float)
    return np.power(10.0, numeric) if scale_mode == ASR_LOG_SCALE else numeric


def _value_from_radius(radius: float, *, scale_mode: ASRScaleMode) -> float:
    numeric = float(radius)
    return float(10.0**numeric) if scale_mode == ASR_LOG_SCALE else numeric

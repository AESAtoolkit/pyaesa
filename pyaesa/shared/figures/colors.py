"""Shared deterministic color policies for figure families."""

import colorsys

from matplotlib.colors import to_hex, to_rgb

HIGH_CONTRAST_COLORS = (
    "#0072B2",
    "#D55E00",
    "#7A3E9D",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#00A6D6",
    "#8B1A1A",
    "#4D4D4D",
    "#708090",
    "#1B3A8A",
    "#FF1493",
    "#000000",
    "#A6761D",
    "#E31A1C",
    "#6A3D9A",
    "#17BECF",
    "#7F7F7F",
    "#FF6F61",
    "#5A2A83",
    "#2F6B3F",
    "#9C3B00",
    "#2C7FB8",
    "#6E2C2C",
)
DEFAULT_SINGLE_SERIES_COLOR = "#0072B2"
MULTI_METHOD_LINE_ALPHA = 0.68


def distinct_colors(count: int) -> list[str]:
    """Return deterministic visually separated colors without palette cycling."""
    total = max(0, int(count))
    if total == 0:
        return []
    if total == 1:
        return [DEFAULT_SINGLE_SERIES_COLOR]
    if total <= len(HIGH_CONTRAST_COLORS):
        return list(HIGH_CONTRAST_COLORS[:total])
    selected: list[str] = [str(color) for color in HIGH_CONTRAST_COLORS]
    candidates = _candidate_colors(count=total)
    while len(selected) < total:
        next_color = max(
            candidates,
            key=lambda color: min(
                _rgb_distance(color, selected_color) for selected_color in selected
            ),
        )
        selected.append(next_color)
        candidates.remove(next_color)
    return selected


def single_or_distinct_colors(labels: list[str]) -> dict[str, str]:
    """Return blue for one visible label and method colors otherwise."""
    unique_labels = list(dict.fromkeys(str(label) for label in labels))
    return {
        label: color
        for label, color in zip(
            unique_labels,
            distinct_colors(len(unique_labels)),
            strict=True,
        )
    }


def _candidate_colors(*, count: int) -> list[str]:
    candidates: list[str] = []
    steps = max(48, int(count) * 4)
    for lightness in (0.34, 0.52, 0.68):
        for saturation in (0.82, 0.62):
            for index in range(steps):
                hue = (0.07 + index * 0.61803398875) % 1.0
                candidates.append(to_hex(colorsys.hls_to_rgb(hue, lightness, saturation)))
    return list(dict.fromkeys(candidates))


def _rgb_distance(left: str, right: str) -> float:
    left_rgb = to_rgb(left)
    right_rgb = to_rgb(right)
    return (
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left_rgb, right_rgb, strict=True)
        )
        ** 0.5
    )

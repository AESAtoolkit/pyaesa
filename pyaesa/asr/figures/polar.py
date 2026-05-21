"""ASR polar figure renderer."""

from pathlib import Path
import textwrap
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyaesa.asr.figures.axis import ASR_LOG_SCALE, ASRScaleMode, positive_asr_values
from pyaesa.asr.figures.frequency import format_fnt_math_label
from pyaesa.asr.figures.polar_artists import (
    render_risk_background,
    render_threshold_arcs,
    render_uncertainty_glyph,
    violin_density_peak,
)
from pyaesa.asr.figures.polar_layout import (
    render_bottom_legend,
    render_impact_labels,
    render_polar_title,
    render_polar_tick_marks,
)
from pyaesa.asr.figures.threshold_contract import (
    build_asr_threshold_contract,
    has_max_asr_threshold,
)
from pyaesa.shared.figures.lcia_metadata import load_lcia_metadata
from pyaesa.shared.figures.save import save_figure

PolarStyle = Literal["deterministic", "violin", "whisker"]
NON_PB_POLAR_LABEL_WIDTH = 16
_POLAR_BOTTOM = 0.285
_POLAR_DETERMINISTIC_NOTE_BOTTOM = 0.315


def render_asr_polar(
    *,
    frame: pd.DataFrame,
    values: dict[str, np.ndarray],
    summaries: dict[str, dict[str, float]] | None = None,
    frequencies: dict[str, float] | None = None,
    output_stem: Path,
    title: str,
    lcia_method: str,
    style: PolarStyle,
    dpi: int,
    output_format: str,
    scale_mode: ASRScaleMode = ASR_LOG_SCALE,
    deterministic_note: str | None = None,
) -> list[Path]:
    """Render one ASR polar checkpoint from persisted ASR figure values."""
    impacts = _ordered_impacts(frame=frame, lcia_method=lcia_method)
    radial_values, zero_count = _radial_values(
        values=values,
        impacts=impacts,
        context=title,
        scale_mode=scale_mode,
    )
    density_scale = _density_scale(radial_values=radial_values, style=style)
    show_max_threshold = has_max_asr_threshold(frame=frame)
    ratios = _max_ratios(lcia_method=lcia_method, impacts=impacts, enabled=show_max_threshold)
    r_min, r_max = _radial_limits(
        radial_values=radial_values,
        ratios=ratios,
        scale_mode=scale_mode,
    )
    theta_bounds = np.linspace(0.0, 2.0 * np.pi, len(impacts) + 1)
    widths = theta_bounds[1:] - theta_bounds[:-1]
    fig, raw_axis = plt.subplots(figsize=(9.0, 10.4), subplot_kw={"projection": "polar"})
    bottom = _POLAR_BOTTOM
    if (
        style == "deterministic"
        and deterministic_note is not None
        and str(deterministic_note).strip()
    ):
        bottom = _POLAR_DETERMINISTIC_NOTE_BOTTOM
    fig.subplots_adjust(bottom=bottom, top=0.93)
    axis: Any = raw_axis
    axis.set_theta_offset(np.pi / 2.0)
    axis.set_theta_direction(-1)
    axis.set_ylim(r_min, r_max)
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
    max_radii = np.asarray(
        [
            _radius_from_value(max(ratios[impact], 1.0000001), scale_mode=scale_mode)
            for impact in impacts
        ]
    )
    render_threshold_arcs(
        axis,
        theta_bounds=theta_bounds,
        max_radii=max_radii,
        has_max_threshold=show_max_threshold,
        scale_mode=scale_mode,
    )
    labels = [_impact_label(lcia_method=lcia_method, impact=impact) for impact in impacts]
    frequency_labels: list[str | None] = (
        [_frequency_label(frequencies[impact]) for impact in impacts]
        if frequencies is not None
        else [None for _impact in impacts]
    )
    label_bottom_y = render_impact_labels(
        fig,
        axis,
        theta_bounds=theta_bounds,
        labels=labels,
        frequency_labels=frequency_labels,
        r_max=r_max,
    )
    render_polar_tick_marks(
        axis,
        theta_bounds=theta_bounds,
        r_min=r_min,
        r_max=r_max,
        scale_mode=scale_mode,
    )
    render_polar_title(fig, axis, title=title)
    for index, impact in enumerate(impacts):
        theta0 = float(theta_bounds[index])
        theta1 = float(theta_bounds[index + 1])
        theta_mid = 0.5 * (theta0 + theta1)
        payload = radial_values[impact]
        representative = (
            float(np.nanmin(payload)) if style == "deterministic" else float(np.nanmedian(payload))
        )
        render_risk_background(
            axis,
            theta0=theta0,
            theta1=theta1,
            r_min=r_min,
            r_end=representative,
            max_ratio=ratios[impact],
            scale_mode=scale_mode,
        )
        if style != "deterministic":
            render_uncertainty_glyph(
                axis,
                theta_mid=theta_mid,
                sector_width=float(widths[index]),
                radial_payload=payload,
                summary=_summary_for_impact(summaries=summaries, impact=impact),
                max_ratio=ratios[impact],
                density_scale=density_scale,
                style=str(style),
                scale_mode=scale_mode,
            )
    axis.axis("off")
    contract = build_asr_threshold_contract(
        cc_source=lcia_method,
        has_max_threshold=show_max_threshold,
    )
    render_bottom_legend(
        fig,
        label_bottom_y=label_bottom_y,
        style=str(style),
        min_label=contract.min_line_label,
        max_label=contract.max_line_label,
        lower_zone_label=contract.lower_zone_label,
        middle_zone_label=contract.middle_zone_label,
        upper_zone_label=contract.upper_zone_label,
        fnt_label=contract.fnt_label,
        deterministic_note=deterministic_note,
    )
    del zero_count
    paths = save_figure(fig, output_stem, dpi=dpi, output_format=output_format)
    plt.close(fig)
    return paths


def _ordered_impacts(*, frame: pd.DataFrame, lcia_method: str) -> list[str]:
    observed = {
        str(value).strip()
        for value in frame["impact"].dropna().astype(str).tolist()
        if str(value).strip()
    }
    metadata = load_lcia_metadata(lcia_method)
    ordered = [impact for impact in metadata.impacts if impact in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


def _impact_label(*, lcia_method: str, impact: str) -> str:
    metadata = load_lcia_metadata(lcia_method)
    raw = str(metadata.labels.get(impact, impact)).strip()
    without_code = _label_without_parenthetical_code(raw)
    return _wrap_two_line_label(without_code, width=NON_PB_POLAR_LABEL_WIDTH)


def _label_without_parenthetical_code(label: str) -> str:
    if label.endswith(")") and " (" in label:
        return label.rsplit(" (", maxsplit=1)[0].strip()
    return label


def _wrap_two_line_label(label: str, *, width: int) -> str:
    wrapped = textwrap.wrap(
        label,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
        max_lines=2,
        placeholder="",
    )
    return "\n".join(part.strip() for part in wrapped if part.strip())


def _max_ratios(*, lcia_method: str, impacts: list[str], enabled: bool) -> dict[str, float]:
    if not enabled:
        return {impact: 1.0 for impact in impacts}
    metadata = load_lcia_metadata(lcia_method)
    return {impact: max(float(metadata.ratios.get(impact, 1.0)), 1.0) for impact in impacts}


def _radial_values(
    *,
    values: dict[str, np.ndarray],
    impacts: list[str],
    context: str,
    scale_mode: ASRScaleMode,
) -> tuple[dict[str, np.ndarray], int]:
    radial_values: dict[str, np.ndarray] = {}
    zero_count = 0
    for impact in impacts:
        numeric = np.asarray(values[impact], dtype=np.float64)
        if np.any(numeric < 0.0):
            raise ValueError(
                "ASR polar figures cannot render negative ASR values. "
                f"Impact='{impact}', figure='{context}'."
            )
        if scale_mode == ASR_LOG_SCALE:
            positive, omitted = positive_asr_values(
                numeric,
                context=f"{context} polar ASR values for impact '{impact}'",
            )
            radial_values[impact] = np.log10(positive.astype(np.float64, copy=False))
            zero_count += omitted
        else:
            finite = numeric[np.isfinite(numeric)]
            radial_values[impact] = finite.astype(np.float64, copy=False)
            zero_count += int(np.count_nonzero(finite == 0.0))
    return radial_values, zero_count


def _radial_limits(
    *,
    radial_values: dict[str, np.ndarray],
    ratios: dict[str, float],
    scale_mode: ASRScaleMode,
) -> tuple[float, float]:
    observed = [payload for payload in radial_values.values() if payload.size]
    all_values = np.concatenate(observed)
    max_radii = np.asarray(
        [
            _radius_from_value(max(ratio, 1.0000001), scale_mode=scale_mode)
            for ratio in ratios.values()
        ]
    )
    if scale_mode == ASR_LOG_SCALE:
        lower = float(min(np.nanmin(all_values), -4.0) - 0.2)
        upper = float(max(np.nanmax(all_values), np.nanmax(max_radii), 2.0) + 0.2)
        return lower, upper
    upper = float(max(np.nanmax(all_values), np.nanmax(max_radii), 1.0))
    return 0.0, upper * 1.18 if upper > 0.0 else 1.0


def _density_scale(*, radial_values: dict[str, np.ndarray], style: str) -> float:
    if style != "violin":
        return 1.0
    peaks = [violin_density_peak(payload) for payload in radial_values.values()]
    return max(peaks, default=1.0)


def _summary_for_impact(
    *, summaries: dict[str, dict[str, float]] | None, impact: str
) -> dict[str, float]:
    return {} if summaries is None else summaries.get(impact, {})


def _frequency_label(value: float) -> str:
    return format_fnt_math_label(float(value))


def _radius_from_value(value: float, *, scale_mode: ASRScaleMode) -> float:
    numeric = float(value)
    return float(np.log10(numeric)) if scale_mode == ASR_LOG_SCALE else numeric

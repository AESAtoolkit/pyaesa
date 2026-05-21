"""ASR axis scale and tick policies."""

import math
import warnings
from typing import Literal

import numpy as np
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    LogFormatterMathtext,
    LogLocator,
    NullFormatter,
    PercentFormatter,
)

from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.scientific_ticks import scientific_tick_formatter

ASRScaleMode = Literal["normal", "log"]
ASR_NORMAL_SCALE: ASRScaleMode = "normal"
ASR_LOG_SCALE: ASRScaleMode = "log"
ASR_LOG_SWITCH_THRESHOLD = 10.0
_ASR_THRESHOLD = 1.0
_NORMAL_THRESHOLD_ONLY_MARGIN = 0.06
_ZERO_LOG_SCALE_WARNING = (
    "ASR figure values include zero values and values above 10. "
    "A log ASR axis would make the high ASR range more readable, but log scaling "
    "is not valid for zero values; therefore using a normal ASR axis."
)


def positive_asr_values(values: object, *, context: str) -> tuple[np.ndarray, int]:
    """Return finite positive ASR values and count omitted zeros."""
    del context
    numeric = np.asarray(values, dtype=np.float64)
    zeros = int(np.count_nonzero(numeric == 0.0))
    return numeric[numeric > 0.0], zeros


def resolve_asr_scale_mode(values: object) -> ASRScaleMode:
    """Return the shared ASR figure scale mode for one visible output scope."""
    numeric = np.asarray(values, dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return ASR_NORMAL_SCALE
    if np.any(finite < 0.0):
        return ASR_NORMAL_SCALE
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return ASR_NORMAL_SCALE
    if np.any(finite == 0.0):
        if np.any(positive > ASR_LOG_SWITCH_THRESHOLD):
            warnings.warn(_ZERO_LOG_SCALE_WARNING, UserWarning, stacklevel=2)
        return ASR_NORMAL_SCALE
    if np.any(positive > ASR_LOG_SWITCH_THRESHOLD):
        return ASR_LOG_SCALE
    return ASR_NORMAL_SCALE


def resolve_asr_log_limits(values: object, *, context: str) -> tuple[tuple[float, float], int]:
    """Return shared log scale limits for ASR value axes."""
    positive, zero_count = positive_asr_values(values, context=context)
    data_min = float(np.nanmin(positive))
    data_max = float(np.nanmax(positive))
    lower = min(data_min, 1.0) / 1.35
    upper = max(data_max, 1.0) * 1.35
    return (lower, upper), zero_count


def resolve_asr_normal_limits(values: object) -> tuple[float, float]:
    """Return padded normal scale ASR limits that keep threshold visible."""
    numeric = np.asarray(values, dtype=np.float64)
    valid = numeric[np.isfinite(numeric)]
    lower = min(float(np.min(valid)), _ASR_THRESHOLD)
    upper = max(float(np.max(valid)), _ASR_THRESHOLD)
    if lower == upper:
        return (
            _ASR_THRESHOLD - _NORMAL_THRESHOLD_ONLY_MARGIN,
            _ASR_THRESHOLD + _NORMAL_THRESHOLD_ONLY_MARGIN,
        )
    span = upper - lower
    padded_lower = lower - 0.12 * span
    padded_upper = upper + 0.12 * span
    threshold_span = padded_upper - padded_lower
    threshold_margin = max(0.02, 0.08 * threshold_span)
    if padded_lower > _ASR_THRESHOLD - threshold_margin:
        padded_lower = _ASR_THRESHOLD - threshold_margin
    if np.all(valid >= 0.0):
        padded_lower = max(0.0, padded_lower)
    return (padded_lower, padded_upper)


def resolve_positive_log_limits(values: object, *, context: str) -> tuple[float, float]:
    """Return data driven log scale limits for positive non-ASR value axes."""
    positive, _zero_count = positive_asr_values(values, context=context)
    data_min = float(np.nanmin(positive))
    data_max = float(np.nanmax(positive))
    return data_min / 1.35, data_max * 1.35


def apply_asr_normal_axis(
    axis,
    *,
    values: object,
    limits: tuple[float, float] | None = None,
) -> None:
    """Apply the shared normal scale policy for ASR value axes."""
    lower, upper = limits if limits is not None else resolve_asr_normal_limits(values)
    lower = min(float(lower), _ASR_THRESHOLD)
    upper = max(float(upper), _ASR_THRESHOLD)
    axis.set_ylim(lower, upper)
    axis.yaxis.set_major_locator(FixedLocator(normal_asr_ticks(lower=lower, upper=upper).tolist()))
    axis.yaxis.set_major_formatter(FuncFormatter(scientific_tick_formatter))


def apply_asr_log_axis(
    axis,
    *,
    values: object,
    context: str,
    limits: tuple[float, float] | None = None,
) -> int:
    """Apply the shared log scale policy for ASR value axes."""
    resolved_limits, zero_count = resolve_asr_log_limits(values, context=context)
    axis.set_yscale("log")
    lower, upper = limits if limits is not None else resolved_limits
    axis.set_ylim(lower, upper)
    axis.yaxis.set_major_locator(FixedLocator(_major_log_ticks(lower=lower, upper=upper)))
    axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    axis.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=[float(value) for value in range(2, 10)], numticks=100)
    )
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(axis="y", which="major", length=5.0, width=0.8)
    axis.tick_params(axis="y", which="minor", length=3.0, width=0.6)
    return zero_count


def apply_positive_log_axis(axis, *, limits: tuple[float, float]) -> None:
    """Apply fixed positive log limits without ASR threshold anchoring."""
    lower, upper = limits
    axis.set_yscale("log")
    axis.set_ylim(lower, upper)
    axis.yaxis.set_major_locator(FixedLocator(_major_log_ticks(lower=lower, upper=upper)))
    axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    axis.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=[float(value) for value in range(2, 10)], numticks=100)
    )
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(axis="y", which="major", length=5.0, width=0.8)
    axis.tick_params(axis="y", which="minor", length=3.0, width=0.6)


def apply_frequency_axis(axis) -> None:
    """Apply ASR frequency of no-transgression axis formatting."""
    axis.set_ylim(-5.0, 105.0)
    axis.set_yticks([0.0, 25.0, 50.0, 75.0, 100.0])
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    axis.set_ylabel(format_scientific_figure_text("fNT"))
    axis.set_xlabel("")
    axis.grid(alpha=0.25)


def normal_asr_ticks(*, lower: float, upper: float) -> np.ndarray:
    """Return one regular normal scale tick grid that includes ASR threshold."""
    low, high = sorted((float(lower), float(upper)))
    low = min(low, _ASR_THRESHOLD)
    high = max(high, _ASR_THRESHOLD)
    span = high - low
    if span <= 0.0:
        return np.asarray([_ASR_THRESHOLD], dtype=np.float64)
    step = _normal_asr_tick_step(span)
    first = math.floor(low / step)
    last = math.ceil(high / step)
    ticks = np.arange(first, last + 1, dtype=np.float64) * step
    margin = max(span, 1.0) * 1e-12
    ticks = ticks[(ticks >= low - margin) & (ticks <= high + margin)]
    return np.sort(np.unique(np.round(ticks, 12)))


def _normal_asr_tick_step(span: float) -> float:
    if span <= 0.2:
        return 0.02
    if span <= 0.5:
        return 0.05
    if span <= 1.2:
        return 0.1
    if span <= 2.5:
        return 0.25
    if span <= 5.0:
        return 0.5
    return 1.0


def normal_asr_tick_text(value: float) -> str:
    """Return one compact normal scale ASR tick label."""
    numeric = float(value)
    rounded_int = round(numeric)
    if np.isclose(numeric, rounded_int, rtol=0.0, atol=1e-9):
        return str(int(rounded_int))
    return f"{numeric:.1f}"


def _major_log_ticks(*, lower: float, upper: float) -> list[float]:
    start = int(math.floor(math.log10(float(lower))))
    stop = int(math.ceil(math.log10(float(upper))))
    exponents = list(range(start, stop + 1))
    exponents.append(0)
    return [10.0 ** float(exponent) for exponent in sorted(set(exponents))]

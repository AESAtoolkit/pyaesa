"""Shared y-axis contract helpers for figure families."""

from typing import Literal

import numpy as np

AxisPolicy = Literal["nonnegative", "signed"]


def resolve_axis_ylim(
    *,
    values: np.ndarray,
    context: str,
    policy: AxisPolicy,
) -> tuple[float, float]:
    """Return y-axis limits for one explicit figure-family value contract."""
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return (0.0, 1.0)
    if policy == "nonnegative":
        return require_nonnegative_figure_ylim(values=valid, context=context)
    return signed_figure_ylim(values=valid)


def require_nonnegative_figure_ylim(
    *,
    values: np.ndarray,
    context: str,
) -> tuple[float, float]:
    """Return zero-based limits for a figure family defined only on nonnegative values."""
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return (0.0, 1.0)
    if float(np.min(valid)) < 0.0:
        raise ValueError(
            f"{context} contains negative values, but this figure family is defined only for "
            "nonnegative values."
        )
    upper = float(np.max(valid))
    if upper == 0.0:
        return (0.0, 1.0)
    return (0.0, upper * 1.12)


def apply_zero_floor_if_nonnegative(*, axis, minimum_value: float | None) -> None:
    """Start an existing y axis at zero when all plotted values are nonnegative."""
    if minimum_value is None or float(minimum_value) < 0.0:
        return
    _bottom, top = axis.get_ylim()
    axis.set_ylim(bottom=0.0, top=top)


def signed_figure_ylim(*, values: np.ndarray) -> tuple[float, float]:
    """Return symmetric padded limits for a figure family that allows signed values."""
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return (-1.0, 1.0)
    lower = float(np.min(valid))
    upper = float(np.max(valid))
    if lower == upper:
        padding = 1.0 if lower == 0.0 else abs(lower) * 0.12
        return (lower - padding, upper + padding)
    span = upper - lower
    padding = span * 0.12
    return (lower - padding, upper + padding)

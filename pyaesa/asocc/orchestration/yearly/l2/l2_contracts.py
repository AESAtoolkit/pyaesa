"""Runtime contract guards for L2 orchestration."""

from typing import cast

import pandas as pd

from .l2_types import _L2ComputeInputs


def require_compute_inputs(
    *,
    inputs: _L2ComputeInputs | None,
    where: str,
) -> _L2ComputeInputs:
    """Return typed L2 inputs produced by branch planning."""
    del where
    return cast(_L2ComputeInputs, inputs)


def require_frame(
    *,
    frame: pd.DataFrame | None,
    where: str,
    subject: str,
) -> pd.DataFrame:
    """Return a DataFrame produced by the planned L2 branch."""
    del where, subject
    return cast(pd.DataFrame, frame)


def require_ref_year(
    *,
    ref_year: int | None,
    where: str,
) -> int:
    """Return a required AR reference year."""
    del where
    return int(cast(int, ref_year))


def require_required_indices(
    *,
    required_indices: tuple[str, ...] | None,
    where: str,
) -> tuple[str, ...]:
    """Return required output indices for L2 weighting."""
    del where
    return cast(tuple[str, ...], required_indices)


def require_weight_axis(
    *,
    weight_axis: str | None,
    where: str,
) -> str:
    """Return the resolved L2 weighting axis."""
    del where
    return cast(str, weight_axis)


def require_weights(
    *,
    weights: pd.Series | None,
    where: str,
) -> pd.Series:
    """Return required L1 weights for a two step L2 branch."""
    del where
    return cast(pd.Series, weights)

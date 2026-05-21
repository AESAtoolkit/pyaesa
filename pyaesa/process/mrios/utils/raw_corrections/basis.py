"""Value added helpers for raw MRIO corrections."""

from pathlib import Path
from typing import cast

import pandas as pd


def positive_factor_inputs_basis_from_frame(f_frame: pd.DataFrame) -> pd.Series:
    """Return the predictor basis from raw ``factor_inputs.F``.

    Args:
        f_frame: Raw ``factor_inputs.F`` table with ``(region, sector)``
            columns.

    Returns:
        Series indexed by ``(r_p, s_p)`` containing the sum of positive factor
        input categories for each product.

    """
    numeric = cast(pd.DataFrame, f_frame.apply(pd.to_numeric, errors="coerce")).fillna(0.0)
    positive = cast(pd.DataFrame, numeric.clip(lower=0.0))
    basis = cast(pd.Series, positive.sum(axis=0, min_count=1))
    basis.index = pd.MultiIndex.from_arrays(
        [
            basis.index.get_level_values(0).map(str),
            basis.index.get_level_values(1).map(str),
        ],
        names=["r_p", "s_p"],
    )
    return cast(pd.Series, basis.astype(float))


def corrected_values_dir() -> Path:
    """Return the raw corrected values directory."""
    return Path(__file__).resolve().parent / "corrected_values"

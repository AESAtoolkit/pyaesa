"""Shared contracts for population and GDP raw downloads."""

from typing import Sequence

import pandas as pd

from pyaesa.shared.tabular.wide_tables import detect_year_columns

POP_SSP_INDICATOR = "Population"
GDP_SSP_INDICATOR = "GDP|PPP"
DEFAULT_SSP_INDICATORS: Sequence[str] = (
    POP_SSP_INDICATOR,
    GDP_SSP_INDICATOR,
)
FUTURE_YEARS = list(range(2025, 2101))

POP_WB_INDICATOR = "Population"
GDP_WB_INDICATOR = "GDP|PPP"
POP_WB_UNIT = "Persons"
GDP_WB_UNIT = "USD_2021/yr"
PAST_YEAR_MIN = 1995


def resolve_historical_years_from_frame(
    frame: pd.DataFrame,
    *,
    minimum_year: int = PAST_YEAR_MIN,
) -> list[int]:
    """Return sorted historical year columns available in ``frame``.

    Args:
        frame (pandas.DataFrame): Wide population or GDP table containing
            year like columns.
        minimum_year (int): Inclusive lower bound for retained historical
            years.

    Returns:
        list[int]: Sorted available historical years at or above
        ``minimum_year``.

    Raises:
        RuntimeError: If ``frame`` does not expose any historical year
            columns within the requested floor.
    """
    years = sorted(
        year
        for year in (int(column) for column in detect_year_columns(frame))
        if year >= int(minimum_year)
    )
    if not years:
        raise RuntimeError(
            "Population or GDP table does not expose any historical year columns "
            f"at or above {int(minimum_year)}."
        )
    return years

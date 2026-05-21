"""Shared aSoCC route transition policy for downstream figures."""

import pandas as pd

from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_PROSPECTIVE_TIME_ROUTE_VALUES,
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.tabular.scalars import is_display_missing

ASOCC_TRANSITION_SERIES_EXCLUDED_COLUMNS = frozenset(
    {
        ASOCC_SSP_SCENARIO_COLUMN,
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
        "reference_year",
        "l2_reuse_year",
        "asocc_ssp_start_year",
    }
)


def asocc_transition_year(group: pd.DataFrame) -> int | None:
    """Return the first prospective year when retrospective rows are also visible."""
    mask = pd.Series(False, index=group.index)
    if ASOCC_SSP_SCENARIO_COLUMN in group.columns:
        mask = mask | ~group[ASOCC_SSP_SCENARIO_COLUMN].map(is_display_missing)
    route = pd.Series(group.loc[:, ASOCC_TIME_ROUTE_PUBLIC_COLUMN], copy=False)
    mask = mask | route.astype("string").isin(ASOCC_PROSPECTIVE_TIME_ROUTE_VALUES)
    years = pd.Series(pd.to_numeric(group.loc[:, "year"], errors="raise"), copy=False).astype(int)
    prospective_by_year = mask.groupby(years, sort=True).any()
    if not bool(prospective_by_year.any()) or bool(prospective_by_year.all()):
        return None
    return int(prospective_by_year.loc[prospective_by_year].index.min())

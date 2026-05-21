"""Canonical aSoCC time route metadata."""

from collections.abc import Iterable

import pandas as pd

from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_PROSPECTIVE_TIME_ROUTE_VALUES,
    ASOCC_TIME_ROUTE_HISTORICAL,
    ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
    ASOCC_TIME_ROUTE_REGRESSION,
)
from pyaesa.shared.tabular.scalars import is_display_missing


def asocc_time_route_from_projection_subfolder(projection_subfolder: str | None) -> str:
    """Return the canonical aSoCC time route for one deterministic output route."""
    if projection_subfolder is None:
        return ASOCC_TIME_ROUTE_HISTORICAL
    text = str(projection_subfolder).strip()
    return {
        ASOCC_TIME_ROUTE_HISTORICAL_REUSE: ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
        ASOCC_TIME_ROUTE_REGRESSION: ASOCC_TIME_ROUTE_REGRESSION,
    }[text]


def collapse_asocc_time_route(values: Iterable[object]) -> object:
    """Return the visible aSoCC route for one collapsed public figure row."""
    visible = [
        str(value).strip()
        for value in values
        if not is_display_missing(value) and str(value).strip()
    ]
    prospective = [value for value in visible if value in ASOCC_PROSPECTIVE_TIME_ROUTE_VALUES]
    return next(iter(prospective), next(iter(visible), pd.NA))

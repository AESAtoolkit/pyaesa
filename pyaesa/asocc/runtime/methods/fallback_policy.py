"""Family-local historical fallback policies for deterministic aSoCC."""

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HistoricalYearFallback:
    """Resolved historical year for one intentional scientific fallback."""

    requested_year: int
    resolved_year: int

    @property
    def used_fallback(self) -> bool:
        """Return whether the resolved year differs from the requested year."""
        return int(self.resolved_year) != int(self.requested_year)


def _sorted_unique_years(*, years: list[int]) -> list[int]:
    """Return sorted unique integer years."""
    return sorted({int(year) for year in years})


def resolve_latest_available_historical_year(
    *,
    requested_year: int,
    available_years: list[int],
) -> HistoricalYearFallback | None:
    """Resolve the latest available historical year up to ``requested_year``."""
    normalized_years = _sorted_unique_years(years=available_years)
    pos = bisect_right(normalized_years, int(requested_year)) - 1
    if pos < 0:
        return None
    return HistoricalYearFallback(
        requested_year=int(requested_year),
        resolved_year=int(normalized_years[pos]),
    )


def resolve_latest_previous_nonzero_series(
    *,
    requested_year: int,
    available_years: list[int],
    load_series: Callable[[int], pd.Series | None],
    is_zero_placeholder: Callable[[pd.Series], bool],
) -> tuple[pd.Series | None, HistoricalYearFallback | None]:
    """Resolve one RP=1 style non-zero historical series fallback.

    The requested year payload is returned unchanged when it is missing or when it
    does not match the zero-placeholder condition. If the requested payload is an
    all-zero placeholder, the latest previous non-zero year is reused.
    """
    current_value = load_series(int(requested_year))
    if current_value is None or not is_zero_placeholder(current_value):
        return current_value, None

    for candidate_year in reversed(_sorted_unique_years(years=available_years)):
        if int(candidate_year) >= int(requested_year):
            continue
        previous_value = load_series(int(candidate_year))
        if previous_value is None or is_zero_placeholder(previous_value):
            continue
        return (
            previous_value.copy(),
            HistoricalYearFallback(
                requested_year=int(requested_year),
                resolved_year=int(candidate_year),
            ),
        )
    return current_value, None

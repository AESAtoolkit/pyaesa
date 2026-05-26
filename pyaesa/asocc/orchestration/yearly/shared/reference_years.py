"""Shared AR reference year candidate policy for yearly orchestration."""

from ....data.source_schema import default_historical_cutoff_for_source


def ar_reference_year_candidates(
    *,
    source: str,
    historical_years: tuple[int, ...] | list[int],
    reference_years: tuple[int, ...] | list[int] | None,
    year: int,
) -> list[int]:
    """Return AR reference candidates available at the computed year."""
    year_int = int(year)
    if reference_years:
        raw = [int(value) for value in reference_years]
    else:
        default_cutoff = default_historical_cutoff_for_source(source)
        raw = [
            int(candidate)
            for candidate in historical_years
            if default_cutoff is None or int(candidate) <= default_cutoff
        ]
    return [int(candidate) for candidate in raw if int(candidate) <= year_int]

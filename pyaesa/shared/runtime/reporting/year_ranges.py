"""Shared compact year range formatting for public runtime reports."""

from collections.abc import Sequence


def format_year_ranges(years: Sequence[int]) -> str:
    """Return compact year ranges for public runtime summaries."""
    if not years:
        return "[]"
    ordered = sorted(set(int(year) for year in years))
    ranges: list[str] = []
    start = ordered[0]
    previous = ordered[0]
    for year in ordered[1:]:
        if year == previous + 1:
            previous = year
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = year
        previous = year
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ", ".join(ranges)

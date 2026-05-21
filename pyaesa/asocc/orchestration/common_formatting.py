"""Shared formatting used across orchestration stages."""


def format_year_ranges(years: list[int]) -> str:
    """Format sorted years into compact ranges (e.g. 1995-2004, 2006).

    This is used by both setup time validation messages and run time summaries
    so users see one consistent year range format across commands.
    """
    if not years:
        return "[]"
    ordered = sorted(set(int(y) for y in years))
    ranges: list[str] = []
    start = ordered[0]
    prev = ordered[0]
    for year in ordered[1:]:
        if year == prev + 1:
            prev = year
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = year
        prev = year
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def format_year_scope(years: list[int]) -> str:
    """Return a human readable year scope clause.

    Examples:
        - ``for year 2022``
        - ``for years 2022-2024, 2026``
    """
    ordered = sorted(set(int(y) for y in years))
    if not ordered:
        return ""
    label = "year" if len(ordered) == 1 else "years"
    return f"for {label} {format_year_ranges(ordered)}"

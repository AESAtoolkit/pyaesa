"""Shared checkpoint year helpers for deterministic and uncertainty figures."""


def unique_figure_years(years: list[int] | None) -> list[int]:
    """Return sorted distinct figure years."""
    if years is None:
        return []
    return sorted({int(year) for year in years})


def has_exact_single_year_scope(years: list[int] | None) -> bool:
    """Return whether one figure request targets exactly one explicit year."""
    return len(unique_figure_years(years)) == 1


def default_checkpoint_years(years: list[int]) -> list[int]:
    """Return first year, last year, and every fifth year from the first."""
    unique_years = unique_figure_years(years)
    if not unique_years:
        return []
    first_year = unique_years[0]
    last_year = unique_years[-1]
    checkpoints = {
        int(year)
        for year in unique_years
        if int(year) in {first_year, last_year} or (int(year) - first_year) % 5 == 0
    }
    return sorted(checkpoints)

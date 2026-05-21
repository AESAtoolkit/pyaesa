"""Study period normalization and validation ownership for AR6 processing."""

from pyaesa.shared.selectors.time_selectors import normalize_optional_year_selector


def resolve_study_period(
    years: int | list[int] | range,
) -> list[int]:
    """Normalize API ``years`` input into ``[start_year, end_year]``."""
    resolved_years = normalize_optional_year_selector(years, name="years")
    if not resolved_years:
        raise ValueError("years must resolve to at least two consecutive years.")
    if len(resolved_years) < 2:
        raise ValueError(
            f"years must resolve to at least two consecutive years. Got {resolved_years}."
        )
    expected = list(range(int(resolved_years[0]), int(resolved_years[-1]) + 1))
    if resolved_years != expected:
        raise ValueError(
            f"years must represent consecutive years with no gaps. Got {resolved_years}."
        )
    return [int(resolved_years[0]), int(resolved_years[-1])]


def validate_study_period_in_ar6(study_period: list[int], ar6_years: list[int]) -> None:
    """Fail when the study period lies outside downloaded AR6 year coverage."""
    if not ar6_years:
        raise ValueError("No AR6 years found in the downloaded explorer CSV.")
    min_year = int(min(ar6_years))
    max_year = int(max(ar6_years))
    if int(study_period[0]) < min_year or int(study_period[1]) > max_year:
        raise ValueError(
            "years must be within AR6 available years "
            f"[{min_year}, {max_year}]. Got {study_period}."
        )

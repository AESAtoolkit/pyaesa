"""Canonicalize public year style selectors used across entrypoint contracts.

This module owns shared normalization rules for optional year selectors and
regression windows so deterministic and uncertainty entrypoints persist
comparable time selector identities.
"""

from typing import Any

YearSelector = int | list[int] | range | None


def normalize_requested_years(years: int | list[int] | range) -> list[int]:
    """Return one concrete ordered list of integer years from a public selector."""
    if isinstance(years, range):
        return [int(year) for year in years]
    if isinstance(years, list):
        return [int(year) for year in years]
    return [int(years)]


def normalize_optional_year_selector(
    value: YearSelector,
    *,
    name: str,
) -> list[int] | None:
    """Normalize one optional year selector to a sorted unique list.

    Args:
        value: Public year selector value.
        name: Argument label used in validation messages.

    Returns:
        ``None`` when the selector is omitted, otherwise a sorted unique
        integer list.

    Raises:
        ValueError: If ``value`` does not use one supported selector type.
    """
    if value is None:
        return None
    if isinstance(value, int):
        years = [int(value)]
    elif isinstance(value, range):
        years = [int(year) for year in value]
    elif isinstance(value, list):
        years = [int(year) for year in value]
    else:
        raise ValueError(f"Unsupported {name} selector type '{type(value).__name__}'.")
    return sorted(set(years))


def normalize_optional_reg_window_selector(
    value: list[int] | range | None,
    *,
    name: str = "reg_window",
) -> list[int] | None:
    """Normalize one optional regression window selector to an integer list.

    Args:
        value: Regression window selector.
        name: Argument label used in validation messages.

    Returns:
        ``None`` when omitted, otherwise the integer years in the provided
        selector order.

    Raises:
        ValueError: If ``value`` does not use one supported selector type.
    """
    if value is None:
        return None
    if isinstance(value, range):
        return [int(year) for year in value]
    if isinstance(value, list):
        return [int(year) for year in value]
    raise ValueError(f"Unsupported {name} selector type '{type(value).__name__}'.")


def normalize_reg_window_for_storage(
    value: list[int] | range | tuple[int, int] | None,
    *,
    name: str = "reg_window",
) -> list[int] | None:
    """Normalize one regression window selector to a persisted year list.

    Args:
        value: Regression window selector accepted by public/runtime layers.
        name: Argument label used in validation messages.

    Returns:
        ``None`` when omitted, otherwise the full consecutive integer year
        list represented by the selector.

    Raises:
        ValueError: If ``value`` is empty, non consecutive, or uses an
            unsupported selector type.
    """
    if value is None:
        return None
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} bounds must contain exactly two years.")
        start_year = int(value[0])
        end_year = int(value[1])
        if start_year > end_year:
            raise ValueError(f"{name} must satisfy start_year <= end_year.")
        return list(range(start_year, end_year + 1))
    years = normalize_optional_reg_window_selector(value, name=name)
    if years is None or not years:
        raise ValueError(f"{name} must contain at least one year when provided.")
    expected = list(range(int(years[0]), int(years[-1]) + 1))
    if years != expected:
        raise ValueError(f"{name} must define consecutive years with step 1.")
    return years


def normalize_time_selector_mapping(
    value: dict[str, Any] | None,
    *,
    year_keys: tuple[str, ...] = ("years", "reference_years", "l2_reuse_years"),
    reg_window_keys: tuple[str, ...] = ("reg_window",),
) -> dict[str, Any] | None:
    """Return a shallow mapping copy with canonicalized time selector fields."""
    if value is None:
        return None
    out = dict(value)
    for key in year_keys:
        if key in out:
            out[key] = normalize_optional_year_selector(out[key], name=key)
    for key in reg_window_keys:
        if key in out:
            out[key] = normalize_optional_reg_window_selector(out[key], name=key)
    return out

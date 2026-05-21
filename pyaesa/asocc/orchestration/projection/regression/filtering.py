"""Filter operations shared by projection payload builders."""

from collections.abc import Mapping, Sequence


def selected_values_for_level(
    *,
    filters: Mapping[str, list[str] | None],
    level: str,
) -> list[str] | None:
    """Return normalized selected values for one index level."""
    values = filters.get(level)
    if not values:
        return None
    return [str(value) for value in values]


def selected_values_for_levels(
    *,
    filters: Mapping[str, list[str] | None],
    levels: Sequence[str],
) -> dict[str, list[str] | None]:
    """Return normalized selected values per requested container level."""
    return {
        str(level): selected_values_for_level(filters=filters, level=str(level)) for level in levels
    }

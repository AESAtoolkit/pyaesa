"""Shared orchestration ownership for selected L2 method scope."""

from collections.abc import Sequence


def _unique_l2_methods_in_scope(
    *,
    selected_l2_one_step: Sequence[str],
    combined: Sequence[tuple[str, str]],
) -> list[str]:
    """Return deterministic unique L2 method names preserving first occurrence order."""
    ordered = [*selected_l2_one_step, *(name for name, _ in combined)]
    unique: list[str] = []
    seen: set[str] = set()
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def _max_historical_mrio_year(*, historical_years: Sequence[int]) -> int | None:
    """Return the latest historical MRIO year when one exists."""
    return max((int(year) for year in historical_years), default=None)

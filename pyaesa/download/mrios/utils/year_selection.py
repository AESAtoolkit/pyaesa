"""Shared MRIO year selection contracts."""

from collections.abc import Sequence

from .source_registry import default_years_for_source

YearSelection = int | range | Sequence[int] | None


def normalize_mrio_years(selection: YearSelection, *, source_key: str) -> list[int]:
    """Return a sorted list of years from one user year selection."""
    if selection is None:
        return default_years_for_source(source_key)
    if isinstance(selection, int):
        return [selection]
    if isinstance(selection, range):
        return list(selection)
    if isinstance(selection, Sequence) and not isinstance(selection, (str, bytes, bytearray)):
        years = sorted({int(item) for item in selection})
        return years
    raise ValueError("years must be None, an int, a sequence of years, or a range.")

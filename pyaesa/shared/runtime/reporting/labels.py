"""Shared labels for runtime summaries."""

from collections.abc import Collection


def plural_label(count: int, singular: str, plural: str | None = None) -> str:
    """Return the singular or plural label for ``count``.

    Args:
        count: Number that controls the label form.
        singular: Label used when ``count`` is one.
        plural: Label used otherwise. When omitted, ``s`` is appended to
            ``singular``.

    Returns:
        The label matching ``count``.
    """
    if count == 1:
        return singular
    if plural is not None:
        return plural
    return f"{singular}s"


def count_line(count: int, singular: str, plural: str | None = None) -> str:
    """Return a labelled count summary line."""
    return f"{plural_label(count, singular, plural)}: {count}"


def figures_available_line(count: int) -> str:
    """Return a figure count summary line."""
    return count_line(count, "Figure available", "Figures available")


def output_files_available_line(count: int) -> str:
    """Return an output file count summary line."""
    return count_line(count, "Output file available", "Output files available")


def value_count(values: Collection[object]) -> int:
    """Return the number of selected values for label selection."""
    return len(values)


def labelled_values_line(
    singular: str,
    plural: str,
    values: Collection[object],
    formatted_values: str,
) -> str:
    """Return a singular or plural label followed by formatted values."""
    return f"{plural_label(value_count(values), singular, plural)}: {formatted_values}"

"""Shared selector labels for aggregated public identities."""

from collections.abc import Iterable


def aggregate_selector_label(values: Iterable[object]) -> str:
    """Return one deterministic full label for aggregated selector values."""
    return ", ".join(sorted({str(value) for value in values}))


def aggregate_selector_label_or_none(values: object) -> str | None:
    """Return the aggregate selector label when more than one value is selected."""
    if values is None:
        return None
    if isinstance(values, str):
        labels = [values]
    elif isinstance(values, Iterable):
        labels = list(values)
    else:
        labels = [values]
    if len(labels) <= 1:
        return None
    return aggregate_selector_label(labels)

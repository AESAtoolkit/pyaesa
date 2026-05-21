"""Shared folder inventory lines for public runtime summaries."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputInventoryItem:
    """One public output folder or interpretation artifact listed in a summary."""

    folder: str
    content: str


def inventory_item(*, folder: str, content: str) -> OutputInventoryItem:
    """Return one normalized public inventory item."""
    clean_folder = str(folder).strip()
    clean_content = str(content).strip().rstrip(".")
    return OutputInventoryItem(folder=clean_folder, content=clean_content)


def inventory_lines(items: Iterable[OutputInventoryItem]) -> tuple[str, ...]:
    """Return grouped inventory lines for public summary sections."""
    grouped: dict[str, list[str]] = {}
    for item in items:
        if not item.folder or not item.content:
            continue
        entries = grouped.setdefault(item.folder, [])
        if item.content not in entries:
            entries.append(item.content)
    if not grouped:
        return ()

    lines = ["Output folders:"]
    lines.extend(f"{folder}: {'; '.join(contents)}." for folder, contents in grouped.items())
    return tuple(lines)

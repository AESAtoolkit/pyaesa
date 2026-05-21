"""Shared runtime figure-progress helpers."""

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pyaesa.shared.runtime.text import compact_user_text
from pyaesa.shared.runtime.reporting.status import StatusSink, TransientStatusPrinter

ItemT = TypeVar("ItemT")


def render_with_progress(
    *,
    source: str,
    items: list[ItemT],
    describe: Callable[[ItemT], str],
    render: Callable[[ItemT], list[Path]],
    status: StatusSink | None = None,
) -> list[Path]:
    """Render a sequence with transient public figure generation status lines."""
    if not items:
        return []
    own_status = status is None
    if own_status:
        status = TransientStatusPrinter(source)
    paths: list[Path] = []
    try:
        total = len(items)
        for index, item in enumerate(items, start=1):
            description = compact_user_text(describe(item), max_chars=70)
            status.show(f"[{source}] Generating figures {index}/{total}: {description}")
            paths.extend(render(item))
    finally:
        status.clear_transient()
        if own_status:
            status.finish()
    return paths

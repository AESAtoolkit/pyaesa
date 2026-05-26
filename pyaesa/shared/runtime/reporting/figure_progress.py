"""Shared runtime figure-progress helpers."""

import gc
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

from pyaesa.shared.runtime.text import compact_user_text
from pyaesa.shared.runtime.reporting.status import StatusSink, TransientStatusPrinter
from pyaesa.shared.runtime.reporting.labels import plural_label

ItemT = TypeVar("ItemT")


def render_with_progress(
    *,
    source: str,
    items: Iterable[ItemT],
    describe: Callable[[ItemT], str],
    render: Callable[[ItemT], list[Path]],
    total: int,
    item_count: Callable[[ItemT], int] | None = None,
    status: StatusSink | None = None,
) -> list[Path]:
    """Render a sequence with transient public figure generation status lines."""
    planned_total = int(total)
    if planned_total == 0:
        return []
    own_status = status is None
    if own_status:
        status = TransientStatusPrinter(source)
    paths: list[Path] = []
    try:
        rendered_count = 0
        for item in items:
            description = compact_user_text(describe(item), max_chars=70)
            planned_item_count = _item_count(item, item_count=item_count)
            start = rendered_count + 1
            end = rendered_count + planned_item_count
            count = f"{start}/{planned_total}" if start == end else f"{start}-{end}/{planned_total}"
            noun = plural_label(planned_item_count, "figure")
            status.show(f"[{source}] Generating {noun} {count}: {description}")
            rendered = render(item)
            rendered_count += len(rendered)
            paths.extend(rendered)
            del rendered
            gc.collect()
        status.log_message(_generated_message(source, len(paths)))
    finally:
        status.clear_transient()
        if own_status:
            status.finish()
    return paths


def _item_count(item: ItemT, *, item_count: Callable[[ItemT], int] | None) -> int:
    if item_count is None:
        return 1
    return max(1, int(item_count(item)))


def _generated_message(source: str, file_count: int) -> str:
    noun = plural_label(file_count, "figure")
    return f"[{source}] Generated {noun} {file_count}/{file_count}."

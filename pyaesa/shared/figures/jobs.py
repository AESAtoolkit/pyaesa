"""Shared logical job planning helpers for figure rendering."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from pyaesa.shared.runtime.reporting.figure_progress import render_with_progress
from pyaesa.shared.runtime.reporting.status import StatusSink


@dataclass(frozen=True)
class PlannedFigureJob:
    """One planned figure render step and its expected output file count."""

    kind: str
    label: str
    render: Callable[[], list[Path]]
    planned_outputs: int = 1
    warning_contexts: tuple[str, ...] = ()


def render_figure_jobs(
    *,
    source: str,
    jobs: Callable[[], Iterable[PlannedFigureJob]],
    status: StatusSink | None = None,
) -> list[Path]:
    """Render planned figure jobs through one shared counted progress loop."""
    total = sum(max(1, int(job.planned_outputs)) for job in jobs())
    return render_with_progress(
        source=source,
        items=jobs(),
        describe=lambda job: job.label,
        render=lambda job: job.render(),
        total=total,
        item_count=lambda job: job.planned_outputs,
        status=status,
    )

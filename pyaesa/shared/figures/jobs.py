"""Shared logical job planning helpers for figure rendering."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pyaesa.shared.runtime.reporting.figure_progress import render_with_progress
from pyaesa.shared.runtime.reporting.status import StatusSink


@dataclass(frozen=True)
class PlannedFigureJob:
    """One logical figure rendering job."""

    kind: str
    label: str
    render: Callable[[], list[Path]]


def render_figure_jobs(
    *,
    source: str,
    jobs: list[PlannedFigureJob],
    status: StatusSink | None = None,
) -> list[Path]:
    """Render planned figure jobs through one shared progress loop."""
    return render_with_progress(
        source=source,
        items=jobs,
        describe=lambda job: job.label,
        render=lambda job: job.render(),
        status=status,
    )

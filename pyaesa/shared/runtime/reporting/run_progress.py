"""Live progress for uncertainty run and Sobol evaluation loops."""

from pyaesa.shared.runtime.reporting.display import short_source
from pyaesa.shared.runtime.reporting.progress import YearProgressPrinter
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.runtime.text import compact_user_text


class RunProgressPrinter:
    """Render one replaceable progress line for run scale uncertainty work."""

    def __init__(
        self,
        *,
        source: str,
        action: str,
        enabled: bool = True,
        status: StatusSink | None = None,
    ) -> None:
        self._source = str(source)
        self._action = str(action)
        self._status = status if enabled else None
        self._printer = (
            YearProgressPrinter(
                source=source,
                action="status",
                total=0,
                show_timing=False,
            )
            if enabled and status is None
            else None
        )

    def begin(self, *, label: str) -> None:
        """Show the currently running run batch or Sobol checkpoint."""
        if self._status is not None:
            self._status.show(self._line(label=label))
            return
        if self._printer is None:
            return
        self._printer.log_message(self._line(label=label), persistent=False)

    def complete(self, *, label: str, persistent: bool = True) -> None:
        """Mark the current run batch or Sobol checkpoint complete."""
        if self._status is not None:
            self._status.log_message(self._line(label=label), persistent=persistent)
            return
        if self._printer is None:
            return
        self._printer.log_message(self._line(label=label), persistent=persistent)

    def show(self, message: str) -> None:
        """Render a transient message on the same run progress line."""
        self.log_message(message, persistent=False)

    def log_message(self, message: str, *, persistent: bool = True) -> None:
        """Render one run progress message with explicit persistence."""
        if self._status is not None:
            self._status.log_message(str(message), persistent=persistent)
            return
        if self._printer is None:
            return
        self._printer.log_message(str(message), persistent=persistent)

    def skip(self) -> None:
        """Leave the current transient line unchanged for skipped work."""

    def clear_transient(self) -> None:
        """Clear the current transient progress line."""
        if self._status is not None:
            self._status.clear_transient()
            return
        if self._printer is None:
            return
        self._printer.clear_transient()

    def finish(self) -> None:
        """Finalize the live progress line."""
        if self._status is not None:
            return
        if self._printer is None:
            return
        self._printer.finish()

    def _line(self, *, label: str) -> str:
        """Return the complete transient progress line."""
        return compact_user_text(f"[{short_source(self._source)}] {self._action} {str(label)}")


def monte_carlo_run_progress(
    *,
    source: str,
    enabled: bool = True,
    status: StatusSink | None = None,
) -> RunProgressPrinter:
    """Return live progress for Monte Carlo run batches."""
    return RunProgressPrinter(
        source=source,
        action="Monte Carlo",
        enabled=enabled,
        status=status,
    )


def monte_carlo_run_progress_label(
    *,
    completed: int,
    max_runs: int,
    mode: str = "convergence",
    component: bool = False,
) -> str:
    """Return a convergence safe Monte Carlo run progress label."""
    if str(mode) == "fixed":
        return f"completed fixed runs {int(completed)} of {int(max_runs)}"
    checkpoint = "component checkpoint checked" if component else "checkpoint checked"
    return f"completed runs {int(completed)}; max {int(max_runs)}; {checkpoint}"


def monte_carlo_run_drawing_label(
    *,
    start: int,
    stop: int,
    max_runs: int,
    mode: str = "convergence",
    component: bool = False,
) -> str:
    """Return a Monte Carlo run label for the active draw interval."""
    first = int(start) + 1
    last = int(stop)
    if str(mode) == "fixed":
        return f"drawing fixed runs {first} to {last} of {int(max_runs)}"
    checkpoint = "component checkpoint" if component else "checkpoint"
    return f"drawing runs {first} to {last}; max {int(max_runs)}; {checkpoint}"


def sobol_progress(
    *,
    source: str,
    enabled: bool = True,
    status: StatusSink | None = None,
) -> RunProgressPrinter:
    """Return live progress for Sobol base sample checkpoints."""
    return RunProgressPrinter(
        source=source,
        action="Sobol",
        enabled=enabled,
        status=status,
    )


def visible_status_for_run_work(
    *,
    progress: RunProgressPrinter,
    fallback: StatusSink | None,
    progress_enabled: bool,
) -> StatusSink | None:
    """Return the visible status sink for non run work owned by a run."""
    if progress_enabled:
        return progress
    return fallback

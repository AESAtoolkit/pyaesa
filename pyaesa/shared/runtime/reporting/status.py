"""Emit short-lived status lines for staged deterministic public functions.

This helper module supports scope limited, user facing runtime messages for
multi stage public workflows where progress should be visible during execution
without persisting a full progress log or polluting returned reports.
"""

from typing import Protocol

from pyaesa.shared.runtime.reporting.progress import YearProgressPrinter
from pyaesa.shared.runtime.text import compact_user_text


class StatusSink(Protocol):
    """Minimal live status interface used by workflow and figure progress."""

    def show(self, message: str) -> None:
        """Render one transient status message."""

    def log_message(self, message: str, *, persistent: bool = True) -> None:
        """Render one status message with explicit persistence."""

    def clear_transient(self) -> None:
        """Clear the current transient status message."""

    def finish(self) -> None:
        """Finalize the live status line."""


class TransientStatusPrinter:
    """Render one replaceable status line for short workflow stages.

    The printer uses the same terminal and notebook live update machinery as
    :class:`pyaesa.shared.runtime.reporting.progress.YearProgressPrinter`, but
    without any year or item counter. Repeated calls replace the current line
    instead of appending new ones.
    """

    def __init__(self, label: str) -> None:
        self._printer = YearProgressPrinter(
            source=label,
            action="status",
            total=0,
            show_timing=False,
        )

    def show(self, message: str) -> None:
        """Render one transient status line."""
        self.log_message(message, persistent=False)

    def log_message(self, message: str, *, persistent: bool = True) -> None:
        """Render one status line with explicit persistence."""
        self._printer.log_message(compact_user_text(message), persistent=persistent)

    def clear_transient(self) -> None:
        """Clear the current transient status line."""
        self._printer.clear_transient()

    def finish(self) -> None:
        """Finalize the live status line."""
        self._printer.finish()

"""Render phase level status lines for chained public entrypoints.

This helper module provides persistent orchestration messages for public
workflows that auto run upstream stages and need a compact, user-facing phase
trace.
"""

import re
from typing import cast

from pyaesa.shared.runtime.text import compact_user_text
from pyaesa.shared.runtime.reporting.composite_phase_index import phase_label_for_owner
from pyaesa.shared.runtime.reporting.progress import YearProgressPrinter


_OWNER_PREFIX_RE = re.compile(r"^\[(?P<owner>[^\]]+)\]\s*(?P<message>.*)$")


class PhasePrinter:
    """Emit phase scoped runtime lines for orchestration style public workflows."""

    def __init__(self, source: str) -> None:
        self._source = str(source)
        self._printers: dict[str, YearProgressPrinter] = {}
        self._current_label: str | None = None
        self._current_owner: str | None = None
        self._announced: set[str] = set()
        self._active_work: set[str] = set()
        self._visible_expected: set[str] = set()

    def _printer_for(self, label: str) -> YearProgressPrinter:
        """Return the display owner for one public phase label."""
        clean_label = str(label).strip()
        printer = self._printers.get(clean_label)
        if printer is None:
            printer = YearProgressPrinter(
                source=self._source,
                action="phase",
                total=0,
                show_timing=False,
            )
            self._printers[clean_label] = printer
        self._current_label = clean_label
        return printer

    def _ensure_announced(self, label: str) -> YearProgressPrinter:
        """Return the phase printer after rendering its title if needed."""
        printer = self._printer_for(label)
        if label not in self._announced:
            printer.log_message(label, persistent=True)
            self._announced.add(label)
        return printer

    def _current_printer(self) -> YearProgressPrinter:
        """Return the current phase display owner."""
        return self._ensure_announced(cast(str, self._current_label))

    @staticmethod
    def _split_owner(message: str, owner: str | None) -> tuple[str | None, str]:
        """Return explicit owner and message text from a status line."""
        text = str(message).strip()
        match = _OWNER_PREFIX_RE.match(text)
        if match is not None:
            parsed_owner = match.group("owner").strip()
            parsed_message = match.group("message").strip()
            return parsed_owner or owner, parsed_message
        return owner, text

    @staticmethod
    def _is_exact_reuse_detail(detail: str) -> bool:
        """Return whether a phase completion detail describes exact reuse."""
        return "reused exactly" in str(detail).lower()

    def _select_owner_phase(self, owner: str | None) -> None:
        """Select the canonical phase for a nested owner when needed."""
        phase_label = phase_label_for_owner(owner)
        if phase_label is not None:
            self._printer_for(phase_label)

    def announce(self, label: str, detail: str | None = None) -> None:
        """Select the active phase and subphase owner without printing."""
        self._printer_for(label)
        self._current_owner = None if detail is None else str(detail).strip() or None

    def expect_visible(self, label: str) -> None:
        """Mark a phase as visibly owned by a composite cascade."""
        clean_label = str(label).strip()
        self._visible_expected.add(clean_label)

    def status(self, message: str, *, owner: str | None = None) -> None:
        """Render one transient status line in the current phase section."""
        line_owner, text = self._split_owner(message, owner or self._current_owner)
        if not text:
            return
        self._select_owner_phase(line_owner)
        prefix = "" if line_owner is None else f"[{line_owner}] "
        self._current_printer().log_message(
            compact_user_text(f"{prefix}{text}"),
            persistent=False,
        )
        self._active_work.add(cast(str, self._current_label))

    def show(self, message: str) -> None:
        """Render one transient status message for shared status sinks."""
        self.status(message)

    def log_message(self, message: str, *, persistent: bool = True) -> None:
        """Render a progress compatible line in the current phase section."""
        line_owner, text = self._split_owner(message, self._current_owner)
        if not text:
            return
        self._select_owner_phase(line_owner)
        prefix = "" if line_owner is None else f"[{line_owner}] "
        self._current_printer().log_message(
            compact_user_text(f"{prefix}{text}"),
            persistent=persistent,
        )
        if not persistent and self._current_label is not None:
            self._active_work.add(self._current_label)

    def clear_transient(self) -> None:
        """Clear the current phase transient line."""
        if self._current_label is None:
            return
        self._printers[self._current_label].clear_transient()

    def complete(self, detail: str, *, owner: str | None = None) -> None:
        """Print one completion line in the current phase display section."""
        text = str(detail).strip()
        if not text:
            return
        line_owner, message = self._split_owner(text, owner or self._current_owner)
        self._select_owner_phase(line_owner)
        exact_reuse = self._is_exact_reuse_detail(text)
        current_label = self._current_label
        visible = current_label is not None and (
            current_label in self._active_work or current_label in self._visible_expected
        )
        if exact_reuse and not visible:
            return
        prefix = "" if line_owner is None else f"[{line_owner}] "
        self._current_printer().log_message(f"{prefix}{message}", persistent=True)

    def finish(self) -> None:
        """Finalize all phase display sections owned by this session."""
        for printer in self._printers.values():
            printer.finish()


class NullPhasePrinter:
    """No op PhasePrinter used when phase output is suppressed."""

    def announce(self, label: str, detail: str | None = None) -> None:
        """Silently discard the phase heading."""

    def expect_visible(self, label: str) -> None:
        """Silently discard visible phase expectation."""

    def status(self, message: str, *, owner: str | None = None) -> None:
        """Silently discard one transient phase status line."""

    def show(self, message: str) -> None:
        """Silently discard one transient status message."""

    def log_message(self, message: str, *, persistent: bool = True) -> None:
        """Silently discard one progress compatible line."""

    def clear_transient(self) -> None:
        """Silently discard the transient clear request."""

    def complete(self, detail: str, *, owner: str | None = None) -> None:
        """Silently discard the completion line."""

    def finish(self) -> None:
        """Silently discard the finalization request."""

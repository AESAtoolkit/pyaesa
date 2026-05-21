"""Render progress feedback for long year indexed deterministic workflows.

This module provides the shared progress printer used by package routines that
iterate through ordered study years and need compact, deterministic status
updates without changing the scientific computation itself.
"""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Callable, cast

from pyaesa.shared.runtime.reporting.display import (
    DEFAULT_IPY_SUPPORT,
    default_ipython_support,
    format_hms,
    format_hms_or_na,
    render_ipy_text,
    short_source,
    supports_ipy_display,
    supports_live_refresh,
    supports_multiline_rerender,
    write_to_stream,
)
from pyaesa.shared.runtime.text import compact_user_text, wrap_user_text_lines


@dataclass
class YearProgressPrinter:
    """Render one dynamic terminal line for a year-by-year loop.

    The line is refreshed every second while a year is running so elapsed time
    and ETA are live without requiring the caller to emit heartbeat updates.
    """

    source: str
    action: str
    total: int
    show_timing: bool = True
    approx_eta: bool = False
    use_ipy_display: bool = True
    stream: Any = field(default=None, repr=False)
    env: Mapping[str, str] | None = field(default=None, repr=False)
    clock: Callable[[], float] = field(default=perf_counter, repr=False)
    event_factory: Callable[[], Event] = field(default=Event, repr=False)
    thread_factory: Callable[..., Thread] = field(default=Thread, repr=False)
    get_ipython_func: Callable[[], Any | None] | None | object = field(
        default=DEFAULT_IPY_SUPPORT,
        repr=False,
    )
    ipy_display_handle_cls: type[Any] | None | object = field(
        default=DEFAULT_IPY_SUPPORT,
        repr=False,
    )
    ipy_html_cls: type[Any] | None | object = field(default=DEFAULT_IPY_SUPPORT, repr=False)
    _start: float = field(default=0.0, init=False)
    _elapsed_completed: float = field(default=0.0, init=False)
    _done: int = field(default=0, init=False)
    _done_timed: int = field(default=0, init=False)
    _active: bool = field(default=False, init=False)
    _year_start: float | None = field(default=None, init=False)
    _year_active: Any | None = field(default=None, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _ticker_stop: Event | None = field(default=None, init=False, repr=False)
    _ticker_thread: Thread | None = field(default=None, init=False, repr=False)
    _last_line1_width: int = field(default=0, init=False)
    _last_line2_width: int = field(default=0, init=False)
    _last_single_width: int = field(default=0, init=False)
    _live_refresh: bool = field(default=False, init=False)
    _ipy_display: bool = field(default=False, init=False)
    _ipy_handle: Any | None = field(default=None, init=False, repr=False)
    _multiline_rerender: bool = field(default=False, init=False)
    _message_history: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize effective refresh mode for terminal and notebook outputs."""
        self._start = self.clock()
        if self.stream is None:
            self.stream = sys.stdout
        if self.env is None:
            self.env = os.environ
        self._live_refresh = supports_live_refresh(self.stream)
        self._resolve_default_ipy_support()
        self._ipy_display = supports_ipy_display(
            get_ipython_func=self.get_ipython_func,
            display_handle_cls=self.ipy_display_handle_cls,
        )
        self._multiline_rerender = supports_multiline_rerender(
            stream=self.stream,
            env=self.env,
        )
        if not self.use_ipy_display:
            self._ipy_display = False
            return
        if self._ipy_display:
            self._live_refresh = True
            self._multiline_rerender = False

    def _render_ipy(self, text: str) -> None:
        """Update one notebook output block in place."""
        display_handle_cls: type[Any] | None = (
            None
            if self.ipy_display_handle_cls is DEFAULT_IPY_SUPPORT
            else cast(type[Any] | None, self.ipy_display_handle_cls)
        )
        html_cls: type[Any] | None = (
            None
            if self.ipy_html_cls is DEFAULT_IPY_SUPPORT
            else cast(type[Any] | None, self.ipy_html_cls)
        )
        handle, rendered = render_ipy_text(
            text=text,
            handle=self._ipy_handle,
            display_handle_cls=display_handle_cls,
            html_cls=html_cls,
        )
        if rendered:
            self._ipy_handle = handle
            return
        self._ipy_display = False
        self._last_single_width = max(self._last_single_width, len(text))
        padded = text.ljust(self._last_single_width)
        write_to_stream(self.stream, f"\r{padded}")

    def _resolve_default_ipy_support(self) -> None:
        """Load optional IPython helpers only for progress printer instances."""
        if (
            self.get_ipython_func is not DEFAULT_IPY_SUPPORT
            and self.ipy_display_handle_cls is not DEFAULT_IPY_SUPPORT
            and self.ipy_html_cls is not DEFAULT_IPY_SUPPORT
        ):
            return
        get_ipython_func, display_handle_cls, html_cls = default_ipython_support()
        if self.get_ipython_func is DEFAULT_IPY_SUPPORT:
            self.get_ipython_func = get_ipython_func
        if self.ipy_display_handle_cls is DEFAULT_IPY_SUPPORT:
            self.ipy_display_handle_cls = display_handle_cls
        if self.ipy_html_cls is DEFAULT_IPY_SUPPORT:
            self.ipy_html_cls = html_cls

    @staticmethod
    def _format_unit_label(year: Any) -> str:
        """Return one stable progress label for a year or batch identifier."""
        try:
            return str(int(year))
        except (TypeError, ValueError):
            return compact_user_text(year, max_chars=48)

    def _render(
        self,
        *,
        year: Any,
        progressed_done: int,
        timed_done: int,
        shown_done: int,
    ) -> None:
        source_label = short_source(self.source)
        unit_label = self._format_unit_label(year)
        if not self.show_timing:
            simple_line = (
                f"[{source_label}] {self.action} {unit_label} {int(shown_done)}/{int(self.total)}"
            )
            if self._ipy_display:
                self._render_ipy("\n".join([*self._message_history, simple_line]))
                self._active = True
                return
            self._last_single_width = max(self._last_single_width, len(simple_line))
            padded = simple_line.ljust(self._last_single_width)
            write_to_stream(self.stream, f"\r{padded}")
            self._active = True
            return

        in_active_year = self._year_start is not None and self._year_active == year
        current_year_elapsed = (
            self.clock() - self._year_start if in_active_year and self._year_start else None
        )
        elapsed = self.clock() - self._start
        avg = self._elapsed_completed / float(timed_done) if timed_done > 0 else None
        remaining = max(0, int(self.total) - int(progressed_done))
        years_after_current = max(0, remaining - 1)
        if avg is None:
            eta = None
        else:
            current_left = 0.0
            if current_year_elapsed is not None:
                current_left = max(0.0, float(avg) - float(current_year_elapsed))
            eta = current_left + float(avg) * float(years_after_current)
        eta_label = "eta~" if self.approx_eta else "eta"
        line1 = (
            f"[{source_label}] {self.action} {unit_label} "
            f"{int(shown_done)}/{int(self.total)} | "
            f"yr={format_hms(current_year_elapsed)} "
            f"avg/yr={format_hms_or_na(avg)} "
        )
        line2 = f"elapsed={format_hms(elapsed)} {eta_label}={format_hms_or_na(eta)}"
        single_line = (
            f"[{source_label}] {self.action} {unit_label} "
            f"{int(shown_done)}/{int(self.total)} | "
            f"avg/yr={format_hms_or_na(avg)} "
            f"{eta_label}={format_hms_or_na(eta)}"
        )
        if self._ipy_display:
            self._render_ipy("\n".join([*self._message_history, f"{line1}\n{line2}"]))
            self._active = True
            return
        if not self._multiline_rerender:
            self._last_single_width = max(self._last_single_width, len(single_line))
            padded = single_line.ljust(self._last_single_width)
            write_to_stream(self.stream, f"\r{padded}")
            self._active = True
            return
        self._last_line1_width = max(self._last_line1_width, len(line1))
        self._last_line2_width = max(self._last_line2_width, len(line2))
        padded1 = line1.ljust(self._last_line1_width)
        padded2 = line2.ljust(self._last_line2_width)
        if self._active:
            write_to_stream(self.stream, "\r\x1b[2K\x1b[1A\r\x1b[2K")
            write_to_stream(self.stream, f"{padded1}\n{padded2}")
        else:
            write_to_stream(self.stream, f"\r{padded1}\n{padded2}")
        self._active = True

    def _stop_ticker(self) -> None:
        """Stop live refresh ticker if present."""
        with self._lock:
            stop = self._ticker_stop
            thread = self._ticker_thread
            self._ticker_stop = None
            self._ticker_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.2)

    def _ticker_loop(self, stop: Event) -> None:
        """Refresh live progress once per second for the active year."""
        while not stop.wait(1.0):
            with self._lock:
                if self.total <= 0 or self._year_active is None or self._year_start is None:
                    continue
                shown_done = min(int(self.total), int(self._done) + 1)
                self._render(
                    year=self._year_active,
                    progressed_done=int(self._done),
                    timed_done=int(self._done_timed),
                    shown_done=shown_done,
                )

    def _start_ticker(self) -> None:
        """Start live refresh ticker for the active year."""
        if not self._live_refresh:
            return
        self._stop_ticker()
        with self._lock:
            if self.total <= 0 or self._year_active is None or self._year_start is None:
                return
            stop = self.event_factory()
            thread = self.thread_factory(
                target=self._ticker_loop,
                args=(stop,),
                daemon=True,
            )
            self._ticker_stop = stop
            self._ticker_thread = thread
        thread.start()

    def begin_year(self, year: Any) -> None:
        """Show the currently running year or batch with its 1-based run position."""
        if self.total <= 0:
            return
        with self._lock:
            self._year_start = self.clock()
            self._year_active = year
            shown_done = min(int(self.total), int(self._done) + 1)
            self._render(
                year=year,
                progressed_done=self._done,
                timed_done=self._done_timed,
                shown_done=shown_done,
            )
        self._start_ticker()

    def complete_year(self, year: Any) -> None:
        """Mark one year or batch complete and update the dynamic line."""
        if self.total <= 0:
            return
        self._stop_ticker()
        with self._lock:
            if self._year_start is not None:
                self._elapsed_completed += max(0.0, self.clock() - self._year_start)
            self._done = min(int(self.total), int(self._done) + 1)
            self._done_timed += 1
            if not self.show_timing:
                self._year_start = None
                self._year_active = None
                return
            self._render(
                year=year,
                progressed_done=self._done,
                timed_done=self._done_timed,
                shown_done=self._done,
            )
            self._year_start = None
            self._year_active = None

    def skip_year(self) -> None:
        """Advance internal position for a skipped year without printing."""
        if self.total <= 0:
            return
        with self._lock:
            self._done = min(int(self.total), int(self._done) + 1)

    def log_message(self, message: str, *, persistent: bool = True) -> None:
        """Print a message while preserving progress rendering semantics."""
        self._stop_ticker()
        resume_live_line = False
        resume_year: Any = 0
        resume_shown_done = 0
        with self._lock:
            if persistent:
                message_lines = wrap_user_text_lines(str(message).split("\n"))
                message_text = "\n".join(message_lines)
                active_year = (
                    self.total > 0
                    and self._year_active is not None
                    and self._year_start is not None
                )
                if self._ipy_display:
                    self._message_history.extend(message_lines)
                    if active_year:
                        resume_year = self._year_active
                        resume_shown_done = min(int(self.total), int(self._done) + 1)
                        self._render(
                            year=resume_year,
                            progressed_done=int(self._done),
                            timed_done=int(self._done_timed),
                            shown_done=resume_shown_done,
                        )
                        resume_live_line = bool(self._live_refresh)
                    else:
                        self._render_ipy("\n".join(self._message_history))
                        self._active = False
                else:
                    if self._active:
                        if self._multiline_rerender and self._last_line2_width > 0:
                            write_to_stream(self.stream, "\r\x1b[2K\x1b[1A\r\x1b[2K")
                        else:
                            clear_width = max(
                                1,
                                int(self._last_single_width or 0),
                                int(self._last_line1_width or 0),
                                int(self._last_line2_width or 0),
                            )
                            write_to_stream(self.stream, f"\r{' ' * clear_width}\r")
                        self._active = False
                        self._last_line1_width = 0
                        self._last_line2_width = 0
                        self._last_single_width = 0
                        self._ipy_handle = None
                    write_to_stream(self.stream, message_text, end="\n")
                    if active_year and (self._live_refresh or self._ipy_display):
                        resume_live_line = True
                        resume_year = self._year_active
                        resume_shown_done = min(int(self.total), int(self._done) + 1)
                        self._render(
                            year=resume_year,
                            progressed_done=int(self._done),
                            timed_done=int(self._done_timed),
                            shown_done=resume_shown_done,
                        )
            else:
                text = compact_user_text(message)
                if self._ipy_display:
                    self._render_ipy("\n".join([*self._message_history, text]))
                    self._active = True
                else:
                    self._last_single_width = max(self._last_single_width, len(text))
                    padded = text.ljust(self._last_single_width)
                    write_to_stream(self.stream, f"\r{padded}")
                    self._active = True
                if (
                    self.total > 0
                    and self._year_active is not None
                    and self._year_start is not None
                    and self._live_refresh
                ):
                    resume_live_line = True
        if resume_live_line:
            self._start_ticker()

    def show(self, message: str) -> None:
        """Render one transient status message on this progress line."""
        self.log_message(message, persistent=False)

    def clear_transient(self) -> None:
        """Clear the current transient line while keeping persistent history."""
        self._stop_ticker()
        with self._lock:
            if not self._active:
                return
            if self._ipy_display:
                self._render_ipy("\n".join(self._message_history))
            else:
                clear_width = max(
                    1,
                    int(self._last_single_width or 0),
                    int(self._last_line1_width or 0),
                    int(self._last_line2_width or 0),
                )
                write_to_stream(self.stream, f"\r{' ' * clear_width}\r")
            self._active = False
            self._last_line1_width = 0
            self._last_line2_width = 0
            self._last_single_width = 0
            if not self._ipy_display:
                self._ipy_handle = None

    def finish(self) -> None:
        """Finalize the dynamic line and append one newline if needed."""
        self._stop_ticker()
        with self._lock:
            if self._active:
                if not self._ipy_display:
                    write_to_stream(self.stream, "", end="\n")
                self._active = False
                self._last_line1_width = 0
                self._last_line2_width = 0
                self._last_single_width = 0
                self._ipy_handle = None
            self._message_history = []


@dataclass
class StatusProgressPrinter:
    """Expose year progress semantics through an existing status sink.

    This adapter lets long deterministic loops share the caller's phase display
    instead of creating an independent notebook or terminal cell.
    """

    source: str
    action: str
    total: int
    status: Any
    _done: int = field(default=0, init=False)

    def _line(self, year: Any, shown_done: int) -> str:
        source_label = short_source(self.source)
        unit_label = YearProgressPrinter._format_unit_label(year)
        return f"[{source_label}] {self.action} {unit_label} {shown_done}/{int(self.total)}"

    def begin_year(self, year: Any) -> None:
        """Show the currently running year or batch in the owning status sink."""
        if self.total <= 0:
            return
        shown_done = min(int(self.total), int(self._done) + 1)
        self.status.show(self._line(year, shown_done))

    def complete_year(self, year: Any) -> None:
        """Mark one year or batch complete without closing the owning sink."""
        if self.total <= 0:
            return
        self._done = min(int(self.total), int(self._done) + 1)

    def skip_year(self) -> None:
        """Advance internal position for a skipped year without printing."""
        if self.total <= 0:
            return
        self._done = min(int(self.total), int(self._done) + 1)

    def log_message(self, message: str, *, persistent: bool = True) -> None:
        """Render one message through the owning status sink."""
        log_message = getattr(self.status, "log_message", None)
        if persistent and callable(log_message):
            log_message(str(message), persistent=True)
            return
        self.status.show(str(message))

    def show(self, message: str) -> None:
        """Render one transient status message."""
        self.log_message(message, persistent=False)

    def clear_transient(self) -> None:
        """Clear the transient line owned by the wrapped status sink."""
        self.status.clear_transient()

    def finish(self) -> None:
        """Leave the owning status sink open for later phase messages."""

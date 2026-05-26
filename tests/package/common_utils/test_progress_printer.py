import io
import sys
from itertools import count
from typing import Any, cast

from pyaesa.shared.runtime.reporting import progress as mod
from pyaesa.shared.runtime.text import (
    compact_user_text,
    extend_user_text_lines,
    join_user_text_lines,
    print_user_text_line,
    wrap_user_text_lines,
)


class ZMQInteractiveShell:
    pass


class _TTYStream(io.StringIO):
    def __init__(self, *, is_tty: bool) -> None:
        super().__init__()
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


class _SequenceClock:
    def __init__(self, *, start: float = 0.0, step: float = 1.0) -> None:
        self._values = count(start, step)

    def __call__(self) -> float:
        return float(next(self._values))


class _FakeEvent:
    def __init__(self, *, stop_after: int = 1) -> None:
        self._stop_after = stop_after
        self._calls = 0
        self.set_called = False

    def set(self) -> None:
        self.set_called = True

    def wait(self, _seconds: float) -> bool:
        self._calls += 1
        return self._calls > self._stop_after


class _FakeThread:
    def __init__(self, target=None, args=(), daemon=None) -> None:
        self._target = target
        self._args = args
        self.daemon = daemon
        self.started = False
        self.joined = False
        self._alive = True

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout=None) -> None:
        self.joined = True
        self._alive = False


class _GoodHandle:
    def __init__(self) -> None:
        self.display_calls = 0
        self.update_calls = 0

    def display(self, _obj) -> None:
        self.display_calls += 1

    def update(self, _obj) -> None:
        self.update_calls += 1


class _BadHandle:
    def display(self, _obj) -> None:
        raise RuntimeError("display failed")

    def update(self, _obj) -> None:
        raise RuntimeError("update failed")


class _DummyHTML:
    def __init__(self, text: str) -> None:
        self.text = text


def test_terminal_render_and_message_paths() -> None:
    default_stream_printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=0,
        use_ipy_display=False,
    )
    assert default_stream_printer.stream is sys.stdout

    stream = _TTYStream(is_tty=True)
    clock = _SequenceClock()
    created_events: list[_FakeEvent] = []
    created_threads: list[_FakeThread] = []

    def event_factory() -> _FakeEvent:
        event = _FakeEvent(stop_after=2)
        created_events.append(event)
        return event

    def thread_factory(*args, **kwargs) -> _FakeThread:
        thread = _FakeThread(*args, **kwargs)
        created_threads.append(thread)
        return thread

    printer = mod.YearProgressPrinter(
        source="oecd_v2025",
        action="processing",
        total=2,
        use_ipy_display=False,
        stream=stream,
        env={"TERM": "xterm"},
        clock=clock,
        event_factory=cast(Any, event_factory),
        thread_factory=cast(Any, thread_factory),
    )
    printer.begin_year(2020)
    printer.log_message("hello")
    printer.complete_year(2020)
    printer.skip_year()
    printer.finish()

    output = stream.getvalue()
    assert "hello" in output
    assert "\x1b[2K" in output
    assert created_events and created_events[0].set_called is True
    assert created_threads and created_threads[0].started is True
    assert created_threads[0].joined is True

    single_line_stream = _TTYStream(is_tty=True)
    single_line_printer = mod.YearProgressPrinter(
        source="very_long_source_name_for_test",
        action="processing",
        total=3,
        use_ipy_display=False,
        stream=single_line_stream,
        env={"TERM": "dumb"},
        clock=_SequenceClock(),
    )
    single_line_printer.begin_year(2021)
    single_line_printer.log_message("status", persistent=False)
    single_line_printer.complete_year(2021)
    single_line_printer.finish()
    assert single_line_stream.getvalue().strip()

    simple_stream = _TTYStream(is_tty=False)
    simple_printer = mod.YearProgressPrinter(
        source="oecd",
        action="downloading",
        total=1,
        show_timing=False,
        use_ipy_display=False,
        stream=simple_stream,
        clock=_SequenceClock(),
    )
    simple_printer.begin_year(2022)
    simple_printer.complete_year(2022)
    simple_printer.finish()
    simple_lines = [line for line in simple_stream.getvalue().splitlines() if line.strip()]
    assert len(simple_lines) == 1

    bundled_stream = _TTYStream(is_tty=False)
    bundled_printer = mod.YearProgressPrinter(
        source="oecd",
        action="downloading",
        total=1,
        show_timing=False,
        use_ipy_display=False,
        stream=bundled_stream,
        clock=_SequenceClock(),
    )
    bundled_printer.begin_year("1995-2000")
    bundled_printer.finish()
    assert bundled_stream.getvalue().strip()

    start_guard_printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=1,
        use_ipy_display=False,
        stream=_TTYStream(is_tty=True),
        env={"TERM": "xterm"},
        clock=_SequenceClock(start=1.0),
    )
    start_guard_printer._start_ticker()
    assert start_guard_printer._ticker_thread is None

    no_start_printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=1,
        use_ipy_display=False,
        stream=_TTYStream(is_tty=False),
        clock=_SequenceClock(start=1.0),
    )
    no_start_printer.complete_year(2025)
    assert no_start_printer._done == 1
    assert no_start_printer._done_timed == 1

    passive_log_printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=1,
        use_ipy_display=False,
        stream=_TTYStream(is_tty=False),
        clock=_SequenceClock(start=1.0),
    )
    passive_log_printer.log_message("plain-terminal")
    assert "plain-terminal" in passive_log_printer.stream.getvalue()


def test_progress_printer_covers_simple_ipy_eta_and_terminal_clear_paths() -> None:
    simple_ipy = mod.YearProgressPrinter(
        source="oecd",
        action="downloading",
        total=2,
        show_timing=False,
        use_ipy_display=True,
        stream=_TTYStream(is_tty=False),
        clock=_SequenceClock(start=1.0),
        get_ipython_func=lambda: ZMQInteractiveShell(),
        ipy_display_handle_cls=_GoodHandle,
        ipy_html_cls=_DummyHTML,
    )
    simple_ipy.begin_year(2028)
    assert isinstance(simple_ipy._ipy_handle, _GoodHandle)
    simple_ipy.finish()

    mod.YearProgressPrinter(
        source="oecd",
        action="downloading",
        total=1,
        use_ipy_display=False,
        stream=_TTYStream(is_tty=False),
        get_ipython_func=lambda: None,
    )
    mod.YearProgressPrinter(
        source="oecd",
        action="downloading",
        total=1,
        use_ipy_display=False,
        stream=_TTYStream(is_tty=False),
        ipy_display_handle_cls=_GoodHandle,
    )
    mod.YearProgressPrinter(
        source="oecd",
        action="downloading",
        total=1,
        use_ipy_display=False,
        stream=_TTYStream(is_tty=False),
        ipy_html_cls=_DummyHTML,
    )

    eta_stream = _TTYStream(is_tty=True)
    eta_printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=3,
        approx_eta=True,
        use_ipy_display=False,
        stream=eta_stream,
        env={"TERM": "xterm"},
        clock=_SequenceClock(start=0.0, step=5.0),
        event_factory=cast(Any, lambda: _FakeEvent(stop_after=1)),
        thread_factory=cast(Any, _FakeThread),
    )
    eta_printer.begin_year(2020)
    eta_printer.complete_year(2020)
    eta_printer.begin_year(2021)
    assert eta_stream.getvalue().strip()
    eta_printer.finish()

    clear_stream = _TTYStream(is_tty=True)
    clear_printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=2,
        use_ipy_display=False,
        stream=clear_stream,
        env={"TERM": "dumb"},
        clock=_SequenceClock(start=1.0),
    )
    clear_printer.begin_year(2029)
    clear_printer.log_message("cleared")
    assert clear_stream.getvalue().strip()
    clear_printer.finish()

    no_live = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=1,
        use_ipy_display=False,
        stream=_TTYStream(is_tty=False),
        clock=_SequenceClock(start=1.0),
    )
    no_live._start_ticker()
    assert no_live._ticker_thread is None


def test_ipy_render_paths_and_ticker_loop() -> None:
    ipy_stream = _TTYStream(is_tty=False)
    clock = _SequenceClock()
    printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=2,
        use_ipy_display=True,
        stream=ipy_stream,
        clock=clock,
        get_ipython_func=lambda: ZMQInteractiveShell(),
        ipy_display_handle_cls=_GoodHandle,
        ipy_html_cls=_DummyHTML,
    )
    printer._render_ipy("line-1")
    handle = printer._ipy_handle
    assert isinstance(handle, _GoodHandle)
    printer._render_ipy("line-2")
    assert handle.display_calls == 1
    assert handle.update_calls == 1

    printer.begin_year(2023)
    printer.log_message("note")
    printer.log_message("status", persistent=False)
    assert printer._message_history == ["note"]
    assert printer._active is True
    printer.clear_transient()
    assert printer._ipy_handle is handle
    printer.log_message("after-clear")
    assert handle.display_calls == 1
    assert handle.update_calls >= 3
    printer.finish()
    assert printer._message_history == []

    inactive_printer = mod.YearProgressPrinter(
        source="oecd",
        action="downloading",
        total=1,
        show_timing=False,
        use_ipy_display=True,
        stream=_TTYStream(is_tty=False),
        clock=_SequenceClock(start=1.0),
        get_ipython_func=lambda: ZMQInteractiveShell(),
        ipy_display_handle_cls=_GoodHandle,
        ipy_html_cls=_DummyHTML,
    )
    inactive_printer.log_message("queued")
    assert inactive_printer._message_history == ["queued"]
    inactive_printer.begin_year(2026)
    assert inactive_printer._active is True

    fallback_stream = _TTYStream(is_tty=False)
    fallback_printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=1,
        use_ipy_display=True,
        stream=fallback_stream,
        clock=_SequenceClock(),
        get_ipython_func=lambda: ZMQInteractiveShell(),
        ipy_display_handle_cls=None,
        ipy_html_cls=None,
    )
    fallback_printer._render_ipy("fallback")
    assert fallback_printer._ipy_display is False
    assert "fallback" in fallback_stream.getvalue()

    bad_stream = _TTYStream(is_tty=False)
    bad_printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=1,
        use_ipy_display=True,
        stream=bad_stream,
        clock=_SequenceClock(),
        get_ipython_func=lambda: ZMQInteractiveShell(),
        ipy_display_handle_cls=_BadHandle,
        ipy_html_cls=_DummyHTML,
    )
    bad_printer._render_ipy("boom")
    assert bad_printer._ipy_display is False
    assert "boom" in bad_stream.getvalue()

    ticker_printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=2,
        use_ipy_display=False,
        stream=_TTYStream(is_tty=False),
        clock=_SequenceClock(start=1.0),
    )
    ticker_printer.begin_year(2024)
    ticker_printer._ticker_loop(cast(Any, _FakeEvent(stop_after=1)))
    assert "2024" in ticker_printer.stream.getvalue()

    idle_printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=2,
        use_ipy_display=False,
        stream=_TTYStream(is_tty=False),
        clock=_SequenceClock(),
    )
    idle_printer._ticker_loop(cast(Any, _FakeEvent(stop_after=1)))
    assert idle_printer._done == 0


def test_zero_total_and_inactive_finish_paths() -> None:
    zero_stream = _TTYStream(is_tty=False)
    zero_printer = mod.YearProgressPrinter(
        source="x",
        action="a",
        total=0,
        use_ipy_display=False,
        stream=zero_stream,
        clock=_SequenceClock(),
    )
    zero_printer.begin_year(2020)
    zero_printer.complete_year(2020)
    zero_printer.skip_year()
    zero_printer.finish()
    zero_printer.finish()
    assert zero_stream.getvalue() == ""

    active_stream = _TTYStream(is_tty=False)
    active_printer = mod.YearProgressPrinter(
        source="x",
        action="a",
        total=1,
        use_ipy_display=False,
        stream=active_stream,
        clock=_SequenceClock(),
    )
    active_printer.show("clear me")
    active_printer.clear_transient()
    assert active_printer._active is False
    assert active_printer._ipy_handle is None
    assert "\r" in active_stream.getvalue()


def test_terminal_transient_status_resumes_live_line() -> None:
    stream = _TTYStream(is_tty=True)
    created_events: list[_FakeEvent] = []

    def event_factory() -> _FakeEvent:
        event = _FakeEvent(stop_after=1)
        created_events.append(event)
        return event

    printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=2,
        use_ipy_display=False,
        stream=stream,
        env={"TERM": "xterm"},
        clock=_SequenceClock(start=1.0),
        event_factory=cast(Any, event_factory),
        thread_factory=cast(Any, _FakeThread),
    )
    printer.begin_year(2027)
    printer.log_message("transient", persistent=False)
    assert len(created_events) == 2
    assert created_events[0].set_called is True
    assert created_events[1].set_called is False


def test_terminal_transient_status_without_active_year_does_not_restart_ticker() -> None:
    stream = _TTYStream(is_tty=False)
    printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=1,
        use_ipy_display=False,
        stream=stream,
        clock=_SequenceClock(start=1.0),
    )
    printer.log_message("idle", persistent=False)
    assert printer._ticker_thread is None
    assert "idle" in stream.getvalue()


def test_persistent_messages_are_wrapped_for_terminal_and_notebook() -> None:
    long_message = (
        "Persistent progress messages are stored in user visible terminal and "
        "workbook output history, so they should wrap before being persisted."
    )

    stream = _TTYStream(is_tty=False)
    terminal_printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=1,
        use_ipy_display=False,
        stream=stream,
        clock=_SequenceClock(start=1.0),
    )
    terminal_printer.log_message(long_message)
    terminal_lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(terminal_lines) > 1
    assert all(len(line) <= 100 for line in terminal_lines)

    notebook_printer = mod.YearProgressPrinter(
        source="oecd",
        action="processing",
        total=1,
        use_ipy_display=True,
        stream=_TTYStream(is_tty=False),
        clock=_SequenceClock(start=1.0),
        get_ipython_func=lambda: ZMQInteractiveShell(),
        ipy_display_handle_cls=_GoodHandle,
        ipy_html_cls=_DummyHTML,
    )
    notebook_printer.log_message(long_message)
    assert len(notebook_printer._message_history) > 1
    assert all(len(line) <= 100 for line in notebook_printer._message_history)


def test_user_text_helpers_wrap_and_compact(capsys) -> None:
    wrapped = wrap_user_text_lines(
        [
            "  - alpha beta gamma delta",
            "",
            "plain alpha beta gamma",
        ],
        width=16,
    )
    assert wrapped == [
        "  - alpha beta",
        "    gamma delta",
        "",
        "plain alpha beta",
        "gamma",
    ]

    assert join_user_text_lines(["alpha beta gamma"], width=12) == "alpha beta\ngamma"
    assert join_user_text_lines(["alpha beta"], width=12, trailing_newline=True).endswith("\n")

    target: list[str] = []
    extend_user_text_lines(target, "- alpha beta gamma", width=10)
    assert target == ["- alpha", "  beta", "  gamma"]

    print_user_text_line("alpha beta gamma", width=10)
    assert capsys.readouterr().out == "alpha beta\ngamma\n"

    assert compact_user_text(" alpha   beta ") == "alpha beta"
    assert compact_user_text("abcdef", max_chars=2) == "ab"
    assert compact_user_text("abcdef", max_chars=5) == "ab..."
